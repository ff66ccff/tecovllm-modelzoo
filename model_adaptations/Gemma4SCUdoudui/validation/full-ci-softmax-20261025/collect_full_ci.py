"""Archive the terminal CI with source/protected/library identities; never rerun it."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil

base = Path('/root/tmpwork/fix-20261024/models')
run = base / 'runs/pr3-ci-softmax-out-20261025'
public = base / 'pr3/model_adaptations/Gemma4SCUdoudui'
ci = base / 'pr3/tools/ci_pipline'
target = base / 'full-ci-softmax-final-receipts'
result = json.loads((run / 'result.json').read_text())
assert result['tag'] == run.name
assert not target.exists(), 'Do not overwrite an earlier receipt archive'
target.mkdir()

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save_raw(source, destination):
    data = source.read_bytes()
    record = dict(original_path=str(source), sha256=hashlib.sha256(data).hexdigest(),
                  raw_utf8=data.decode('utf-8'),
                  format='Verbatim UTF-8 bytes encoded as a JSON string; decode raw_utf8 to restore')
    destination.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')

before = json.loads((run / 'protected-before.json').read_text())
after = json.loads((run / 'protected-after.json').read_text())
assert before == after
assert all(digest(base / 'pr3' / path) == sha for path, sha in before.items())
runtime = json.loads((base / 'softmax-runtime-source-manifest.json').read_text())
assert all(digest(base / 'pr3' / path) == sha for path, sha in runtime.items())
assert digest(public / 'runtime/custom_ops/prefill_attention/op.py') == (
    'ac2a3b35d264110b87de2c5c5a6c1b33bb8a985c88d2ab79dd8a42545b86664d')
for name in ('result.json', 'run-configuration.json', 'protected-before.json', 'protected-after.json'):
    shutil.copy2(run / name, target / name)
(target / 'runtime-source-manifest.json').write_text(json.dumps(runtime, indent=2) + '\n')
save_raw(run / 'run_ci.log', target / 'run_ci.log.json')
save_raw(ci / 'vllm_serve.log', target / 'vllm_serve.log.json')
module = importlib.util.spec_from_file_location('immutable_result_parser', ci / 'result_analyse.py')
parser = importlib.util.module_from_spec(module)
module.loader.exec_module(parser)
receipt = dict(passed=False, clean_container=False, performance_claim=False,
               result=result, protected_scripts_unchanged=True, runtime_matches=True,
               parser_sha256=digest(ci / 'result_analyse.py'),
               precision_configuration=dict(dataset='mmlu_pro', subset='computer science',
                                            limit=200, eval_batch_size=8, do_sample=True,
                                            temperature=1.0, max_tokens=1024,
                                            scope='Original protected prec.sh, distinct from greedy32 gate'))
try:
    configuration = json.loads((run / 'run-configuration.json').read_text())
    installed = Path(configuration['ops_install']) / 'tecoops'
    extensions = list(installed.glob('_torch_ext*.so'))
    assert len(extensions) == 1
    actual_core = digest(installed / 'libteco_ops.so')
    actual_extension = digest(extensions[0])
    assert actual_core == configuration['source_manifest']['core_sha256']
    assert actual_extension == configuration['source_manifest']['extension_sha256']
    receipt['installed_artifacts_after_ci'] = dict(core_sha256=actual_core,
                                                  extension_sha256=actual_extension)
    precision_dirs = sorted((ci / 'prec_logs').glob('gemma-4-12B-it_*'))
    speed_dirs = sorted((ci / 'speed_logs').glob('gemma-4-12B-it_*'))
    assert len(precision_dirs) == 1 and len(speed_dirs) == 1
    receipt['precision_directory'] = str(precision_dirs[0])
    receipt['speed_directory'] = str(speed_dirs[0])
    receipt['precision'] = parser.parse_precision_dir(str(precision_dirs[0]))
    receipt['performance'] = parser.parse_performance_dir(str(speed_dirs[0]))
    precision_log = (precision_dirs[0] / 'mmlu_pro.log').read_text()
    counts = re.findall(r'Evaluating\[mmlu_pro\][^\r\n]*?\b(\d+)/200', precision_log)
    assert counts and int(counts[-1]) == 200, 'Must complete original 200 questions'
    receipt['completed_precision_questions'] = 200
    receipt['raw_total_success_failed'] = {}
    for log in speed_dirs[0].glob('*.log'):
        summary = [line for line in log.read_text().splitlines() if 'Total / Success / Failed' in line]
        assert len(summary) == 1, log
        counts = tuple(parser._extract_total_success_failed(summary[0]))
        required = (1, 1, 0) if log.stem == 'warmup' else (10, 10, 0)
        assert counts == required, (log.name, counts)
        receipt['raw_total_success_failed'][log.stem] = counts
    for phase, directory in (('precision', precision_dirs[0]), ('speed', speed_dirs[0])):
        archive = target / phase
        archive.mkdir()
        for log in directory.glob('*.log'):
            save_raw(log, archive / (log.name + '.json'))
    for name in ('precision_result.json', 'performance_result.json'):
        shutil.copy2(ci / name, target / name)
    assert result['passed'] and result['exit_code'] == 0 and result['protected_scripts_unchanged']
    assert '===== CI 完成 =====' in (run / 'run_ci.log').read_text()
    assert 0.73 <= receipt['precision']['mmlu_pro'] <= 0.93
    assert len([name for name in receipt['performance'] if name.startswith('T')]) == 4
    receipt['passed'] = True
finally:
    (target / 'verification.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(receipt, ensure_ascii=False))
