"""Run fixed greedy32/steady and the unchanged official speed script on one service."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path('/root/tmpwork/fix-20261024/models')
REPO = ROOT / 'pr3'
ADAPT = REPO / 'model_adaptations/Gemma4SCUdoudui'
OUT = ROOT / 'runs/pr3-softmax-out-preflight-20261025-r2'
OUT.mkdir(parents=True, exist_ok=False)
env = os.environ.copy()
env.update(PATH='/home/py312/bin:' + env['PATH'], MODEL_ROOT='/gpfs/model',
           SDAA_VISIBLE_DEVICES='0,1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
           NO_PROXY='localhost,127.0.0.1,0.0.0.0', GEMMA4_EVAL_TOKENIZER='1')
env.pop('SCUD_DECODE_CENSUS', None)
env['PYTHONPATH'] = str(ADAPT / 'eval_overlay') + ':' + str(ROOT.parent / 'team/src/install') + ':' + str(REPO)
env['LD_LIBRARY_PATH'] = '/home/py312/lib/python3.12/site-packages/torch/lib:' + str(ROOT.parent / 'team/src/install/tecoops') + ':' + env.get('LD_LIBRARY_PATH', '')
protected = [REPO / 'tools/ci_pipline' / x for x in
             ('run_ci.py', 'prec.sh', 'speed.sh', 'precision_baseline.json', 'result_analyse.py')]
hashes = lambda: {str(x.relative_to(REPO)): hashlib.sha256(x.read_bytes()).hexdigest() for x in protected}
before = hashes()
(OUT / 'protected-before.json').write_text(json.dumps(before, indent=2) + '\n')
old = json.loads((ROOT / 'runs/pr3-gate3-20261024/model-gate.json').read_text())
(OUT / 'frozen-token-ids.json').write_text(json.dumps(dict(prompt=old['prompt'], expected_token_ids=old['rows'][0]['token_ids']), indent=2) + '\n')
started = time.monotonic()
result = dict(passed=False, performance_claim=False, clean_container=False,
              source_sha256=hashlib.sha256((ADAPT / 'runtime/custom_ops/prefill_attention/op.py').read_bytes()).hexdigest())
server = None
try:
    with (OUT / 'service.log').open('w') as log:
        server = subprocess.Popen(['bash', str(ADAPT / 'run.sh')], cwd=REPO, env=env,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    result['service_pid'] = server.pid
    for _ in range(200):
        if server.poll() is not None:
            raise RuntimeError('service exited before health')
        try:
            with urllib.request.urlopen('http://127.0.0.1:8003/health', timeout=3) as response:
                if response.status == 200:
                    break
        except Exception:
            time.sleep(3)
    else:
        raise TimeoutError('health did not arrive in 600s')
    with (OUT / 'greedy-steady.log').open('w') as log:
        rc = subprocess.call([sys.executable, str(ROOT / 'model_client.py'), '--base', 'http://127.0.0.1:8003',
                              '--model', 'gemma-4-12B-it', '--output', str(OUT / 'model-gate.json'),
                              '--seconds', '300', '--expected', str(OUT / 'frozen-token-ids.json')],
                             cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    result['steady_exit_code'] = rc
    if rc:
        raise RuntimeError('greedy32/steady failed')
    script = REPO / 'tools/ci_pipline'
    with (OUT / 'speed.log').open('w') as log:
        rc = subprocess.call(['bash', str(script / 'speed.sh'), 'gemma-4-12B-it',
                              '/gpfs/model/google/gemma-4-12B-it', '0.0.0.0', '8003'],
                             cwd=script, env=env, stdout=log, stderr=subprocess.STDOUT)
    result['speed_exit_code'] = rc
    if rc:
        raise RuntimeError('official speed script failed')
    speed_dir = max((script / 'speed_logs').glob('gemma-4-12B-it_*'), key=lambda x: x.stat().st_mtime)
    result['speed_directory'] = str(speed_dir)
    spec = importlib.util.spec_from_file_location('official_result_analyse', script / 'result_analyse.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result['official_speed_results'] = module.parse_performance_dir(str(speed_dir))
    result['passed'] = True
except BaseException as exc:
    result['error'] = repr(exc)
    raise
finally:
    if server is not None:
        try:
            os.killpg(server.pid, signal.SIGTERM)
            time.sleep(5)
            os.killpg(server.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    after = hashes()
    (OUT / 'protected-after.json').write_text(json.dumps(after, indent=2) + '\n')
    result.update(wall_seconds=time.monotonic() - started, protected_scripts_unchanged=before == after)
    (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
if not result['passed'] or not result['protected_scripts_unchanged']:
    sys.exit(1)
