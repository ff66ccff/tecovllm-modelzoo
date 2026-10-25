"""After the full CI releases its flock, verify shared entry and native memory samples."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

root = Path('/root/tmpwork/fix-20261024/models')
output = root / 'shared-memory-queue-result.json'
result = dict(passed=False, start_time=time.time(), performance_claim=False,
              tag='pr3-shared-memory-gate-20261025')
try:
    previous = json.loads((root / 'softmax-full-ci-queue-result.json').read_text())
    complete = json.loads((root / 'runs/pr3-ci-softmax-out-20261025/result.json').read_text())
    assert previous['passed'] and complete['passed'] and complete['protected_scripts_unchanged']
    source = root / 'pr3/model_adaptations/Gemma4SCUdoudui/runtime/custom_ops/prefill_attention/op.py'
    result['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    assert result['source_sha256'] == 'ac2a3b35d264110b87de2c5c5a6c1b33bb8a985c88d2ab79dd8a42545b86664d'
    print('FULL_CI_PASSED_STARTING_SHARED_MEMORY_GATE', flush=True)
    rc = subprocess.call([sys.executable, '-u', str(root / 'run_shared_memory_gate.py'),
                          '--pr', '3', '--mode', 'gate', '--seconds', '0', '--shared',
                          '--ops-install', '/root/tmpwork/fix-20261024/team/src/install',
                          '--tag', result['tag']])
    assert rc == 0, rc
    run = root / 'runs' / result['tag']
    gate = json.loads((run / 'model-gate.json').read_text())
    frozen = json.loads((root / 'runs/pr3-gate3-20261024/model-gate.json').read_text())
    assert gate['passed'] and all(row['token_ids'] == frozen['rows'][0]['token_ids'] for row in gate['rows'])
    result['frozen_token_ids_identical'] = True
    raw = run / 'device-memory.jsonl'
    samples = [json.loads(line) for line in raw.read_text().splitlines()]
    good = [sample for sample in samples if sample.get('returncode') == 0 and sample.get('stdout')]
    assert good, 'Native teco-smi must produce real readings'
    device = [int(value) for sample in good for value in re.findall(r'^\s*Used\s*:\s*(\d+) MB', sample['stdout'], re.M)]
    processes = [int(value) for sample in good for value in re.findall(r'^\s*Used Memory\s*:\s*(\d+) MB', sample['stdout'], re.M)]
    assert device and max(device) > 0 and processes
    result['memory'] = dict(command=['teco-smi', '-q', '-d', 'MEMORY,PIDS'],
                          valid_samples=len(good), raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                          maximum_sampled_card_used_vendor_MB=max(device),
                          maximum_sampled_process_used_vendor_MB=max(processes),
                          scope='Native vendor-reported sampled peak during this shared-entry startup and greedy32 gate only; not allocator high-water mark or complete CI peak',
                          original_ci_query='--query-device=memory.used returned empty output; no peak inferred from those old samples')
    assert hashlib.sha256(source.read_bytes()).hexdigest() == result['source_sha256']
    result['passed'] = True
finally:
    result['finish_time'] = time.time()
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
if not result['passed']:
    sys.exit(1)
