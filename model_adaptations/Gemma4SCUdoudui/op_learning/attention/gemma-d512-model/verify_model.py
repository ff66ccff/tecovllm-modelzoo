# BSD 3-Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/gemma4'))
from http_probe import completions, DEFAULT_PROMPT

p = argparse.ArgumentParser()
p.add_argument('--out', required=True)
p.add_argument('--base', default='http://127.0.0.1:8001')
p.add_argument('--steady', type=float, default=300)
p.add_argument('--concurrency', action='store_true')
a = p.parse_args()
base = a.base
baseline_file = Path(__file__).with_name('baseline_token_ids.json')
expected = json.loads(baseline_file.read_text())['token_ids']
result = dict(passed=False, prompt=DEFAULT_PROMPT, expected_token_ids=expected,
              baseline_file=str(baseline_file), smoke=[], benchmark=[], concurrency=[], steady=[])

def save():
    Path(a.out).write_text(json.dumps(result, indent=2))

def run(label):
    start = time.monotonic()
    r = completions(base, DEFAULT_PROMPT, 32)
    ch = r['choices'][0]
    row = dict(label=label, seconds=time.monotonic() - start,
               token_ids=ch.get('token_ids'), text=ch.get('text'),
               finish_reason=ch.get('finish_reason'), usage=r.get('usage'))
    assert row['token_ids'] == expected and r['usage']['completion_tokens'] == 32, row
    return row

try:
    for i in range(2):
        result['smoke'].append(run(f'smoke_{i}'))
        save()
        print(json.dumps(result['smoke'][-1]), flush=True)
    for i in range(3):
        result['benchmark'].append(run(f'benchmark_{i}'))
        save()
        print(json.dumps(result['benchmark'][-1]), flush=True)
    result['median_seconds'] = statistics.median(r['seconds'] for r in result['benchmark'])
    if a.concurrency:
        for n in (1, 4, 8):
            t0 = time.monotonic()
            with ThreadPoolExecutor(max_workers=n) as pool:
                requests = list(pool.map(run, [f'concurrency_{n}_{i}' for i in range(n)]))
            row = dict(concurrency=n, wall_seconds=time.monotonic() - t0, requests=requests)
            result['concurrency'].append(row)
            save()
            print(json.dumps(dict(concurrency=n, wall_seconds=row['wall_seconds'], passed=True)), flush=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < a.steady:
        n = (1, 4, 8)[len(result['steady']) % 3]
        with ThreadPoolExecutor(max_workers=n) as pool:
            requests = list(pool.map(run, [f'steady_{len(result["steady"])}_{i}' for i in range(n)]))
        result['steady'].append(dict(concurrency=n, requests=requests))
        result['steady_seconds'] = time.monotonic() - t0
        save()
        print(json.dumps(dict(steady_rounds=len(result['steady']),
                              seconds=result['steady_seconds'], concurrency=n, passed=True)), flush=True)
    result['passed'] = True
except Exception as error:
    result['error'] = repr(error)
    raise
finally:
    save()
print(json.dumps(dict(passed=True, median_seconds=result['median_seconds'],
                      steady_seconds=result.get('steady_seconds', 0),
                      requests=sum(len(r['requests']) for r in result['steady']))), flush=True)
