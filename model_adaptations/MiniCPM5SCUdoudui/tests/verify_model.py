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
import time
import urllib.request

p=argparse.ArgumentParser()
p.add_argument('--base',default='http://127.0.0.1:8014')
p.add_argument('--out',required=True)
p.add_argument('--expected')
p.add_argument('--steady',type=float,default=0)
p.add_argument('--concurrency',action='store_true')
p.add_argument('--receipt',action='store_true')
a=p.parse_args()
prompt='Explain why reproducible benchmarks matter.'
expected=json.loads(Path(a.expected).read_text())['expected_token_ids'] if a.expected else None
result=dict(passed=False,prompt=prompt,expected_token_ids=expected,smoke=[],concurrency=[],steady=[])
def save(): Path(a.out).write_text(json.dumps(result,indent=2)+'\n')
def post(path,payload):
    req=urllib.request.Request(a.base+path,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=120) as response: return json.load(response)
def run(label):
    t0=time.monotonic()
    response=post('/v1/completions',dict(model='MiniCPM5-1B',prompt=prompt,max_tokens=32,temperature=0,
                                     top_p=1,top_k=-1,seed=0,return_token_ids=True))
    row=dict(label=label,seconds=time.monotonic()-t0,token_ids=response['choices'][0].get('token_ids'),
             text=response['choices'][0].get('text'),usage=response['usage'])
    assert len(row['token_ids'])==32 and row['usage']['completion_tokens']==32, row
    if expected is not None: assert row['token_ids']==expected, row
    return row
try:
    if a.receipt:
        result['worker_receipt_before']=post('/collective_rpc',{'method':'minicpm_receipt'})
        save()
    for i in range(2):
        row=run('smoke_'+str(i))
        if expected is None:
            expected=row['token_ids'];result['expected_token_ids']=expected
        result['smoke'].append(row);save()
        print(json.dumps(dict(smoke=i,all32_match=True,seconds=row['seconds'])),flush=True)
    if a.concurrency:
        for n in (4,8):
            with ThreadPoolExecutor(max_workers=n) as pool: rows=list(pool.map(run,[f'concurrency{n}_{i}' for i in range(n)]))
            result['concurrency'].append(dict(concurrency=n,requests=rows));save()
            print(json.dumps(dict(concurrency=n,all32_match=True)),flush=True)
    t0=time.monotonic()
    while time.monotonic()-t0<a.steady:
        n=(1,4,8)[len(result['steady'])%3]
        with ThreadPoolExecutor(max_workers=n) as pool: rows=list(pool.map(run,[f'steady{len(result["steady"])}_{i}' for i in range(n)]))
        result['steady'].append(dict(concurrency=n,requests=rows));result['steady_seconds']=time.monotonic()-t0;save()
        print(json.dumps(dict(rounds=len(result['steady']),seconds=result['steady_seconds'],all32_match=True)),flush=True)
    if a.receipt:
        result['worker_receipt']=post('/collective_rpc',{'method':'minicpm_receipt'})
    result['passed']=True
except Exception as error:
    result['error']=repr(error)
    raise
finally: save()
print(json.dumps(dict(passed=True,steady_seconds=result.get('steady_seconds',0),
                      steady_requests=sum(len(r['requests']) for r in result['steady']))),flush=True)
