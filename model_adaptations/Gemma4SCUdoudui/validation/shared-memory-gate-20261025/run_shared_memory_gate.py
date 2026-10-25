"""Run a single model under the caller's GPU flock; preserve official scripts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.request

ROOT=Path('/root/tmpwork/fix-20261024/models')
MODELS={3:('Gemma4SCUdoudui','gemma-4-12B-it',8003),4:('MiniCPM5SCUdoudui','MiniCPM5-1B',8000),5:('HyMT2SCUdoudui','Hy-MT2-1.8B',8001),6:('InternVL3_5SCUdoudui','InternVL3_5-8B',8002)}
p=argparse.ArgumentParser()
p.add_argument('--pr',type=int,required=True,choices=MODELS)
p.add_argument('--mode',choices=['census','gate','ci'],required=True)
p.add_argument('--ops-install',required=True)
p.add_argument('--seconds',type=int,default=300)
p.add_argument('--tag',required=True)
p.add_argument('--shared', action='store_true')
a=p.parse_args()
assert a.pr == 3 and a.mode == 'gate' and a.shared, 'Dedicated shared-entry gate only'
name,model,port=MODELS[a.pr]
checkout=ROOT/f'pr{a.pr}'
public=checkout/'model_adaptations'/name
entry=checkout/'model_adaptations/SCUdoudui/run.sh'
output=ROOT/'runs'/a.tag
output.mkdir(parents=True,exist_ok=False)
env=os.environ.copy()
env['PATH']='/home/py312/bin:'+env['PATH']
env['MODEL_ROOT']='/gpfs/model'
env['PYTHONPATH']=a.ops_install+':'+str(checkout)
env['HF_HUB_OFFLINE']='1'
env['TRANSFORMERS_OFFLINE']='1'
env['NO_PROXY']='localhost,127.0.0.1,0.0.0.0'
env['SDAA_VISIBLE_DEVICES']='0,1' if a.pr in (3,6) else '0'
env['LD_LIBRARY_PATH']='/home/py312/lib/python3.12/site-packages/torch/lib:'+str(Path(a.ops_install)/'tecoops')+':'+env.get('LD_LIBRARY_PATH','')
if a.mode=='census':env['SCUD_DECODE_CENSUS']=str(output/'census')
else:env.pop('SCUD_DECODE_CENSUS',None)
protected=[checkout/'tools/ci_pipline'/x for x in ['run_ci.py','prec.sh','speed.sh','precision_baseline.json','result_analyse.py']]
before={str(x.relative_to(checkout)):hashlib.sha256(x.read_bytes()).hexdigest() for x in protected}
(output/'protected-before.json').write_text(json.dumps(before,indent=2)+'\n')
(output/'run-configuration.json').write_text(json.dumps(dict(pr=a.pr,mode=a.mode,served_model_name=model,port=port,visible_devices=env['SDAA_VISIBLE_DEVICES'],ops_install=a.ops_install,public=str(public),entry=str(entry),source_manifest=json.loads((public/'validation/ops-artifacts.json').read_text()),clean_container=False),indent=2)+'\n')

def descendants(pid):
    table={}
    for proc in Path('/proc').glob('[0-9]*'):
        try:table[int(proc.name)]=int((proc/'stat').read_text().split(') ',1)[1].split()[1])
        except (OSError,ValueError,IndexError):pass
    found={pid}
    changed=True
    while changed:
        changed=False
        for child,parent in table.items():
            if parent in found and child not in found:found.add(child);changed=True
    return found

process=None
monitor_stop=threading.Event()
def monitor_memory():
    with open(output/'device-memory.jsonl','w') as stream:
        while not monitor_stop.is_set():
            try:
                sample=subprocess.run(['teco-smi','-q','-d','MEMORY,PIDS'],capture_output=True,text=True,timeout=15)
                stream.write(json.dumps(dict(timestamp=time.time(),returncode=sample.returncode,stdout=sample.stdout,stderr=sample.stderr))+'\n')
                stream.flush()
            except Exception as exc:
                stream.write(json.dumps(dict(timestamp=time.time(),error=repr(exc)))+'\n');stream.flush()
            monitor_stop.wait(3)
monitor=threading.Thread(target=monitor_memory,daemon=True)
started=time.monotonic()
status=dict(passed=False,pr=a.pr,mode=a.mode,tag=a.tag,clean_container=False,entry=str(entry))
try:
    command=([sys.executable,'-u',str(checkout/'tools/ci_pipline/run_ci.py'),str(entry)]
             if a.mode=='ci' else ['bash',str(entry)])
    log=open(output/('run_ci.log' if a.mode=='ci' else 'serve.log'),'w')
    process=subprocess.Popen(command,cwd=checkout,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    monitor.start()
    status['pid']=process.pid
    if a.mode=='ci':
        try:rc=process.wait(timeout=14400)
        except subprocess.TimeoutExpired:rc=124
        status.update(exit_code=rc,passed=rc==0)
    else:
        deadline=time.monotonic()+600
        while time.monotonic()<deadline:
            if process.poll() is not None:raise RuntimeError('model service exited before health: '+str(process.returncode))
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=3) as response:
                    if response.status==200:break
            except Exception:time.sleep(3)
        else:raise TimeoutError('service did not become healthy in 600 seconds')
        status['startup_seconds']=time.monotonic()-started
        client=[sys.executable,str(ROOT/'model_client.py'),'--base',f'http://127.0.0.1:{port}','--model',model,'--output',str(output/'model-gate.json'),'--seconds',str(0 if a.mode=='census' else a.seconds)]
        if a.pr==4:client+=['--expected',str(public/'validation/baseline_token_ids.json')]
        with open(output/'client.log','w') as clientlog:
            rc=subprocess.call(client,env=env,stdout=clientlog,stderr=subprocess.STDOUT)
        status.update(exit_code=rc,passed=rc==0)
except BaseException as exc:
    status['error']=repr(exc)
    raise
finally:
    monitor_stop.set()
    if monitor.is_alive():monitor.join(timeout=20)
    if process is not None:
        owned=descendants(process.pid)
        for pid in sorted(owned,reverse=True):
            try:os.kill(pid,signal.SIGTERM)
            except ProcessLookupError:pass
        time.sleep(5)
        for pid in sorted(owned,reverse=True):
            try:os.kill(pid,signal.SIGKILL)
            except ProcessLookupError:pass
    status['wall_seconds']=time.monotonic()-started
    after={str(x.relative_to(checkout)):hashlib.sha256(x.read_bytes()).hexdigest() for x in protected}
    status['protected_scripts_unchanged']=before==after
    (output/'protected-after.json').write_text(json.dumps(after,indent=2)+'\n')
    (output/'result.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status),flush=True)
if not status['passed'] or not status['protected_scripts_unchanged']:sys.exit(1)
