"""CPU-only public eval tokenizer gate with independent tokenizer.json oracle."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

public=Path(__file__).resolve().parents[1]
model=Path(os.environ.get("GEMMA_PATH","/gpfs/model/google/gemma-4-12B-it"))
assert Path(sys.executable).resolve()==Path('/home/py312/bin/python').resolve()
watched=[model/name for name in ('config.json','tokenizer_config.json','tokenizer.json')]
watched += [Path('/home/py312/lib/python3.12/site-packages')/name for name in
 ('transformers/tokenization_utils_base.py','transformers/models/gemma/tokenization_gemma_fast.py','evalscope/perf/plugin/datasets/utils.py')]
def hashes(): return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
before=hashes()
child=r"""
import json, os, runpy, sys
from pathlib import Path
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from tokenizers import Tokenizer
model=sys.argv[1]
active=sys.argv[2]=='enabled'
method=PreTrainedTokenizerBase._set_model_specific_special_tokens
for _ in range(2):
 runpy.run_path(str(Path(os.environ['PYTHONPATH'])/'sitecustomize.py'))
 assert PreTrainedTokenizerBase._set_model_specific_special_tokens is method
if active:
 assert getattr(PreTrainedTokenizerBase,'_gemma4_list_shim',False)
 assert not hasattr(method._gemma4_orig,'_gemma4_orig')
else:
 assert not getattr(PreTrainedTokenizerBase,'_gemma4_list_shim',False)
 assert not hasattr(method,'_gemma4_orig')
if not active:
 try:
  AutoTokenizer.from_pretrained(model,trust_remote_code=True,local_files_only=True)
 except AttributeError as exc:
  assert 'keys' in str(exc) and 'list' in str(exc),str(exc)
  print(json.dumps({'baseline_failed_as_expected':True,'disabled_no_mutation':True,'repeated_initialization':2,'error':str(exc)}))
 else: raise RuntimeError('expected unpatched list/dict tokenizer failure')
else:
 tok=AutoTokenizer.from_pretrained(model,trust_remote_code=True,local_files_only=True)
 oracle=Tokenizer.from_file(str(Path(model)/'tokenizer.json'))
 texts=['','Why is the sky blue?','四川大学：可复现评测。','line1\nline2','<|video|>','emoji 🌍 café']
 cases=[]
 for text in texts:
  ids=tok.encode(text,add_special_tokens=False)
  expected=oracle.encode(text,add_special_tokens=False).ids
  assert ids==expected,(text,ids,expected)
  cases.append({'text':text,'ids':ids,'oracle_equal':True})
 assert getattr(PreTrainedTokenizerBase,'_gemma4_list_shim',False)
 unexpected=[name for name in sys.modules if name=='vllm' or name.startswith('vllm.') ]
 assert not unexpected,unexpected
 print(json.dumps({'loaded':True,'idempotent_initialization':True,'repeated_initialization':2,'single_wrapper':True,'cases':cases,'no_vllm_import':True,'torch_sdaa_auto_registered':('torch_sdaa' in sys.modules),'device_execution':False}))
"""
results={}
for stage in ('disabled','enabled'):
 env=dict(os.environ)
 env['PYTHONPATH']=str(public/'eval_overlay')
 env['GEMMA4_EVAL_TOKENIZER']='1' if stage=='enabled' else '0'
 env.pop('GEMMA4_OVERLAY',None)
 env['HF_HUB_OFFLINE']='1';env['TRANSFORMERS_OFFLINE']='1'
 proc=subprocess.run([sys.executable,'-c',child,str(model),stage],env=env,capture_output=True,text=True,check=False)
 if proc.returncode: raise RuntimeError(proc.stderr)
 results[stage]=json.loads(proc.stdout.strip().splitlines()[-1])
assert before==hashes()
result={'passed':True,'client_only':True,'model_path':str(model),'results':results,'unchanged_source_sha256':before,'model_smoke_run':False,'full_ci_run':False,'performance_claim':False}
output=Path(sys.argv[1]) if len(sys.argv)>1 else public/'validation/eval_tokenizer_20261005.json'
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'passed':True,'oracle_cases':len(results['enabled']['cases']),'unchanged_sources':len(before),'output':str(output)}))
