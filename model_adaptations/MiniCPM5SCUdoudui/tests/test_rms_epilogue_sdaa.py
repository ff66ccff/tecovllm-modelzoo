"""MiniCPM E29 RMS focused reproduction; original FP16 oracle and structural gates."""
import argparse,json,os,statistics,sys,time
from pathlib import Path
import hashlib
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def signatures(path):
 d=json.loads(Path(path).read_text());assert d['model']=='MiniCPM5-1B' and d['capture_scope']=='own_real_sdaa_fixed_prompt';cases=d['cases'];assert cases and all(c['source']=='model_actual' and c['shape'] in ([1,1536],[7,1536]) and c['eps']==1e-6 and c['mode'] in ('plain','add') for c in cases);assert len(cases)==4 and {(tuple(c['shape']),c['mode']) for c in cases}=={((n,1536),m) for n in (1,7) for m in ('plain','add')};return cases
def package(variant):
 p=Path(os.environ['MINICPM_RMS_PACKAGE_ROOT']).resolve();assert (p/'tecoops/libteco_ops.so').is_file();files=[p/'tecoops/__init__.py',p/'tecoops/libteco_ops.so']+list((p/'tecoops').glob('_torch_ext.cpython-312-*.so'));assert len(files)==3;return p,{z.name:sha(z) for z in files}
p=argparse.ArgumentParser();p.add_argument('--metadata',required=True,type=Path);p.add_argument('--out',type=Path);p.add_argument('--preflight',action='store_true');p.add_argument('--warmup',type=int,default=5);p.add_argument('--loops',type=int,default=20);a=p.parse_args();a.variant='isolated-package'
cases=signatures(a.metadata);pkg,digests=package(a.variant)
if a.preflight:print(json.dumps({'own_cases':cases,'package':str(pkg),'gpu_run':False}));raise SystemExit(0)
assert os.environ.get('SDAA_VISIBLE_DEVICES'), 'Explicit assigned device required'
assert a.out and a.warmup>0 and a.loops>0
sys.path.insert(0,str(pkg));sys.path.insert(1,str(Path(__file__).resolve().parents[1]/'runtime'))
import numpy as np
import torch,torch_sdaa,tecoops
import tecoops._torch_ext as native
from custom_ops.compile_safe import official_ops
assert Path(sys.executable).resolve()==Path('/home/py312/bin/python').resolve()
assert Path(tecoops.__file__).resolve().parent==pkg/'tecoops' and Path(native.__file__).resolve().parent==pkg/'tecoops'
mapped={Path(l.split()[-1]).resolve() for l in Path('/proc/self/maps').read_text().splitlines() if l.split()[-1].startswith('/') and Path(l.split()[-1]).name=='libteco_ops.so'}
assert mapped=={(pkg/'tecoops/libteco_ops.so').resolve()} and torch.sdaa.device_count()==1
assert all(callable(getattr(tecoops,n,None)) for n in ('rms_norm','reshape_and_cache','flash_attn_varlen_func'))
mapped_ext={Path(l.split()[-1]).resolve() for l in Path('/proc/self/maps').read_text().splitlines() if l.split()[-1].startswith('/') and Path(l.split()[-1]).name==Path(native.__file__).name}
assert mapped_ext=={Path(native.__file__).resolve()}
ops=official_ops(tecoops);assert ops is not None
report={'variant':a.variant,'metadata_sha256':sha(a.metadata),'seed':20261008,'warmup':a.warmup,'loops':a.loops,'package':str(pkg),'sha256':digests,'mapped_cores':[str(z) for z in mapped],'extension_path':str(Path(native.__file__).resolve()),'mapped_extensions':[str(z) for z in mapped_ext],'process_pid':os.getpid(),'process_pgid':os.getpgid(0),'cases':[],'passed':False,'model_performance_claim':False}
a.out.parent.mkdir(parents=True,exist_ok=True)
def save():a.out.write_text(json.dumps(report,indent=2)+'\n')
# Additional synthetic stress is explicitly separate from own model targets.
controls=[{'shape':[257,cases[0]['shape'][1]],'mode':'plain','eps':cases[0]['eps'],'source':'additional_reuse_stress'}, {'shape':[3,130],'mode':'plain','eps':1e-6,'source':'additional_even_tail'}]
controls.extend(dict(c,mode='add',source=c['source']+'_add') for c in list(controls))
try:
 for ci,c in enumerate(cases+controls):
  rows,hidden=c['shape'];add=c['mode']=='add';eps=c['eps'];g=torch.Generator(device='cpu').manual_seed(20261008+ci)
  xc=torch.randn(rows,hidden,generator=g,dtype=torch.float16);rc=torch.randn(rows,hidden,generator=g,dtype=torch.float16);wc=(torch.rand(hidden,generator=g)*.5+.75).half()
  x,r,w=xc.to('sdaa'),rc.to('sdaa'),wc.to('sdaa');out=torch.full_like(x,float('nan'));rout=torch.full_like(x,float('nan')) if add else None
  def raw():tecoops.rms_norm(x,w,r if add else None,out,rout,eps)
  def opaque(x,w,r):
   o=torch.empty_like(x)
   if add:
    ro=torch.empty_like(x);ops.rms_norm_add(x,w,r,o,ro,eps);return o,ro
   ops.rms_norm(x,w,o,eps);return (o,)
  raw();torch.sdaa.synchronize();got=out.cpu().numpy().copy();res=rout.cpu().numpy().copy() if add else None
  z=xc.float()+rc.float() if add else xc.float();inv=torch.rsqrt(z.square().mean(-1,keepdim=True)+eps);num=z.half().float() if add else z;ref=(num.double()*inv.double()*wc.double()).half().float().numpy()
  err=float(np.max(np.abs(got.astype(np.float32)-ref)));tol=.01 if add else .005;assert np.isfinite(got).all() and err<tol
  re=None
  if add:re=float(np.max(np.abs(res.astype(np.float32)-z.half().float().numpy())));assert re<.01
  inputsha=hashlib.sha256(xc.numpy().tobytes()+rc.numpy().tobytes()+wc.numpy().tobytes()).hexdigest()
  for run in (opaque,torch.compile(opaque,backend='eager',fullgraph=True)):
   outputs=run(x,w,r);torch.sdaa.synchronize();assert np.array_equal(outputs[0].cpu().numpy().view(np.uint16),got.view(np.uint16))
   if add:assert np.array_equal(outputs[1].cpu().numpy().view(np.uint16),res.view(np.uint16))
  from torch._subclasses.fake_tensor import FakeTensorMode
  with FakeTensorMode() as mode:opaque(mode.from_tensor(x),mode.from_tensor(w),mode.from_tensor(r))
  for repeat in range(16):
   out.fill_(float('nan'))
   if add:rout.fill_(float('nan'))
   raw();torch.sdaa.synchronize();assert np.array_equal(out.cpu().numpy().view(np.uint16),got.view(np.uint16))
   if add:assert np.array_equal(rout.cpu().numpy().view(np.uint16),res.view(np.uint16))
  stream=torch.sdaa.Stream()
  with torch.sdaa.stream(stream):raw()
  stream.synchronize();assert np.array_equal(out.cpu().numpy().view(np.uint16),got.view(np.uint16))
  if add:assert np.array_equal(rout.cpu().numpy().view(np.uint16),res.view(np.uint16))
  assert torch.equal(x.cpu(),xc) and torch.equal(r.cpu(),rc) and torch.equal(w.cpu(),wc)
  for _ in range(a.warmup):raw()
  torch.sdaa.synchronize();times=[]
  for _ in range(3):
   torch.sdaa.synchronize();start=time.perf_counter()
   for _ in range(a.loops):raw()
   torch.sdaa.synchronize();times.append((time.perf_counter()-start)*1000/a.loops)
  report['cases'].append(dict(c,input_sha256=inputsha,maxabs=err,tolerance=tol,residual_maxabs=re,raw_ms=times,median_ms=statistics.median(times),raw_opaque_fullgraph_fake=True,default_nondefault_stream=True,reuse16=True,inputs_unchanged=True));save()
 report['passed']=True
except Exception as e:report['error']=repr(e);raise
finally:save()
