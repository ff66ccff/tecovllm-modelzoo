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

"""Build the archived D512 integration in a new directory with vendor Python."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


def main():
    if Path(sys.executable).resolve() != Path('/home/py312/bin/python').resolve():
        raise RuntimeError('use /home/py312/bin/python')
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-repo', type=Path, required=True)
    p.add_argument('--teco-hal', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    output = a.output.resolve()
    hal = a.teco_hal.resolve(strict=True)
    if output.exists() or not (hal / 'include').is_dir() or not (hal / 'lib').is_dir():
        raise RuntimeError('output must be new and teco-hal must contain include/lib')
    patch = Path(__file__).resolve().parents[2] / 'op_learning/attention/gemma-d512-model/official_combined.patch'
    commit = subprocess.check_output(['git', '-C', str(a.source_repo), 'rev-parse', 'de27305^{commit}'], text=True).strip()
    blob = subprocess.check_output(['git', '-C', str(a.source_repo), 'archive', commit])
    output.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
        archive.extractall(output, filter='data')
    apply = ['git', '-C', str(output), 'apply', '--unidiff-zero', '--ignore-space-change']
    subprocess.run(apply + ['--check', str(patch)], check=True)
    subprocess.run(apply + [str(patch)], check=True)
    (output / 'thirdparty').mkdir(exist_ok=True)
    (output / 'thirdparty/teco-hal').symlink_to(hal, target_is_directory=True)
    provenance = dict(official_base=commit, patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest(),
                      teco_hal=str(hal), python=str(Path(sys.executable).resolve()), output=str(output))
    if not a.prepare_only:
        subprocess.run(['teco-smi'], check=True)
        provenance['disk_free_bytes_before_build'] = shutil.disk_usage(output).free
        env = dict(os.environ, WITH_TORCH='ON', WITH_INFERENCE_PLUGIN='OFF', MAX_JOBS='4')
        with (output / 'build.log').open('w') as log:
            subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'],
                           cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        extension = output / 'api/tecoops/_torch_ext.cpython-312-loongarch64-linux-gnu.so'
        core = extension.parent / 'libteco_gemma_flash.so'
        provenance.update(extension=str(extension), extension_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(),
                          core=str(core), core_sha256=hashlib.sha256(core.read_bytes()).hexdigest())
        dynamic = subprocess.check_output(['readelf', '-d', str(extension)], text=True)
        if 'Shared library: [libteco_gemma_flash.so]' not in dynamic or 'Shared library: [libteco_ops.so]' in dynamic:
            raise RuntimeError('unexpected core DT_NEEDED')
        (output / 'library_dependencies.log').write_text(dynamic)
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance), flush=True)


if __name__ == '__main__':
    main()
