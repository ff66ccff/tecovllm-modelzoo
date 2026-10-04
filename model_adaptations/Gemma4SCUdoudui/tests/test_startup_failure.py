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

"""The process must not reach main after a selected D512 initialization fails."""
import json
import os
from pathlib import Path
import subprocess
import sys

bundle = Path(__file__).resolve().parents[1]
records = []
for label, extension, digest in (
    ("missing_selected_library", "/nonexistent/gemma-d512/_torch_ext.cpython-312-loongarch64-linux-gnu.so", "0" * 64),
    ("incomplete_selection", "/nonexistent/gemma-d512/_torch_ext.cpython-312-loongarch64-linux-gnu.so", None),
):
    env = dict(os.environ)
    env.update(PYTHONPATH=str(bundle / "overlay") + os.pathsep + str(bundle / "runtime"),
               GEMMA4_OVERLAY="1", GEMMA_D512_OFFICIAL_EXTENSION=extension)
    if digest is None:
        env.pop("GEMMA_D512_OFFICIAL_SHA256", None)
    else:
        env["GEMMA_D512_OFFICIAL_SHA256"] = digest
    run = subprocess.run(
        ["/home/py312/bin/python", "-c", "print('GEMMA_MAIN_REACHED')"],
        env=env, capture_output=True, text=True, timeout=120,
    )
    if run.returncode != 1 or "GEMMA_MAIN_REACHED" in run.stdout:
        raise AssertionError((label, run.returncode, run.stdout, run.stderr))
    if "[gemma4-overlay] FAILED:" not in run.stderr:
        raise AssertionError((label, "missing initialization diagnostic", run.stderr))
    records.append({"case": label, "exit_code": run.returncode, "main_unreachable": True})
print(json.dumps({"passed": True, "cases": records}))
