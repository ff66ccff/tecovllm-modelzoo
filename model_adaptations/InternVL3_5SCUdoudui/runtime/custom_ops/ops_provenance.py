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

"""Verify the installed operator bundle once, before model construction.

Export names alone cannot distinguish an old wheel from the tested build. The
receipt binds both host extension and mapped device core to a reviewed manifest.
No environment lookup, hashing or backend decision occurs during forward.
"""
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_bundle(tecoops, torch, manifest, required, mapped):
    """Validate capabilities and bytes; arguments permit device-free negative tests."""
    if manifest.get("schema") != 1 or len(manifest.get("source_commit", "")) != 40:
        raise RuntimeError("invalid operator manifest; record the full tested source commit")
    missing = [name for name in required if not callable(getattr(tecoops, name, None))]
    for name in ("rms_norm_out", "rms_norm_add_out"):
        if not callable(getattr(getattr(torch.ops, "tecoops_ops", None), name, None)):
            missing.append("torch.ops.tecoops_ops." + name)
    if missing:
        raise RuntimeError("operator bundle lacks required APIs: " + ", ".join(missing))
    cores = sorted({str(Path(path).resolve()) for path in mapped})
    if len(cores) != 1:
        raise RuntimeError("expected one mapped libteco_ops.so; found " + repr(cores))
    extension = Path(tecoops._torch_ext.__file__).resolve()
    actual = {"core_sha256": digest(cores[0]), "extension_sha256": digest(extension)}
    for name, value in actual.items():
        if value != manifest.get(name):
            raise RuntimeError(name + " differs from the tested operator bundle; refusing an old or mixed wheel")
    return dict(actual, source_commit=manifest["source_commit"], core=cores[0], extension=str(extension))


def require_ops_bundle(required):
    import torch
    import tecoops

    manifest_path = Path(os.environ.get("SCUD_OPS_MANIFEST", str(
        Path(__file__).resolve().parents[2] / "validation" / "ops-artifacts.json")))
    manifest = json.loads(manifest_path.read_text())
    mapped = []
    with open("/proc/self/maps") as stream:
        for line in stream:
            fields = line.split()
            if fields and fields[-1].endswith("/libteco_ops.so"):
                mapped.append(fields[-1])
    result = verify_bundle(tecoops, torch, manifest, required, mapped)
    result["manifest"] = str(manifest_path.resolve())
    print("SCUD_OPS_PROVENANCE " + json.dumps(result, sort_keys=True), flush=True)
    return result
