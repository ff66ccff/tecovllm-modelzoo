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

"""CPU checks for old-wheel and mixed-core rejection at process startup."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

path = Path(__file__).resolve().parents[1] / "runtime/custom_ops/ops_provenance.py"
spec = importlib.util.spec_from_file_location("ops_provenance", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BundleGuard(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.core = Path(self.tmp.name) / "libteco_ops.so"
        self.ext = Path(self.tmp.name) / "_torch_ext.so"
        self.core.write_bytes(b"tested core")
        self.ext.write_bytes(b"tested extension")
        self.manifest = dict(schema=1, source_commit="a" * 40,
            core_sha256=module.digest(self.core), extension_sha256=module.digest(self.ext))
        self.ops = NS(linear_stream=lambda: None, _torch_ext=NS(__file__=str(self.ext)))
        self.torch = NS(ops=NS(tecoops_ops=NS(rms_norm_out=lambda: None, rms_norm_add_out=lambda: None)))

    def check(self, mapped=None):
        return module.verify_bundle(self.ops, self.torch, self.manifest, ["linear_stream"],
            [self.core] if mapped is None else mapped)

    def test_exact_bundle_passes(self):
        self.assertEqual(self.check()["source_commit"], "a" * 40)

    def test_old_core_with_identical_exports_is_rejected(self):
        self.core.write_bytes(b"old core, same export names")
        with self.assertRaisesRegex(RuntimeError, "core_sha256 differs"):
            self.check()

    def test_old_extension_is_rejected(self):
        self.ext.write_bytes(b"old extension")
        with self.assertRaisesRegex(RuntimeError, "extension_sha256 differs"):
            self.check()

    def test_missing_non_rms_api_is_rejected(self):
        self.ops.linear_stream = None
        with self.assertRaisesRegex(RuntimeError, "linear_stream"):
            self.check()

    def test_missing_rms_out_is_rejected(self):
        self.torch.ops.tecoops_ops.rms_norm_add_out = None
        with self.assertRaisesRegex(RuntimeError, "rms_norm_add_out"):
            self.check()

    def test_two_cores_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "expected one mapped"):
            self.check([self.core, self.core.parent / "other/libteco_ops.so"])


if __name__ == "__main__":
    unittest.main()
