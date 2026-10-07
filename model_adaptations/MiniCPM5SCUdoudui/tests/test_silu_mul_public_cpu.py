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

"""CPU/Fake/fullgraph contract tests for the public MiniCPM SwiGLU bridge."""
import os
import subprocess
import sys
import unittest
from pathlib import Path

import custom_ops
import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensorMode

public_custom_ops = Path(__file__).resolve().parents[1] / "runtime" / "custom_ops"
if str(public_custom_ops) not in custom_ops.__path__:
    custom_ops.__path__.append(str(public_custom_ops))

from custom_ops.silu_mul.op import silu_mul


def reference(x):
    width = x.shape[-1] // 2
    return F.silu(x[..., :width]) * x[..., width:]


class TestPublicSiluMul(unittest.TestCase):
    def test_random_fp16_matches_reference(self):
        torch.manual_seed(20261007)
        x = torch.randn(4, 9216, dtype=torch.float16)
        got = silu_mul(x)
        torch.testing.assert_close(got, reference(x), rtol=0, atol=0)
        self.assertEqual(tuple(got.shape), (4, 4608))
        self.assertEqual(got.dtype, torch.float16)

    def test_extreme_fp16_matches_reference(self):
        gate = torch.tensor([-100.0, -20.0, -5.0, -1.0, 0.0, 0.5, 1.0, 5.0, 20.0, 100.0], dtype=torch.float16)
        up = torch.tensor([1.0, -1.0, 2.0, -2.0, 3.0, 4.0, -5.0, 10.0, -10.0, 1.0], dtype=torch.float16)
        x = torch.cat((gate, up)).view(1, -1)
        torch.testing.assert_close(silu_mul(x), reference(x), rtol=0, atol=0)

    def test_rejects_invalid_layout_width_dtype_and_rank(self):
        noncontiguous = torch.randn(12, 4, dtype=torch.float16).t()
        with self.assertRaisesRegex((RuntimeError, ValueError), "contiguous"):
            silu_mul(noncontiguous)
        with self.assertRaisesRegex((RuntimeError, ValueError), "even"):
            silu_mul(torch.randn(2, 7, dtype=torch.float16))
        with self.assertRaisesRegex((RuntimeError, TypeError), "FP16"):
            silu_mul(torch.randn(2, 8, dtype=torch.float32))
        with self.assertRaisesRegex((RuntimeError, ValueError), "rank"):
            silu_mul(torch.randn(8, dtype=torch.float16))

    def test_fake_shape_dtype_device(self):
        with FakeTensorMode():
            x = torch.empty((3, 9216), dtype=torch.float16)
            out = silu_mul(x)
        self.assertEqual(tuple(out.shape), (3, 4608))
        self.assertEqual(out.dtype, torch.float16)
        self.assertEqual(out.device, x.device)

    def test_fullgraph_cpu_eager(self):
        torch.manual_seed(20261007)
        x = torch.randn(2, 9216, dtype=torch.float16)
        compiled = torch.compile(silu_mul, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(x), reference(x), rtol=0, atol=0)


    def _run_vendor_profile_bootstrap(self, cli_args=(), validated_dtype=None):
        repo_root = Path(__file__).resolve().parents[3]
        overlay = Path(__file__).resolve().parents[1] / "runtime" / "overlay"
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(overlay), str(repo_root), env.get("PYTHONPATH", "")]
        )
        env["TMPDIR"] = "/tmp/hm26"
        env["MINICPM_OP_PROFILE"] = "official"
        env["MINICPM_SILU_MUL_PROFILE"] = "vendor_swiglu"
        if validated_dtype is None:
            env.pop("MINICPM_SILU_MUL_DTYPE_VALIDATED", None)
        else:
            env["MINICPM_SILU_MUL_DTYPE_VALIDATED"] = validated_dtype
        return subprocess.run(
            [sys.executable, "-c", "pass", *cli_args],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_fp16_marker_does_not_hide_bad_cli_dtype(self):
        result = self._run_vendor_profile_bootstrap(
            ("--dtype", "bfloat16"), validated_dtype="fp16"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("only supports explicit FP16 dtype", result.stderr)

    def test_worker_bootstrap_inherits_fp16_and_uses_public_module(self):
        result = self._run_vendor_profile_bootstrap(validated_dtype="fp16")
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        public_op = (
            Path(__file__).resolve().parents[1]
            / "runtime"
            / "custom_ops"
            / "silu_mul"
            / "op.py"
        ).resolve()
        self.assertIn(str(public_op), result.stderr)

    def test_missing_dtype_and_marker_is_rejected(self):
        result = self._run_vendor_profile_bootstrap()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires explicit FP16 dtype", result.stderr)


if __name__ == "__main__":
    unittest.main()
