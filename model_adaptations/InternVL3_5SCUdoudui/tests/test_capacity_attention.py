# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
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
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.
"""Bounded-memory FP32 oracle gate for InternVL's declared max prefill shape."""

import hashlib
from pathlib import Path
import sys
import unittest

import torch
import torch_sdaa  # noqa: F401

ADAPTATION = Path(__file__).resolve().parents[1]
PUBLIC_RUNTIME = ADAPTATION / "runtime"
sys.path.insert(0, str(PUBLIC_RUNTIME))
from custom_ops.prefill_attention import op as public_prefill  # noqa: E402

SEQ, HQ, HKV, D, BLOCK = 4352, 16, 4, 128, 32
DEVICE = "sdaa:0"
NRMSE_TOL = 8 * 2 ** -11
MAXABS_TOL = 0.06


def pack_cache(values):
    num_blocks = (SEQ + BLOCK - 1) // BLOCK
    packed = values.permute(1, 0, 2).reshape(HKV, num_blocks, BLOCK, D)
    return packed.permute(1, 0, 2, 3).contiguous().to(DEVICE)


def cpu_reference(q, k, v, chunk=32):
    """Independent explicit-causal FP32 attention; bounds score memory by query chunks."""
    torch.set_num_threads(min(4, torch.get_num_threads()))
    k_heads = k.float().permute(1, 0, 2)
    v_heads = v.float().permute(1, 0, 2)
    keys = torch.arange(SEQ, dtype=torch.int64)
    result = torch.empty((SEQ, HQ, D), dtype=torch.float32)
    group = HQ // HKV
    for start in range(0, SEQ, chunk):
        end = min(start + chunk, SEQ)
        q_group = q[start:end].float().permute(1, 0, 2).reshape(HKV, group, end - start, D)
        scores = torch.matmul(q_group, k_heads[:, None].transpose(-1, -2)) * (D ** -0.5)
        query_positions = torch.arange(start, end, dtype=torch.int64)
        scores.masked_fill_(
            keys[None, None, None, :] > query_positions[None, None, :, None],
            float("-inf"),
        )
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v_heads[:, None, :, :])
        result[start:end] = out.reshape(HQ, end - start, D).permute(1, 0, 2)
    return result


class TestInternVLCapacityPrefill(unittest.TestCase):
    def test_public_prefill_q16_kv4_d128_seq4352(self):
        expected_op = (PUBLIC_RUNTIME / "custom_ops/prefill_attention/op.py").resolve()
        actual_op = Path(public_prefill.__file__).resolve()
        self.assertEqual(actual_op, expected_op, f"wrong operator binding: {actual_op}")
        op_sha256 = hashlib.sha256(actual_op.read_bytes()).hexdigest()
        print(f"[binding] file={actual_op} sha256={op_sha256}", flush=True)

        generator = torch.Generator(device="cpu").manual_seed(4352)
        q_cpu = torch.randn((SEQ, HQ, D), dtype=torch.float16, generator=generator)
        k_cpu = torch.randn((SEQ, HKV, D), dtype=torch.float16, generator=generator)
        v_cpu = torch.randn((SEQ, HKV, D), dtype=torch.float16, generator=generator)
        expected = cpu_reference(q_cpu, k_cpu, v_cpu)

        q = q_cpu.to(DEVICE)
        key_cache = pack_cache(k_cpu)
        value_cache = pack_cache(v_cpu)
        block_table = torch.arange((SEQ + BLOCK - 1) // BLOCK, dtype=torch.int32, device=DEVICE)[None, :]
        query_start_loc = torch.tensor([0, SEQ], dtype=torch.int32, device=DEVICE)
        seq_lens = torch.tensor([SEQ], dtype=torch.int32, device=DEVICE)

        actual = public_prefill.sdpa_prefill_attention(
            q, key_cache, value_cache, block_table, query_start_loc,
            seq_lens, D ** -0.5, True,
        )
        torch.sdaa.synchronize()
        actual_cpu = actual.cpu().float()
        difference = (actual_cpu - expected).abs()
        nrmse = (difference.square().mean().sqrt() / expected.square().mean().sqrt()).item()
        max_abs = difference.max().item()
        print(
            f"[capacity prefill] seq={SEQ} hq={HQ} hkv={HKV} d={D} "
            f"nrmse={nrmse:.6e} (tol {NRMSE_TOL:.6e}) "
            f"max_abs={max_abs:.6f} (tol {MAXABS_TOL})",
            flush=True,
        )
        self.assertLessEqual(nrmse, NRMSE_TOL)
        self.assertLessEqual(max_abs, MAXABS_TOL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
