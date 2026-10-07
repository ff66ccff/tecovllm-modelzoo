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

"""Focused real-device validation for the public MiniCPM SwiGLU bridge.

Run on one explicit SDAA device. The original minimum-normal gate × 2048 case
is kept bit-for-bit; its old separate-SDAA result is reported independently
from both CPU references because that path flushes the intermediate to zero.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
PUBLIC_CUSTOM_OPS = (
    REPO_ROOT
    / "model_adaptations"
    / "MiniCPM5SCUdoudui"
    / "runtime"
    / "custom_ops"
).resolve()
import custom_ops  # noqa: E402

if str(PUBLIC_CUSTOM_OPS) in custom_ops.__path__:
    custom_ops.__path__.remove(str(PUBLIC_CUSTOM_OPS))
custom_ops.__path__.insert(0, str(PUBLIC_CUSTOM_OPS))
silu_mul_module = importlib.import_module("custom_ops.silu_mul.op")
if PUBLIC_CUSTOM_OPS not in Path(silu_mul_module.__file__).resolve().parents:
    raise RuntimeError(f"expected public MiniCPM op, got {silu_mul_module.__file__}")
from custom_ops.silu_mul.op import bind_vendor_silu_mul, silu_mul  # noqa: E402


EDGE_GATE_BITS = [
    52480, 50432, 48128, 32768, 0, 1, 1024, 14336, 17664, 18688, 19712, 21760
]
EDGE_UP_BITS = [
    26624, 59392, 18688, 15360, 15360, 26624, 26624, 51456, 22080, 27874, 27100, 24528
]
RTOL = 0.001
ATOL = 0.001


def parse_device(value: str) -> torch.device:
    try:
        device = torch.device(value)
    except Exception as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if device.type != "sdaa" or device.index is None:
        raise argparse.ArgumentTypeError("use one explicit logical SDAA device such as sdaa:0")
    return device


def fp16_from_bits(values: list[int]) -> torch.Tensor:
    signed = [value if value < 0x8000 else value - 0x10000 for value in values]
    return torch.tensor(signed, dtype=torch.int16).view(torch.float16)


def fp16_bits(tensor: torch.Tensor) -> list[int]:
    signed = tensor.detach().to(device="cpu").contiguous().view(torch.int16).to(torch.int32)
    return [value & 0xFFFF for value in signed.reshape(-1).tolist()]


def make_original_edge_input() -> torch.Tensor:
    packed_width = 9216
    width = packed_width // 2
    x = torch.zeros((1, packed_width), dtype=torch.float16)
    gate = fp16_from_bits(EDGE_GATE_BITS)
    up = fp16_from_bits(EDGE_UP_BITS)
    x[0, : len(gate)] = gate
    x[0, width : width + len(up)] = up
    if fp16_bits(x[0, : len(gate)]) != EDGE_GATE_BITS:
        raise AssertionError("original gate edge input bits changed")
    if fp16_bits(x[0, width : width + len(up)]) != EDGE_UP_BITS:
        raise AssertionError("original up edge input bits changed")
    return x


def references(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    width = x.shape[-1] // 2
    gate = x[..., :width]
    up = x[..., width:]
    fp32_final_fp16 = (F.silu(gate.float()) * up.float()).to(torch.float16)
    fp16_separate = F.silu(gate) * up
    return fp32_final_fp16, fp16_separate


def check_one(x_cpu: torch.Tensor, device: torch.device, label: str, calls: dict[str, int]) -> dict:
    x_device = x_cpu.to(device)
    original_storage = x_device.detach().cpu().contiguous().view(torch.int16).clone()
    expected_fp32, expected_fp16 = references(x_cpu)
    candidate = silu_mul(x_device)
    old_sdaa_separate = F.silu(x_device[..., : x_device.shape[-1] // 2]) * x_device[..., x_device.shape[-1] // 2 :]
    torch.sdaa.synchronize(device.index)
    candidate_cpu = candidate.cpu()
    old_cpu = old_sdaa_separate.cpu()
    torch.testing.assert_close(candidate_cpu, expected_fp32, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(candidate_cpu, expected_fp16, rtol=RTOL, atol=ATOL)
    input_unchanged = torch.equal(
        original_storage,
        x_device.detach().cpu().contiguous().view(torch.int16),
    )
    if not input_unchanged:
        raise AssertionError(f"{label}: candidate changed its input")
    old_matches = bool(torch.allclose(candidate_cpu, old_cpu, rtol=RTOL, atol=ATOL))
    return {
        "label": label,
        "shape": list(x_cpu.shape),
        "dtype": "torch.float16",
        "stride": list(x_cpu.stride()),
        "candidate_vendor_calls_cumulative": calls["count"],
        "input_unchanged": input_unchanged,
        "max_abs_vs_cpu_float32_final_fp16": float((candidate_cpu - expected_fp32).abs().max()),
        "max_abs_vs_cpu_fp16_separate": float((candidate_cpu - expected_fp16).abs().max()),
        "max_abs_vs_old_sdaa_separate": float((candidate_cpu - old_cpu).abs().max()),
        "old_sdaa_separate_matches_at_tolerance": old_matches,
        "rtol": RTOL,
        "atol": ATOL,
        "edge_gate_bits": EDGE_GATE_BITS if label == "original_min_normal_times_2048" else None,
        "edge_up_bits": EDGE_UP_BITS if label == "original_min_normal_times_2048" else None,
        "candidate_edge": candidate_cpu[0, :12].tolist() if label == "original_min_normal_times_2048" else None,
        "cpu_fp32_final_fp16_edge": expected_fp32[0, :12].tolist() if label == "original_min_normal_times_2048" else None,
        "cpu_fp16_separate_edge": expected_fp16[0, :12].tolist() if label == "original_min_normal_times_2048" else None,
        "old_sdaa_separate_edge": old_cpu[0, :12].tolist() if label == "original_min_normal_times_2048" else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=parse_device, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch_sdaa  # noqa: F401

    torch.sdaa.set_device(args.device.index)
    if not torch.sdaa.is_available():
        raise RuntimeError("SDAA is unavailable")
    vendor = torch.ops.sdaa.swiglu
    calls = {"count": 0}

    def counted_vendor(*args, **kwargs):
        calls["count"] += 1
        return vendor(*args, **kwargs)

    bind_vendor_silu_mul(counted_vendor)
    sync = lambda: torch.sdaa.synchronize(args.device.index)
    rows = []
    rows.append(check_one(make_original_edge_input(), args.device, "original_min_normal_times_2048", calls))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20261007)
    rows.append(
        check_one(
            torch.randn((7, 9216), generator=generator, dtype=torch.float16),
            args.device,
            "captured_shape_7x9216",
            calls,
        )
    )
    sync()
    result = {
        "passed_cpu_oracles": True,
        "device": str(args.device),
        "public_op_file": str(Path(silu_mul_module.__file__).resolve()),
        "vendor_calls": calls["count"],
        "tolerance": {"rtol": RTOL, "atol": ATOL},
        "old_sdaa_separate_all_matches_at_tolerance": all(
            row["old_sdaa_separate_matches_at_tolerance"] for row in rows
        ),
        "known_numeric_boundary": (
            "The original minimum-normal gate × 2048 case remains unchanged. "
            "The fused candidate is checked independently against both CPU references; "
            "the old separate SDAA comparison is reported separately and may differ "
            "because the intermediate SiLU flushes to zero."
        ),
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if calls["count"] != len(rows):
        raise RuntimeError(f"expected {len(rows)} actual vendor calls, got {calls['count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
