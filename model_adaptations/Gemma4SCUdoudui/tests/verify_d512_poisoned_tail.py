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

"""Supplemental poison test for unused pages and partial D512 cache tails."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

TASK_DIR = Path(__file__).resolve().parent
if TASK_DIR.name == "tests" and TASK_DIR.parent.name == "Gemma4SCUdoudui":
    ADAPTER_ROOT = Path(__file__).resolve().parent.parent
else:
    REPO_ROOT = TASK_DIR.parents[2]
    ADAPTER_ROOT = REPO_ROOT / "model_adaptations/Gemma4SCUdoudui"
sys.path.insert(0, str(TASK_DIR))
from d512_build_provenance import validate_build_provenance
PUBLIC_RUNTIME = ADAPTER_ROOT / "runtime"
PUBLIC_OP = PUBLIC_RUNTIME / "custom_ops/d512_official.py"
EXPECTED_PUBLIC_OP_SHA256 = "f0326a3d3bc89535e51c7ca96b468cfc88fa89053c7113a48e504b6cdd4facc8"
EXPECTED_EXTENSION_SHA256 = "9339b3756b49f9a4e2ec32e98f6b5e2f812d7790ceab360601238f77f517c7cb"
EXPECTED_CORE_SHA256 = "34afac5eded9a1d474a53c32bf19d71fe025ef81ad69fe0848ed0cbc616d0a6a"
MAX_CONTEXT = 4352
PAGE_SIZE = 32
MAX_ABS_TOLERANCE = 0.02

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", required=True)
parser.add_argument("--build-provenance", type=Path)
args = parser.parse_args()

sys.path.insert(0, str(PUBLIC_RUNTIME))

import torch
import torch_sdaa
import tecoops
from custom_ops import d512_official as op

torch.set_num_threads(4)
torch.manual_seed(20261006)
random.seed(20261006)

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

assert Path(op.__file__).resolve() == PUBLIC_OP.resolve(), op.__file__
public_op_sha256 = sha256(PUBLIC_OP)
assert public_op_sha256 == EXPECTED_PUBLIC_OP_SHA256, public_op_sha256
assert op.OFFICIAL_DECODE is not None, "the accepted isolated extension must be selected"

global_tecoops_file = str(Path(tecoops.__file__).resolve())
global_tecoops_kernel = tecoops.flash_attn_varlen_func
extension_path = Path(os.environ["GEMMA_D512_OFFICIAL_EXTENSION"]).resolve(strict=True)
extension_sha256 = sha256(extension_path)
core_path = extension_path.with_name("libteco_gemma_flash.so").resolve(strict=True)
core_sha256 = sha256(core_path)
if args.build_provenance is None:
    assert extension_sha256 == EXPECTED_EXTENSION_SHA256, extension_sha256
    assert core_sha256 == EXPECTED_CORE_SHA256, core_sha256
    build_provenance_file = None
    build_provenance_sha256 = None
else:
    build_provenance_path = args.build_provenance.resolve(strict=True)
    validate_build_provenance(
        build_provenance_path,
        extension_sha256=extension_sha256,
        core_sha256=core_sha256,
        python_executable=sys.executable,
    )
    build_provenance_file = str(build_provenance_path)
    build_provenance_sha256 = sha256(build_provenance_path)
assert os.environ["GEMMA_D512_OFFICIAL_SHA256"].lower() == extension_sha256
extension_module = __import__("_gemma_d512_official._torch_ext", fromlist=[""])
assert Path(extension_module.__file__).resolve() == extension_path

binding_probe = SimpleNamespace(
    head_size=512, num_heads=8, num_kv_heads=1,
    _gemma4_window_size=None, logits_soft_cap=0, sinks=None,
    kv_cache_dtype="auto", scale=1.0,
)
op.bind_global_decode(binding_probe)
assert binding_probe._gemma4_decode_fn is op.decode_forward

def invoke(q, k, v, cu, seq, block_table, out):
    op.OFFICIAL_DECODE(q, k, v, cu, seq, block_table, MAX_CONTEXT, 1.0, out)
    return out

compiled_invoke = torch.compile(invoke, backend="eager", fullgraph=True)
CASES = [
    ("partial_4195", [4195]),
    ("partial_4351", [4351]),
    ("mixed_partial_pages", [4195, 4351]),
]
rows = []

for case_name, lengths in CASES:
    n = len(lengths)
    page_counts = [(length + PAGE_SIZE - 1) // PAGE_SIZE for length in lengths]
    block_count = sum(page_counts) + 11
    table_width = (max(lengths) + PAGE_SIZE - 1) // PAGE_SIZE
    q_cpu = torch.randn(n, 8, 512).half()
    q_float = q_cpu.float()
    q_cpu = (q_float * torch.rsqrt(q_float.square().mean(-1, keepdim=True) + 1e-6)).half()
    k_control_cpu = torch.randn(block_count, 1, PAGE_SIZE, 512).half()
    k_float = k_control_cpu.float()
    k_control_cpu = (k_float * torch.rsqrt(k_float.square().mean(-1, keepdim=True) + 1e-6)).half()
    v_control_cpu = torch.randn(block_count, 1, PAGE_SIZE, 512).half()
    v_float = v_control_cpu.float()
    v_control_cpu = (v_float * torch.rsqrt(v_float.square().mean(-1, keepdim=True) + 1e-6)).half()

    pages = list(range(block_count))
    random.shuffle(pages)
    block_table_cpu = torch.full((n + 1, table_width), -1, dtype=torch.int32)
    references = []
    used_physical_pages = set()
    poison_tail_offsets = []
    cursor = 0
    for row, length in enumerate(lengths):
        count = page_counts[row]
        selected_pages = pages[cursor:cursor + count]
        cursor += count
        used_physical_pages.update(selected_pages)
        block_table_cpu[row, :count] = torch.tensor(selected_pages, dtype=torch.int32)
        key_rows = k_control_cpu[selected_pages, 0].reshape(-1, 512)[:length].float()
        value_rows = v_control_cpu[selected_pages, 0].reshape(-1, 512)[:length].float()
        references.append(torch.softmax(q_cpu[row].float() @ key_rows.T, dim=-1) @ value_rows)
        valid_tail = length % PAGE_SIZE
        if valid_tail:
            poison_tail_offsets.append({
                "row": row,
                "physical_page": selected_pages[-1],
                "first_poisoned_offset": valid_tail,
                "poisoned_offsets": PAGE_SIZE - valid_tail,
            })
    reference = torch.stack(references)

    k_poison_cpu = k_control_cpu.clone()
    v_poison_cpu = v_control_cpu.clone()
    unused_physical_pages = sorted(set(range(block_count)) - used_physical_pages)
    if unused_physical_pages:
        k_poison_cpu[unused_physical_pages] = float("nan")
        v_poison_cpu[unused_physical_pages] = float("nan")
    for tail in poison_tail_offsets:
        page_id = tail["physical_page"]
        start = tail["first_poisoned_offset"]
        k_poison_cpu[page_id, 0, start:, :] = float("nan")
        v_poison_cpu[page_id, 0, start:, :] = float("nan")

    for nondefault_stream in (False, True):
        def run_once(k_cpu, v_cpu):
            q_storage = torch.empty(n, 8, 1024, dtype=torch.float16, device="sdaa")
            query = q_storage[..., ::2]
            key_cache = torch.empty_like(k_cpu, device="sdaa")
            value_cache = torch.empty_like(v_cpu, device="sdaa")
            query_start_loc = torch.empty(n + 2, dtype=torch.int32, device="sdaa")
            seq_lens = torch.empty(n + 1, dtype=torch.int32, device="sdaa")
            block_table = torch.empty_like(block_table_cpu, device="sdaa")
            output = torch.empty(n, 8, 512, dtype=torch.float16, device="sdaa")

            stream = torch.sdaa.Stream() if nondefault_stream else torch.sdaa.current_stream()
            stream.wait_stream(torch.sdaa.current_stream())
            with torch.sdaa.stream(stream):
                query.copy_(q_cpu)
                key_cache.copy_(k_cpu)
                value_cache.copy_(v_cpu)
                query_start_loc.copy_(
                    torch.tensor(list(range(n + 1)) + [n], dtype=torch.int32)
                )
                seq_lens.copy_(torch.tensor(lengths + [0], dtype=torch.int32))
                block_table.copy_(block_table_cpu)
            stream.synchronize()

            key_before_bits = key_cache.view(torch.int16).cpu().clone()
            value_before_bits = value_cache.view(torch.int16).cpu().clone()
            with torch.sdaa.stream(stream):
                if nondefault_stream:
                    compiled_invoke(
                        query, key_cache, value_cache,
                        query_start_loc[:n + 1], seq_lens[:n],
                        block_table[:n], output,
                    )
                else:
                    metadata = SimpleNamespace(
                        query_start_loc=query_start_loc,
                        seq_lens=seq_lens,
                        block_table=block_table,
                        max_model_len=MAX_CONTEXT,
                    )
                    impl = SimpleNamespace(scale=1.0)
                    op.decode_forward(
                        impl, output, query, key_cache, value_cache,
                        metadata, n, key_cache.stride(0),
                    )
            stream.synchronize()

            actual = output.cpu().float()
            key_after_bits = key_cache.view(torch.int16).cpu()
            value_after_bits = value_cache.view(torch.int16).cpu()
            assert torch.equal(key_after_bits, key_before_bits), "key cache bits changed"
            assert torch.equal(value_after_bits, value_before_bits), "value cache bits changed"
            assert torch.isfinite(actual).all(), "output contains nonfinite values"
            error = (actual - reference).abs().max().item()
            assert error <= MAX_ABS_TOLERANCE, (case_name, nondefault_stream, error)
            result_cpu = actual.clone()
            del q_storage, query, key_cache, value_cache, query_start_loc
            del seq_lens, block_table, output, key_before_bits, value_before_bits
            del key_after_bits, value_after_bits, actual
            torch.sdaa.synchronize()
            return result_cpu, error

        control_output, control_error = run_once(k_control_cpu, v_control_cpu)
        poisoned_output, poisoned_error = run_once(k_poison_cpu, v_poison_cpu)
        control_delta = (poisoned_output - control_output).abs().max().item()
        assert torch.isfinite(poisoned_output).all()
        assert control_delta <= MAX_ABS_TOLERANCE, (
            case_name, nondefault_stream, control_delta, MAX_ABS_TOLERANCE
        )
        rows.append({
            "case": case_name,
            "lengths": lengths,
            "max_seq_len_argument": MAX_CONTEXT,
            "block_size": PAGE_SIZE,
            "block_table_width": table_width,
            "block_count": block_count,
            "used_physical_page_count": len(used_physical_pages),
            "unused_physical_page_ids_poisoned": unused_physical_pages,
            "tail_page_offsets_poisoned": poison_tail_offsets,
            "nondefault_stream": nondefault_stream,
            "compiled_fullgraph_eager": nondefault_stream,
            "permuted_physical_pages": True,
            "poison_kind": "FP16 NaN",
            "control_max_abs_error": control_error,
            "poisoned_max_abs_error": poisoned_error,
            "control_vs_poisoned_max_abs_delta": control_delta,
            "absolute_tolerance": MAX_ABS_TOLERANCE,
            "output_finite": True,
            "cache_bitwise_exact_via_int16_view": True,
        })
        print(json.dumps(rows[-1]), flush=True)

assert str(Path(tecoops.__file__).resolve()) == global_tecoops_file
assert tecoops.flash_attn_varlen_func is global_tecoops_kernel
assert len(rows) == 6

result = {
    "schema": "gemma-d512-budget4352-poisoned-cache/v1",
    "passed": True,
    "device": "sdaa",
    "repo_head_at_preparation": "f49de6300778ccc340c2d4b4d8969ce941b320eb",
    "public_runtime_operator": str(PUBLIC_OP),
    "public_runtime_operator_sha256": public_op_sha256,
    "extension": str(extension_path),
    "extension_sha256": extension_sha256,
    "core": str(core_path),
    "core_sha256": core_sha256,
    "build_provenance_file": build_provenance_file,
    "build_provenance_sha256": build_provenance_sha256,
    "max_context": MAX_CONTEXT,
    "page_size": PAGE_SIZE,
    "cases": rows,
    "case_count": len(rows),
    "stream_invocation_count": len(rows) * 2,
    "max_abs_tolerance": MAX_ABS_TOLERANCE,
    "cache_comparison": "bitwise equality through int16 view, safe for NaN payloads",
    "global_tecoops_unchanged": global_tecoops_file,
}
output_path = Path(args.out)
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
