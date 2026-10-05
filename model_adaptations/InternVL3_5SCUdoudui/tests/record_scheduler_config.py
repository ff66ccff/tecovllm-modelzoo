#!/home/py312/bin/python
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
"""Read-only API client that records actual InternVL worker scheduler config."""
import argparse
import json
from pathlib import Path
import sys
import urllib.request


def rpc(base_url, method):
    request = urllib.request.Request(
        base_url.rstrip("/") + "/collective_rpc",
        data=json.dumps({"method": method}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def main():
    expected_python = Path("/home/py312/bin/python").resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError(f"must use vendor Python {expected_python}, got {sys.executable}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8002")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    response = rpc(args.url, "read_scheduler_config")
    workers = response.get("results") if isinstance(response, dict) else None
    workers = workers if isinstance(workers, list) else []
    passed = (
        len(workers) == 2
        and all(
            row.get("max_model_len") == 4352
            and row.get("max_num_batched_tokens") == 4352
            and row.get("enable_chunked_prefill") is False
            and row.get("model_dtype") == "torch.float16"
            and row.get("tensor_parallel_size") == 2
            for row in workers
        )
    )
    record = {
        "passed": passed,
        "expected_tp_workers": 2,
        "expected_max_model_len": 4352,
        "expected_max_num_batched_tokens": 4352,
        "expected_chunked_prefill": False,
        "weights_devices_observed": [row.get("weight_device") for row in workers],
        "response": response,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, sort_keys=True))
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
