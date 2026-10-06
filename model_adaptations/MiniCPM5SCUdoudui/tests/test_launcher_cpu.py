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


"""Startup-only argument checks with interpreter invocation intercepted."""
from pathlib import Path
import os, subprocess, tempfile, sys, importlib.util, json, hashlib
b=Path(__file__).resolve().parents[1]
source=(b/'run.sh').read_text()
needle='exec /home/py312/bin/python -m vllm.entrypoints.cli.main serve "${serve_args[@]}"'
assert source.count(needle)==1
mock=source.replace(needle,"printf '%s\\n' serve \"${serve_args[@]}\"")
with tempfile.TemporaryDirectory(prefix='minicpm-shim-',dir='/dev/shm') as d:
    path=Path(d)/'run.sh';path.write_text(mock)
    def invoke(profile=None, *args):
        env = dict(os.environ)
        for key in ("MINICPM_OP_PROFILE", "MODEL_ROOT", "MINICPM5_MODEL", "MINICPM_HOST", "MINICPM_PORT", "MINICPM_MAX_MODEL_LEN"):
            env.pop(key, None)
        if profile is not None:
            env["MINICPM_OP_PROFILE"] = profile
        return subprocess.run(["bash", str(path), *args], env=env, capture_output=True, text=True)


    default = invoke()
    assert default.returncode == 0, default.stderr
    official = invoke("official")
    assert official.returncode == 0, official.stderr
    default_args = [
        "serve",
        "/gpfs/model/OpenBMB/MiniCPM5-1B",
        "--served-model-name",
        "MiniCPM5-1B",
        "--tensor-parallel-size",
        "1",
        "--port",
        "8000",
        "--host",
        "0.0.0.0",
        "--dtype",
        "float16",
        "--trust-remote-code",
        "--no-enable-prefix-caching",
        "--max-model-len",
        "32768",
    ]
    expected_default = "\n".join(default_args) + "\n"
    assert default.stdout == official.stdout == expected_default, (default.stdout, official.stdout)
    print("DEFAULT_OFFICIAL_ARGS_BYTE_FOR_BYTE_UNCHANGED")

    fast = invoke("fast_attention")
    assert fast.returncode == 0, fast.stderr
    fast_args = fast.stdout.splitlines()
    assert "--no-enable-chunked-prefill" in fast_args
    assert "--no-enable-prefix-caching" in fast_args
    assert fast_args.count("--max-num-seqs") == 1
    max_index = fast_args.index("--max-num-seqs")
    assert fast_args[max_index + 1] == "1"
    print("FAST_PROFILE_FIXED_MAX_NUM_SEQS_1_AND_UNCHUNKED_OK")

    for allowed in (("--max-num-seqs", "1"), ("--max-num-seqs=1",), ("--max-num-seqs", "1", "--max-num-seqs=1")):
        result = invoke("fast_attention", *allowed)
        assert result.returncode == 0, (allowed, result.stderr)
        values = result.stdout.splitlines()
        assert values.count("--max-num-seqs") == 1, (allowed, values)
        index = values.index("--max-num-seqs")
        assert values[index + 1] == "1", (allowed, values)
    print("FAST_PROFILE_ACCEPTS_BOTH_MAX_NUM_SEQS_1_FORMS_AND_NORMALIZES_DUPLICATES")

    invalid = (
        ("--max-num-seqs",),
        ("--max-num-seqs", "2"),
        ("--max-num-seqs=2",),
        ("--max-num-seqs=",),
        ("--max-num-seqs", "--port", "9000"),
        ("--max-num-seqs", "1", "--max-num-seqs=2"),
        ("--enable-chunked-prefill",),
        ("--enable-prefix-caching",),
        ("--enable-chunked-prefill=true",),
    )
    for args in invalid:
        result = invoke("fast_attention", *args)
        assert result.returncode != 0 and "FATAL" in result.stderr, (args, result.returncode, result.stderr)
    assert invoke("unknown").returncode != 0
    print("FAST_PROFILE_REJECTS_INVALID_OR_MISSING_MAX_NUM_SEQS_AND_UNSAFE_FLAGS")

if len(sys.argv)>1:
    parser=Path(sys.argv[1]);sys.path.insert(0,str(parser.parent))
    spec=importlib.util.spec_from_file_location('actual_official_ci',parser)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    got=module.parse_run_sh(str(b/'run.sh'))
    assert got=={'model_name':'MiniCPM5-1B','model_path':'/gpfs/model/OpenBMB/MiniCPM5-1B','port':8000,'host':'0.0.0.0'},got
    print('ACTUAL_OFFICIAL_PARSE_RUN_SH_OK '+json.dumps(got))
    print('PARSER_SOURCE_SHA256 '+hashlib.sha256(parser.read_bytes()).hexdigest())
