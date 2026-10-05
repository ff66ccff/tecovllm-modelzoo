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
"""Test-only worker receipt for InternVL context-budget validation."""
import hashlib
import importlib
from pathlib import Path

from custom_ops.peak_probe import PeakProbe


class InternVLBudgetProbe(PeakProbe):
    def read_scheduler_config(self):
        config = self.vllm_config
        scheduler = config.scheduler_config
        model = config.model_config
        source = Path(importlib.import_module("vllm.engine.arg_utils").__file__).resolve()
        runner = getattr(self, "model_runner", None)
        model_object = getattr(runner, "model", None)
        try:
            parameter = next(model_object.parameters())
            weight_device = str(parameter.device)
        except (AttributeError, StopIteration, TypeError):
            weight_device = None
        import torch

        return {
            "rank": int(self.rank),
            "local_rank": int(self.local_rank),
            "max_model_len": int(model.max_model_len),
            "max_num_batched_tokens": int(scheduler.max_num_batched_tokens),
            "max_num_seqs": int(scheduler.max_num_seqs),
            "enable_chunked_prefill": bool(scheduler.enable_chunked_prefill),
            "model_dtype": str(model.dtype),
            "tensor_parallel_size": int(config.parallel_config.tensor_parallel_size),
            "configured_device": str(config.device_config.device),
            "weight_device": weight_device,
            "sdaa_current_device": int(torch.sdaa.current_device()),
            "arg_utils_path": str(source),
            "arg_utils_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "default_resolution_lines": "2354-2405",
        }
