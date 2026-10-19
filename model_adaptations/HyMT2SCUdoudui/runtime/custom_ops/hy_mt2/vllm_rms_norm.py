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

"""C++ TORCH_LIBRARY binding for Hy-MT2 RMSNorm API (replaces Python custom_op)."""

import torch
import tecoops  # noqa: F401  ensure library loaded

# [c42-modeladopt] C++ TORCH_LIBRARY entry points replacing python custom_op (-50 us/call dispatch recovery)
_cpp_rms_norm_out = getattr(getattr(torch.ops, "tecoops_ops", None), "rms_norm_out", None)
_cpp_rms_norm_add_out = getattr(getattr(torch.ops, "tecoops_ops", None), "rms_norm_add_out", None)
if _cpp_rms_norm_out is None or _cpp_rms_norm_add_out is None:
    raise RuntimeError(
        "tecoops_ops.rms_norm_out / rms_norm_add_out not found in torch.ops (fail-closed)"
    )

rms_norm = _cpp_rms_norm_out
rms_norm_add = _cpp_rms_norm_add_out
