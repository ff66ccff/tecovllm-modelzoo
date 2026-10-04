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

"""Mutation/Fake boundary for the existing Hy-MT2 vendor RMSNorm API."""

import torch
import tecoops


@torch.library.custom_op(
    "hy_mt2_public::rms_norm", mutates_args=("out",), device_types="sdaa"
)
def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    out: torch.Tensor,
    eps: float,
) -> None:
    tecoops.rms_norm(x, weight, None, out, None, eps)


@rms_norm.register_fake
def _rms_norm_fake(x, weight, out, eps):
    return None


@torch.library.custom_op(
    "hy_mt2_public::rms_norm_add",
    mutates_args=("out", "residual_out"),
    device_types="sdaa",
)
def rms_norm_add(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    residual: torch.Tensor,
    out: torch.Tensor,
    residual_out: torch.Tensor,
    eps: float,
) -> None:
    tecoops.rms_norm(x, weight, residual, out, residual_out, eps)


@rms_norm_add.register_fake
def _rms_norm_add_fake(x, weight, residual, out, residual_out, eps):
    return None
