# BSD 3-Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
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

"""Gemma4 RMSNorm -> tecoops C++ kernel (delivery layer 3, model call site).

``GEMMA4_RMSNORM_STATS`` (read once, at import) selects the shipped form:
**``whole`` is the default** = the delivered single-kernel whole chain
(``tecoops.rms_norm``); ``compat`` = the reference-exact split
(``tecoops.rms_norm_compat`` + eager statistics), kept as a fallback switch.
See ``op.py`` for the measurement behind each.
"""
from custom_ops.rms_norm.op import (
    STATS_MODE,
    TecoopsRMSNorm,
    mapped_libteco_ops,
    rms_norm_epilogue,
    rms_norm_epilogue_add,
    rms_norm_whole,
    rms_norm_whole_add,
)

__all__ = ["TecoopsRMSNorm", "STATS_MODE", "rms_norm_epilogue", "rms_norm_epilogue_add",
           "rms_norm_whole", "rms_norm_whole_add", "mapped_libteco_ops"]
