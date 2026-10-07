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

"""Validate a selected Gemma D512 extension against its build receipt."""
import json
from pathlib import Path
import sys

EXPECTED_OFFICIAL_BASE = "de27305efed0a17ae926d21d5415d8b915614649"
EXPECTED_PATCH_SHA256 = "9756790105110f8fd4ed610d8e12c9269d2992a331fb8a0a530134204b3b7600"
VENDOR_PYTHON = Path("/home/py312/bin/python")


def _resolved_path(value, field):
    if not isinstance(value, str) or not value:
        raise ValueError(f"provenance {field} must be a non-empty path")
    try:
        return Path(value).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"provenance {field} path cannot be resolved") from exc


def validate_build_provenance(
    provenance_path,
    *,
    extension_sha256,
    core_sha256,
    python_executable=None,
):
    """Validate build identities and the bytes of the currently selected libraries.

    Provenance extension/core paths are informational: copied files are accepted
    when their selected bytes match the recorded digests.
    """
    try:
        with Path(provenance_path).open(encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("build provenance must be readable JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("build provenance must be a JSON object")

    if data.get("official_base") != EXPECTED_OFFICIAL_BASE:
        raise ValueError("provenance official_base mismatch")
    patch_sha256 = data.get("patch_sha256")
    if not isinstance(patch_sha256, str) or patch_sha256.lower() != EXPECTED_PATCH_SHA256:
        raise ValueError("provenance patch_sha256 mismatch")

    vendor_python = _resolved_path(str(VENDOR_PYTHON), "vendor Python")
    current_python = _resolved_path(
        str(python_executable or sys.executable), "running Python"
    )
    recorded_python = _resolved_path(data.get("python"), "python")
    if current_python != vendor_python:
        raise ValueError("running Python realpath is not the vendor Python")
    if recorded_python != vendor_python:
        raise ValueError("provenance Python realpath is not the vendor Python")

    for field, selected_digest in (
        ("extension_sha256", extension_sha256),
        ("core_sha256", core_sha256),
    ):
        recorded_digest = data.get(field)
        if (
            not isinstance(recorded_digest, str)
            or not isinstance(selected_digest, str)
            or recorded_digest.lower() != selected_digest.lower()
        ):
            raise ValueError(f"provenance {field} does not match selected bytes")
    return data
