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

"""CPU-only checks for public D512 build provenance validation."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d512_build_provenance import validate_build_provenance


class BuildProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.vendor_python = Path("/home/py312/bin/python").resolve(strict=True)
        self.extension_sha256 = hashlib.sha256(b"selected extension bytes").hexdigest()
        self.core_sha256 = hashlib.sha256(b"selected core bytes").hexdigest()
        self.receipt = {
            "official_base": "de27305efed0a17ae926d21d5415d8b915614649",
            "patch_sha256": "9756790105110f8fd4ed610d8e12c9269d2992a331fb8a0a530134204b3b7600",
            "python": str(self.vendor_python),
            "extension": "/original/build/path/_torch_ext.so",
            "extension_sha256": self.extension_sha256,
            "core": "/original/build/path/libteco_gemma_flash.so",
            "core_sha256": self.core_sha256,
        }
        self.receipt_path = self.root / "provenance.json"
        self.write_receipt()

    def write_receipt(self):
        self.receipt_path.write_text(json.dumps(self.receipt), encoding="utf-8")

    def validate(self, **kwargs):
        return validate_build_provenance(
            self.receipt_path,
            extension_sha256=self.extension_sha256,
            core_sha256=self.core_sha256,
            **kwargs,
        )

    def test_accepts_selected_bytes_after_copy_to_a_different_path(self):
        self.assertEqual(self.validate()["official_base"], self.receipt["official_base"])

    def test_rejects_wrong_official_base(self):
        self.receipt["official_base"] = "0" * 40
        self.write_receipt()
        with self.assertRaisesRegex(ValueError, "official_base"):
            self.validate()

    def test_rejects_wrong_patch_digest(self):
        self.receipt["patch_sha256"] = "0" * 64
        self.write_receipt()
        with self.assertRaisesRegex(ValueError, "patch_sha256"):
            self.validate()

    def test_rejects_wrong_provenance_python_realpath(self):
        other_python = self.root / "other-python"
        other_python.write_bytes(b"not a real interpreter")
        self.receipt["python"] = str(other_python)
        self.write_receipt()
        with self.assertRaisesRegex(ValueError, "Python realpath"):
            self.validate()

    def test_rejects_wrong_running_python_realpath(self):
        other_python = self.root / "other-running-python"
        other_python.write_bytes(b"not a real interpreter")
        with self.assertRaisesRegex(ValueError, "running Python realpath"):
            self.validate(python_executable=str(other_python))

    def test_rejects_extension_or_core_digest_mismatch(self):
        self.receipt["extension_sha256"] = "0" * 64
        self.write_receipt()
        with self.assertRaisesRegex(ValueError, "extension_sha256"):
            self.validate()

        self.receipt["extension_sha256"] = self.extension_sha256
        self.receipt["core_sha256"] = "0" * 64
        self.write_receipt()
        with self.assertRaisesRegex(ValueError, "core_sha256"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
