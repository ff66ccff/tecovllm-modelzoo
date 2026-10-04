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

"""Init-time-bound call accounting for the official `tecoops` ABI.

Purpose
-------
Leg 1 of CP3 requires proof that the official operators are *actually invoked*
inside a real model forward and that the invocation counts match the model's
layer/step arithmetic.  A single "I was imported" receipt cannot prove that, so
this module binds a per-API call counter.

Hot-path contract (AGENTS.md rule 6)
------------------------------------
All decision making happens exactly once, at module import (process
initialization):

* ``TECOOPS_CALL_RECEIPT`` is read once via ``os.environ`` at import time.
* When it is unset the *original* pybind function object is returned unchanged,
  so production forwards contain no counter, no branch, no ``getenv`` and no
  extra Python frame.
* When it is set a counting binding is returned.  Its forward path performs one
  dict read + one dict write; there is no backend/fallback ``if-else``.

The tally is flushed on normal interpreter exit and on ``SIGTERM``/``SIGINT``
so a gracefully stopped engine core still yields exact numbers.  If the file is
absent the harness must report *blocked* rather than invent a count.
"""

import atexit
import os
import signal
import sys

_RECEIPT_PATH = os.environ.get("TECOOPS_CALL_RECEIPT")


class _Tally:
    """Mutable per-process call counters for the official ABI."""

    __slots__ = ("counts", "path", "_flushed")

    def __init__(self, path: str) -> None:
        self.counts: dict[str, int] = {}
        self.path = path
        self._flushed = False

    def bump(self, name: str) -> None:
        counts = self.counts
        counts[name] = counts.get(name, 0) + 1

    def flush(self, *_args: object) -> None:
        if self._flushed:
            return
        self._flushed = True
        try:
            with open(self.path, "a", encoding="utf-8") as handle:
                for name in sorted(self.counts):
                    handle.write(f"CALL {name}\n")
                    handle.write(f"COUNT {name} {self.counts[name]}\n")
        except OSError as exc:  # pragma: no cover - diagnostics must not crash
            sys.stderr.write(f"[custom_ops.receipt] flush failed: {exc}\n")

    def snapshot(self) -> dict[str, int]:
        return dict(self.counts)


def _make_terminating_handler(tally: "_Tally"):
    """Flush the tally, then re-raise the signal with default disposition."""

    def handler(signum, _frame):
        tally.flush()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    return handler


_TALLY = _Tally(_RECEIPT_PATH) if _RECEIPT_PATH else None

if _TALLY is not None:
    atexit.register(_TALLY.flush)
    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_sig, _make_terminating_handler(_TALLY))
        except (ValueError, OSError):  # non-main thread or restricted platform
            pass


def receipt_enabled() -> bool:
    """True when this process was launched with call accounting enabled."""
    return _TALLY is not None


def bind_official_api(fn, name: str):
    """Return ``fn`` unchanged, or a counting binding when receipts are on.

    Called during process initialization only.
    """
    if _TALLY is None:
        return fn

    tally = _TALLY

    def counted(*args, **kwargs):
        tally.bump(name)
        return fn(*args, **kwargs)

    counted.__name__ = "counted_" + name.replace(".", "_")
    counted.__doc__ = f"Counting binding for {name} (diagnostic launch only)."
    counted.__wrapped__ = fn
    return counted


def snapshot() -> dict[str, int]:
    """In-process view of the tally (used by single-process harnesses)."""
    return _TALLY.snapshot() if _TALLY is not None else {}


def receipt_path() -> str | None:
    return _RECEIPT_PATH
