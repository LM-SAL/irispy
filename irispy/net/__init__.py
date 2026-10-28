"""
Importing `irispy.net` adds `irispy.net.IRISClient` to ``Fido`` and the
`irispy.net.attrs` to `sunpy.net.attrs` as ``a.iris``.
"""

from irispy.net.client import IRISClient

__all__ = ["IRISClient"]
