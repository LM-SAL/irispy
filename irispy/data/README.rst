Data directory
**************

This directory contains data files included with the package source code distribution.
Note that this is intended only for relatively small files - large files should be externally hosted and downloaded as needed.

``iris_lines.ecsv`` contains NIST and CHIANTI transitions, approximate reference-model strength rankings, and published IRIS line identifications.
Query it with `irispy.utils.lines.get_lines`; regenerate it with ``tools/make_line_database.py``.
