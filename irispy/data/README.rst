Data directory
**************

This directory contains data files included with the package source code distribution.
Note that this is intended only for relatively small files - large files should be externally hosted and downloaded as needed.

``iris_lines.ecsv`` is the IRIS spectral line database read by `irispy.utils.lines.get_lines`: NIST wavelengths with solar strengths predicted from CHIANTI, and the lines documented in the IRIS literature with their references.
It is regenerated with ``tools/make_line_database.py``; see the line database documentation.
