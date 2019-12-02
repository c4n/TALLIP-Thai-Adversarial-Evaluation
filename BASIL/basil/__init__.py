#Make sure that Python 3.5 is the minimum requirement
import sys
if sys.version_info < (3, 5):
    raise RuntimeError("BASIL requires Python 3.5 or later")

from basil.version import VERSION as __version__
