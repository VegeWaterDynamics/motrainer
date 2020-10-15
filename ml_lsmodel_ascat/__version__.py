import os
import sys

# To update the package version number, edit CITATION.cff
here = os.path.abspath(os.path.dirname(__file__))
citationfile = os.path.join(here, '..', 'CITATION.cff')
with open(citationfile, 'r') as cff:
    for line in cff:
        if 'version:' in line:
            __version__ = line.replace('version:', '').strip().strip('"')
