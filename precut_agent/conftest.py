# Repo root on sys.path so `import precut.x` works from the tests.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
