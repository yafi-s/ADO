"""Run the display's object-memory tests, which are written for Node, with the rest of the suite."""
from pathlib import Path
import shutil
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("node"), "needs Node")
class ObjectMemoryTests(unittest.TestCase):
    def test_object_memory(self):
        run = subprocess.run(["node", "--test", "tests/test_object_memory.mjs"], cwd=ROOT,
                             capture_output=True, text=True, encoding="utf-8", timeout=120)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == "__main__":
    unittest.main()
