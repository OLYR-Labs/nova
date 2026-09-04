import tempfile
import unittest
from pathlib import Path

from core.self_developer import SelfDevelopmentEngine
from core.thought_engine import ThoughtEngine


class SelfDevelopmentTests(unittest.TestCase):

    def test_thought_state_persists(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = ThoughtEngine(directory)
            engine.reset("test goal")
            engine.transition("CODE", "write module", "verify")
            engine.observe("source inspected")
            engine.record_failure({"error": "example"})
            engine.learn("always verify generated code")

            restored = ThoughtEngine(directory)
            state = restored.snapshot()

            self.assertEqual(state["goal"], "test goal")
            self.assertEqual(state["phase"], "CODE")
            self.assertEqual(state["current_step"], "write module")
            self.assertIn("source inspected", state["observations"])
            self.assertEqual(state["failures"][0]["error"], "example")
            self.assertIn("always verify generated code", state["lessons"])

    def test_path_boundary_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = SelfDevelopmentEngine(directory)
            with self.assertRaises(PermissionError):
                engine.workspace.safe_path("../outside.py")

    def test_sensitive_files_are_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = SelfDevelopmentEngine(directory)
            self.assertFalse(engine._allowed(".env"))
            self.assertFalse(engine._allowed(".env.local"))
            self.assertFalse(engine._allowed("secrets.json"))
            self.assertTrue(engine._allowed("core/example.py"))


if __name__ == "__main__":
    unittest.main()
