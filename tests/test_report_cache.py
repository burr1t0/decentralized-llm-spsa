import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from src import real_benchmarks, report_figures


class TestPolicyCache(unittest.TestCase):
    def test_cache_is_parquet_and_round_trips(self):
        rows = [{"method": "a_spsa", "seed": 3, "rho": 0.85, "phi": 0.12},
                {"method": "spsa", "seed": 4, "rho": 0.85, "phi": 0.34}]
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(report_figures, "DATA", Path(tmp)), \
                    mock.patch.object(report_figures, "_run_jobs", return_value=rows) as run, \
                    mock.patch.object(report_figures, "_policy_cfg", return_value={"rho": 0.85, "test_seeds": [3, 4]}), \
                    mock.patch.object(report_figures, "_policy_jobs", return_value=[]):
                first = report_figures.policy_operational_data(1)
                second = report_figures.policy_operational_data(1)
            files = sorted(p.suffix for p in Path(tmp).iterdir())
        self.assertEqual(files, [".parquet"])
        self.assertEqual(run.call_count, 1)  # second call read the cache
        pd.testing.assert_frame_equal(first, second)

    def test_report_figures_has_no_pickle(self):
        text = Path(report_figures.__file__).read_text(encoding="utf-8")
        self.assertNotIn("read_pickle", text)
        self.assertNotIn("to_pickle", text)


class _Evil:
    def __reduce__(self):
        import os
        return (os.system, ("echo pwned > /dev/null",))


class TestRestrictedPickle(unittest.TestCase):
    def _dump(self, obj, path):
        with open(path, "wb") as handle:
            pickle.dump(obj, handle)

    def test_dataframe_round_trips(self):
        frame = pd.DataFrame({"sample_id": ["a", "b"], "m|total_cost": [0.1, 0.2], "n": [1, 2]})
        frame["when"] = pd.to_datetime(["2024-01-01", "2024-01-02"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.pkl"
            frame.to_pickle(path)
            pd.testing.assert_frame_equal(real_benchmarks.read_trusted_dataframe(path), frame)

    def test_code_execution_payload_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evil.pkl"
            self._dump(_Evil(), path)
            with self.assertRaises(pickle.UnpicklingError):
                real_benchmarks.read_trusted_dataframe(path)

    def test_non_dataframe_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "list.pkl"
            self._dump([1, 2, 3], path)
            with self.assertRaises(ValueError):
                real_benchmarks.read_trusted_dataframe(path)


if __name__ == "__main__":
    unittest.main()
