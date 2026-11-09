import unittest
from pathlib import Path

from grader_contracts.pandas_tasks import TitanicInput
from pandas_tasks import analyze_titanic


DATA = Path(__file__).parent / "titanic.csv"


class TitanicTests(unittest.TestCase):
    def test_dataset_summary(self):
        result = analyze_titanic(TitanicInput(str(DATA)))

        self.assertEqual(result.row_count, 891)
        self.assertEqual(result.missing_by_column["Age"], 177)
        self.assertEqual(result.missing_by_column["Cabin"], 687)
        self.assertEqual(result.adults_over_30_count, 305)
        self.assertEqual(result.highest_fares, [512.3292, 512.3292, 512.3292, 263.0, 263.0])

    def test_grouped_values(self):
        result = analyze_titanic(TitanicInput(str(DATA)))

        self.assertEqual(set(result.mean_age_by_pclass), {1, 2, 3})
        self.assertAlmostEqual(result.mean_age_by_pclass[1], 38.233440860215055)
        self.assertAlmostEqual(result.survival_rate_by_pclass[3], 0.24236252545824846)


if __name__ == "__main__":
    unittest.main()
