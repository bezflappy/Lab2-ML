import unittest
from pathlib import Path

from correlation_tasks import analyze_brain_correlations, plot_brain_correlations
from grader_contracts.correlation_tasks import BrainDataInput

try:
    import matplotlib.pyplot  # noqa: F401
    import seaborn  # noqa: F401
except ImportError:
    PLOTTING_AVAILABLE = False
else:
    PLOTTING_AVAILABLE = True


DATA = Path(__file__).parent / "brainsize.txt"


class BrainCorrelationTests(unittest.TestCase):
    def test_group_counts_and_features(self):
        result = analyze_brain_correlations(BrainDataInput(str(DATA)))

        self.assertEqual((result.men_count, result.women_count), (20, 20))
        self.assertEqual(
            set(result.men_mri_correlation),
            {"FSIQ", "VIQ", "PIQ", "Weight", "Height"},
        )
        self.assertEqual(result.strongest_mri_feature, "PIQ")

    def test_correlations_ignore_missing_values(self):
        result = analyze_brain_correlations(BrainDataInput(str(DATA)))

        self.assertAlmostEqual(result.women_mri_correlation["Weight"], 0.4462713526616431)
        self.assertAlmostEqual(result.men_mri_correlation["PIQ"], 0.5682370032144163)

    @unittest.skipUnless(PLOTTING_AVAILABLE, "Matplotlib or Seaborn is not installed")
    def test_visual_output(self):
        """Show scatter plots and a heatmap; close the windows to finish the tests."""
        figures = plot_brain_correlations(BrainDataInput(str(DATA)))
        self.assertEqual(len(figures), 2)


if __name__ == "__main__":
    unittest.main()
