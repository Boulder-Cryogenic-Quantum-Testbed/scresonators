import cryores.analysis.base as analysis
import cryores.experiments.data.base as data
import cryores.instruments.base as instruments


class ResonatorMinAnalyzer(analysis.Analyzer):
    def analyze(self, dataset: data.Dataset) -> analysis.AnalysisResults:
        # Simply grab the minimum value from the data.
        return analysis.AnalysisResults(
            lowest_resonance=dataset.data[instruments.COL_S21_AMP].min())
