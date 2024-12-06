""" Test the TBEvaluator class."""
import pytest

from topobenchmark.evaluator import TBEvaluator

class TestTBEvaluator:
    """ Test the TBXEvaluator class."""

    def setup_method(self):
        """ Setup the test."""
        self.evaluator_multilable = TBEvaluator(task="multilabel classification")
        self.evaluator_regression = TBEvaluator(task="regression")
        with pytest.raises(ValueError):
            TBEvaluator(task="wrong")
        repr = self.evaluator_multilable.__repr__()