import json
from pathlib import Path
import unittest
import joblib
import numpy as np
import pandas as pd
from modeling import FreightModel, features, fill_coordinates
from score import validate_predictions, validate_december

ROOT = Path(__file__).resolve().parent

class SolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = pd.read_csv(ROOT / 'train-test.csv')
        cls.validation = pd.read_csv(ROOT / 'validation.csv')
        cls.selection = json.loads((ROOT / 'artifacts/selection.json').read_text())
        cls.model = joblib.load(ROOT / f"artifacts/{cls.selection['primary']}.joblib")

    def test_official_contract(self):
        validate_predictions(pd.read_csv(ROOT / 'validation_predictions.csv'))
        validate_december(pd.read_csv(ROOT / 'data/december_chart_inputs.csv'))

    def test_predictions_match_saved_model_and_ids(self):
        output = pd.read_csv(ROOT / 'validation_predictions.csv').set_index('load_id')
        expected = self.model.predict(self.validation)
        np.testing.assert_allclose(output.loc[self.validation.load_id, 'predicted_rate'], expected, atol=.00501, rtol=0)

    def test_target_and_id_cannot_influence_features(self):
        batch = self.validation.head(20).copy()
        expected = self.model.predict(batch)
        batch['posted_rate'] = -999999
        batch['load_id'] = 'unrelated-id'
        np.testing.assert_array_equal(expected, self.model.predict(batch))

    def test_missing_invalid_weight_and_unseen_city(self):
        batch = self.validation.head(3).copy()
        batch.loc[batch.index[0], 'weight'] = np.nan
        batch.loc[batch.index[1], 'weight'] = -32000
        batch.loc[batch.index[2], 'pickup'] = 'Unseen city'
        original = self.model.medians.copy()
        pred = self.model.predict(batch)
        self.assertTrue(np.isfinite(pred).all() and (pred > 0).all())
        pd.testing.assert_series_equal(original, self.model.medians)
        self.assertTrue(features(batch).weight.iloc[:2].isna().all())

    def test_batch_independence(self):
        batch = self.validation.head(4)
        np.testing.assert_allclose(self.model.predict(batch), [self.model.predict(batch.iloc[[i]])[0] for i in range(4)], atol=1e-8)

    def test_december_input_preserved_and_model_reproducible(self):
        original = pd.read_csv(ROOT / 'december-chart-inputs.csv')
        completed = pd.read_csv(ROOT / 'data/december_chart_inputs.csv')
        pd.testing.assert_frame_equal(original.drop(columns='predicted_rate'), completed.drop(columns='predicted_rate'))
        model = joblib.load(ROOT / f"artifacts/{self.selection['core']}.joblib")
        pred = model.predict(fill_coordinates(original, self.train))
        np.testing.assert_allclose(completed.predicted_rate, pred, atol=.00501, rtol=0)

if __name__ == '__main__':
    unittest.main()
