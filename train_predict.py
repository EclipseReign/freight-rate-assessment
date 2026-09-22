"""Refit frozen model choices on all development data and create submission files."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
from modeling import FreightModel, fill_coordinates

ROOT = Path(__file__).resolve().parent

def main():
    artifacts = ROOT / 'artifacts'
    selection = json.loads((artifacts / 'selection.json').read_text())
    train = pd.read_csv(ROOT / 'train-test.csv')
    validation = pd.read_csv(ROOT / 'validation.csv')
    template = pd.read_csv(ROOT / 'validation-predictions-template.csv')
    december = pd.read_csv(ROOT / 'december-chart-inputs.csv')
    assert train.load_id.is_unique and validation.load_id.is_unique and template.load_id.is_unique
    assert set(template.load_id) == set(validation.load_id)
    models = {}
    for name in dict.fromkeys([selection['primary'], selection['core']]):
        models[name] = FreightModel(**selection['configurations'][name]).fit(train)
        joblib.dump(models[name], artifacts / f'{name}.joblib')
    predictions = pd.Series(models[selection['primary']].predict(validation), index=validation.load_id)
    template['predicted_rate'] = template.load_id.map(predictions)
    assert np.isfinite(template.predicted_rate).all() and template.predicted_rate.gt(0).all()
    template.to_csv(ROOT / 'validation_predictions.csv', index=False, float_format='%.2f')
    chart_inputs = fill_coordinates(december, train)
    december['predicted_rate'] = models[selection['core']].predict(chart_inputs)
    data = ROOT / 'data'
    data.mkdir(exist_ok=True)
    december.to_csv(data / 'december_chart_inputs.csv', index=False, float_format='%.2f')
    subprocess.run([sys.executable, str(ROOT / 'score.py'), '--predictions', str(ROOT / 'validation_predictions.csv'),
        '--december-predictions', str(data / 'december_chart_inputs.csv'), '--output-dir', str(ROOT / 'scorer_results')], check=True)
    sources = ['train-test.csv', 'validation.csv', 'validation-predictions-template.csv',
        'december-chart-inputs.csv', 'score.py', 'freight-rate-ml-assessment.pdf', 'readme.md', 'requirements.txt']
    manifest = dict(seed=42, python=sys.version, input_sha256={name:hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
        versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','catboost','matplotlib','reportlab','joblib']},
        selected_models={k:selection[k] for k in ['primary','core']})
    (artifacts / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Submission files and fitted models saved.', flush=True)

if __name__ == '__main__':
    main()
