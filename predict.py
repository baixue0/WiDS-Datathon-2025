import pandas as pd
import pickle
import os
import json
import numpy as np

with open('config/SETTINGS.json') as f:
    config = json.load(f)

test = pd.read_csv(config['TEST_DATA_CLEAN_PATH'])

with open(os.path.join(config['MODEL_DIR'], 'best_model.pkl'), 'rb') as f:
    model_type, model, scaler = pickle.load(f)

if model_type == 'autoencoder':
    test_scaled = scaler.transform(test)
    reconstructions = model.predict(test_scaled)
    mse = np.mean((test_scaled - reconstructions) ** 2, axis=1)
    preds = mse
else:
    preds = model.predict_proba(test)[:, 1]

os.makedirs(config['SUBMISSION_DIR'], exist_ok=True)
pd.DataFrame({'prediction': preds}).to_csv(os.path.join(config['SUBMISSION_DIR'], 'submission.csv'), index=False)
