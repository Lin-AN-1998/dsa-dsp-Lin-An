from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.preprocess import preprocess


def inference(models_dir: Path, model_name: str = 'RandomizedSearchCV',
              data: pd.DataFrame = None, filepath: Path = None) -> np.ndarray:
    if filepath:
        data = pd.read_csv(filepath, index_col='Id')
    processed_data = preprocess(data, models_dir)
    model = joblib.load(models_dir / f'{model_name}.joblib')
    predictions = model.predict(processed_data)
    return predictions
