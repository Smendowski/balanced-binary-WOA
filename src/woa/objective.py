import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def mse_objective_fn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    selected_features: np.ndarray,
) -> float:
    # selected features is a one hot vector that needs to be mapped into list with indices
    selected_features = np.where(np.array(selected_features) == 1)[0].tolist()
    X_train_subset = X_train.iloc[:, selected_features]
    X_test_subset = X_test.iloc[:, selected_features]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_subset, y_train)

    y_pred = model.predict(X_test_subset)

    return mean_squared_error(y_test, y_pred)
