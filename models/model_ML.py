import xgboost as xgb
from sklearn.model_selection import train_test_split

def model_XGB(X, y):
    best_params = {
        'objective': 'reg:squarederror',
        'subsample': 1.0,
        'reg_lambda': 1.0,
        'reg_alpha': 0.05,
        'n_estimators': 1500,
        'min_child_weight': 6,
        'max_depth': 10,
        'learning_rate': 0.01,
        'gamma': 0.1,
        'colsample_bytree': 0.7,
        'n_jobs': -1,
    }

    model = xgb.XGBRegressor(**best_params)

    model.fit(X, y)

    return model
