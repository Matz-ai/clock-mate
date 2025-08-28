from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_union
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import RobustScaler
from sklearn.compose import make_column_selector



def scaling(X):

    num_transformer = make_pipeline(RobustScaler())

    pipelinee = make_column_transformer(
    (num_transformer, make_column_selector(dtype_include=['float64','int64'])),
    remainder='passthrough'
)

    X_scaled = pipelinee.fit_transform(X)
    return X_scaled
