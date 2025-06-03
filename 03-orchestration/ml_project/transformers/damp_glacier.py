from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import mlflow
import pickle

mlflow.set_tracking_uri("http://172.17.0.1:5000")
mlflow.set_experiment("nyc-taxi-experiment")

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your transformation logic here

    categorical = ['PULocationID', 'DOLocationID']

    train_dicts = data[categorical].to_dict(orient='records')

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    
    target = 'duration'
    y_train = data[target].values

    with mlflow.start_run():
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_train)

        with open ("models/lin_reg.bin", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact(local_path='models/lin_reg.bin', artifact_path='artifacts_local')

    #print(lr.intercept_)

    return data
    

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
