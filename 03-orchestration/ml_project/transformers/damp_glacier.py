from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import mlflow
import pickle

#mlflow.set_tracking_uri("http://172.17.0.1:5000")
mlflow.set_tracking_uri("file:./mlruns")
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

    with mlflow.start_run():
        #lr = LinearRegression()
        with open ("/home/src/ml_project/models/lin_reg.bin", "rb") as f:
            lr = pickle.load(f)

        # lr.fit(X_train, y_train)

        # y_pred = lr.predict(X_train)

        mlflow.sklearn.log_model(lr, artifact_path="artifacts_local")
        #mlflow.log_artifact(local_path='models/lin_reg.bin', artifact_path='artifacts_local')
    #print(lr.intercept_)

    return data
    

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
