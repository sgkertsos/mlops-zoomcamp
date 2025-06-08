import mlflow
import pandas as pd
logged_model = 'runs:/1b4de8d17b0143efac006cdb84900efd/models_pickle'

@data_loader
def load_data_from_api(*args, **kwargs):
    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    return loaded_model

# Data

# Predict on a Pandas DataFrame.
#loaded_model.predict(pd.DataFrame(data))