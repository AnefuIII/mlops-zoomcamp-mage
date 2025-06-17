import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import sklearn # Keep for sklearn.__version__
from pathlib import Path # To ensure models folder exists if this is run standalone

# This decorator tells Mage that this function is a custom block
# It receives outputs from upstream blocks
@custom
def train_linear_model(data_tuple, *args, **kwargs):
    """
    Trains a Linear Regression model and logs it with MLflow.

    Args:
        data_tuple (tuple): A tuple containing (X, y, dv) from the prepare_features block.
    """
    X, y, dv = data_tuple # Unpack the tuple from the previous block

    # MLflow Setup (can also be done via environment variables or pipeline settings)
    # Ensure MLflow Tracking Server is running at http://localhost:5000 in your Codespace
    mlflow.set_tracking_uri('http://localhost:5000')
    mlflow.set_experiment('nyc-orchestration')

    # Ensure models_folder exists if not already
    models_folder = Path('models')
    models_folder.mkdir(exist_ok=True) # Mage often creates a default artifact dir, but good for local saves

    with mlflow.start_run() as run:
        mlflow.set_tag('Developer', 'OcheAI')
        mlflow.log_param("sklearn_version", sklearn.__version__)

        model = LinearRegression()
        model.fit(X, y)
        print("Linear Regression model trained.")

        # Log intercept and coefficients
        mlflow.log_metric("intercept", model.intercept_)
        print(f"The intercept of the Linear Regression model is: {model.intercept_}")

        # You've already saved DictVectorizer in prepare_features.py,
        # so here you can just log it as an artifact from its saved location.
        mlflow.log_artifact(local_path='models/DictVectorizer.b', artifact_path='DictVectorizer')
        print("DictVectorizer logged as MLflow artifact.")

        # Evaluate model and log metrics
        y_pred = model.predict(X) # Predicting on training data for logging purposes here,
                                # ideally you'd use a validation set
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mlflow.log_metric("rmse", rmse)
        print(f"RMSE (on training data): {rmse}")

        # Log the scikit-learn LinearRegression model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="lin_reg_model",
            signature=mlflow.models.infer_signature(X, y_pred),
            input_example=X[:5].toarray() if hasattr(X, 'toarray') else X[:5] # For sparse matrices
        )
        print("Linear Regression model logged to MLflow.")

        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"MLflow Artifact URI: {mlflow.get_artifact_uri()}")