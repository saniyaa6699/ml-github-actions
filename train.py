import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Start MLflow run
mlflow.start_run()

# Load data
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Log parameters
mlflow.log_param("model", "RandomForest")

# Log metrics
accuracy = model.score(X_test, y_test)
mlflow.log_metric("accuracy", accuracy)

# Save model
mlflow.sklearn.log_model(model, "model")

# End run
mlflow.end_run()