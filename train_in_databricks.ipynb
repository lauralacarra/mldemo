import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

def main(data_path="https://azuremlexamples.blob.core.windows.net/datasets/credit_card/default_of_credit_card_clients.csv",
          test_train_ratio=0.25, n_estimators=100, learning_rate=0.1, registered_model_name="credit_card_model"):
    # Get the data
    credit_df = pd.read_csv(data_path, header=1, index_col=0)
    # split train and test 
    train_df, test_df = train_test_split(credit_df, test_size=test_train_ratio)

    y_train = train_df.pop("default payment next month")
    X_train = train_df.values

    y_test = test_df.pop("default payment next month")
    X_test = test_df.values

    # Train the model
    clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
    clf.fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy as a target loss metric
    accuracy = accuracy_score(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    print(cr)

    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    with mlflow.start_run():
        mlflow.log_metric("num_samples", credit_df.shape[0])
        mlflow.log_metric("num_features", credit_df.shape[1] - 1)
        mlflow.log_metric("accuracy", accuracy)

        signature = infer_signature(X_test, y_pred)

        # Registering the model via MLFlow
        mlflow.sklearn.log_model(
            sk_model=clf,
            registered_model_name=registered_model_name,
            artifact_path=registered_model_name,
            signature=signature
        )

if __name__ == "__main__":
    main()


