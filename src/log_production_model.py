from src.get_data import read_params
import argparse
import mlflow
from mlflow.tracking import MlflowClient
from pprint import pprint
import joblib
import os


def log_production_model(config_path):
    config = read_params(config_path)
    mlflow_config = config["mlflow_config"]
    
    model_name = mlflow_config["registered_model_name"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    # Set the tracking URI
    mlflow.set_tracking_uri(remote_server_uri)

    # Fetch all runs for the experiment
    experiment_id = mlflow.get_experiment_by_name(mlflow_config["experiment_name"]).experiment_id
    runs = mlflow.search_runs(experiment_ids=[experiment_id])

    if runs.empty:
        print("No runs found for the given experiment. Exiting.")
        return

    # Find the run with the lowest MAE
    if "metrics.mae" not in runs.columns:
        print("No 'metrics.mae' column found in runs. Exiting.")
        return

    lowest = runs["metrics.mae"].min()
    lowest_run_id = runs[runs["metrics.mae"] == lowest]["run_id"].iloc[0]

    #print(f'The lowest run id is {lowest_run_id}')

    client = MlflowClient()

    logged_model = None
    for mv in client.search_model_versions(f"name='{model_name}'"):
        mv = dict(mv)
        if mv["run_id"] == lowest_run_id:
            current_version = mv["version"]
            logged_model = mv["source"]
            pprint(mv, indent=4)
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Production"
            )
            print(f"Model version {current_version} moved to Production.")
        else:
            current_version = mv["version"]
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Staging"
            )
            print(f"Model version {current_version} moved to Staging.")

    if logged_model:
        # Load the production model and save it locally
        loaded_model = mlflow.pyfunc.load_model(logged_model)
        model_path = config["webapp_model_dir"]
        joblib.dump(loaded_model, model_path)
        print(f"Model saved to {model_path}.")
    else:
        print("No valid model found to load.")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = log_production_model(config_path=parsed_args.config)