import argparse
import logging
import os

import mlflow
import mlflow.keras
from src.utils.all_utils import create_directory, read_yaml
from src.utils.callbacks import get_callbacks
from src.utils.data_management import train_valid_generator
from src.utils.models import get_unique_path_to_save_model, load_full_model

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'stage_04_log.log'), level=logging.INFO, format=logging_str,
                    filemode="a")

def train_model(config_path: str, params_path: str):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]

    train_model_dir_path = os.path.join(artifacts_dir, artifacts["TRAINED_MODEL_DIR"])

    create_directory([train_model_dir_path])

    untrained_full_model_path = os.path.join(artifacts_dir, artifacts["BASE_MODEL_DIR"], artifacts["UPDATED_BASE_MODEL_NAME"])

    model = load_full_model(untrained_full_model_path)

    callback_dir_path  = os.path.join(artifacts_dir, artifacts["CALLBACKS_DIR"])
    callbacks = get_callbacks(callback_dir_path)

    train_generator, valid_generator = train_valid_generator(
        data_dir=artifacts["DATA_DIR"],
        IMAGE_SIZE=tuple(params["IMAGE_SIZE"][:-1]),
        BATCH_SIZE=params["BATCH_SIZE"],
        do_data_augmentation=params["AUGMENTATION"]
    )

    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = valid_generator.samples // valid_generator.batch_size

    with mlflow.start_run():
        mlflow.log_params(params)

        model.fit(
            train_generator,
            validation_data=valid_generator,
            epochs=params["EPOCHS"], 
            steps_per_epoch=steps_per_epoch, 
            validation_steps=validation_steps,
            callbacks=callbacks
        )

        mlflow.keras.log_model(model, "model")

        trained_model_dir = os.path.join(artifacts_dir, artifacts["TRAINED_MODEL_DIR"])
        create_directory([trained_model_dir])

        model_file_path = get_unique_path_to_save_model(trained_model_dir)
        model.save(model_file_path)

        metrics = {
                    'val_loss': model.history.history['val_loss'][-1],
                    'val_acc': model.history.history['val_accuracy'][-1],
                    'train_loss': model.history.history['loss'][-1],
                    'train_acc': model.history.history['accuracy'][-1]
                }
        mlflow.log_metrics(metrics)

        mlflow.log_artifact(model_file_path, "model")
        
        logging.info(f"training completed and model is saved with run_id: {mlflow.active_run().info.run_id}")

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")

    parsed_args = args.parse_args()

    try:
        logging.info(">>>>> stage four started")
        train_model(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info("stage four completed! training completed and model is saved >>>>>\n\n")
    except Exception as e:
        logging.exception(e)
        raise e
