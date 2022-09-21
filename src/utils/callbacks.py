import tensorflow as tf
import os
import joblib
import logging
from src.utils.all_utils import get_timestamp

#-------------------------------------------------------------------------------
#   create_and_save_tensorboard_callback
#-------------------------------------------------------------------------------


def create_and_save_tensorboard_callback(callbacks_dir, tensorboard_log_dir):
    '''
    create_and_save_tensorboard_callback allows users to save their tenserflow callbacks from tensorboard_log_dir to callbacks_dir
    
    Parameters
    ----------
    INPUT : callbacks_dir : varible storing the callbacks_dir location  || DICT,
            tensorboard_log_dir : location to fetch the tensorboard_log_dir || STRING

    Output : The reports are saved in the  callbacks_dir location
    '''
    unique_name = get_timestamp("tb_logs")

    tb_running_log_dir = os.path.join(tensorboard_log_dir, unique_name)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)

    tb_callback_filepath = os.path.join(callbacks_dir, "tensorboard_cb.cb")
    joblib.dump(tensorboard_callback, tb_callback_filepath)
    logging.info(f"tensorboard callback is being saved at {tb_callback_filepath}")
    
#-------------------------------------------------------------------------------
#   create_and_save_checkpoint_callback
#-------------------------------------------------------------------------------

def create_and_save_checkpoint_callback(callbacks_dir, checkpoint_dir):
    '''
    create_and_save_checkpoint_callback allows users to save their tenserflow checkpint to checkpoint_dir
    
    Parameters
    ----------
    INPUT : callbacks_dir : varible storing the callbacks_dir location
            checkpoint_dir : location to store the checkpoint_dir

    Output : The reports are saved in the  callbacks_dir location
    '''
    checkpoint_file_path = os.path.join(checkpoint_dir, "ckpt_model.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_file_path,
        save_best_only=True
    )

    ckpt_callback_filepath = os.path.join(callbacks_dir, "checkpoint_cb.cb")
    joblib.dump(checkpoint_callback, ckpt_callback_filepath)
    logging.info(f"tensorboard callback is being saved at {ckpt_callback_filepath}")

#-------------------------------------------------------------------------------
#   get_callbacks
#-------------------------------------------------------------------------------

def get_callbacks(callback_dir_path):
    '''
    get_callbacks allows users to fetch their tenserflow callback
    Parameters
    ----------
    INPUT : callback_dir_path : varible storing the callbacks_dir location
            
    Output : Loading the saved callbacks
    '''

    callback_path = [
        os.path.join(callback_dir_path, bin_file) for bin_file in os.listdir(callback_dir_path) if bin_file.endswith(".cb")
    ]

    callbacks = [
        joblib.load(path) for path in callback_path
    ]

    logging.info(f"saved callbacks are loaded from {callback_dir_path}")

    return callbacks