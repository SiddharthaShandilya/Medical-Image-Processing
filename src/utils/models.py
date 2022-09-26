import tensorflow as tf
import os
import logging
from tensorflow.python.keras.backend import flatten
from tensorflow.keras.applications import InceptionV3
from src.utils.all_utils import get_timestamp

#-------------------------------------------------------------------------------
#   get_inception_model
#-------------------------------------------------------------------------------


def get_inception_model(input_shape, model_path):
    '''
    get_inception_model allows users to fetch their InceptionV3 model from tenserflow
    Parameters
    ----------
    INPUT : input_shape : Pass the image input_shpae for training
            model_path : variable storing the model location
            
    Output : InceptionV3 model
    '''

    model = tf.keras.applications.InceptionV3(
        input_shape=input_shape,
        weights="imagenet",
        include_top=False
    )

    model.save(model_path)
    logging.info(f"InceptionV3 base model saved at: {model_path}")
    return model

#-------------------------------------------------------------------------------
#   prepare_model
#-------------------------------------------------------------------------------


def prepare_model(model, CLASSES, freeze_all, freeze_till, learning_rate):
    '''
    prepare_model allows users to make changes in the InceptionV3 model such as changing their learning rate or train all the weights from scratch.
    Parameters
    ----------
    INPUT : model : Model prepared for training
            CLASSES : dimensionality of the output space. || INT
            freeze_all : Keep the weights same as their original model || INT
            freeze_till : Layer till which you need to keep the weights || INT
            learning_rate : Learning rate of the model || float

    Output : Fully Customized model ready for training
    '''  
    if freeze_all:
        for layer in model.layers:
            layer.trainable = False
    elif (freeze_till is not None) and (freeze_till > 0):
        for layer in model.layers[:-freeze_till]:
            layer.trainable = False

    ## add our fully connected layers
    flatten_in = tf.keras.layers.Flatten()(model.output)
    prediction = tf.keras.layers.Dense(
        units=CLASSES,
        activation="softmax"
    )(flatten_in)

    full_model = tf.keras.models.Model(
        inputs = model.input,
        outputs = prediction
    )

    full_model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics = ["accuracy"]
    )

    logging.info("custom model is compiled and ready to be trained")
    full_model.summary()
    return full_model

#-------------------------------------------------------------------------------
#   load_full_model
#-------------------------------------------------------------------------------


def load_full_model(untrained_full_model_path):
    '''
    load_full_model allows users to load the model from mentioned path
    Parameters
    ----------
    INPUT : untrained_full_model_path : Location of the untrained model || String

    Output : Model from the location.
    '''  
    model = tf.keras.models.load_model(untrained_full_model_path)
    logging.info(f"untrained model is read from: {untrained_full_model_path}")
    return model

#-------------------------------------------------------------------------------
#   get_unique_path_to_save_model
#-------------------------------------------------------------------------------

def get_unique_path_to_save_model(trained_model_dir, model_name="inceptionv3_covid_ct_model"):
    '''
    get_unique_path_to_save_model allows users to create multiple models path on the basis of timestamp. 
    Parameters
    ----------
    INPUT : trained_model_dir : Location of the trained model directory || String
            model_name : Name for the new model || String

    Output : New path for everytime a new model is trained.
    '''  
    timestamp = get_timestamp(model_name)
    unique_model_name = f"{timestamp}_.h5"
    unique_model_path = os.path.join(trained_model_dir, unique_model_name)
    return unique_model_path

