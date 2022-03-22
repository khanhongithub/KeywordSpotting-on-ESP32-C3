# Includes

import tensorflow as tf
import numpy as np

# Further imports are NOT allowed, please use the APIs in `tf`, `tf.keras` and `tf.keras.layers`!


def create_micro_kws_student_model(
    model_settings: dict, model_name: str = "micro_kws_student"
) -> tf.keras.Model:
    """Builds a MicroKWS model with the keras API.

    Arguments
    ---------
    model_settings : dict
        Dict of different settings for model training.

    Returns
    -------
    model : tf.keras.Model
        Model of the 'Student' architecture.
    """

    # Get relevant model setting.
    input_frequency_size = model_settings["dct_coefficient_count"]
    input_time_size = model_settings["spectrogram_length"]

    inputs = tf.keras.Input(shape=(model_settings["fingerprint_size"]), name="input")

    ### ENTER STUDENT CODE BELOW ###

    ### ENTER STUDENT CODE ABOVE ###

    model = tf.keras.Model(inputs, output)
    model._name = model_name
    return model
