# Imports
from typing import Optional, List
import tensorflow as tf

# Further imports are NOT allowed, please use the APIs in `tf`, `tf.keras` and `tf.keras.callbacks`!


def get_early_stopping_callback() -> Optional[tf.keras.callbacks.Callback]:
    """Create a EarlyStopping callback using the Keras API.

    Requirements:
    - Patience whould be 10 or higher
    - Monitored value: "val_loss"

    Returns
    -------
    callback : tf.keras.callbacks.Callback
        The created callback.
    """

    callback = None

    ### ENTER STUDENT CODE BELOW ###

    ### ENTER STUDENT CODE ABOVE ###

    return callback


def get_student_callbacks() -> List[tf.keras.callbacks.Callback]:
    """Combine all callbacks into a list.

    Returns
    -------
    callbacks : List[tf.keras.callbacks.Callback]
        All implemented callbacks.
    """

    callbacks = []

    early_stopping_callback = get_early_stopping_callback()

    if early_stopping_callback:
        callbacks.append(early_stopping_callback)

    return callbacks
