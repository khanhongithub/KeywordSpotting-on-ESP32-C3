# Imports
import tensorflow as tf

# Further imports are NOT allowed, please use the APIs in `tf`, `tf.keras` and `tf.keras.backend`!


def recall(matrix: int, idx: int) -> int:
    """Calclate the recall metric for a given confusion matrix and category.

    Arguments
    ---------
    matrix : ?
        The confusion matrix for the trained model. (rows: real labels, cols: predicted labels)
    idx : int
        The category index (0: silence, 1: unknown, 2: ...)

    Returns
    -------
    recall : ?
        The calculated recall value (between 0 and 1)
    """

    recall = None

    ### ENTER STUDENT CODE BELOW ###

    ### ENTER STUDENT CODE ABOVE ###

    return recall


def precision(matrix, idx):

    precision = None

    ### ENTER STUDENT CODE BELOW ###

    ### ENTER STUDENT CODE ABOVE ###

    return precision


def f1_score(matrix, idx):

    f1_score = None

    ### ENTER STUDENT CODE BELOW ###

    ### ENTER STUDENT CODE ABOVE ###

    return f1_score


def get_student_metrics(matrix, idx):

    return {
        "recall": recall(matrix, idx).numpy(),
        "precision": precision(matrix, idx).numpy(),
        "f1_score": f1_score(matrix, idx).numpy(),
    }
