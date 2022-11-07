import numpy as np
import tensorflow as tf

from student.metrics import recall, precision, f1_score

MATRIX_1 = tf.math.confusion_matrix(
    [0, 1, 0, 1, 0, 1, 0, 1],
    [1, 1, 0, 0, 1, 1, 0, 0],
    num_classes=2,
)

MATRIX_2 = tf.math.confusion_matrix(
    [2, 3, 3, 3, 1, 2, 1, 1, 1, 3, 2, 2, 1, 2, 3, 1, 2, 1, 2, 1, 2, 3, 2, 0, 0, 0],
    [2, 2, 1, 3, 0, 2, 0, 1, 0, 3, 3, 2, 0, 2, 3, 2, 2, 0, 2, 1, 3, 2, 2, 1, 0, 0],
    num_classes=4,
)


def test_metrics_precision():
    # matrix 1
    np.testing.assert_almost_equal(precision(MATRIX_1, 0), 0.5)
    np.testing.assert_almost_equal(precision(MATRIX_1, 1), 0.5)

    # matrix 2
    np.testing.assert_almost_equal(precision(MATRIX_2, 0), 0.2857142857142857)
    np.testing.assert_almost_equal(precision(MATRIX_2, 1), 0.5)
    np.testing.assert_almost_equal(precision(MATRIX_2, 2), 0.7)
    np.testing.assert_almost_equal(precision(MATRIX_2, 3), 0.6)


def test_metrics_recall():
    # matrix 1
    np.testing.assert_almost_equal(recall(MATRIX_1, 0), 0.5)
    np.testing.assert_almost_equal(recall(MATRIX_1, 1), 0.5)

    # matrix 2
    np.testing.assert_almost_equal(recall(MATRIX_2, 0), 0.6666666666666666)
    np.testing.assert_almost_equal(recall(MATRIX_2, 1), 0.25)
    np.testing.assert_almost_equal(recall(MATRIX_2, 2), 0.7777777777777778)
    np.testing.assert_almost_equal(recall(MATRIX_2, 3), 0.5)


def test_metrics_f1_score():
    # matrix 1
    np.testing.assert_almost_equal(f1_score(MATRIX_1, 0), 0.5)
    np.testing.assert_almost_equal(f1_score(MATRIX_1, 1), 0.5)

    # matrix 2
    np.testing.assert_almost_equal(f1_score(MATRIX_2, 0), 0.4)
    np.testing.assert_almost_equal(f1_score(MATRIX_2, 1), 0.3333333333333333)
    np.testing.assert_almost_equal(f1_score(MATRIX_2, 2), 0.7368421052631577)
    np.testing.assert_almost_equal(f1_score(MATRIX_2, 3), 0.5454545454545454)
