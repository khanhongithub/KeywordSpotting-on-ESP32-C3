# Copyright © 2021 Arm Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications Copyright 2022 Chair of Electronic Design Automation, TUM
"""Model definitions for simple keyword spotting."""

import tensorflow as tf

from student.model import create_micro_kws_student_model


def prepare_model_settings(
    label_count,
    sample_rate,
    clip_duration_ms,
    window_size_ms,
    window_stride_ms,
    dct_coefficient_count,
):
    """Calculates common settings needed for all models.

    Args:
        label_count: How many classes are to be recognized.
        sample_rate: Number of audio samples per second.
        clip_duration_ms: Length of each audio clip to be analyzed.
        window_size_ms: Duration of frequency analysis window.
        window_stride_ms: How far to move in time between frequency windows.
        dct_coefficient_count: Number of frequency bins to use for analysis.

    Returns:
        Dictionary containing common settings.
    """
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = desired_samples - window_size_samples
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
    fingerprint_size = dct_coefficient_count * spectrogram_length

    return {
        "desired_samples": desired_samples,
        "window_size_samples": window_size_samples,
        "window_stride_samples": window_stride_samples,
        "spectrogram_length": spectrogram_length,
        "dct_coefficient_count": dct_coefficient_count,
        "fingerprint_size": fingerprint_size,
        "label_count": label_count,
        "sample_rate": sample_rate,
    }


def get_model(model_settings, model_architecture, model_name="micro_kws"):
    """Builds a tf.keras model of the requested architecture compatible with the settings.

    Args:
        model_settings: Dictionary of information about the model.
        model_architecture: String specifying which kind of model to create.

    Returns:
        A tf.keras Model with the requested architecture.

    Raises:
        Exception: If the architecture type isn't recognized.
    """

    if model_architecture == "micro_kws_xs":
        return create_micro_kws_xs_model(model_settings, model_name=model_name)
    if model_architecture == "micro_kws_student":
        return create_micro_kws_student_model(model_settings, model_name=model_name)
    else:
        raise Exception(f"model_architecture argument {model_architecture} not recognized")


def create_micro_kws_xs_model(model_settings, model_name="micro_kws_xs"):
    """Builds a model with a single depthwise-convolution layer followed by a single fully-connected layer.
    Args:
        model_settings: Dict of different settings for model training.
    Returns:
        tf.keras Model of the 'CNN' architecture.
    """

    # Get relevant model setting.
    input_frequency_size = model_settings["dct_coefficient_count"]
    input_time_size = model_settings["spectrogram_length"]

    inputs = tf.keras.Input(shape=(model_settings["fingerprint_size"]), name="input")

    # Reshape the flattened input.
    x = tf.reshape(inputs, shape=(-1, input_time_size, input_frequency_size, 1))

    # First convolution.
    x = tf.keras.layers.DepthwiseConv2D(
        depth_multiplier=4,
        kernel_size=(5, 4),
        strides=(2, 2),
        padding="SAME",
        activation="relu",
    )(x)

    # Flatten for fully connected layers.
    x = tf.keras.layers.Flatten()(x)

    # Output fully connected.
    output = tf.keras.layers.Dense(units=model_settings["label_count"], activation="softmax")(x)

    model = tf.keras.Model(inputs, output)
    model._name = model_name
    return model
