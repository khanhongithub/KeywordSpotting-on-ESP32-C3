import collections

from student.estimate import (
    estimate_conv2d_macs,
    estimate_depthwise_conv2d_macs,
    estimate_fully_connected_macs,
    estimate_rom,
    estimate_ram,
)

MyTensor = collections.namedtuple("MyTensor", ("idx", "name", "shape", "dtype", "is_const"))
MyLayer = collections.namedtuple("MyLayer", ("idx", "name", "inputs", "outputs"))

CHANNEL_MULT = 8

# Test tensors
INPUT = MyTensor(0, "x0", [1, 32, 32, 1], "int8", False)
WEIGHTS_1 = MyTensor(1, "w1", [1, 2, 2, 8], "int8", True)
BIAS_1 = MyTensor(2, "b1", [8], "int32", True)
INTERMEDIATE_1 = MyTensor(3, "x1", [1, 32, 32, 8], "int8", False)
WEIGHTS_2 = MyTensor(4, "w2", [16, 3, 3, 8], "int8", True)
BIAS_2 = MyTensor(5, "b2", [16], "int32", True)
INTERMEDIATE_2 = MyTensor(6, "x2", [1, 16, 16, 16], "int8", False)
SHAPE = MyTensor(7, "s", [2], "int32", True)
INTERMEDIATE_2_ = MyTensor(8, "x2_", [1, 4096], "int8", False)
WEIGHTS_3 = MyTensor(9, "w3", [4096, 10], "int8", True)
BIAS_3 = MyTensor(10, "b3", [10], "int32", True)
OUTPUT = MyTensor(11, "x3", [1, 10], "int8", False)
PLACEHOLDER = MyTensor(-1, "p", [0, 0], "int8", False)
ALL_TENSORS = [
    INPUT,
    WEIGHTS_1,
    BIAS_1,
    INTERMEDIATE_1,
    WEIGHTS_2,
    BIAS_2,
    INTERMEDIATE_2,
    SHAPE,
    INTERMEDIATE_2_,
    WEIGHTS_3,
    BIAS_3,
    OUTPUT,
]

# Test layers
DEPTHWISE_CONV2D = MyLayer(
    0, "DepthwiseConv2D", [INPUT.idx, WEIGHTS_1.idx, BIAS_1.idx], [INTERMEDIATE_1.idx]
)
CONV2D = MyLayer(1, "Conv2D", [INTERMEDIATE_1.idx, WEIGHTS_2.idx, BIAS_2.idx], [INTERMEDIATE_2.idx])
RESHAPE = MyLayer(2, "Reshape", [INTERMEDIATE_2.idx, SHAPE.idx], [INTERMEDIATE_2_.idx])
FULLY_CONNECTED = MyLayer(
    3, "FullyConnected", [INTERMEDIATE_2_.idx, WEIGHTS_3.idx, BIAS_3.idx], [OUTPUT.idx]
)
ALL_LAYERS = [DEPTHWISE_CONV2D, CONV2D, RESHAPE, FULLY_CONNECTED]


def test_estimate_macs():
    # Trivial cases
    assert estimate_depthwise_conv2d_macs([1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], 1) == 1
    assert estimate_conv2d_macs([1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]) == 1
    assert estimate_fully_connected_macs([1, 1], [1, 1], [1, 1]) == 1

    # Single layers
    assert (
        estimate_depthwise_conv2d_macs(
            INPUT.shape, WEIGHTS_1.shape, INTERMEDIATE_1.shape, CHANNEL_MULT
        )
        == 32768
    )
    assert (
        estimate_conv2d_macs(INTERMEDIATE_1.shape, WEIGHTS_2.shape, INTERMEDIATE_2.shape) == 294912
    )
    assert (
        estimate_fully_connected_macs(INTERMEDIATE_2_.shape, WEIGHTS_3.shape, OUTPUT.shape) == 40960
    )


def test_estimate_rom():
    # Trivial cases
    assert estimate_rom([MyTensor(0, "foo", [1, 1, 1, 1], "int8", False)]) == 0

    # Single layer
    assert estimate_rom([INPUT, WEIGHTS_1, BIAS_1, INTERMEDIATE_1]) == 64
    assert estimate_rom([INTERMEDIATE_1, WEIGHTS_2, BIAS_2, INTERMEDIATE_2]) == 1216
    assert estimate_rom([INTERMEDIATE_2, SHAPE, INTERMEDIATE_2_]) == 8
    assert estimate_rom([INTERMEDIATE_2_, WEIGHTS_3, BIAS_3, OUTPUT]) == 41000

    # Multi layer
    assert estimate_rom(ALL_TENSORS) == 42288


def test_estimate_ram():
    # Trvial cases
    assert estimate_ram(
        [MyTensor(0, "foo", [1, 1, 1, 1], "int8", True)], [MyLayer(0, "bar", [0], [0])]
    ) == (0, 0)

    # Single layer
    # assert estimate_ram(ALL_TENSORS, [DEPTHWISE_CONV2D]) == (9216, 9216)
    # assert estimate_ram(ALL_TENSORS, [CONV2D]) == (123, 456)
    # assert estimate_ram(ALL_TENSORS, [RESHAPE]) == (123, 456)
    # assert estimate_ram(ALL_TENSORS, [FULLY_CONNECTED]) == (123, 456)

    # Multi layer
    assert estimate_ram(ALL_TENSORS, ALL_LAYERS) == (12288, 17418)
