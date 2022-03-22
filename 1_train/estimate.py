import argparse

import tflite.Model

from student.estimate import (
    estimate_conv2d_macs,
    estimate_depthwise_conv2d_macs,
    estimate_fully_connected_macs,
    estimate_rom,
    estimate_ram,
)

import numpy as np

from student.estimate import (
    estimate_rom,
    estimate_ram,
    estimate_conv2d_macs,
    estimate_depthwise_conv2d_macs,
    estimate_fully_connected_macs,
    MyTensor,
    MyLayer,
)


TYPES_MAP = {
    tflite.TensorType.INT8: "int8",
    tflite.TensorType.INT32: "int32",
}
# Load tflite flatbuffer to object tree.


def load_model(filename):
    with open(filename, "rb") as f:
        buf = bytearray(f.read())

    model = tflite.Model.GetRootAsModel(buf, 0)
    return model


def get_tensors(m, g):
    tensors = []
    for tensor_idx in range(g.TensorsLength()):
        tensor = g.Tensors(tensor_idx)
        assert tensor.Type() in [
            tflite.TensorType.INT8,
            tflite.TensorType.INT32,
        ], "Unsupported Tensor type (Needs quantized model)"
        shape = tensor.ShapeAsNumpy().tolist()
        name = tensor.Name()
        dtype = TYPES_MAP[tensor.Type()]
        buf = m.Buffers(tensor.Buffer())
        is_const = buf.DataLength() > 0
        tensors.append(MyTensor(tensor_idx, name, shape, dtype, is_const))
    return tensors


def get_layers(m, g):
    layers = []
    for op_idx in range(g.OperatorsLength()):
        op = g.Operators(op_idx)
        op_code = m.OperatorCodes(op.OpcodeIndex())
        op_code_id = op_code.BuiltinCode()
        name = tflite.opcode2name(op_code_id)
        inputs = []
        for input_idx in range(op.InputsLength()):
            tensor_index = op.Inputs(input_idx)
            inputs.append(tensor_index)
        outputs = []
        for output_idx in range(op.OutputsLength()):
            tensor_index = op.Outputs(output_idx)
            outputs.append(tensor_index)

        layers.append(MyLayer(op_idx, name, inputs, outputs))
    return layers


def estimate_macs_per_layer(op, m, graph):
    op_code = m.OperatorCodes(op.OpcodeIndex())
    # ignoring types here (assume int8/int32)
    if op_code.BuiltinCode() == tflite.BuiltinOperator.CONV_2D:
        in_tensor = graph.Tensors(op.Inputs(0))
        in_shape = in_tensor.ShapeAsNumpy().tolist()
        kernel_tensor = graph.Tensors(op.Inputs(1))
        kernel_shape = kernel_tensor.ShapeAsNumpy().tolist()
        out_tensor = graph.Tensors(op.Outputs(0))
        out_shape = out_tensor.ShapeAsNumpy().tolist()
        assert len(in_shape) == 4, "Unsupported shape"  # TODO: allow more shapes
        input_n = in_shape[0]
        input_c = in_shape[3]
        assert len(kernel_shape) == 4, "Unsupported shape"  # TODO: allow more shapes
        kernel_n = kernel_shape[0]
        kernel_c = kernel_shape[3]
        assert len(out_shape) == 4, "Unsupported shape"  # TODO: allow more shapes
        output_n = out_shape[0]
        output_c = out_shape[3]
        assert input_n == output_n, "Shape missmatch"
        assert kernel_n == output_c, "Shape missmatch"
        assert input_c % kernel_c == 0, "Shape mismatch"
        macs = estimate_conv2d_macs(in_shape, kernel_shape, out_shape)
        return macs

    elif op_code.BuiltinCode() == tflite.BuiltinOperator.DEPTHWISE_CONV_2D:
        in_tensor = graph.Tensors(op.Inputs(0))
        in_shape = in_tensor.ShapeAsNumpy().tolist()
        kernel_tensor = graph.Tensors(op.Inputs(1))
        kernel_shape = kernel_tensor.ShapeAsNumpy().tolist()
        out_tensor = graph.Tensors(op.Outputs(0))
        out_shape = out_tensor.ShapeAsNumpy().tolist()
        assert len(in_shape) == 4, "Unsupported shape"  # TODO: allow more shapes
        input_n = in_shape[0]
        input_c = in_shape[3]
        assert len(kernel_shape) == 4, "Unsupported shape"  # TODO: allow more shapes
        kernel_n = kernel_shape[0]
        kernel_c = kernel_shape[3]
        assert op.BuiltinOptionsType() == tflite.BuiltinOptions.DepthwiseConv2DOptions
        op_options = op.BuiltinOptions()
        options = tflite.DepthwiseConv2DOptions()
        options.Init(op_options.Bytes, op_options.Pos)
        channel_mult = options.DepthMultiplier()
        assert len(out_shape) == 4, "Unsupported shape"  # TODO: allow more shapes
        output_n = out_shape[0]
        output_c = out_shape[3]
        assert input_n == output_n, "Shape missmatch"
        assert kernel_c == output_c, "Shape missmatch"
        assert output_c == input_c * channel_mult, "Shape mismatch"
        # See: https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d
        # See: https://www.tensorflow.org/api_docs/python/tf/keras/layers/DepthwiseConv2D
        # See: https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h

        macs = estimate_depthwise_conv2d_macs(in_shape, kernel_shape, out_shape, channel_mult)
        return macs

    elif op_code.BuiltinCode() == tflite.BuiltinOperator.FULLY_CONNECTED:
        in_tensor = graph.Tensors(op.Inputs(0))
        in_shape = in_tensor.ShapeAsNumpy().tolist()
        filter_tensor = graph.Tensors(op.Inputs(1))
        filter_shape = np.flip(filter_tensor.ShapeAsNumpy()).tolist()
        out_tensor = graph.Tensors(op.Outputs(0))
        out_shape = out_tensor.ShapeAsNumpy().tolist()
        filter_h = filter_shape[0]
        filter_w = filter_shape[1]
        output_h = out_shape[0]
        output_w = out_shape[1]
        if len(in_shape) == 2:
            input_h = in_shape[0]
            input_w = in_shape[1]
            assert input_h == output_h, "Dimension missmatch"
            assert input_w == filter_h, "Dimension missmatch"
        else:
            input_h = output_h
            input_w = filter_h
        assert filter_w == output_w, "Dimension missmatch"

        macs = estimate_fully_connected_macs(in_shape, filter_shape, out_shape)
        return macs

    elif op_code.BuiltinCode() in [
        tflite.BuiltinOperator.AVERAGE_POOL_2D,
        tflite.BuiltinOperator.MAX_POOL_2D,
        tflite.BuiltinOperator.RESHAPE,
        tflite.BuiltinOperator.SOFTMAX,
        tflite.BuiltinOperator.MUL,
        tflite.BuiltinOperator.ADD,
    ]:
        return 0
    else:
        name = tflite.opcode2name(op_code.BuiltinCode())
        assert False, f"Unsupported operator type: {name}"


def estimate_model_macs(m):
    assert m.SubgraphsLength() == 1, "Multi-subgraph models are currently not supported"
    graph = m.Subgraphs(0)
    assert graph.InputsLength() == 1, "Multi-input models are currently not supported"
    input_tensor = graph.Tensors(graph.Inputs(0))
    assert input_tensor.Type() in [
        tflite.TensorType.INT8,
        tflite.TensorType.INT32,
    ], "Unsupported Input type (Needs quantized model)"
    output_tensors = []
    for out_idx in range(graph.OutputsLength()):
        output_tensor = graph.Tensors(graph.Outputs(0))
        output_tensors.append(output_tensor)
    is_quantized = output_tensor.Type() in [
        tflite.TensorType.INT8,
        tflite.TensorType.INT32,
    ]
    assert is_quantized, "Unsupported Output type (Needs quantized model)"
    total_macs = 0
    for op_idx in range(graph.OperatorsLength()):
        op = graph.Operators(op_idx)
        num_macs = estimate_macs_per_layer(op, m, graph)
        total_macs += num_macs
    return total_macs


def estimate_model_rom(m):
    assert m.SubgraphsLength() == 1, "Multi-subgraph models are currently not supported"
    graph = m.Subgraphs(0)
    assert graph.InputsLength() == 1, "Multi-input model are currently not supported"
    input_tensor = graph.Tensors(graph.Inputs(0))
    assert input_tensor.Type() in [
        tflite.TensorType.INT8,
        tflite.TensorType.INT32,
    ], "Unsupported Input type (Needs quantized model)"
    output_tensors = []
    for out_idx in range(graph.OutputsLength()):
        output_tensor = graph.Tensors(graph.Outputs(0))
        output_tensors.append(output_tensor)
    is_quantized = output_tensor.Type() in [
        tflite.TensorType.INT8,
        tflite.TensorType.INT32,
    ]
    assert is_quantized, "Unsupported Output type (Needs quantized model)"

    tensors = get_tensors(m, graph)

    return estimate_rom(tensors)


def estimate_model_ram(m):
    assert m.SubgraphsLength() == 1, "Multi-subgraph models are currently not supported"
    graph = m.Subgraphs(0)
    assert graph.InputsLength() == 1, "Multi-input model are currently not supported"
    input_tensor = graph.Tensors(graph.Inputs(0))
    assert input_tensor.Type() in [
        tflite.TensorType.INT8,
        tflite.TensorType.INT32,
    ], "Unsupported Input type (Needs quantized model)"
    output_tensors = []
    for out_idx in range(graph.OutputsLength()):
        output_tensor = graph.Tensors(graph.Outputs(0))
        output_tensors.append(output_tensor)
    is_quantized = output_tensor.Type() in [
        tflite.TensorType.INT8,
        tflite.TensorType.INT32,
    ]
    assert is_quantized, "Unsupported Output type (Needs quantized model)"

    tensors = get_tensors(m, graph)
    layers = get_layers(m, graph)

    return estimate_ram(tensors, layers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument(
        "--out", type=str, default=None, help="File which should contain the determined accuracy"
    )
    args = parser.parse_args()

    m = load_model(args.model)

    estimated_rom = estimate_model_rom(m)
    estimated_ram_wc, estimated_ram_bc = estimate_model_ram(m)
    estimated_macs = estimate_model_macs(m)

    estimations = f"""ROM={estimated_rom}
RAM_BC={estimated_ram_bc}
RAM_WC={estimated_ram_wc}
MACS={estimated_macs}
"""

    # TODO: support float

    print("Estimations:")
    print(estimations)

    if args.out:
        with open(args.out, "w") as handle:
            handle.write(estimations)


if __name__ == "__main__":
    main()
