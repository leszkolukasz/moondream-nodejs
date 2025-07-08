import os
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

# As onnxruntime on react-native does not support float16, we need to convert the model to float32.
# Maybe later I will figure out a better way to do this.
def convert_onnx(input_path: str, output_path: str):
    model = onnx.load(input_path)

    for input_tensor in model.graph.input:
        if input_tensor.type.tensor_type.elem_type == TensorProto.FLOAT16:
            input_tensor.type.tensor_type.elem_type = TensorProto.FLOAT

    for value in model.graph.value_info:
        if value.type.tensor_type.elem_type == TensorProto.FLOAT16:
            value.type.tensor_type.elem_type = TensorProto.FLOAT

    for output in model.graph.output:
        if output.type.tensor_type.elem_type == TensorProto.FLOAT16:
            output.type.tensor_type.elem_type = TensorProto.FLOAT

    new_initializers = []
    for init in model.graph.initializer:
        if init.data_type == TensorProto.FLOAT16:
            arr = numpy_helper.to_array(init).astype(np.float32)
            new_init = numpy_helper.from_array(arr, init.name)
            new_initializers.append(new_init)
        else:
            new_initializers.append(init)

    del model.graph.initializer[:]
    model.graph.initializer.extend(new_initializers)

    # Float32 -> Float16 casts are now Float32 -> Float32 casts.
    # They are useless, but it's easier that removing them.
    for node in model.graph.node:
        if node.op_type == "Cast":
            attr = next((a for a in node.attribute if a.name == "to"), None)
            if attr and attr.i == TensorProto.FLOAT16:
                attr.i = TensorProto.FLOAT

    onnx.checker.check_model(model)
    onnx.save(model, output_path)

def convert_npy(input_path: str, output_path: str):
    arr = np.load(input_path)
    
    with open(output_path, 'w') as f:
        json.dump({
            "shape": arr.shape,
            "data": arr.flatten().tolist()
        }, f)


def copy_file(input_path: str, output_path: str):
    shutil.copy(input_path, output_path)

def convert(src_dir: str, dest_dir):
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)

    if src_path == dest_path:
        raise ValueError("Source and destination directories must be different.")

    if not src_path.exists():
        raise FileNotFoundError(f"Source directory {src_dir} does not exist.")

    if not dest_path.exists():
        dest_path.mkdir(parents=True, exist_ok=True)

    for file in [
        "coord_encoder.onnx",
        "coord_decoder.onnx",
        "size_encoder.onnx",
        "size_decoder.onnx",
        "text_encoder.onnx",
        "text_decoder.onnx",
        "vision_encoder.onnx",
        "vision_projection.onnx"
    ]:
        convert_onnx(src_path / file, dest_path / file)

    for file in [
        "config.json",
        "tokenizer.json"
    ]:
        copy_file(src_path / file, dest_path / file)

    convert_npy(
        src_path / "initial_kv_cache.npy",
        dest_path / "initial_kv_cache.json"
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert ONNX models to float32.")
    parser.add_argument("src_dir", type=str, help="Source directory containing ONNX models.")
    parser.add_argument("dest_dir", type=str, help="Destination directory for converted ONNX models.")

    args = parser.parse_args()
    convert(args.src_dir, args.dest_dir)