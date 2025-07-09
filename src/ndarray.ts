// @ts-expect-error dasdad
import * as nj from "numjs";
import { TypedTensor, Tensor } from "onnxruntime-node";
// @ts-expect-error adsd
import resizeLib from "ndarray-resize";

const totalSize = (shape: number[]): number => {
  return shape.reduce((acc, dim) => acc * dim, 1);
};

const posToIndices = (pos: number, strides: number[]): number[] => {
  const indices: number[] = [];
  for (let i = 0; i < strides.length; i++) {
    const stride = strides[i];
    indices.push(Math.floor(pos / stride));
    pos %= stride;
  }
  return indices;
};

export const swapaxes = (
  arr: nj.NdArray,
  axe1: number,
  axe2: number
): nj.NdArray => {
  const newShape = [...arr.shape];
  newShape[axe1] = arr.shape[axe2];
  newShape[axe2] = arr.shape[axe1];

  const newArr = nj.zeros(newShape);

  const size = totalSize(newShape);
  for (let i = 0; i < size; i++) {
    const newIndices = posToIndices(i, newArr.selection.stride);
    const oldIndices = [...newIndices];
    oldIndices[axe1] = newIndices[axe2];
    oldIndices[axe2] = newIndices[axe1];

    newArr.set(...newIndices, arr.get(...oldIndices));
  }

  return newArr;
};

export const resize = (
  arr: nj.NdArray,
  options: { targetWidth: number; targetHeight: number; algorithm?: string }
): nj.NdArray => {
  options.algorithm = options.algorithm ?? "bicubic";
  const resized = resizeLib(arr.selection, options);
  return new nj.NdArray(
    resized.data,
    resized.shape,
    resized.stride,
    resized.offset
  );
};

export const toTensor = (arr: nj.NdArray): TypedTensor<"float32"> => {
  const typedArray = Float32Array.from(arr.flatten().tolist());
  return new Tensor("float32", typedArray, arr.shape);
};
