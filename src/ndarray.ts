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

// Input: (height, width, channels)
export const resize = (
  arr: nj.NdArray,
  options: { targetWidth: number; targetHeight: number; algorithm?: string }
): nj.NdArray => {
  options.algorithm = options.algorithm ?? "bilinear";
  const resized = resizeLib(arr.selection, {
    ...options,
    targetWidth: options.targetHeight,
    targetHeight: options.targetWidth, // Library expects (width, height, ...) but input is of shape (height, width, ...)
  });
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

export const fromTensor = (tensor: TypedTensor<"float32">): nj.NdArray => {
  const arr = Array.from(tensor.data);
  return nj.array(arr).reshape(tensor.dims);
};

// nj does not not support concatenation along arbitrary axes directly,
// but it can be simulated by swapping axes, concatenating, and swapping back.
export const concatenateAlongAxis = (
  arr: nj.NdArray[],
  axis: number
): nj.NdArray => {
  const lastAxis = arr[0].shape.length - 1;

  if (axis == lastAxis) {
    return nj.concatenate(arr);
  }

  let swapped = arr.map((a) => swapaxes(a, axis, lastAxis));

  const shape = swapped[0].shape;
  if (shape.length > 3) {
    // Concatenation in nj is hardcoded up to 3D arrays ¯\_(ツ)_/¯
    const newShape = shape.slice(0, -1);
    swapped = swapped.map((a) => {
      return a.reshape(totalSize(newShape), a.shape[lastAxis]);
    });
  }

  let concatenated = nj.concatenate(swapped);

  if (shape.length > 3) {
    // Reshape back to the original shape
    const newShape = shape.slice(0, -1);
    concatenated = concatenated.reshape(...newShape, concatenated.shape[1]);
  }

  return swapaxes(concatenated, lastAxis, axis);
};

// in-place
export const assignSlice = (
  arr: nj.NdArray,
  slice: ([number, number] | null)[],
  value: nj.NdArray
) => {
  const normalizedSlice = slice.map((s, i) => {
    if (s == null) {
      return [0, arr.shape[i]];
    }

    return s;
  });

  const sliceSize = normalizedSlice.reduce(
    (acc, [start, end]) => acc * (end - start),
    1
  );

  if (sliceSize !== totalSize(value.shape)) {
    throw new Error(
      `Slice size ${sliceSize} does not match value size ${value.size}`
    );
  }

  const arrSize = totalSize(arr.shape);
  const arrStride = arr.selection.stride;

  for (let i = 0; i < arrSize; i++) {
    const indices = posToIndices(i, arrStride);
    let valid = true;

    for (let j = 0; j < normalizedSlice.length; j++) {
      const [start, end] = normalizedSlice[j];
      if (indices[j] < start || indices[j] >= end) {
        valid = false;
        break;
      }
    }

    if (valid) {
      const valueIndices = indices.map((idx, j) => {
        const [start] = normalizedSlice[j];
        return idx - start;
      });

      arr.set(...indices, value.get(...valueIndices));
    }
  }
};
