import * as RNFS from "react-native-fs";
// import { readFile } from "fs/promises";
import * as jpeg from "jpeg-js";
// @ts-expect-error No types
import * as nj from "numjs";
import { PATCH_SIZE } from "./constants";
import { concatenateAlongAxis, resize } from "./ndarray";
import { assert } from "./utils";

export type EncodedImage = {
  kvCache: nj.NdArray; // (layer, K/V, batch, head, seq_len, dim)
};

export const loadImage = async (imageURI: string): Promise<jpeg.UintArrRet> => {
  const imageData = await RNFS.readFile(imageURI, "base64");
  // const imageData = await readFile(imageURI, { encoding: "base64" });
  const buffer = Uint8Array.from(atob(imageData), (c) => c.charCodeAt(0));
  const image = jpeg.decode(buffer, { useTArray: true });
  return image;
};

// Output: (height, width, 3) tensor
export const imageToTensor = (image: jpeg.UintArrRet): nj.NdArray => {
  const { width, height, data } = image;
  const tensor = nj.zeros([height, width, 3]);

  for (let i = 0; i < data.length; i += 4) {
    const x = (i / 4) % width;
    const y = Math.floor(i / 4 / width);
    tensor.set(y, x, 0, data[i]); // Red
    tensor.set(y, x, 1, data[i + 1]); // Green
    tensor.set(y, x, 2, data[i + 2]); // Blue
  }

  return tensor;
};

const normalize = (
  arr: nj.NdArray,
  mean: number = 0.5,
  std: number = 0.5 // 0.5?
): nj.NdArray => {
  return arr.divide(255.0).subtract(mean).divide(std);
};

// Input: (height, width, 3) tensor
// Output: patches (num patches, patchSize, patchSize, 3), template (rows, cols)
export const createPatches = (
  imageArr: nj.NdArray,
  patchSize: number = PATCH_SIZE
): [nj.NdArray, [number, number]] => {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [height, width, channels] = imageArr.shape;

  let selectedTemplate: [number, number] = [1, 1];
  const candidateTemplates: [number, number][] = [
    [1, 2],
    [2, 1],
    [2, 2],
  ];

  const patches: nj.NdArray[] = [
    normalize(
      resize(imageArr, {
        targetWidth: patchSize,
        targetHeight: patchSize,
      })
    ),
  ];

  if (Math.max(width, height) >= patchSize * 1.4) {
    const aspectRatio = height / width;
    selectedTemplate = candidateTemplates.reduce((prev, curr) => {
      return Math.abs(curr[0] / curr[1] - aspectRatio) <
        Math.abs(prev[0] / prev[1] - aspectRatio)
        ? curr
        : prev;
    });

    const [patchHeight, patchWidth] = [
      Math.floor(height / selectedTemplate[0]),
      Math.floor(width / selectedTemplate[1]),
    ];

    for (let row = 0; row < selectedTemplate[0]; row++) {
      for (let col = 0; col < selectedTemplate[1]; col++) {
        const rowStart = row * patchHeight;
        const rowEnd = (row + 1) * patchHeight;
        const colStart = col * patchWidth;
        const colEnd = (col + 1) * patchWidth;

        const cropped = imageArr.slice(
          [rowStart, rowEnd],
          [colStart, colEnd],
          null
        );

        const patch = resize(cropped, {
          targetWidth: patchSize,
          targetHeight: patchSize,
        });

        patches.push(normalize(patch));
      }
    }
  }

  console.log("Chosen template:", selectedTemplate);

  return [nj.stack(patches), selectedTemplate];
};

export const adaptiveAvgPooling2D = (
  input: nj.NdArray,
  outputSize: [number, number]
) => {
  const [inputHeight, inputWidth, channels] = input.shape;
  const [outputHeight, outputWidth] = outputSize;

  const stride_h = Math.floor(inputHeight / outputHeight);
  const stride_w = Math.floor(inputWidth / outputWidth);
  const kernel_h = inputHeight - (outputHeight - 1) * stride_h;
  const kernel_w = inputWidth - (outputWidth - 1) * stride_w;

  const output = nj.zeros([outputHeight, outputWidth, channels]);

  for (let i = 0; i < outputHeight; i++) {
    for (let j = 0; j < outputWidth; j++) {
      const h_start = i * stride_h;
      const w_start = j * stride_w;
      const h_end = h_start + kernel_h;
      const w_end = w_start + kernel_w;

      const patch = input.slice([h_start, h_end], [w_start, w_end], null);
      let sum = nj.applyOverAxis(patch, nj.sum, { axis: 0 });
      sum = nj.applyOverAxis(sum, nj.sum, { axis: 0 });
      const avg = sum.divide(patch.shape[0] * patch.shape[1]); // mean(axis=(0, 1))

      for (let c = 0; c < channels; c++) {
        output.set(i, j, c, avg.get(c));
      }
    }
  }

  return output;
};

// Input: (num patches, 729, 720)
// Output: (729, 2*720)
export const processPatchEmbeddings = (
  patchEmb: nj.NdArray,
  patchTemplate: [number, number]
): nj.NdArray => {
  const globalPatchEmb: nj.NdArray = patchEmb
    .slice([0, 1], null, null)
    .reshape(729, 720);

  if (patchTemplate[0] === 1 && patchTemplate[1] === 1) {
    patchEmb = nj.concatenate([globalPatchEmb, globalPatchEmb]);
  } else {
    const seqLen = patchEmb.shape[1];
    const w = Math.round(Math.sqrt(seqLen));
    assert(w * w === 729);

    const rows: nj.NdArray[] = [];

    // Creates a grid of sub-patches
    for (let r = 0; r < patchTemplate[0]; r++) {
      const row: nj.NdArray[] = [];
      for (let c = 0; c < patchTemplate[1]; c++) {
        const idx = r * patchTemplate[1] + c;
        const patch = patchEmb
          .slice([idx, idx + 1], null, null)
          .reshape(w, w, 720);

        row.push(patch);
      }
      rows.push(concatenateAlongAxis(row, 1));
    }

    patchEmb = concatenateAlongAxis(rows, 0);
    patchEmb = adaptiveAvgPooling2D(patchEmb, [w, w]);
    patchEmb = patchEmb.reshape(w * w, 720);
    patchEmb = nj.concatenate([globalPatchEmb, patchEmb]);
  }

  return patchEmb;
};
