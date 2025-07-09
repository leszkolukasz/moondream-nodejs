// import * as RNFS from "react-native-fs";
import { readFile } from "fs/promises"; // TODO: remove
import * as jpeg from "jpeg-js";
// @ts-expect-error adsd
import * as nj from "numjs";
import { PATCH_SIZE } from "./constants";
import { resize } from "./ndarray";

export const loadImage = async (imageURI: string): Promise<jpeg.UintArrRet> => {
  // const imageData = await RNFS.readFile(imageURI, "base64");
  const imageData = await readFile(imageURI, { encoding: "base64" });
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
  tensor: nj.NdArray,
  mean: number = 0.5,
  std: number = 0.5 // 0.5?
): nj.NdArray => {
  return tensor.subtract(mean).divide(std);
};

// Input: (height, width, 3) tensor
// Output: patches (num patches, patchSize, patchSize, 3), template (rows, cols)
export const createPatches = (
  imageTensor: nj.NdArray,
  patchSize: number = PATCH_SIZE
): [nj.NdArray[], [number, number]] => {
  const [height, width, channels] = imageTensor.shape;

  let selectedTemplate: [number, number] = [1, 1];
  const candidateTemplates: [number, number][] = [
    [1, 2],
    [2, 1],
    [2, 2],
  ];

  const patches: nj.NdArray[] = [
    normalize(
      resize(imageTensor, {
        targetWidth: patchSize,
        targetHeight: patchSize,
        algorithm: "bicubic",
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

    const [pathHeight, patchWidth] = [
      Math.floor(height / selectedTemplate[0]),
      Math.floor(width / selectedTemplate[1]),
    ];

    for (let row = 0; row < selectedTemplate[0]; row++) {
      for (let col = 0; col < selectedTemplate[1]; col++) {
        const cropped = imageTensor.slice(
          row * pathHeight,
          (row + 1) * pathHeight,
          col * patchWidth,
          (col + 1) * patchWidth,
          channels
        );

        const patch = resize(cropped, {
          targetWidth: patchSize,
          targetHeight: patchSize,
          algorithm: "bicubic",
        });

        patches.push(normalize(patch));
      }
    }
  }

  return [nj.stack(patches), selectedTemplate];
};
