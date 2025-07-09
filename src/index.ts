/* eslint-disable @typescript-eslint/no-explicit-any */

import { InferenceSession, TypedTensor, Tensor } from "onnxruntime-node"; // TODO: change to onnxruntime-react-native
// import * as RNFS from "react-native-fs";
import { readFile } from "fs/promises"; // TODO: remove
import { ModelConfig } from "./types";
import { joinPath, loadONNX, snakeToCamel } from "./utils";
import {
  loadImage,
  imageToTensor,
  createPatches,
  processPatchEmbeddings,
  EncodedImage,
} from "./image";
import { Tokenizer } from "./tokenizer";
import { MAX_IMAGE_SIZE } from "./constants";
// @ts-expect-error adsd
import * as nj from "numjs";
import {
  concatenateAlongAxis,
  fromTensor,
  resize,
  swapaxes,
  toTensor,
} from "./ndarray";

export class Moondream {
  visionEncoder!: InferenceSession;
  visionProjection!: InferenceSession;
  textEncoder!: InferenceSession;
  textDecoder!: InferenceSession;
  sizeEncoder!: InferenceSession;
  sizeDecoder!: InferenceSession;
  coordEncoder!: InferenceSession;
  coordDecoder!: InferenceSession;
  tokenizer!: Tokenizer;
  initialKVCache!: TypedTensor<"float32">;
  config!: ModelConfig;

  static async load(modelPath: string): Promise<Moondream> {
    const instance = new Moondream();

    const ortSettings: InferenceSession.SessionOptions = {
      executionProviders: ["cuda"],
      graphOptimizationLevel: "all",
      logSeverityLevel: 3,
    };

    console.log("Loading ONNX models...");

    for (const file of [
      "coord_encoder.onnx",
      "coord_decoder.onnx",
      "size_encoder.onnx",
      "size_decoder.onnx",
      "text_encoder.onnx",
      "text_decoder.onnx",
      "vision_encoder.onnx",
      "vision_projection.onnx",
    ]) {
      const fileName = snakeToCamel(file.split(".")[0]);
      (instance as any)[fileName] = await loadONNX(
        joinPath(modelPath, file),
        ortSettings
      );
    }

    instance.config = JSON.parse(
      // await RNFS.readFile(joinPath(modelPath, "config.json"), "utf8")
      await readFile(joinPath(modelPath, "config.json"), "utf8")
    );

    const initialKVCacheJJSON = JSON.parse(
      // await RNFS.readFile(joinPath(modelPath, "initial_kv_cache.json"), "utf8")
      await readFile(joinPath(modelPath, "initial_kv_cache.json"), "utf8")
    );

    instance.initialKVCache = new Tensor(
      "float32",
      Float32Array.from(initialKVCacheJJSON.data),
      initialKVCacheJJSON.shape
    );

    instance.tokenizer = Tokenizer.fromConfig(
      JSON.parse(
        // await RNFS.readFile(joinPath(modelPath, "tokenizer.json"), "utf8")
        await readFile(joinPath(modelPath, "tokenizer.json"), "utf8")
      )
    );

    return instance;
  }

  async encodeImage(imageURI: string): Promise<EncodedImage> {
    const image = await loadImage(imageURI);
    let imageTensor = imageToTensor(image);

    const scale = MAX_IMAGE_SIZE / Math.max(image.width, image.height);

    if (scale < 1) {
      console.log(
        `Resizing image from ${image.width}x${image.height} with scale ${scale}`
      );
      const targetWidth = Math.floor(image.width * scale);
      const targetHeight = Math.floor(image.height * scale);
      imageTensor = resize(imageTensor, {
        targetWidth,
        targetHeight,
        algorithm: "bicubic",
      });
    }

    // eslint-disable-next-line prefer-const
    let [patches, patchTemplate] = createPatches(imageTensor);
    patches = swapaxes(patches, 2, 3);
    patches = swapaxes(patches, 1, 2); // (num patches, 3, patchSize, patchSize)

    let result = await this.visionEncoder.run({
      input: toTensor(patches),
    });

    let patchEmb = fromTensor(result.output as TypedTensor<"float32">); // (num patches, 729, 720)
    patchEmb = processPatchEmbeddings(patchEmb, patchTemplate);

    patchEmb = patchEmb.reshape(1, 729, 1440);
    result = await this.visionProjection.run({
      input: toTensor(patchEmb),
    });

    const inputEmb = fromTensor(result.output as TypedTensor<"float32">); // (1, 729, 1024)

    result = await this.textDecoder.run({
      input_embeds: toTensor(inputEmb),
      kv_cache: this.initialKVCache, // (24, 2, 1, 16, 1, 64)
    });

    const newKVCache = fromTensor(
      result.new_kv_cache as TypedTensor<"float32">
    ); // (24, 2, 1, 16, 729, 64)

    const kvCache = concatenateAlongAxis(
      [fromTensor(this.initialKVCache), newKVCache],
      4
    );

    return { kvCache };
  }
}

const moondram = await Moondream.load("../moondream-mobile/assets/models");
const res = await moondram.encodeImage("jojo.jpg");
console.log(res);
// let x = 638 / 2 + 120;
// let y = 900 / 2;
// console.log(image.get(0, y, x), image.get(1, y, x), image.get(2, y, x));
