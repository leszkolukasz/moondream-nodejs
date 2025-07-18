/* eslint-disable @typescript-eslint/no-explicit-any */

import { InferenceSession, TypedTensor, Tensor } from "onnxruntime-node";
// import {
//   InferenceSession,
//   TypedTensor,
//   Tensor,
// } from "onnxruntime-react-native";
// import * as RNFS from "react-native-fs";
import { readFile } from "fs/promises";
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
import { CONTEXT_WINDOW, MAX_IMAGE_SIZE } from "./constants";
// @ts-expect-error No types
import * as nj from "numjs";
import {
  assignSlice,
  concatenateAlongAxis,
  fromTensor,
  resize,
  swapaxes,
  toTensor,
} from "./ndarray";

// Creates a copy of the encoded image kv cache with max sequence length 2048.
const prepareKVCache = (encodedImage: EncodedImage): nj.NdArray => {
  const newShape = [...encodedImage.kvCache.shape];
  newShape[4] = CONTEXT_WINDOW;

  const newKVCache = nj.zeros(newShape);
  assignSlice(
    newKVCache,
    [null, null, null, null, [0, encodedImage.kvCache.shape[4]], null],
    encodedImage.kvCache
  );

  return newKVCache;
};

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
      executionProviders: ["cuda"], //["xnnpack"]
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
      });
    }

    // eslint-disable-next-line prefer-const
    let [patches, patchTemplate] = createPatches(imageTensor);
    patches = swapaxes(patches, 2, 3);
    patches = swapaxes(patches, 1, 2); // (num patches, 3, patchSize, patchSize)

    let result = await this.visionEncoder.run({
      input: toTensor(patches),
    });

    console.log("Vision encoder finished");

    let patchEmb = fromTensor(result.output as TypedTensor<"float32">); // (num patches, 729, 720)
    patchEmb = processPatchEmbeddings(patchEmb, patchTemplate);

    console.log("Patch processing finished");

    patchEmb = patchEmb.reshape(1, 729, 1440);
    result = await this.visionProjection.run({
      input: toTensor(patchEmb),
    });

    console.log("Vision projection finished");

    const inputEmb = fromTensor(result.output as TypedTensor<"float32">); // (1, 729, 1024)

    result = await this.textDecoder.run({
      input_embeds: toTensor(inputEmb),
      kv_cache: this.initialKVCache, // (24, 2, 1, 16, 1, 64)
    });

    console.log("Text decoder finished");

    let newKVCache = fromTensor(result.new_kv_cache as TypedTensor<"float32">); // (24, 2, 1, 16, 729, 64)

    newKVCache = newKVCache.slice(
      null,
      null,
      null,
      null,
      [0, 5], // For speedup
      null
    );

    const kvCache = concatenateAlongAxis(
      [fromTensor(this.initialKVCache), newKVCache],
      4
    );

    return { kvCache };
  }

  async generate(
    inputEmbeds: nj.NdArray, // (1, seq_len, 1024)
    encodedImage: EncodedImage,
    maxTokens: number
  ): Promise<string> {
    console.log("Generating text...");
    let kvSize = encodedImage.kvCache.shape[4];
    const kvCache = prepareKVCache(encodedImage);
    let generatedTokens = 0;
    let inputLen = inputEmbeds.shape[1];

    let text = "";

    inputEmbeds = toTensor(inputEmbeds);

    console.log("Starting generation loop...");

    while (generatedTokens < maxTokens) {
      const x = toTensor(
        kvCache.slice(null, null, null, null, [0, kvSize], null)
      );

      let result = await this.textDecoder.run({
        input_embeds: inputEmbeds,
        kv_cache: x,
      });

      const kvCacheUpdate = fromTensor(
        result.new_kv_cache as TypedTensor<"float32">
      );

      assignSlice(
        kvCache,
        [null, null, null, null, [kvSize, kvSize + inputLen], null],
        kvCacheUpdate
      );
      kvSize += inputLen;

      const logits = fromTensor(result.logits as TypedTensor<"float32">); // (1, 51200)
      const nextToken = nj.argmax(logits)[1];

      const decoded = this.tokenizer.decode([nextToken]);

      text += decoded;
      generatedTokens += 1;

      result = await this.textEncoder.run({
        input_ids: new Tensor(
          "int64",
          BigInt64Array.from([BigInt(nextToken)]),
          [1, 1]
        ),
      });

      inputEmbeds = result.input_embeds as TypedTensor<"float32">;
      inputLen = 1;

      console.log(text);
    }

    return text;
  }

  async caption(
    imageURI: string,
    length: keyof ModelConfig["templates"]["caption"],
    maxTokens: number = 50
  ): Promise<string> {
    const template = this.config.templates.caption[length];

    const result = await this.textEncoder.run({
      input_ids: new Tensor(
        "int64",
        BigInt64Array.from(template, (x) => BigInt(x)),
        [1, template.length]
      ),
    });

    const inputEmbeds = fromTensor(
      result.input_embeds as TypedTensor<"float32">
    );
    const encodedImage = await this.encodeImage(imageURI);

    console.log("Image encoded");

    return this.generate(inputEmbeds, encodedImage, maxTokens);
  }
}
