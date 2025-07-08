/* eslint-disable @typescript-eslint/no-explicit-any */

import {
  InferenceSession,
  TypedTensor,
  Tensor,
} from "onnxruntime-react-native";
import * as RNFS from "react-native-fs";
import { ModelConfig } from "./types";
import { joinPath, snakeToCamel } from "./utils";
import { Tokenizer } from "./tokenizer";
// import * as jpeg from "jpeg-js";
// import * as png from "fast-png";
// import { Image } from "image-js";
// import { decode as atob } from "base-64";

const loadONNX = async (
  path: string,
  ortSettings: InferenceSession.SessionOptions
): Promise<InferenceSession> => {
  return InferenceSession.create(path, ortSettings);
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
      executionProviders: ["xnnpack"],
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
      await RNFS.readFile(joinPath(modelPath, "config.json"), "utf8")
    );

    const initialKVCacheJJSON = JSON.parse(
      await RNFS.readFile(joinPath(modelPath, "initial_kv_cache.json"), "utf8")
    );

    instance.initialKVCache = new Tensor(
      "float32",
      Float32Array.from(initialKVCacheJJSON.data),
      initialKVCacheJJSON.shape
    );

    instance.tokenizer = Tokenizer.fromConfig(
      JSON.parse(
        await RNFS.readFile(joinPath(modelPath, "tokenizer.json"), "utf8")
      )
    );

    return instance;
  }

  async encodeImage(imageURI: string): Promise<any> {
    const imageData = await RNFS.readFile(imageURI, "base64");
    console.log(imageData);
    // const buffer = Uint8Array.from(atob(imageData), (c) => c.charCodeAt(0));
    // const buffer2 = Buffer.from(imageData, "base64");
    // const dataURL = `data:image/jpeg;base64,${imageData}`;
    // console.log(buffer);
    // console.log(buffer2);
    // const image = jpeg.decode(buffer, { useTArray: true });
    // const image = png.decode(buffer);
    // const image = await Image.load(imageData);

    // console.log(image);

    // return image;
  }
}
