// import { InferenceSession } from "onnxruntime-react-native";
import { InferenceSession } from "onnxruntime-node";

export const joinPath = (a: string, b: string): string => {
  if (a.endsWith("/")) {
    a = a.slice(0, -1);
  }

  if (b.startsWith("/")) {
    b = b.slice(1);
  }

  return `${a}/${b}`;
};

export const snakeToCamel = (str: string) => {
  return str.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
};

export const loadONNX = async (
  path: string,
  ortSettings: InferenceSession.SessionOptions
): Promise<InferenceSession> => {
  return InferenceSession.create(path, ortSettings);
};

export const assert = (condition: boolean, message: string = "") => {
  if (!condition) {
    throw message || "Assertion failed";
  }
};
