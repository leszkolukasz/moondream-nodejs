import { InferenceSession } from "onnxruntime-react-native";

export const getSession = async (modelPath: string) => {
  return await InferenceSession.create(modelPath, {
    executionProviders: ["xnnpack"],
  });
};
