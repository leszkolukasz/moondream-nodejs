## Moondream NodeJS

This is a NodeJS/Bun implementation of inference engine for [Moondream](https://github.com/vikhyat/moondream) model.

> [!NOTE]
> In the beginning, this was meant to run on react-native, but as it turns out it's hard to find good libraries for image processing/tensor manipulation that do not depend on browser/nodejs APIs. One can still try to use this code in react-native by just commenting out/uncommenting the runtime dependent parts and installing `onnxruntime-react-native` package. See [moondream-mobile](https://github.com/leszkolukasz/moondream-mobile) for a semi-working example.

## Usage

The weights are from the `.mf` file which, at least in the past, could be found in the moondream repo. Folder `scripts` contains a `unpack.py` script that unpacks `.mf` file into `.onnx`, `.json`, `.npy` files. File `convert.py` converts those files to format that can be run on NodeJS/React Native that is:

- `.npy` files are converted to `.json` files
- `.onnx` operations are cast to Float32 as by default they are Float16 which is not supported by `onnxruntime-react-native`

Alternatively, one can find converted files in my [huggingface repo](https://huggingface.co/whistleroosh/moondream-0.5B).

File `app.ts` contains an example of how to use the library. It loads the model and runs inference on an image. To run use command:

```bash
bun src/app.ts
```

> [!NOTE]
> While this project is mean to run on NodeJS, bun implements a lot of NodeJS APIs and supports typescript natively, so using it simplified execution.
