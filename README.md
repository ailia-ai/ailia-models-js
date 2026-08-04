`ailia.js` demos and tutorial
================

`ailia.js` is the JavaScript version of the ailia SDK. It runs neural networks entirely inside the browser using WebAssembly, so there is no server-side inference and no data upload.

This repository holds four runnable demos plus a tutorial on the `ailia.js` standard API.

Live demo
---------

You can try the models in your browser at [https://js.ailia.ai/](https://js.ailia.ai/).

Demos in this repository
------------------------

| Demo | Task | Model files |
| --- | --- | --- |
| [`blazeface/`](blazeface/) | Face detection | `blazeface.onnx` |
| [`facemesh_v2/`](facemesh_v2/) | 478 point face mesh | `face_detector.onnx`, `face_landmarks_detector.onnx` |
| [`blazehand/`](blazehand/) | Hand landmark detection | `blazepalm.onnx`, `blazehand.onnx` |
| [`u2netp/`](u2netp/) | Background removal | `u2netp_opset11.onnx` |
| [`mobilenet_ssd/`](mobilenet_ssd/) | Object detection (20 VOC categories) | `mb2-ssd-lite.onnx` |
| [`yolox/`](yolox/) | Object detection (80 COCO categories) | `yolox_tiny.onnx` |

Each demo runs on a bundled sample image (click the image to substitute your own) or on your camera.

FaceMesh-V2's face detector is MediaPipe's short range model, which expects the face to fill a good part of the frame. The face in the shared sample image is smaller than that, so the demo searches a sub-region of the picture and maps the landmarks back, drawing them over the whole image.

Repository layout
-----------------

```
index.html          landing page linking to the four demos
style.css           single stylesheet shared by every page
ailia-js/           the ailia.js SDK (git submodule)
common/             modules shared across demos
  constants.js
  demoUtils.js      model loading, image to tensor conversion, run buttons
  blazeUtils.js     anchor decoding and non-max suppression, shared by the Blaze models
  blazefaceAnchors.js the 896 SSD anchors, shared by Blaze Face and FaceMesh-V2
  detectionUtils.js draws detector boxes, one colour per category
images/             sample images
<model>/            one folder per demo: index.html, its helper modules, its
                    weights and the model's LICENSE
```

Every model folder holds its own `.onnx` weights and `.onnx.prototxt` model file next to its `index.html`, so a relative `fetch()` from the page finds them without any path configuration.

Running the demos locally
-------------------------

Clone with the submodule, otherwise `ailia-js/` will be empty:

```bash
git clone --recurse-submodules https://github.com/ailia-ai/ailia-models-js.git
```

If you have already cloned without it:

```bash
git submodule update --init --recursive
```

The pages must be served over HTTP. Opening them as `file://` fails, because ES module imports and the `fetch()` calls that load the model files are both blocked on that scheme:

```bash
python3 -m http.server 8000
```

Then open <http://localhost:8000/>.

Two ways to call `ailia.js`
---------------------------

- `blazeface/` and `blazehand/` use the standard API, `ailia.run` with preprocessing and postprocessing callbacks, which is what the tutorial below explains. `ailia.run` owns the input and output buffers, so the callbacks only have to fill and read tensors. The Blaze models still need anchor decoding and weighted non-max suppression, and that work lives in their postprocessing callbacks. Blaze Hand also shows how to vary the batch size at run time, by passing the input shape as `ailia.run`'s third argument: one crop per detected hand.

- `facemesh_v2/` chains two models on `ailia.run`: a face detector, then the mesh on a rotated crop of each face. Its models are converted from TFLite and take NHWC input, so it passes `channelsLast` to the shared image conversion, and the mesh reports no default input shape so the shape is passed to `ailia.run` explicitly.

- `u2netp/` also uses `ailia.run`. Its preprocessing needs per channel mean and standard deviation rather than one of the shared normalization modes, so it writes the tensor itself in the preprocessing callback.

- `mobilenet_ssd/` and `yolox/` use `AiliaDetector` instead. It wraps the detector algorithm, the category list and the thresholds for a known model type, and decodes the boxes natively, so there is no postprocessing to write. Moving these to `ailia.run` would mean reimplementing the SSD and YOLOX decoding in JavaScript.

License
-------

`ailia.js` is a commercial library, and commercial use requires a license. See `ailia-js/LICENSE_AILIA_EN.pdf` and contact [ailia Inc.](https://ailia.ai/) for licensing.

The model weights are redistributed here under their own upstream licenses, and each model folder carries the matching `LICENSE` file:

| Model | License |
| --- | --- |
| [`blazeface/LICENSE`](blazeface/LICENSE) | Apache 2.0 |
| [`facemesh_v2/LICENSE`](facemesh_v2/LICENSE) | Apache 2.0 (MediaPipe) |
| [`blazehand/LICENSE`](blazehand/LICENSE) | Apache 2.0 (MediaPipe), via the BlazeFace-PyTorch conversion |
| [`u2netp/LICENSE`](u2netp/LICENSE) | Apache 2.0 |
| [`mobilenet_ssd/LICENSE`](mobilenet_ssd/LICENSE) | MIT |
| [`yolox/LICENSE`](yolox/LICENSE) | Apache 2.0 |

These cover the models only. They say nothing about `ailia.js` itself, which stays under the commercial license above.

Tutorial: the standard API
================

This tutorial will show you how to use `ailia.js` in an ES6 module.

(Of course this can be adapted for old-style JavaScript)

First, put the files `ailia.js`, `ailia_s.js`, `ailia_s.wasm`, `ailia_simd_s.js`, and `ailia_simd_s.wasm` in the folder of your server. In this repository those files live in `ailia-js/`, which is why the demo pages import from `../ailia-js/ailia.js` rather than `./ailia.js`.

As a support we'll use the example file `example_mnist_standard_api.html`.

This uses the model called MNIST, which is a convolutional neural network trained to recognize hand-drawn digits.

Note that this file, and the `testmnist.jpg` it reads, are **not** part of this repository. The snippets below are a self-contained walkthrough: to try it out, put the HTML file in the same folder as the other files above, as well as the image file `testmnist.jpg`, and open it in a browser.

The typical usage of `ailia.js` can be decomposed in the following steps:

1. import Ailia module
2. initialize the module
3. instantiate an ailia object
4. load model and weights files
5. define or import both a preprocessing and a postprocessing function, to process the input and the output of the neural network
6. perform the inference on some input
7. call the destructor on the ailia object


Here is the full JavaScript content of `example_mnist_standard_api.html`. Its imports assume `ailia.js` and `demoUtils.js` sit next to the HTML file; adjust the paths to wherever you put them. You don't need to understand everything at the first read, as a detailed explanation of the parts specific to `ailia.js` follows.

```javascript
import {Ailia} from './ailia.js';
import {readFileAsync, distinguishModelFiles, readImageData} from './demoUtils.js';

await Ailia.initialize();
let ailia = new Ailia();

modelfiles.addEventListener('change', async () => {
    await loadModel();
    myResult.innerText = '';
});

async function loadModel() {
    if (modelFiles.files.length !== 2) {
       return;
    }

    let {modelFile, weightsFile} = distinguishModelFiles(modelFiles.files);

    if (modelFile === undefined) {
        console.log('Error: could not find a model file with the extension .onnx.prototxt');
        return;
    }
    if (weightsFile === undefined) {
        console.log('Error: could not find a weights file with the extension .onnx');
        return;
    }

    console.log(`model file: ${modelFile.name}`);
    console.log(`weights file: ${weightsFile.name}`);

    let [modelFileContents, weightsFileContents] = await Promise.all([
        readFileAsync(modelFile),
        readFileAsync(weightsFile)
    ]);

    let loadStatus = ailia.loadModelAndWeights(modelFileContents, weightsFileContents);
    if (loadStatus) {
        console.log('model files loaded');
        document.querySelector('button.myButton').removeAttribute("hidden");
    }
}

function buttonClicked() {
    return new Promise((resolve) =>
        document.querySelector('.myButton').addEventListener('click', () => resolve())
    );
}

await buttonClicked();
console.log('button clicked');

let inputPicturePath = 'testmnist.jpg';
let inputPicData = await readImageData(inputPicturePath);

let preprocessFactory = function (inputPictureData) {
    let preprocessingFunc = function (inputBuffers) {
        for (let y = 0; y<inputPictureData.height; ++y) {
            const lineOffset = y * inputPictureData.width;
            for (let x = 0; x<inputPictureData.width; ++x) {
                let red = inputPictureData.data[(lineOffset + x) * 4];
                inputBuffers.byIndex[0].buffer.Float32[lineOffset + x] = red - 128;
            }
        }
    };
    return preprocessingFunc;
};

let postprocessCb = function (outputBuffers) {
    let result = [];
    for (const floatValue of outputBuffers.byIndex[0].buffer.Float32) {
        result.push(floatValue);
    }
    return result;
};


try {

    let result = ailia.run(preprocessFactory(inputPicData), postprocessCb, [[1, 1, inputPicData.width, inputPicData.height]]);

    for (const [index, resultItem] of Object.entries(result)) {
        console.log(`prediction score of ${index}: ${resultItem}`);
    }

} catch (e) {
    console.log(e);
} finally {
    ailia.destroy();
}
```

Now here is a detailed explanation of the important parts.

Importing `ailia.js`
-------------------

First, import `Ailia` from your script:

```javascript

import {Ailia} from './ailia.js';

```

(The other imports (from `demoUtils.js`) are not strictly required, those are just convenience helper functions to facilitate the loading of the model files from the browser, as well as the conversion of the input image into an array.)

Initializing `ailia.js`
----------------------

Then you need to initialize the `Ailia` module:

```javascript
await Ailia.initialize();
```

This initialization needs to be done only once for all, even if you plan to have several instances of `Ailia` or even if you want to reload successively different models in the same `Ailia` instances.

Instantiation of an `Ailia` object
------------

Only then (now that you've initialized the `Ailia` module)  you can instantiate an `Ailia` object:

```javascript
let ailia = new Ailia();
```

Load model and weights files
-------------------------

Then, you need to load some model files.

There are many lines present just to provide some mechanism to load the model files from the browser, but the really important line is:

```javascript
let loadStatus = ailia.loadModelAndWeights(modelFileContents, weightsFileContents);

```

`Ailia` needs two files, one _model file_ which contains the model structure but not its weights, and one _weights file_ which contains the whole model including the weights.

Both files are necessary to run the model. However you can load them in two steps, first the model and then the weights one, or even load again a different weights file if you want to change the weights without reloading the model file.

But in this simple example we use the command that loads them both in one single statement, `ailia.loadModelAndWeights(model, weights)`.


Define a preprocessing function
--------------------

The raw convolutional neural network takes in input a multidimentional array of float numbers, which is almost certainly what you do not have as an input.

Rather, you probably want to feed image files, or maybe videos, audio files, etc, to the CNN.

Thus you need a way to convert those media files into the array that the CNN can understand.

This is what the preprocessing function is for, and you need to define one (but you could also import a preexisting one).


```javascript
let preprocessFactory = function (inputPictureData) {
    let preprocessingFunc = function (inputBuffers) {
        for (let y = 0; y<inputPictureData.height; ++y) {
            const lineOffset = y * inputPictureData.width;
            for (let x = 0; x<inputPictureData.width; ++x) {
                let red = inputPictureData.data[(lineOffset + x) * 4];
                inputBuffers.byIndex[0].buffer.Float32[lineOffset + x] = red - 128;
            }
        }
    };
    return preprocessingFunc;
};
```

In order to avoid you to have to create and manage the memory buffers used internally by `Ailia`, the preprocessing function is designed to be passed as a callback, that will take in parameter the object containing the buffers (`inputBuffers`) that `Ailia` created for you.

Then in the body of the callback you just need to fill each of the input buffers (which is accessed like a JS array) with the data as floats.

The argument `inputBuffers` gives you access to all of the input buffers.

Each model can have several inputs (but sometimes only one; theoretically they can also have none, although this would be rare and of dubious usefulness), so to access the input buffers associated with these inputs the `inputBuffers` object provides two ways:

- `inputBuffers.byIndex` is an array of input buffers, numbered in the order that the model defines them. For example you access to the first one by `inputBuffers.byIndex[0]`.

- `inputBuffers.byName` is an object whose properties are the input names and whose values are the corresponding buffers. For example you access to the input named `input_data` by `inputBuffers.byName['input_data']`.

Once you get an input buffer using one of the two methods described above, the buffer object gives you all the necessary to easily fill input data into it.

In our example, `inputBuffers.byIndex[0].buffer.Float32` provides an interface to the input buffer that is in all aspects similar to a normal array, in which the values are stored in 32 bits floats (4 bytes of the raw buffer interpreted as a `float`).

Please note that the preprocessing function completely determines the contents of the input buffers, so that if you need an easy way to make it depend on an input file, you should use a factory pattern. This is exactly what has been done in this example: the factory takes in parameter an image file, and returns a tailored preprocessing callback ready to be passed to the `Ailia` inference instruction (we'll get to that shortly).


Define a postprocessing function
--------------------

Similarly to what we said about input of the raw CNN, its output is also what you almost certainly do not want, as it is a multidimensional array of float numbers.

So you need to extract the actual useful result from these floats, and for that you need a _postprocessing function_, that you will define (but you could also import a preexisting one).

```javascript
let postprocessCb = function (outputBuffers) {
    let result = [];
    for (const floatValue of outputBuffers.byIndex[0].buffer.Float32) {
        result.push(floatValue);
    }
    return result;
};
```

Here the overall design is simpler than for the preprocessing, as you rarely need to use a factory pattern for the postprocessing function.

Similarly to the preprocessing callback, `Ailia` provides to the postprocessing callback a `outputBuffers` object that gives you access to all the output buffers (possibly one).

The only difference being that instead of writing into them you will mainly be reading from them, although the access to each output buffer is still done the same way as with a standard array.

Perform the inference on some input
--------------------------

Just pass to `ailia.run` the preprocessing callback (if necessary, pass some arguments to the preprocessing factory function, here the input image), the postprocessing callback, and the input shapes.

As the CNN can possibly have several inputs, the input shapes argument is an array of input shapes, and each input shape is just an array of integers, which are the dimensions of the shape.

In case of a shape that has less than 4 dimensions, complete with `1` values at the beginning of the shape.


```javascript
let result = ailia.run(preprocessFactory(inputPicData), postprocessCb, [[1, 1, inputPicData.width, inputPicData.height]]);
```

The result contains the values returned by the postprocessing callback, i.e. the output data that you can immediately start exploiting.


Call the destructor on the ailia object
-------------------------

Simply call the `destroy` method on the `ailia` object.

This will take care of freeing all the allocated memory.

```javascript
ailia.destroy();
```

Notice that in the example this instruction is inside a `finally` block, so that, even if some error or some exception is raised after the `ailia` object is created (all that follows its creation is in a `try` block), there will be some cleaning done anyway.
