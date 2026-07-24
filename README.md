Tutorial on how to use `ailia.js`
================

General presentation of the example
-----------------------------------

This tutorial will show you how to use `ailia.js` in an ES6 module.

(Of course this can be adapted for old-style JavaScript)

First, put the files `ailia.js`, `ailia_s.js`, `ailia_s.wasm`, `ailia_simd_s.js`, and `ailia_simd_s.wasm` in the folder of your server.

As a support we'll use the example file `example_mnist_standard_api.html`.

This uses the model called MNIST, which is a convolutional neural network trained to recognize hand-drawn digits.

To try it out, put the HTML file in the same folder as the other files above, as well as the image file `testmnist.jpg`, and open it in a browser.

The typical usage of `ailia.js` can be decomposed in the following steps:

1. import Ailia module
2. initialize the module
3. instantiate an ailia object
4. load model and weights files
5. define or import both a preprocessing and a postprocessing function, to process the input and the output of the neural network
6. perform the inference on some input
7. call the destructor on the ailia object


Here is the full JavaScript content of `example_mnist_standard_api.html`. You don't need to understand everything at the first read, as a detailed explanation of the parts specific to `ailia.js` follows.

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
