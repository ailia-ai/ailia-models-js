const INPUT_SIZE = 320;

// Per channel mean and standard deviation from the U^2-Net repository's
// ToTensorLab transform, applied after the image is scaled to 0..1
const CHANNEL_MEAN = [0.485, 0.456, 0.406];
const CHANNEL_STD = [0.229, 0.224, 0.225];

// Size of the squares in the transparency backdrop, in output pixels
const CHECKER_SIZE = 16;
const CHECKER_LIGHT = '#ffffff';
const CHECKER_DARK = '#d8dce1';

/**
 * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} inputPic
 */
function imageSize(inputPic) {
    return {
        width: inputPic.naturalWidth || inputPic.videoWidth || inputPic.width,
        height: inputPic.naturalHeight || inputPic.videoHeight || inputPic.height
    };
}

/**
 * Scales the input to the model's square input and writes it as an NCHW tensor.
 * The aspect ratio is not preserved, matching the reference implementation, which
 * resizes straight to 320x320.
 *
 * @param {Float32Array} destination
 * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} inputPic
 */
function writeInput(destination, inputPic) {
    const canvas = document.createElement('canvas');
    canvas.width = INPUT_SIZE;
    canvas.height = INPUT_SIZE;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(inputPic, 0, 0, INPUT_SIZE, INPUT_SIZE);
    const { data } = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);

    // The reference divides by the brightest sample rather than by 255, so a
    // dim picture is stretched to the full range before normalization
    let brightest = 0;
    for (let i = 0; i < data.length; i += 4) {
        brightest = Math.max(brightest, data[i], data[i + 1], data[i + 2]);
    }
    if (brightest === 0) {
        brightest = 1;
    }

    const pixelCount = INPUT_SIZE * INPUT_SIZE;
    for (let c = 0; c < 3; c++) {
        for (let i = 0; i < pixelCount; i++) {
            destination[c * pixelCount + i] =
                (data[i * 4 + c] / brightest - CHANNEL_MEAN[c]) / CHANNEL_STD[c];
        }
    }
}

/**
 * Rescales the saliency map to 0..1. The model's output is unbounded, so the
 * reference normalizes against the map's own range.
 *
 * @param {Float32Array} prediction the first output blob, 320x320
 * @returns {Uint8ClampedArray} the map as 8 bit alpha values
 */
function normalizeMask(prediction) {
    const pixelCount = INPUT_SIZE * INPUT_SIZE;
    let min = Infinity;
    let max = -Infinity;

    for (let i = 0; i < pixelCount; i++) {
        if (prediction[i] < min) {
            min = prediction[i];
        }
        if (prediction[i] > max) {
            max = prediction[i];
        }
    }

    const range = max - min || 1;
    const mask = new Uint8ClampedArray(pixelCount);
    for (let i = 0; i < pixelCount; i++) {
        mask[i] = Math.round(255 * (prediction[i] - min) / range);
    }

    return mask;
}

/**
 * Runs U^2-Net and returns the saliency mask at the model's resolution.
 *
 * @param {*} ailia an Ailia instance holding u2netp_opset11.onnx
 * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} inputPic
 * @returns {{size: number, mask: Uint8ClampedArray}}
 */
export function computeU2net(ailia, inputPic) {
    return ailia.run(
        inputBuffers => writeInput(inputBuffers.byIndex[0].buffer.Float32, inputPic),
        // U^2-Net emits the fused map plus six side outputs; only the first is used
        outputBuffers => ({
            size: INPUT_SIZE,
            mask: normalizeMask(outputBuffers.byIndex[0].buffer.Float32)
        })
    );
}

/**
 * Paints the checkerboard that makes the removed background visible.
 */
function drawTransparencyBackdrop(ctx, width, height) {
    ctx.fillStyle = CHECKER_LIGHT;
    ctx.fillRect(0, 0, width, height);

    ctx.fillStyle = CHECKER_DARK;
    for (let y = 0; y < height; y += CHECKER_SIZE) {
        for (let x = 0; x < width; x += CHECKER_SIZE) {
            if (((x / CHECKER_SIZE) + (y / CHECKER_SIZE)) % 2 === 1) {
                ctx.fillRect(x, y, CHECKER_SIZE, CHECKER_SIZE);
            }
        }
    }
}

/**
 * Draws the input with its background removed, over a checkerboard so the
 * transparency is visible. The canvas is filled opaquely, so it can sit on top of
 * the untouched copy of the image that the demo scaffolding draws underneath.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} inputPic
 * @param {{size: number, mask: Uint8ClampedArray}} result from computeU2net
 */
export function plotCutout(canvas, inputPic, result) {
    const width = canvas.width;
    const height = canvas.height;
    const ctx = canvas.getContext('2d');

    // Let the browser interpolate the mask up to the output size
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = result.size;
    maskCanvas.height = result.size;
    const maskImage = new ImageData(result.size, result.size);
    for (let i = 0; i < result.mask.length; i++) {
        maskImage.data[i * 4] = result.mask[i];
        maskImage.data[i * 4 + 3] = 255;
    }
    maskCanvas.getContext('2d').putImageData(maskImage, 0, 0);

    const scaledMask = document.createElement('canvas');
    scaledMask.width = width;
    scaledMask.height = height;
    scaledMask.getContext('2d').drawImage(maskCanvas, 0, 0, width, height);
    const scaledMaskData = scaledMask.getContext('2d').getImageData(0, 0, width, height).data;

    const cutout = document.createElement('canvas');
    cutout.width = width;
    cutout.height = height;
    const cutoutCtx = cutout.getContext('2d');
    cutoutCtx.drawImage(inputPic, 0, 0, width, height);

    const cutoutData = cutoutCtx.getImageData(0, 0, width, height);
    for (let i = 0; i < width * height; i++) {
        cutoutData.data[i * 4 + 3] = scaledMaskData[i * 4];
    }
    cutoutCtx.putImageData(cutoutData, 0, 0);

    drawTransparencyBackdrop(ctx, width, height);
    ctx.drawImage(cutout, 0, 0);
}

/**
 * Draws the saliency mask on its own, which is easier to judge than the cutout
 * when the subject and the background are a similar colour.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {{size: number, mask: Uint8ClampedArray}} result from computeU2net
 */
export function plotMask(canvas, result) {
    const image = new ImageData(result.size, result.size);
    for (let i = 0; i < result.mask.length; i++) {
        image.data[i * 4] = result.mask[i];
        image.data[i * 4 + 1] = result.mask[i];
        image.data[i * 4 + 2] = result.mask[i];
        image.data[i * 4 + 3] = 255;
    }

    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = result.size;
    maskCanvas.height = result.size;
    maskCanvas.getContext('2d').putImageData(image, 0, 0);

    canvas.getContext('2d').drawImage(maskCanvas, 0, 0, canvas.width, canvas.height);
}

export { imageSize };
