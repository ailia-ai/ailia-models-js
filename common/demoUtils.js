import { Ailia } from '../ailia-js/ailia.js';

const VIDEO_STREAM_STATE = {
    STOPPED: 0,
    STOP_REQUESTED: 1,
    RUNNING: 2
};

let videoStreamState = VIDEO_STREAM_STATE.STOPPED;

export function readFileAsync(file) {
    return new Promise((resolve, reject) => {
        let reader = new FileReader();
        reader.onabort = () => reject(new Error('aborted'));
        reader.onerror = () => reject(reader.error);
        reader.onload = () => resolve(reader.result);
        reader.readAsArrayBuffer(file);
    });
}

export function distinguishModelFiles([fileA, fileB]) {
    let modelFile;
    let weightsFile;
    if (fileA.name.endsWith('.onnx')) {
        weightsFile = fileA;
    } else if (fileA.name.endsWith('.onnx.prototxt')) {
        modelFile = fileA;
    }
    if (fileB.name.endsWith('.onnx')) {
        weightsFile = fileB;
    } else if (fileB.name.endsWith('.onnx.prototxt')) {
        modelFile = fileB;
    }
    return {modelFile, weightsFile};
}

export async function readImageData(path) {
    const image = new Image();

    image.crossOrigin = 'anonymous';
    image.src = path;

    await image.decode();

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    ctx.drawImage(image, 0, 0);
    const imageData = ctx.getImageData(0, 0, image.naturalWidth, image.naturalHeight);
    return imageData; // <--- This carries the width, height and the pixel format.
}

export function setStatus(message, className = '') {
    const baseClassName = 'status-container';
    const statusElement = document.querySelector(`.${baseClassName}`);

    statusElement.innerText = message;
    // Overwriting the className like this is easier than manually iterating
    // over potential message styles (info, debug, log, error) to remove
    // them before adding the one (potentially) passed as an argument
    statusElement.className = `${baseClassName} ${className}`;
}

export function getRatioToMaxSizeWithinFrame(width, height, maxWidth, maxHeight) {
    const horizontalMatchRatio = maxWidth / width;
    const verticalMatchRatio = maxHeight / height;

    const finalRatio = (
        width * verticalMatchRatio > maxWidth ? horizontalMatchRatio : verticalMatchRatio
    );

    return finalRatio;
}

export function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function ensureModelFilesPresence(modelName) {
    for (let i = 0; i < 5; i++) {
        try {
            const rep = await fetch(`${modelName}`, { method: 'POST' });

            if (rep.status === 202) {
                i -= 1;
                continue;
            }

            if (rep.status >= 200 && rep.status <= 299) {
                return true;
            }

            if (rep.status >= 400 && rep.status <= 499) {
                return false;
            }

            await delay(1000);
        } catch { }
    }

    return false;
}

export function changeSampleImage(imageElement) {
    imageElement = imageElement || document.querySelector('.input-container img');

    const inputFileElement = document.createElement('input');
    inputFileElement.setAttribute('type', 'file');
    inputFileElement.setAttribute('accept', 'image/*');

    inputFileElement.addEventListener('change', e => {
        if (imageElement.src) {
            URL.revokeObjectURL(imageElement.src);
        }

        const objectURL = URL.createObjectURL(e.target.files[0]);
        imageElement.src = objectURL;
    });

    inputFileElement.click();
}

export function downloadCanvasAsImage(canvas, filename = 'canvas.png') {
    const anchorElement = document.createElement('a');
    anchorElement.href = canvas.toDataURL('image/png').replace(/^data:image\/[^;]/, 'data:application/octet-stream');
    anchorElement.download = filename;

    anchorElement.click();
}

export const Normalization = {
    None: 0,
    Unit: 1, // [0, 1]
    HalfUnitCentered: 2, // [-0.5, 0.5]
    UnitCentered: 3 // [-1, 1]
};

/**
 * Converts one or more images into the NCHW float tensor that ailia expects and
 * writes it straight into a buffer that `ailia.run` provides, so no manual
 * allocation is needed.
 *
 * Defaults to the NCHW channel order. Models converted from TFLite, such as the
 * MediaPipe face detector and face mesh, expect NHWC instead; pass
 * `channelsLast: true` for those.
 *
 * @param {Float32Array} destination the input buffer handed to the preprocessing callback
 * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement|Array} inputImages
 * @param {number} requiredWidth
 * @param {number} requiredHeight
 */
export function writeInputImage(destination, inputImages, requiredWidth, requiredHeight, { toGreyscale: greyscale, normalize, channelsLast } = { toGreyscale: false, normalize: Normalization.None }) {
    if (Array.isArray(inputImages) === false) {
        inputImages = [inputImages];
    }

    const colorCanalCount = (greyscale ? 1 : 3);

    const unsupportedImages = inputImages.filter(image =>
        image.constructor.name !== 'HTMLImageElement' &&
        image.constructor.name !== 'HTMLVideoElement' &&
        image.constructor.name !== 'HTMLCanvasElement' &&
        (image.constructor.name === 'HTMLCanvasElement' && !image.getContext('2d'))
    );

    if (unsupportedImages.length > 0) {
        const errorMessage = unsupportedImages.map(image => (
            (image.constructor.name === 'HTMLCanvasElement' && !image.getContext('2d')) ?
                'unsupported image object type: HTMLCanvasElement with a non-2D context' :
                'unsupported image object type: ' + image
        )).join('\n');

        throw new Error(errorMessage);
    }

    inputImages.forEach((inputImage, inputIndex) => {
        let ctx;

        if (
            inputImage.constructor.name === 'HTMLImageElement' ||
            inputImage.constructor.name === 'HTMLVideoElement' ||
            (
                inputImage.constructor.name === 'HTMLCanvasElement' && (inputImage.width !== requiredWidth || inputImage.height !== requiredHeight)
            )
        ) {
            // Create an empty canvas element
            const canvas = document.createElement('canvas');
            canvas.width = requiredWidth;
            canvas.height = requiredHeight;

            const initialWidth = inputImage.naturalWidth || inputImage.videoWidth || inputImage.width;
            const initialHeight = inputImage.naturalHeight || inputImage.videoHeight || inputImage.height;
            const ratio = getRatioToMaxSizeWithinFrame(
                initialWidth,
                initialHeight,
                requiredWidth,
                requiredHeight
            );
            const finalWidth = ratio * initialWidth;
            const finalHeight = ratio * initialHeight;
            const xOffset = (requiredWidth - finalWidth) / 2;
            const yOffset = (requiredHeight - finalHeight) / 2;

            // Copy the image contents to the canvas, while resizing it
            ctx = canvas.getContext('2d');
            ctx.fillStyle = '#000000';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(inputImage, xOffset, yOffset, finalWidth, finalHeight);
        } else if (inputImage.constructor.name === 'HTMLCanvasElement') {
            ctx = inputImage.getContext('2d');
        }

        // Extract the data array representing the decoded image
        const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);

        for (let c = 0; c < colorCanalCount; ++c) {
            for (let j = 0; j < requiredHeight; ++j) {
                for (let i = 0; i < requiredWidth; ++i) {
                    let value = imageData.data[(j * requiredWidth + i) * 4 + c];

                    if (greyscale) {
                        value *= 0.299;
                        value += 0.587 * imageData.data[(j * requiredWidth + i) * 4 + c + 1] + 0.114 * imageData.data[(j * requiredWidth + i) * 4 + c + 2];
                        value = Math.round(value);
                    }

                    switch (normalize) {
                        case Normalization.None:
                            break;
                        case Normalization.Unit:
                            value /= 255;
                            break;
                        case Normalization.HalfUnitCentered:
                            value = value / 255 - 0.5;
                            break;
                        case Normalization.UnitCentered:
                            value = value / 127.5 - 1;
                            break;
                        default:
                            break;
                    }

                    destination[
                        channelsLast ?
                            ((inputIndex * requiredHeight + j) * requiredWidth + i) * colorCanalCount + c :
                            ((inputIndex * colorCanalCount + c) * requiredHeight + j) * requiredWidth + i
                    ] = value;
                }
            }
        }
    });
}

export async function createAiliaWithModel(modelName, modelFileName) {
    modelFileName = modelFileName ?? modelName;
    //const isModelAvailable = await ensureModelFilesPresence(modelName);

    //if (!isModelAvailable) {
    //    const message = `Could not ensure model presence on the backend: ${modelName}`;
    //    console.error(message);
    //    setStatus(message, 'error');
    //    return null;
    //}

    const modelFiles = await Promise.all([
        `${modelFileName}.onnx`,
        `${modelFileName}.onnx.prototxt`
    ].map(async filename => {
        return await (await fetch(filename)).arrayBuffer();
    }));

    const ailia = new Ailia();
    ailia.loadModelAndWeights(modelFiles[1], modelFiles[0]);

    return ailia;
}

export function setupSampleImageRunButton(runSampleProcessing) {
    const singleRunButton = document.querySelector('.singleImageButton');

    singleRunButton.addEventListener('click', async function () {
        if (videoStreamState !== VIDEO_STREAM_STATE.STOPPED) {
            videoStreamState = VIDEO_STREAM_STATE.STOP_REQUESTED;
        }

        const sampleImage = document.querySelector('.sample-image');

        const containerElement = document.querySelector('.result-container');
        for (const child of Array.from(containerElement.children)) {
            containerElement.removeChild(child);
        }
        const outputOverlayCanvas = document.querySelector('.result-canvas') || document.createElement('canvas');

        await delay(15);

        outputOverlayCanvas.className = 'result-canvas';
        outputOverlayCanvas.width = sampleImage.naturalWidth;
        outputOverlayCanvas.height = sampleImage.naturalHeight;
        // Clearing all the inline styles to reset the outputOverlayCanvas in case it is getting reused
        outputOverlayCanvas.setAttribute('style', '');
        outputOverlayCanvas.style.position = 'absolute';
        outputOverlayCanvas.style.top = 0;
        outputOverlayCanvas.style.bottom = 0;
        outputOverlayCanvas.style.left = 0;
        outputOverlayCanvas.style.right = 0;
        outputOverlayCanvas.style.maxWidth = '100%';
        const imageCanvasElement = document.createElement('canvas');
        imageCanvasElement.width = sampleImage.naturalWidth;
        imageCanvasElement.height = sampleImage.naturalHeight;
        imageCanvasElement.getContext('2d').drawImage(sampleImage, 0, 0);
        containerElement.appendChild(imageCanvasElement);
        containerElement.appendChild(outputOverlayCanvas);

        await runSampleProcessing(sampleImage, outputOverlayCanvas);
    });
}

async function setupVideoCapture() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setStatus('Your browser does not support user media.', 'error');
        return null;
    }

    try {
        return await navigator.mediaDevices.getUserMedia({ video: true });
    } catch (error) {
        setStatus(`Failed to acquire video stream: ${error.message}`, 'error');
        return null;
    }
}

function videoReadyAsync(videoElement) {
    if (videoElement.readyState >= 3) {
        return Promise.resolved();
    } else {
        return new Promise((resolve) => {
            videoElement.addEventListener('canplay', resolve);
        });
    }
}

export function setupVideoStreamRunButton(runVideoStreamProcessing) {
    const cameraButton = document.querySelector('.videoStreamButton');

    cameraButton.addEventListener('click', async () => {
        if (videoStreamState !== VIDEO_STREAM_STATE.STOPPED) {
            return;
        }

        const mediaStream = await setupVideoCapture();

        const containerElement = document.querySelector('.result-container');
        for (const child of Array.from(containerElement.children)) {
            containerElement.removeChild(child);
        }
        const outputOverlayCanvas = document.querySelector('.result-canvas') || document.createElement('canvas');
        outputOverlayCanvas.className = 'result-canvas';
        // Clearing all the inline styles to reset the outputOverlayCanvas in case it is getting reused
        outputOverlayCanvas.setAttribute('style', '');
        outputOverlayCanvas.style.position = 'absolute';
        outputOverlayCanvas.style.top = 0;
        outputOverlayCanvas.style.bottom = 0;
        outputOverlayCanvas.style.left = 0;
        outputOverlayCanvas.style.right = 0;
        outputOverlayCanvas.style.width = '100%';
        const videoElement = document.createElement('video');
        videoElement.style.width = '100%';
        
        // for Unhandled Promise Rejection: NotAllowedError: The request is not allowed by the user agent or the platform in the current context, possibly because the user denied permission.
        videoElement.autoplay = true;
        videoElement.muted = true;
        videoElement.playsInline = true;

        videoElement.srcObject = mediaStream;

        containerElement.appendChild(videoElement);
        containerElement.appendChild(outputOverlayCanvas);

        await videoReadyAsync(videoElement);

        outputOverlayCanvas.width = videoElement.videoWidth;
        outputOverlayCanvas.height = videoElement.videoHeight;

        async function processFrame() {
            if (videoStreamState === VIDEO_STREAM_STATE.STOP_REQUESTED) {
                videoStreamState = VIDEO_STREAM_STATE.STOPPED;

                return;
            }

            await runVideoStreamProcessing(videoElement, outputOverlayCanvas);

            requestAnimationFrame(processFrame);
        }

        videoStreamState = VIDEO_STREAM_STATE.RUNNING;
        requestAnimationFrame(processFrame);

        videoElement.play();
    });
}

/**
 * @param {number} angle
 */
export function normalizeRadians(angle) {
    return angle - 2 * Math.PI * Math.floor((angle - (-Math.PI)) / (2 * Math.PI));
}
