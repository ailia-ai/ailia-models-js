import { weightedNonMaxSuppression, generateBoxDecoder, sigmoid } from './blazeUtils.js';
import { CONSTANTS } from './constants.js';
import { loadInputImage, Normalization } from './demoUtils.js';
import anchors from './blazefaceAnchors.js';

const REQUIRED_INPUT_WIDTH = 128;
const REQUIRED_INPUT_HEIGHT = 128;
const TENSOR_COUNT = 896;
const TENSOR_SIZE = 16;
const FEATURE_COUNT = 6;
const SCORE_THRESHOLD = 100.0;
const MINIMUM_SCORE_THRESHOLD = 0.75;
const MINIMUM_OVERLAP_THRESHOLD = 0.3;
const BLAZEFACE_INPUT_CHANNEL_COUNT = 3;
const BLAZEFACE_FEATURES = {
    RIGHT_EYE: 0,
    LEFT_EYE: 1,
    NOSE: 2,
    MOUTH: 3,
    RIGHT_EAR: 4,
    LEFT_EAR: 5
};

/**
 * @typedef {Object} DetectionBox
 * @property {number} xMin
 * @property {number} xMax
 * @property {number} yMin
 * @property {number} yMax
 * @property {number} score
 * @property {number} area
 */

/**
 * @param {Float32Array} heap
 * @param {number} outputBoxesRef
 * @param {number} outputScoresRef
 */
export function postprocess(heap, outputBoxesRef, outputScoresRef) {
    outputBoxesRef /= CONSTANTS.SIZE_OF_FLOAT;
    outputScoresRef /= CONSTANTS.SIZE_OF_FLOAT;
    let boxes = decodeBoxes(heap, outputBoxesRef);
    const rawScores = [];

    for (let i = 0; i < TENSOR_COUNT; i++) {
        rawScores.push(heap[outputScoresRef + i]);
        boxes[i].score = sigmoid(Math.max(Math.min(heap[outputScoresRef + i], SCORE_THRESHOLD), -SCORE_THRESHOLD));
    }

    boxes = boxes.filter(b => b.score > MINIMUM_SCORE_THRESHOLD);

    const filteredDetections = weightedNonMaxSuppression(boxes, MINIMUM_OVERLAP_THRESHOLD);

    // @ts-ignore
    window.rawScores = rawScores;
    // @ts-ignore
    window.scores = boxes.map(b => b.score);
    // @ts-ignore
    window.boxes = boxes;
    // @ts-ignore
    window.filteredDetections = filteredDetections;

    return filteredDetections;
}

const decodeBoxes = generateBoxDecoder(REQUIRED_INPUT_WIDTH, TENSOR_COUNT, TENSOR_SIZE, FEATURE_COUNT, anchors);

export function computeBlazeface(ailia, inputPic) {
    const { inputImageBufferRef: inputImageBuffer } = loadInputImage(ailia, inputPic, REQUIRED_INPUT_WIDTH, REQUIRED_INPUT_HEIGHT, { normalize: Normalization.UnitCentered });

    const outputShape = ailia.getOutputShape();

    const totalBoxesOutputSize = outputShape.x * outputShape.y * outputShape.z * outputShape.w * CONSTANTS.SIZE_OF_FLOAT;
    const totalInputSize = REQUIRED_INPUT_WIDTH * REQUIRED_INPUT_HEIGHT * BLAZEFACE_INPUT_CHANNEL_COUNT * CONSTANTS.SIZE_OF_FLOAT;

    // ailia.Module._free(outputShapeRef);

    const outputBoxesRef = ailia.Module._malloc(totalBoxesOutputSize * CONSTANTS.SIZE_OF_FLOAT);
    const outputScoresRef = ailia.Module._malloc(1 * CONSTANTS.SIZE_OF_REF);
    const outputSizeRef = ailia.Module._malloc(CONSTANTS.SIZE_OF_REF);

    // ailiaPredict(struct AILIANetwork* net, void* dest, unsigned int dest_size, const void* src, unsigned int src_size);
    ailia.call('ailiaPredict', 'number', [ailia.ailiaDoublePointer.getPointer(), outputBoxesRef, totalBoxesOutputSize, inputImageBuffer, totalInputSize]);

    // ailiaRefOutputBlobBuffer(struct AILIANetwork* net, const void** ptr, unsigned int* size, unsigned int index);
    ailia.call('ailiaRefOutputBlobBuffer', 'number', [ailia.ailiaDoublePointer.getPointer(), outputBoxesRef, outputSizeRef, 0]);

    // ailiaRefOutputBlobBuffer(struct AILIANetwork* net, const void** ptr, unsigned int* size, unsigned int index);
    ailia.call('ailiaRefOutputBlobBuffer', 'number', [ailia.ailiaDoublePointer.getPointer(), outputScoresRef, outputSizeRef, 1]);

    const faces = postprocess(ailia.Module.HEAPF32, ailia.Module.HEAPU32[outputBoxesRef / CONSTANTS.SIZE_OF_U32], ailia.Module.HEAPU32[outputScoresRef / CONSTANTS.SIZE_OF_U32]);

    ailia.Module._free(outputBoxesRef);
    ailia.Module._free(outputScoresRef);
    ailia.Module._free(outputSizeRef);
    ailia.Module._free(inputImageBuffer);

    return faces;
}

/**
 * @param {HTMLCanvasElement} canvas
 * @param {DetectionBox[]} faces
 * @param {HTMLImageElement} inputPic
 */
export function plotFaces(canvas, faces, inputPic) {
    const ctx = canvas.getContext('2d');
    const featureSize = 10;

    const finalWidth = canvas.width;
    const finalHeight = canvas.height;
    const finalSquareLength = Math.max(finalWidth, finalHeight);
    const xOffset = -(finalSquareLength - finalWidth) / 2;
    const yOffset = -(finalSquareLength - finalHeight) / 2;

    if (inputPic) {
        ctx.drawImage(inputPic, 0, 0, canvas.width, canvas.height);
    } else {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    faces.forEach(f => {
        ctx.strokeStyle = 'red';
        ctx.lineWidth = featureSize / 2;
        ctx.strokeRect(
            finalSquareLength * f.xMin + xOffset,
            finalSquareLength * f.yMin + yOffset,
            finalSquareLength * (f.xMax - f.xMin),
            finalSquareLength * (f.yMax - f.yMin)
        );
        ctx.fillStyle = 'blue';
        f.features.forEach((feature, i) => {
            ctx.beginPath();
            ctx.arc(
                finalSquareLength * feature.x + xOffset - featureSize / 2,
                finalSquareLength * feature.y + yOffset - featureSize / 2,
                featureSize,
                0,
                2 * Math.PI
            );
            ctx.fillStyle = 'blue';
            ctx.fill();
            ctx.fillStyle = 'white';
            ctx.font = `${2 * featureSize}px serif`;
            ctx.fillText(i + 'n', finalSquareLength * feature.x + xOffset - featureSize / 2,
                finalSquareLength * feature.y + yOffset - featureSize / 2);
        });
        if (f.name) {
            ctx.fillStyle = 'white';
            ctx.strokeStyle = 'black';
            ctx.font = `${2 * featureSize}px serif`;
            ctx.strokeText(f.name, finalSquareLength * f.xMin + xOffset, finalSquareLength * f.yMin + yOffset - featureSize / 2);
            ctx.fillText(f.name, finalSquareLength * f.xMin + xOffset, finalSquareLength * f.yMin + yOffset - featureSize / 2);
        }
    });
}

function estimateFaceAngle(face) {
    const leftEye = face.features[BLAZEFACE_FEATURES.LEFT_EYE];
    const rightEye = face.features[BLAZEFACE_FEATURES.RIGHT_EYE];
    const mouth = face.features[BLAZEFACE_FEATURES.MOUTH];
    const leftEar = face.features[BLAZEFACE_FEATURES.LEFT_EAR];
    const rightEar = face.features[BLAZEFACE_FEATURES.RIGHT_EAR];

    const mouthToEyesCenter = {
        x: (leftEye.x + rightEye.x) / 2 - mouth.x,
        y: (leftEye.y + rightEye.y) / 2 - mouth.y
    };
    const rightToLeftEarsRotated = {
        x: leftEar.y - rightEar.y,
        y: rightEar.x - leftEar.x
    };

    const estimatedAngle = (
        (Math.atan2(mouthToEyesCenter.y, mouthToEyesCenter.x) +
    Math.atan2(rightToLeftEarsRotated.y, rightToLeftEarsRotated.x)) / 2
    );

    return (estimatedAngle + Math.PI / 2);
}

/**
 * @param {DetectionBox[]} faces
 * @param {HTMLImageElement || HTMLVideoElement || HTMLCanvasElement} inputPic
 */
export function extractFaces(faces, inputPic) {
    const finalWidth = inputPic.naturalWidth || inputPic.videoWidth || inputPic.width;
    const finalHeight = inputPic.naturalHeight || inputPic.videoHeight || inputPic.height;
    const finalSquareLength = Math.max(finalWidth, finalHeight);
    const xOffset = -(finalSquareLength - finalWidth) / 2;
    const yOffset = -(finalSquareLength - finalHeight) / 2;

    return faces.map(face => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const estimatedAngle = estimateFaceAngle(face);
        const addedBoxLengthPercentage = (Math.floor(100 * Math.abs(Math.sin(2 * estimatedAngle))) / 100) * 0.45;

        const originalBoxWidth = finalSquareLength * (face.xMax - face.xMin);
        const addedBoxWidth = originalBoxWidth * addedBoxLengthPercentage;
        const boxWidth = originalBoxWidth + addedBoxWidth;
        const originalBoxHeight = finalSquareLength * (face.yMax - face.yMin);
        const addedBoxHeight = originalBoxHeight * addedBoxLengthPercentage;
        const boxHeight = originalBoxHeight + addedBoxHeight;

        canvas.width = boxWidth;
        canvas.height = boxHeight;

        ctx.translate(canvas.width / 2, canvas.height / 2);
        ctx.rotate(-estimatedAngle);
        ctx.translate(-canvas.width / 2, -canvas.height / 2);

        ctx.drawImage(
            inputPic,

            finalSquareLength * face.xMin + xOffset - (addedBoxWidth / 2),
            finalSquareLength * face.yMin + yOffset - (addedBoxHeight / 2),
            canvas.width,
            canvas.height,

            0,
            0,
            canvas.width,
            canvas.height
        );

        return canvas;
    });
}
