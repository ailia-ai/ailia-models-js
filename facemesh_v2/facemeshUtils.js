import { generateBoxDecoder, weightedNonMaxSuppression, sigmoid } from '../common/blazeUtils.js';
import { writeInputImage, Normalization, normalizeRadians } from '../common/demoUtils.js';
import anchors from '../common/blazefaceAnchors.js';
import { FACE_CONTOURS } from './facemeshContours.js';

// The face detector shares BlazeFace's SSD anchor layout: 128x128 input,
// strides 8/16/16/16, so 16x16x2 + 8x8x6 = 896 anchors with w = h = 1
const DETECTOR_INPUT_SIZE = 128;
const DETECTOR_TENSOR_COUNT = 896;
const DETECTOR_TENSOR_SIZE = 16;
const DETECTOR_KEYPOINT_COUNT = 6;
const DETECTOR_SCORE_CLAMP = 100.0;
const DETECTOR_SCORE_THRESHOLD = 0.5;
const DETECTOR_OVERLAP_THRESHOLD = 0.5;

const LANDMARK_INPUT_SIZE = 256;
const LANDMARK_COUNT = 478;
const LANDMARK_DIMENSIONS = 3;

// MediaPipe grows the detected box before cropping, so the mesh gets some margin
const REGION_OF_INTEREST_SCALE = 1.5;

// The first two detector keypoints are the eyes, which give the head roll
const RIGHT_EYE_KEYPOINT = 0;
const LEFT_EYE_KEYPOINT = 1;

const CONTOUR_COLOURS = {
    face_oval: '#f5f5f5',
    left_eyebrow: '#7ee787',
    right_eyebrow: '#7ee787',
    left_eye: '#58a6ff',
    right_eye: '#58a6ff',
    left_iris: '#ffa657',
    right_iris: '#ffa657',
    nose: '#d2a8ff',
    lips: '#ff7b72'
};

const decodeBoxes = generateBoxDecoder(
    DETECTOR_INPUT_SIZE, DETECTOR_TENSOR_COUNT, DETECTOR_TENSOR_SIZE, DETECTOR_KEYPOINT_COUNT, anchors
);

// Run on the whole input unless a caller narrows it down
export const WHOLE_IMAGE = { x: 0, y: 0, width: 1, height: 1 };

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
 * Runs the short range face detector. Boxes and keypoints come back normalized
 * against the input image, not against the square the model was fed.
 *
 * @param {*} ailia an Ailia instance holding face_detector.onnx
 * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} inputPic
 */
export function detectFaces(ailia, inputPic) {
    const { width, height } = imageSize(inputPic);

    // The detector sees the image letterboxed into a square, so its coordinates
    // are relative to that square and the padding has to be undone
    const squareLength = Math.max(width, height);
    const toImageX = x => (squareLength * x - (squareLength - width) / 2) / width;
    const toImageY = y => (squareLength * y - (squareLength - height) / 2) / height;

    return ailia.run(
        inputBuffers => writeInputImage(
            inputBuffers.byIndex[0].buffer.Float32,
            inputPic,
            DETECTOR_INPUT_SIZE,
            DETECTOR_INPUT_SIZE,
            { normalize: Normalization.UnitCentered, channelsLast: true }
        ),
        outputBuffers => {
            const regressors = outputBuffers.byIndex[0].buffer.Float32;
            const classificators = outputBuffers.byIndex[1].buffer.Float32;

            let boxes = decodeBoxes(regressors);

            for (let i = 0; i < DETECTOR_TENSOR_COUNT; i++) {
                boxes[i].score = sigmoid(
                    Math.max(Math.min(classificators[i], DETECTOR_SCORE_CLAMP), -DETECTOR_SCORE_CLAMP)
                );
            }

            boxes = boxes.filter(box => box.score >= DETECTOR_SCORE_THRESHOLD);

            return weightedNonMaxSuppression(boxes, DETECTOR_OVERLAP_THRESHOLD).map(box => ({
                xMin: toImageX(box.xMin),
                xMax: toImageX(box.xMax),
                yMin: toImageY(box.yMin),
                yMax: toImageY(box.yMax),
                score: box.score,
                features: box.features.map(feature => ({ x: toImageX(feature.x), y: toImageY(feature.y) }))
            }));
        }
    );
}

/**
 * Turns a detection into the rotated, enlarged square the mesh model expects.
 * All values stay normalized against the input image.
 *
 * @param {*} detection one entry returned by detectFaces
 */
export function detectionToRegionOfInterest(detection, imageWidth, imageHeight) {
    const rightEye = detection.features[RIGHT_EYE_KEYPOINT];
    const leftEye = detection.features[LEFT_EYE_KEYPOINT];

    // Measured in pixels, so a non-square image does not skew the angle
    const rotation = normalizeRadians(-Math.atan2(
        -(leftEye.y - rightEye.y) * imageHeight,
        (leftEye.x - rightEye.x) * imageWidth
    ));

    return {
        xCenter: (detection.xMin + detection.xMax) / 2,
        yCenter: (detection.yMin + detection.yMax) / 2,
        width: (detection.xMax - detection.xMin) * REGION_OF_INTEREST_SCALE,
        height: (detection.yMax - detection.yMin) * REGION_OF_INTEREST_SCALE,
        rotation
    };
}

/**
 * Draws the rotated region of interest into a square canvas of the mesh model's
 * input size. The transform is the inverse of the projection that
 * projectLandmarks applies, so the two stay in step.
 */
function cropRegionOfInterest(inputPic, regionOfInterest, imageWidth, imageHeight) {
    const canvas = document.createElement('canvas');
    canvas.width = LANDMARK_INPUT_SIZE;
    canvas.height = LANDMARK_INPUT_SIZE;

    const ctx = canvas.getContext('2d');
    const regionWidth = regionOfInterest.width * imageWidth;
    const regionHeight = regionOfInterest.height * imageHeight;

    // Read bottom up: shift the region's centre to the origin, undo its size and
    // rotation, then scale to the input square and move its centre into place
    ctx.translate(LANDMARK_INPUT_SIZE / 2, LANDMARK_INPUT_SIZE / 2);
    ctx.scale(LANDMARK_INPUT_SIZE, LANDMARK_INPUT_SIZE);
    ctx.rotate(-regionOfInterest.rotation);
    ctx.scale(1 / regionWidth, 1 / regionHeight);
    ctx.translate(-regionOfInterest.xCenter * imageWidth, -regionOfInterest.yCenter * imageHeight);
    ctx.drawImage(inputPic, 0, 0);

    return canvas;
}

/**
 * Maps the mesh output back onto the input image. There is no letterbox to undo,
 * because the crop stretches the region of interest to fill the input square.
 *
 * @param {Float32Array} tensor the 478 x 3 landmark output blob
 */
function projectLandmarks(tensor, regionOfInterest) {
    const cos = Math.cos(regionOfInterest.rotation);
    const sin = Math.sin(regionOfInterest.rotation);
    const landmarks = new Array(LANDMARK_COUNT);

    for (let i = 0; i < LANDMARK_COUNT; i++) {
        const offset = i * LANDMARK_DIMENSIONS;
        // The model reports coordinates in its own input pixels
        const x = tensor[offset] / LANDMARK_INPUT_SIZE - 0.5;
        const y = tensor[offset + 1] / LANDMARK_INPUT_SIZE - 0.5;
        const z = tensor[offset + 2] / LANDMARK_INPUT_SIZE;

        landmarks[i] = {
            x: (cos * x - sin * y) * regionOfInterest.width + regionOfInterest.xCenter,
            y: (sin * x + cos * y) * regionOfInterest.height + regionOfInterest.yCenter,
            // Depth carries the same scale as x, in region of interest units
            z: z * regionOfInterest.width
        };
    }

    return landmarks;
}

/**
 * Copies a normalized sub-region of the input into a canvas of that region's
 * pixel size, so the pipeline can be pointed at part of a larger picture.
 */
function cropRegion(inputPic, region, imageWidth, imageHeight) {
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(region.width * imageWidth);
    canvas.height = Math.round(region.height * imageHeight);

    canvas.getContext('2d').drawImage(
        inputPic,
        region.x * imageWidth, region.y * imageHeight, canvas.width, canvas.height,
        0, 0, canvas.width, canvas.height
    );

    return canvas;
}

/**
 * @param {*} ailia an Ailia instance holding face_landmarks_detector.onnx
 */
function computeFaceLandmarks(ailia, cropCanvas, regionOfInterest) {
    return ailia.run(
        inputBuffers => writeInputImage(
            inputBuffers.byIndex[0].buffer.Float32,
            cropCanvas,
            LANDMARK_INPUT_SIZE,
            LANDMARK_INPUT_SIZE,
            { normalize: Normalization.Unit, channelsLast: true }
        ),
        // The model has two further outputs which the reference implementation in
        // ailia-models reads but never uses, so they are ignored here too
        outputBuffers => projectLandmarks(outputBuffers.byIndex[0].buffer.Float32, regionOfInterest),
        // This model reports no default input shape, so it has to be stated
        [[1, LANDMARK_INPUT_SIZE, LANDMARK_INPUT_SIZE, 3]]
    );
}

/**
 * Detects faces, then runs the mesh on each one.
 *
 * Landmarks are always returned normalized against `inputPic`, whatever region
 * was searched, so they can be drawn straight onto the full picture.
 *
 * @param {*} detectorAilia an Ailia instance holding face_detector.onnx
 * @param {*} landmarkAilia an Ailia instance holding face_landmarks_detector.onnx
 * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} inputPic
 * @param {{x: number, y: number, width: number, height: number}} [region]
 *        normalized part of the input to search, to zoom in on a small face
 */
export function computeFacemesh(detectorAilia, landmarkAilia, inputPic, region = WHOLE_IMAGE) {
    const { width, height } = imageSize(inputPic);
    const searched = region === WHOLE_IMAGE ? inputPic : cropRegion(inputPic, region, width, height);
    const searchedSize = imageSize(searched);

    return detectFaces(detectorAilia, searched).map(detection => {
        const regionOfInterest = detectionToRegionOfInterest(detection, searchedSize.width, searchedSize.height);
        const cropCanvas = cropRegionOfInterest(searched, regionOfInterest, searchedSize.width, searchedSize.height);
        const landmarks = computeFaceLandmarks(landmarkAilia, cropCanvas, regionOfInterest);

        return {
            score: detection.score,
            // Lift the landmarks out of the searched region and back onto the input
            landmarks: landmarks.map(landmark => ({
                x: region.x + landmark.x * region.width,
                y: region.y + landmark.y * region.height,
                z: landmark.z * region.width
            }))
        };
    });
}

/**
 * @param {HTMLCanvasElement} canvas
 * @param {*[]} faces the result of computeFacemesh
 */
export function plotFacemesh(canvas, faces) {
    const ctx = canvas.getContext('2d');
    const featureSize = Math.max(1, Math.round(Math.max(canvas.width, canvas.height) / 512));

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    faces.forEach(({ landmarks }) => {
        const toX = landmark => landmark.x * canvas.width;
        const toY = landmark => landmark.y * canvas.height;

        ctx.lineWidth = featureSize;

        FACE_CONTOURS.forEach(({ name, connections }) => {
            ctx.strokeStyle = CONTOUR_COLOURS[name];
            ctx.beginPath();
            connections.forEach(([from, to]) => {
                ctx.moveTo(toX(landmarks[from]), toY(landmarks[from]));
                ctx.lineTo(toX(landmarks[to]), toY(landmarks[to]));
            });
            ctx.stroke();
        });

        ctx.fillStyle = 'rgba(88, 166, 255, 0.85)';
        landmarks.forEach(landmark => {
            ctx.beginPath();
            ctx.arc(toX(landmark), toY(landmark), featureSize, 0, 2 * Math.PI);
            ctx.fill();
        });
    });
}
