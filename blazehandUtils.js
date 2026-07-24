import { weightedNonMaxSuppression, generateBoxDecoder, sigmoid } from './blazeUtils.js';
import { CONSTANTS } from './constants.js';
import { loadInputImage, Normalization, normalizeRadians } from './demoUtils.js';
import anchors from './blazehandAnchors.js';

const BLAZEPALM_REQUIRED_INPUT_WIDTH = 256;
const BLAZEPALM_REQUIRED_INPUT_HEIGHT = 256;
const BLAZEPALM_TENSOR_COUNT = 2944;
const BLAZEPALM_TENSOR_SIZE = 18;
const BLAZEPALM_FEATURE_COUNT = 7;
const SCORE_THRESHOLD = 100.0;
const MINIMUM_SCORE_THRESHOLD = 0.75;
const MINIMUM_OVERLAP_THRESHOLD = 0.3;

const BLAZEHAND_PART_ID = {
    WRIST: 0,
    THUMB_CMC: 1,
    THUMB_MCP: 2,
    THUMB_IP: 3,
    THUMB_TIP: 4,
    INDEX_FINGER_MCP: 5,
    INDEX_FINGER_PIP: 6,
    INDEX_FINGER_DIP: 7,
    INDEX_FINGER_TIP: 8,
    MIDDLE_FINGER_MCP: 9,
    MIDDLE_FINGER_PIP: 10,
    MIDDLE_FINGER_DIP: 11,
    MIDDLE_FINGER_TIP: 12,
    RING_FINGER_MCP: 13,
    RING_FINGER_PIP: 14,
    RING_FINGER_DIP: 15,
    RING_FINGER_TIP: 16,
    PINKY_FINGER_MCP: 17,
    PINKY_FINGER_PIP: 18,
    PINKY_FINGER_DIP: 19,
    PINKY_FINGER_TIP: 20
};
const BLAZEHAND_LANDMARKS_CONNECTIONS = [[0, 1], [1, 2], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [9, 10], [10, 11], [11, 12], [13, 14], [14, 15], [15, 16], [17, 18], [18, 19], [19, 20], [0, 5], [5, 9], [9, 13], [13, 17], [0, 17]];

const BLAZEHAND_KEYPOINT_1_INDEX = 0;
const BLAZEHAND_KEYPOINT_2_INDEX = 2;
const BLAZEHAND_ANGLE0 = Math.PI / 2;
const BLAZEHAND_DSCALE = 2.6;
const BLAZEHAND_DY = -0.5;
const BLAZEHAND_INPUT_RESOLUTION = 256;
const BLAZEHAND_LANDMARK_COUNT = Object.keys(BLAZEHAND_PART_ID).length;
const BLAZEHAND_DETECTION_THRESHOLD = 0.75;
const BLAZEHAND_RIGHT_HANDEDNESS_THRESHOLD = 0.5;
const BLAZEHAND_LANDMARK_ROI_SCALE = 1.6;
const BLAZEHAND_INPUT_CHANNEL_COUNT = 3;
const BLAZEHAND_FEATURE_COORDINATE_COUNT = 3;

/**
 * @typedef {Object} DetectionBox
 * @property {number} xMin
 * @property {number} xMax
 * @property {number} yMin
 * @property {number} yMax
 * @property {number} score
 * @property {number} area
 * @property {{x: number, y: number}[]} features
 */

/**
 * @typedef {Object} C
 * @property {number} presence
 * @property {number} handedness
 * @property {boolean} isRightHand
 * @property {{x: number, y: number}[]} landmarks
 */

/**
 * @typedef {Object} Transform
 * @property {number} scaleX
 * @property {number} scaleY
 * @property {number} angle
 * @property {number} tX translation along the X axis
 * @property {number} tY translation along the Y axis
 */

/**
 * @param {Float32Array} heap
 * @param {number} outputBoxesRef
 * @param {number} outputScoresRef
 */
export function blazepalmPostprocess(heap, outputBoxesRef, outputScoresRef) {
    let boxes = decodeBoxes(heap, outputBoxesRef);
    const rawScores = [];

    for (let i = 0; i < BLAZEPALM_TENSOR_COUNT; i++) {
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

const decodeBoxes = generateBoxDecoder(BLAZEHAND_INPUT_RESOLUTION, BLAZEPALM_TENSOR_COUNT, BLAZEPALM_TENSOR_SIZE, BLAZEPALM_FEATURE_COUNT, anchors);

export function computeBlazepalm(ailia, inputPic) {
    const { inputImageBufferRef, totalInputSize } = loadInputImage(
        ailia,
        inputPic,
        BLAZEPALM_REQUIRED_INPUT_WIDTH,
        BLAZEPALM_REQUIRED_INPUT_HEIGHT,
        {
            normalize: Normalization.Unit
        }
    );

    const outputShape = ailia.getOutputShape();

    const totalBoxesOutputSize = outputShape.x * outputShape.y * outputShape.z * outputShape.w * CONSTANTS.SIZE_OF_FLOAT;

    const outputBoxesRef = ailia.Module._malloc(totalBoxesOutputSize * CONSTANTS.SIZE_OF_FLOAT);
    const outputScoresRef = ailia.Module._malloc(2 * CONSTANTS.SIZE_OF_FLOAT);
    const outputSizeRef = ailia.Module._malloc(CONSTANTS.SIZE_OF_REF);

    // struct AILIANetwork* net, void* dest, unsigned int dest_size, const void* src, unsigned int src_size
    ailia.call('ailiaPredict', 'number', [ailia.ailiaDoublePointer.getPointer(), outputBoxesRef, totalBoxesOutputSize, inputImageBufferRef, totalInputSize]);

    // ailiaRefOutputBlobBuffer(struct AILIANetwork* net, const void** ptr, unsigned int* size, unsigned int index);
    ailia.call('ailiaRefOutputBlobBuffer', 'number', [ailia.ailiaDoublePointer.getPointer(), outputScoresRef, outputSizeRef, 1]);

    const palms = blazepalmPostprocess(ailia.Module.HEAPF32, outputBoxesRef / CONSTANTS.SIZE_OF_FLOAT, ailia.Module.HEAPU32[outputScoresRef / CONSTANTS.SIZE_OF_U32] / CONSTANTS.SIZE_OF_U32);

    ailia.Module._free(outputBoxesRef);
    ailia.Module._free(outputScoresRef);
    ailia.Module._free(outputSizeRef);
    ailia.Module._free(inputImageBufferRef);

    return palms;
}

/**
 * @param {DetectionBox[]} boxes
 * @returns {DetectionBox[]}
 */
function denormalizeDetectionBoxes(boxes, imageWidth, imageHeight) {
    const finalSquareLength = Math.max(imageWidth, imageHeight);
    const xOffset = (finalSquareLength - imageWidth) / 2;
    const yOffset = -(finalSquareLength - imageHeight) / 2;

    return boxes.map(box => ({
        xMin: finalSquareLength * box.xMin + xOffset,
        xMax: finalSquareLength * box.xMax + xOffset,
        yMin: finalSquareLength * box.yMin + yOffset,
        yMax: finalSquareLength * box.yMax + yOffset,
        features: box.features.map(feature => ({
            x: finalSquareLength * feature.x + xOffset,
            y: finalSquareLength * feature.y + yOffset
        })),
        score: box.score
    }));
}

/**
 * @param {DetectionBox[]} box
 */
function detectionBoxesToRegionsOfInterest(boxes) {
    return boxes.map(box => {
        let xCenter = (box.xMin + box.xMax) / 2;
        let yCenter = (box.yMin + box.yMax) / 2;
        // This assumes that the boxes are squares
        let scale = (box.xMax - box.xMin);

        // Compute the box's rotation
        const x0 = box.features[BLAZEHAND_KEYPOINT_1_INDEX].x;
        const y0 = box.features[BLAZEHAND_KEYPOINT_1_INDEX].y;
        const x1 = box.features[BLAZEHAND_KEYPOINT_2_INDEX].x;
        const y1 = box.features[BLAZEHAND_KEYPOINT_2_INDEX].y;
        const angle = Math.atan2(y0 - y1, x0 - x1) - BLAZEHAND_ANGLE0;

        xCenter += BLAZEHAND_DY * scale * -Math.sin(angle);
        yCenter += BLAZEHAND_DY * scale * Math.cos(angle);
        scale *= BLAZEHAND_DSCALE;

        return {
            xCenter,
            yCenter,
            scale,
            angle
        };
    });
}

/**
 * @param {{x: number, y: number}[]} sp
 * @param {{x: number, y: number}[]} tp
 * @returns {Transform}
 */
function getTransformation(sp, tp) {
    function distance(pt1, pt2) {
        return Math.sqrt((pt2.x - pt1.x) ** 2 + (pt2.y - pt1.y) ** 2);
    }

    function getScales(points) {
        const [p1, p2, p3] = points;
        return [distance(p1, p2), distance(p1, p3)];
    }

    function getAngle(points) {
        const [p0, p1] = points;
        const v0 = [p1.x - p0.x, p1.y - p0.y];

        return Math.atan2(v0[1], v0[0]);
    }

    const scalesSP = getScales(sp);
    const scalesTP = getScales(tp);

    const scaleX = scalesTP[0] / scalesSP[0];
    const scaleY = scalesTP[1] / scalesSP[1];

    const angle = getAngle(tp) - getAngle(sp);

    const tX = -(sp[0].x * scaleX * Math.cos(angle) - sp[0].y * scaleY * Math.sin(angle) - tp[0].x);
    const tY = -(sp[0].x * scaleX * Math.sin(angle) + sp[0].y * scaleY * Math.cos(angle) - tp[0].y);

    return {
        scaleX,
        scaleY,
        angle,
        tX,
        tY
    };
};

/**
 *
 * @param {HTMLImageElement | HTMLVideoElement} inputPic
 * @param {*} regionsOfInterest
 * @returns {{ canvases: HTMLCanvasElement[], transformsToRegionOfInterest: Transform[], transformsFromRegionOfInterest: Transform[] }}
 */
function extractRegionsOfInterest(inputPic, regionsOfInterest) {
    const extractedRegionsOfInterest = {
        canvases: [],
        transformsToRegionOfInterest: [],
        transformsFromRegionOfInterest: []
    };

    for (const regionOfInterest of regionsOfInterest) {
        // Start from the unit square points and transform them according to the region of interest
        // (we can omit the last point because we only need 3 of them)
        const points = [{ x: -1, y: -1 }, { x: -1, y: 1 }, { x: 1, y: -1 }];
        for (const p of points) {
            p.x *= regionOfInterest.scale / 2;
            p.y *= regionOfInterest.scale / 2;
        }
        const cosAngle = Math.cos(regionOfInterest.angle);
        const sinAngle = Math.sin(regionOfInterest.angle);

        for (const p of points) {
            [p.x, p.y] = [
                p.x * cosAngle - p.y * sinAngle + regionOfInterest.xCenter,
                p.x * sinAngle + p.y * cosAngle + regionOfInterest.yCenter
            ];
        }

        const resolution = BLAZEHAND_INPUT_RESOLUTION - 1;
        const imagePoints = [
            { x: 0, y: 0 },
            { x: 0, y: resolution },
            { x: resolution, y: 0 }
        ];
        const transformToRegionOfInterest = getTransformation(points, imagePoints);
        const transformFromRegionOfInterest = getTransformation(imagePoints, points);

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = BLAZEHAND_INPUT_RESOLUTION;
        canvas.height = BLAZEHAND_INPUT_RESOLUTION;

        ctx.translate(transformToRegionOfInterest.tX, transformToRegionOfInterest.tY);
        ctx.scale(transformToRegionOfInterest.scaleX, transformToRegionOfInterest.scaleY);
        ctx.rotate(transformToRegionOfInterest.angle);
        ctx.drawImage(inputPic, 0, 0);

        extractedRegionsOfInterest.canvases.push(canvas);
        extractedRegionsOfInterest.transformsToRegionOfInterest.push(transformToRegionOfInterest);
        extractedRegionsOfInterest.transformsFromRegionOfInterest.push(transformFromRegionOfInterest);
    }

    return extractedRegionsOfInterest;
}

/**
 *
 * @param {Float32Array} heap
 * @param {*} landmarksBufferRef
 * @param {*} transforms
 * @returns
 */
function denormalizeLandmarks(heap, landmarksBufferRef, transforms) {
    const handCount = transforms.length;
    const hands = new Array(handCount);

    function getFloat(hI, lI, cI) {
        const i = (hI * BLAZEHAND_LANDMARK_COUNT + lI) * BLAZEHAND_FEATURE_COORDINATE_COUNT + cI;
        const f = heap[landmarksBufferRef + i];

        return f;
    }

    for (let handIndex = 0; handIndex < handCount; handIndex++) {
        const hand = {
            landmarks: new Array(BLAZEHAND_LANDMARK_COUNT)
        };

        const { scaleX, scaleY, angle, tX, tY } = transforms[handIndex];

        for (let landmarkIndex = 0; landmarkIndex < BLAZEHAND_LANDMARK_COUNT; landmarkIndex++) {
            const x = BLAZEHAND_INPUT_RESOLUTION * getFloat(handIndex, landmarkIndex, 0);
            const y = BLAZEHAND_INPUT_RESOLUTION * getFloat(handIndex, landmarkIndex, 1);

            hand.landmarks[landmarkIndex] = {
                x: (x * scaleX * Math.cos(angle) - y * scaleY * Math.sin(angle) + tX),
                y: (x * scaleX * Math.sin(angle) + y * scaleY * Math.cos(angle) + tY),
                z: getFloat(handIndex, landmarkIndex, 2)
            };
        }

        hands[handIndex] = hand;
    }

    return hands;
}

/**
 * @param {DetectedHand} hand
 */
function computeHandRotation(hand) {
    const { x: x0, y: y0 } = hand.landmarks[BLAZEHAND_PART_ID.WRIST];
    const { x: x1, y: y1 } = hand.landmarks[BLAZEHAND_PART_ID.INDEX_FINGER_PIP];
    const { x: x2, y: y2 } = hand.landmarks[BLAZEHAND_PART_ID.MIDDLE_FINGER_PIP];
    const { x: x3, y: y3 } = hand.landmarks[BLAZEHAND_PART_ID.RING_FINGER_PIP];

    const xC = (x1 + 2 * x2 + x3) / 4;
    const yC = (y1 + 2 * y2 + y3) / 4;

    const angle = normalizeRadians(BLAZEHAND_ANGLE0 - Math.atan2(-(yC - y0), xC - x0));

    return angle;
}

/**
 * @param {{ x: number, y: number }[]} points
 * @returns {{ xMin: number, xMax: number, yMin: number, yMax: number }}
 */
function getBoundingBox(points) {
    return points.reduce((acc, point) => {
        acc.xMin = Math.min(acc.xMin, point.x);
        acc.yMin = Math.min(acc.yMin, point.y);
        acc.xMax = Math.max(acc.xMax, point.x);
        acc.yMax = Math.max(acc.yMax, point.y);

        return acc;
    }, {
        xMin: Number.MAX_SAFE_INTEGER,
        yMin: Number.MAX_SAFE_INTEGER,
        xMax: -Number.MAX_SAFE_INTEGER,
        yMax: -Number.MAX_SAFE_INTEGER
    });
}

// Adpated from the original code in the MediaPipe repository
// https://github.com/google/mediapipe/blob/master/mediapipe/modules/hand_landmark/calculators/hand_landmarks_to_rect_calculator.cc
/**
 * @param {DetectedHand[]} hands
 */
function extractRegionsOfInterestFromLandmarks(hands) {
    return hands.map(hand => {
        const roi = {
            angle: computeHandRotation(hand)
        };
        const angle = normalizeRadians(roi.angle);
        const reverseAngle = normalizeRadians(-angle);
        const cosAngle = Math.cos(angle);
        const sinAngle = Math.sin(angle);
        const cosReverseAngle = Math.cos(reverseAngle);
        const sinReverseAngle = Math.sin(reverseAngle);

        const points = [
            BLAZEHAND_PART_ID.WRIST,
            BLAZEHAND_PART_ID.THUMB_CMC,
            BLAZEHAND_PART_ID.THUMB_MCP,
            BLAZEHAND_PART_ID.THUMB_TIP,
            BLAZEHAND_PART_ID.INDEX_FINGER_MCP,
            BLAZEHAND_PART_ID.INDEX_FINGER_PIP,
            BLAZEHAND_PART_ID.INDEX_FINGER_TIP,
            BLAZEHAND_PART_ID.MIDDLE_FINGER_MCP,
            BLAZEHAND_PART_ID.MIDDLE_FINGER_PIP,
            BLAZEHAND_PART_ID.RING_FINGER_MCP,
            BLAZEHAND_PART_ID.RING_FINGER_PIP,
            BLAZEHAND_PART_ID.PINKY_FINGER_MCP,
            BLAZEHAND_PART_ID.PINKY_FINGER_TIP
        ].map(i => hand.landmarks[i]);

        const axisAlignedBB = getBoundingBox(points);
        const axisAlignedXCenter = (axisAlignedBB.xMin + axisAlignedBB.xMax) / 2;
        const axisAlignedYCenter = (axisAlignedBB.yMin + axisAlignedBB.yMax) / 2;

        const rotatedPoints = points.map(p => ({
            x: (p.x - axisAlignedXCenter) * cosReverseAngle - (p.y - axisAlignedYCenter) * sinReverseAngle,
            y: (p.x - axisAlignedXCenter) * sinReverseAngle + (p.y - axisAlignedYCenter) * cosReverseAngle
        }));
        const rotatedBB = getBoundingBox(rotatedPoints);
        const rotatedXCenter = (rotatedBB.xMin + rotatedBB.xMax) / 2;
        const rotatedYCenter = (rotatedBB.yMin + rotatedBB.yMax) / 2;

        const xCenter = rotatedXCenter * cosAngle - rotatedYCenter * sinAngle + axisAlignedXCenter;
        const yCenter = rotatedXCenter * sinAngle + rotatedYCenter * cosAngle + axisAlignedYCenter;

        const width = rotatedBB.xMax - rotatedBB.xMin;
        const height = rotatedBB.yMax - rotatedBB.yMin;

        return {
            xCenter,
            yCenter,
            scale: BLAZEHAND_LANDMARK_ROI_SCALE * Math.max(width, height),
            angle
        };
    });
}

export function computeBlazehandFromPalms(ailia, inputPic, palms) {
    const denormalizedDetections = denormalizeDetectionBoxes(palms, inputPic.naturalWidth || inputPic.videoWidth, inputPic.naturalHeight || inputPic.videoHeight);
    const detectedRegionsOfInterest = detectionBoxesToRegionsOfInterest(denormalizedDetections);

    return computeBlazehandFromRegionsOfInterest(ailia, inputPic, detectedRegionsOfInterest);
}

export function computeBlazehandFromPreviousLandmarks(ailia, inputPic, hands) {
    const extractedRegionsOfInterest = extractRegionsOfInterestFromLandmarks(hands);

    return computeBlazehandFromRegionsOfInterest(ailia, inputPic, extractedRegionsOfInterest);
}

export function computeBlazehandFromRegionsOfInterest(ailia, inputPic, detectedRegionsOfInterest) {
    const extractedRegionsOfInterest = extractRegionsOfInterest(inputPic, detectedRegionsOfInterest);

    const { inputImageBufferRef, totalInputSize } = loadInputImage(
        ailia,
        extractedRegionsOfInterest.canvases,
        BLAZEHAND_INPUT_RESOLUTION,
        BLAZEHAND_INPUT_RESOLUTION,
        {
            normalize: Normalization.Unit
        }
    );

    const handCount = extractedRegionsOfInterest.canvases.length;

    const handFlagBufferRef = ailia.Module._malloc(handCount * CONSTANTS.SIZE_OF_FLOAT);
    const handFlagBufferSizeRef = ailia.Module._malloc(CONSTANTS.SIZE_OF_REF);

    const handednessBufferRef = ailia.Module._malloc(handCount * CONSTANTS.SIZE_OF_FLOAT);
    const handednessBufferSizeRef = ailia.Module._malloc(CONSTANTS.SIZE_OF_REF);

    const landmarksBufferRef = ailia.Module._malloc(handCount * BLAZEHAND_LANDMARK_COUNT * BLAZEHAND_FEATURE_COORDINATE_COUNT * CONSTANTS.SIZE_OF_FLOAT);
    const landmarksBufferSizeRef = ailia.Module._malloc(CONSTANTS.SIZE_OF_REF);

    ailia.setInputShape(BLAZEHAND_INPUT_RESOLUTION, BLAZEHAND_INPUT_RESOLUTION, BLAZEHAND_INPUT_CHANNEL_COUNT, handCount);

    // struct AILIANetwork* net, void* dest, unsigned int dest_size, const void* src, unsigned int src_size
    ailia.call('ailiaPredict', 'number', [ailia.ailiaDoublePointer.getPointer(), handFlagBufferRef, handFlagBufferSizeRef, inputImageBufferRef, totalInputSize]);

    console.log('Ran the blazehand network to completion');

    // ailiaRefOutputBlobBuffer(struct AILIANetwork* net, const void** ptr, unsigned int* size, unsigned int index);
    ailia.call('ailiaRefOutputBlobBuffer', 'number', [ailia.ailiaDoublePointer.getPointer(), handednessBufferRef, handednessBufferSizeRef, 1]);

    // ailiaRefOutputBlobBuffer(struct AILIANetwork* net, const void** ptr, unsigned int* size, unsigned int index);
    ailia.call('ailiaRefOutputBlobBuffer', 'number', [ailia.ailiaDoublePointer.getPointer(), landmarksBufferRef, landmarksBufferSizeRef, 2]);

    const hands = denormalizeLandmarks(
        ailia.Module.HEAPF32,
        ailia.Module.HEAPU32[landmarksBufferRef / CONSTANTS.SIZE_OF_U32] / CONSTANTS.SIZE_OF_U32,
        extractedRegionsOfInterest.transformsFromRegionOfInterest
    );

    function getHandFlagFloat(hI) {
        return ailia.Module.HEAPF32[handFlagBufferRef / CONSTANTS.SIZE_OF_FLOAT + hI];
    }

    function getHandednessFlag(hI) {
        return ailia.Module.HEAPF32[ailia.Module.HEAPU32[handednessBufferRef / CONSTANTS.SIZE_OF_U32] / CONSTANTS.SIZE_OF_FLOAT + hI];
    }

    hands.forEach((hand, i) => {
        hand.presence = getHandFlagFloat(i);
        hand.handedness = getHandednessFlag(i);
        hand.isRightHand = hand.handedness > BLAZEHAND_RIGHT_HANDEDNESS_THRESHOLD;
    });

    ailia.Module._free(inputImageBufferRef);
    ailia.Module._free(handFlagBufferRef);
    ailia.Module._free(handFlagBufferSizeRef);
    ailia.Module._free(handednessBufferRef);
    ailia.Module._free(handednessBufferSizeRef);
    ailia.Module._free(landmarksBufferRef);
    ailia.Module._free(landmarksBufferSizeRef);

    return hands.filter(hand => hand.presence > BLAZEHAND_DETECTION_THRESHOLD);
}

/**
 * @param {HTMLCanvasElement} canvas
 * @param {DetectionBox[]} palms
 * @param {HTMLImageElement} inputPic
 */
export function plotPalms(canvas, palms, inputPic) {
    const ctx = canvas.getContext('2d');
    const featureSize = 10;

    const finalWidth = canvas.width;
    const finalHeight = canvas.height;
    const finalSquareLength = Math.max(finalWidth, finalHeight);
    const xOffset = (finalSquareLength - finalWidth) / 2;
    const yOffset = -(finalSquareLength - finalHeight) / 2;

    if (inputPic) {
        ctx.drawImage(inputPic, 0, 0, canvas.width, canvas.height);
    } else {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    for (const palm of palms) {
        ctx.strokeStyle = 'red';
        ctx.lineWidth = featureSize / 2;
        ctx.strokeRect(
            finalSquareLength * palm.xMin + xOffset,
            finalSquareLength * palm.yMin + yOffset,
            finalSquareLength * (palm.xMax - palm.xMin),
            finalSquareLength * (palm.yMax - palm.yMin)
        );
        ctx.fillStyle = 'blue';

        for (const feature of palm.features) {
            ctx.beginPath();
            ctx.arc(
                finalSquareLength * feature.x + xOffset - featureSize / 2,
                finalSquareLength * feature.y + yOffset - featureSize / 2,
                featureSize,
                0,
                2 * Math.PI
            );
            ctx.fill();
        }
    }
}

/**
 * @param {HTMLCanvasElement} canvas
 * @param {DetectedHand[]} hands
 * @param {HTMLImageElement} inputPic
 */
export function plotHands(canvas, hands, inputPic) {
    const ctx = canvas.getContext('2d');
    const featureSize = 5;

    if (inputPic) {
        ctx.drawImage(inputPic, 0, 0, canvas.width, canvas.height);
    } else {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    for (const hand of hands) {
        ctx.strokeStyle = 'blue';
        ctx.lineWidth = featureSize / 2;

        for (const connection of BLAZEHAND_LANDMARKS_CONNECTIONS) {
            const [pointIndex1, pointIndex2] = connection;
            const p1 = hand.landmarks[pointIndex1];
            const p2 = hand.landmarks[pointIndex2];
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.closePath();
            ctx.stroke();
        }

        const [xT, yT] = [
            hand.landmarks[0],
            hand.landmarks[5],
            hand.landmarks[17]
        ].reduce((acc, val) => [acc[0] + val.x, acc[1] + val.y], [0, 0]).map(v => v / 3);

        const text = (hand.isRightHand ? 'Right' : 'Left');
        const fontSize = 10 * featureSize;
        ctx.fillStyle = 'black';
        ctx.font = `${fontSize}px serif`;
        const textSize = ctx.measureText(text);
        ctx.fillText(text, xT - textSize.width / 2, yT + fontSize / 2);

        for (const landmark of hand.landmarks) {
            ctx.fillStyle = 'red';
            ctx.beginPath();
            ctx.arc(
                landmark.x,
                landmark.y,
                featureSize,
                0,
                2 * Math.PI
            );
            ctx.fill();
        }
    }
}
