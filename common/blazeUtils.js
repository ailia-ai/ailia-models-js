export function sigmoid(x) {
    return 1.0 / (1.0 + Math.exp(-x));
}

// A box is described by 4 numbers (centre x, centre y, width, height); the
// keypoints follow them in the same tensor.
const BOX_PARAMETER_COUNT = 4;

export function generateBoxDecoder(inputScale, tensorCount, tensorSize, featureCount, anchors) {
    /**
     * @param {Float32Array} predictions the raw regressor output blob
     * @returns {DetectionBox[]}
     */
    return (predictions) => {
        const boxes = new Array(tensorCount);

        const xScale = inputScale;
        const yScale = inputScale;
        const wScale = inputScale;
        const hScale = inputScale;

        function getFloat(tI, cI) {
            return predictions[tI * tensorSize + cI];
        }

        for (let tI = 0; tI < tensorCount; tI++) {
            const xCenter = getFloat(tI, 0) / xScale * anchors[tI][2] + anchors[tI][0];
            const yCenter = getFloat(tI, 1) / yScale * anchors[tI][3] + anchors[tI][1];
            const width = getFloat(tI, 2) / wScale * anchors[tI][2];
            const height = getFloat(tI, 3) / hScale * anchors[tI][3];

            const box = {
                xMin: xCenter - width / 2,
                yMin: yCenter - height / 2,
                xMax: xCenter + width / 2,
                yMax: yCenter + height / 2,
                area: width * height,
                features: []
            };

            for (let featureIndex = 0; featureIndex < featureCount; featureIndex++) {
                const featureOffset = BOX_PARAMETER_COUNT + featureIndex * 2;

                box.features.push({
                    x: getFloat(tI, featureOffset) / xScale * anchors[tI][2] + anchors[tI][0],
                    y: getFloat(tI, featureOffset + 1) / yScale * anchors[tI][3] + anchors[tI][1]
                });
            }

            boxes[tI] = box;
        }

        return boxes;
    };
}

/**
 * @param {DetectionBox[]} boxes
 */
function splitByOverlapSimilarity(boxes, minimumOverlapThreshold) {
    const referenceBox = boxes[0];
    const jaccardOverlaps = boxes.map(b => jaccardOverlap(referenceBox, b));
    const splitResult = jaccardOverlaps.reduce((acc, val, i) => {
        const box = boxes[i];
        if (val > minimumOverlapThreshold || i === 0) {
            acc.overlappingBoxes.push(box);

            acc.mergedBox.score += box.score;

            acc.mergedBox.xMin += box.score * box.xMin;
            acc.mergedBox.xMax += box.score * box.xMax;
            acc.mergedBox.yMin += box.score * box.yMin;
            acc.mergedBox.yMax += box.score * box.yMax;

            for (let i = 0; i < acc.mergedBox.features.length; i++) {
                acc.mergedBox.features[i].x += box.score * box.features[i].x;
                acc.mergedBox.features[i].y += box.score * box.features[i].y;
            }
        } else {
            acc.remainingBoxes.push(box);
        }

        return acc;
    }, {
        mergedBox: {
            xMin: 0,
            yMin: 0,
            xMax: 0,
            yMax: 0,
            score: 0,
            features: new Array(boxes[0].features.length).fill(0).map(_ => ({ x: 0, y: 0 }))
        },
        remainingBoxes: [],
        overlappingBoxes: []
    });

    splitResult.mergedBox.xMin /= splitResult.mergedBox.score;
    splitResult.mergedBox.xMax /= splitResult.mergedBox.score;
    splitResult.mergedBox.yMin /= splitResult.mergedBox.score;
    splitResult.mergedBox.yMax /= splitResult.mergedBox.score;

    for (let i = 0; i < splitResult.mergedBox.features.length; i++) {
        splitResult.mergedBox.features[i].x /= splitResult.mergedBox.score;
        splitResult.mergedBox.features[i].y /= splitResult.mergedBox.score;
    }

    splitResult.mergedBox.score /= splitResult.overlappingBoxes.length;

    return splitResult;
}

/**
 * @param {DetectionBox} box
 * @param {DetectionBox} otherBox
 */
function jaccardOverlap(box, otherBox) {
    const inter = intersection(box, otherBox);
    const union = box.area + otherBox.area - inter;

    return inter / union;
}

/**
 * @param {DetectionBox} box
 * @param {DetectionBox} otherBox
 */
function intersection(box, otherBox) {
    const widthOverlap = Math.max(0, Math.min(box.xMax, otherBox.xMax) - Math.max(box.xMin, otherBox.xMin));
    const heightOverlap = Math.max(0, Math.min(box.yMax, otherBox.yMax) - Math.max(box.yMin, otherBox.yMin));

    return widthOverlap * heightOverlap;
}

/**
 * @param {DetectionBox[]} boxes
 */
export function weightedNonMaxSuppression(boxes, minimumOverlapThreshold) {
    // The input array is duplicated because it will get its elements removed from it when merging them
    let remainingBoxes = [...boxes].sort((a, b) => b.score - a.score);
    const detectedBoxes = [];

    while (remainingBoxes.length > 0) {
        const result = splitByOverlapSimilarity(remainingBoxes, minimumOverlapThreshold);

        detectedBoxes.push(result.mergedBox);
        remainingBoxes = result.remainingBoxes;
    }

    return detectedBoxes;
}
