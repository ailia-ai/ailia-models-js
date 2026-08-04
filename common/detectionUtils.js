const HUE_COUNT = 20;

// 20 hues x 4 bands = 80 distinct styles, which covers COCO's 80 categories
// without repeating. Bands vary saturation and lightness so that two categories
// sharing a hue are still told apart.
const BANDS = [
    { saturation: 80, lightness: 42 },
    { saturation: 45, lightness: 34 },
    { saturation: 95, lightness: 27 },
    { saturation: 60, lightness: 47 }
];

/**
 * @param {number} hue
 * @param {number} saturation
 * @param {number} lightness
 * @returns {number[]} r, g, b in 0-255
 */
function hslToRgb(hue, saturation, lightness) {
    const s = saturation / 100;
    const l = lightness / 100;
    const chroma = (1 - Math.abs(2 * l - 1)) * s;
    const huePrime = hue / 60;
    const x = chroma * (1 - Math.abs((huePrime % 2) - 1));
    const rgb = [
        [chroma, x, 0], [x, chroma, 0], [0, chroma, x],
        [0, x, chroma], [x, 0, chroma], [chroma, 0, x]
    ][Math.floor(huePrime) % 6];
    const m = l - chroma / 2;

    return rgb.map(c => Math.round((c + m) * 255));
}

/**
 * WCAG relative luminance, used to decide whether a label reads better in black
 * or white. A fixed white would be unreadable on the yellow-green hues, which
 * are far brighter than blues at the same lightness.
 *
 * @param {number[]} rgb
 */
function relativeLuminance([r, g, b]) {
    const [rl, gl, bl] = [r, g, b].map(value => {
        const c = value / 255;

        return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
    });

    return 0.2126 * rl + 0.7152 * gl + 0.0722 * bl;
}

/**
 * Stable colour for a detection category, plus a readable label colour.
 *
 * The style is derived from the category id rather than a hash of the name:
 * hashing collides badly at 80 categories (`donut` and `broccoli` landed on the
 * same hue), while indexing keeps consecutive categories 18 degrees apart.
 *
 * @param {number} categoryId
 * @returns {{box: string, label: string}} CSS colours
 */
export function categoryStyle(categoryId) {
    const hue = (categoryId % HUE_COUNT) * (360 / HUE_COUNT);
    const { saturation, lightness } = BANDS[Math.floor(categoryId / HUE_COUNT) % BANDS.length];

    // Black overtakes white in contrast once the background luminance passes
    // sqrt(0.05 * 1.05) - 0.05, so use that crossover rather than a guessed one
    const useBlack = relativeLuminance(hslToRgb(hue, saturation, lightness)) > Math.sqrt(0.05 * 1.05) - 0.05;

    return {
        box: `hsl(${hue}, ${saturation}%, ${lightness}%)`,
        label: useBlack ? '#000000' : '#ffffff'
    };
}

/**
 * Draws detection boxes onto an overlay canvas, one colour per category.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {{categoryId: number, categoryName: string, prob: number, x: number, y: number, w: number, h: number}[]} detections
 * @param {HTMLImageElement} [inputPic] when given, the image is drawn underneath the boxes
 */
export function plotDetections(canvas, detections, inputPic) {
    const ctx = canvas.getContext('2d');
    const featureSize = 10 * (Math.max(1, Math.floor(Math.max(canvas.width, canvas.height) / 512)));

    const finalWidth = canvas.width;
    const finalHeight = canvas.height;
    const finalSquareLength = Math.max(finalWidth, finalHeight);
    // The model is fed the image letterboxed into a square, so the coordinates
    // it returns are relative to that square and the padding has to be undone
    const xOffset = -(finalSquareLength - finalWidth) / 2;
    const yOffset = -(finalSquareLength - finalHeight) / 2;

    if (inputPic) {
        ctx.drawImage(inputPic, 0, 0, canvas.width, canvas.height);
    } else {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    ctx.font = `${2 * featureSize}px sans-serif`;
    ctx.textBaseline = 'middle';

    for (const detection of detections) {
        const style = categoryStyle(detection.categoryId);
        const x = finalSquareLength * detection.x + xOffset;
        const y = finalSquareLength * detection.y + yOffset;

        ctx.strokeStyle = style.box;
        ctx.lineWidth = featureSize / 2;
        ctx.strokeRect(x, y, finalSquareLength * detection.w, finalSquareLength * detection.h);

        const label = `${detection.categoryName} ${Math.round(detection.prob * 100)}%`;
        const padding = featureSize / 2;
        const labelHeight = 2.6 * featureSize;
        // Sit the label above the box, or just inside it when there is no room
        const labelY = y - labelHeight >= 0 ? y - labelHeight : y;

        ctx.fillStyle = style.box;
        ctx.fillRect(x, labelY, ctx.measureText(label).width + 2 * padding, labelHeight);
        ctx.fillStyle = style.label;
        ctx.fillText(label, x + padding, labelY + labelHeight / 2);
    }
}
