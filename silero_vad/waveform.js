const BACKGROUND = '#fbfcfd';
const GRID = '#e4e7ec';
const WAVE = '#8b95a1';
const SPEECH_WAVE = '#0b6bcb';
const SPEECH_BAND = 'rgba(11, 107, 203, 0.12)';

/**
 * Draws a mono signal, highlighting the stretches the VAD called speech.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {Float32Array} samples
 * @param {{start: number, end: number}[]} segments sample offsets, may be empty
 * @param {number} [visibleSamples] length to scale to, for a still filling buffer
 */
export function drawWaveform(canvas, samples, segments = [], visibleSamples = samples.length) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const middle = height / 2;
    const span = Math.max(1, visibleSamples);

    ctx.fillStyle = BACKGROUND;
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = GRID;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, middle);
    ctx.lineTo(width, middle);
    ctx.stroke();

    // Speech bands sit behind the trace
    ctx.fillStyle = SPEECH_BAND;
    for (const segment of segments) {
        const left = width * segment.start / span;
        ctx.fillRect(left, 0, Math.max(1, width * (segment.end - segment.start) / span), height);
    }

    const isSpeech = column => segments.some(segment =>
        column >= width * segment.start / span && column <= width * segment.end / span);

    // One vertical bar per column, spanning that column's min and max
    const samplesPerColumn = Math.max(1, Math.floor(span / width));
    for (let column = 0; column < width; column++) {
        const from = column * samplesPerColumn;
        if (from >= samples.length) {
            break;
        }
        const to = Math.min(samples.length, from + samplesPerColumn);

        let min = 0;
        let max = 0;
        for (let i = from; i < to; i++) {
            if (samples[i] < min) {
                min = samples[i];
            }
            if (samples[i] > max) {
                max = samples[i];
            }
        }

        ctx.strokeStyle = isSpeech(column) ? SPEECH_WAVE : WAVE;
        ctx.beginPath();
        ctx.moveTo(column + 0.5, middle - max * middle);
        ctx.lineTo(column + 0.5, middle - min * middle);
        ctx.stroke();
    }
}

/**
 * @param {Float32Array} samples
 * @param {number} sampleRate
 */
export function play(samples, sampleRate) {
    const context = new AudioContext({ sampleRate });
    const buffer = context.createBuffer(1, Math.max(1, samples.length), sampleRate);
    buffer.copyToChannel(samples, 0);

    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    source.onended = () => context.close();
    source.start();

    return source;
}
