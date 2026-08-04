import { Ailia } from '../ailia-js/ailia.js';

// Silero's 8 kHz configuration: 256 sample windows with 32 samples of the
// previous window prepended as context.
export const SAMPLE_RATE = 8000;
export const WINDOW_SAMPLES = 256;
const CONTEXT_SAMPLES = 32;
const INPUT_SAMPLES = CONTEXT_SAMPLES + WINDOW_SAMPLES;
const STATE_SHAPE = [2, 1, 128];
const STATE_SAMPLES = 2 * 1 * 128;

// Segmentation defaults from get_speech_timestamps in the reference
const SPEECH_THRESHOLD = 0.5;
const SILENCE_THRESHOLD = SPEECH_THRESHOLD - 0.15;
const MIN_SPEECH_MS = 250;
const MIN_SILENCE_MS = 100;
const SPEECH_PAD_MS = 30;

const BYTES_PER_FLOAT = 4;
const BYTES_PER_INT64 = 8;

/**
 * Runs silero-vad one window at a time, carrying the recurrent state across
 * calls.
 *
 * This drives ailia through its blob level calls rather than `ailia.run`, which
 * the other demos use. The model's third input, `sr`, is a rank 0 int64, and
 * `ailia.run` sizes every input buffer as rank-1-or-more float32, so it cannot
 * express that input at all.
 */
export class SileroVad {
    /**
     * @param {*} ailia an Ailia instance holding silero_vad_v6_2.onnx
     */
    constructor(ailia) {
        this.ailia = ailia;

        ailia._buildInputBuffersStructure();
        ailia._buildOutputBuffersStructure();

        const inputBlobs = ailia.inputBuffers.byIndex;
        this.inputBlob = inputBlobs[0].blobIndex;
        this.stateBlob = inputBlobs[1].blobIndex;
        this.sampleRateBlob = inputBlobs[2].blobIndex;

        this.outputBlobs = {};
        for (const blob of ailia.outputBuffers.byIndex) {
            this.outputBlobs[blob.name] = blob.blobIndex;
        }

        ailia._setInputBlobShape(this.inputBlob, [1, INPUT_SAMPLES]);
        ailia._setInputBlobShape(this.stateBlob, STATE_SHAPE);

        this.input = Ailia.allocateNew(INPUT_SAMPLES * BYTES_PER_FLOAT);
        this.state = Ailia.allocateNew(STATE_SAMPLES * BYTES_PER_FLOAT);
        this.nextState = Ailia.allocateNew(STATE_SAMPLES * BYTES_PER_FLOAT);
        this.probability = Ailia.allocateNew(BYTES_PER_FLOAT);

        this.sampleRate = Ailia.allocateNew(BYTES_PER_INT64);
        this.sampleRate.BigInt64[0] = BigInt(SAMPLE_RATE);

        this.context = new Float32Array(CONTEXT_SAMPLES);
        this.reset();
    }

    /** Clears the recurrent state and the carried context. */
    reset() {
        this.state.Float32.fill(0);
        this.context.fill(0);
    }

    /**
     * @param {Float32Array} window exactly WINDOW_SAMPLES samples
     * @returns {number} probability that the window contains speech
     */
    process(window) {
        if (window.length !== WINDOW_SAMPLES) {
            throw new Error(`Expected ${WINDOW_SAMPLES} samples, got ${window.length}`);
        }

        const ailia = this.ailia;

        this.input.Float32.set(this.context, 0);
        this.input.Float32.set(window, CONTEXT_SAMPLES);

        ailia._setInputBlobData(this.input, this.inputBlob);
        ailia._setInputBlobData(this.state, this.stateBlob);
        ailia._setInputBlobData(this.sampleRate, this.sampleRateBlob);
        ailia._update();

        ailia._getBlobData(this.probability, this.outputBlobs.output);
        ailia._getBlobData(this.nextState, this.outputBlobs.stateN);
        this.state.Float32.set(this.nextState.Float32);

        // The tail of this window becomes the next window's context
        this.context.set(this.input.Float32.subarray(INPUT_SAMPLES - CONTEXT_SAMPLES, INPUT_SAMPLES));

        return this.probability.Float32[0];
    }
}

/**
 * Runs the whole signal through the model, one window at a time.
 *
 * @param {SileroVad} vad
 * @param {Float32Array} samples mono audio at SAMPLE_RATE
 * @returns {Float32Array} one probability per window
 */
export function computeSpeechProbabilities(vad, samples) {
    const frameCount = Math.floor(samples.length / WINDOW_SAMPLES);
    const probabilities = new Float32Array(frameCount);

    vad.reset();
    for (let frame = 0; frame < frameCount; frame++) {
        const offset = frame * WINDOW_SAMPLES;
        probabilities[frame] = vad.process(samples.subarray(offset, offset + WINDOW_SAMPLES));
    }

    return probabilities;
}

/**
 * Turns a stream of per window probabilities into speech segments.
 *
 * Follows the hysteresis that get_speech_timestamps uses: a segment opens above
 * the threshold and only closes once the probability has stayed below
 * threshold - 0.15 for long enough. Segments shorter than the minimum are
 * dropped, and the rest are padded without reaching into the one before.
 *
 * Probabilities are pushed one at a time, so the same code serves the recording
 * path, which needs each segment as soon as it closes, and a whole signal.
 */
export class SpeechSegmenter {
    constructor() {
        this.minSpeech = SAMPLE_RATE * MIN_SPEECH_MS / 1000;
        this.minSilence = SAMPLE_RATE * MIN_SILENCE_MS / 1000;
        this.pad = SAMPLE_RATE * SPEECH_PAD_MS / 1000;
        this.reset();
    }

    reset() {
        this.frame = 0;
        this.inSpeech = false;
        this.start = 0;
        this.silenceStart = 0;
        // Padding is held back from the previous segment's unpadded end
        this.previousEnd = 0;
    }

    /**
     * The segment being spoken right now, so it can be shown before it closes.
     * @returns {{start: number, end: number}|null}
     */
    get openSegment() {
        return this.inSpeech ? { start: this.start, end: this.frame * WINDOW_SAMPLES } : null;
    }

    /**
     * @param {number} probability for the next window
     * @returns {{start: number, end: number}|null} a segment, if one just closed
     */
    push(probability) {
        const position = this.frame * WINDOW_SAMPLES;
        this.frame++;

        if (probability >= SPEECH_THRESHOLD) {
            this.silenceStart = 0;
            if (!this.inSpeech) {
                this.inSpeech = true;
                this.start = position;
            }
            return null;
        }

        if (!this.inSpeech || probability >= SILENCE_THRESHOLD) {
            return null;
        }

        if (!this.silenceStart) {
            this.silenceStart = position;
        }
        if (position - this.silenceStart < this.minSilence) {
            return null;
        }

        const end = this.silenceStart;
        const start = this.start;
        this.inSpeech = false;
        this.silenceStart = 0;

        // The silence just waited out is longer than the padding, so `position`
        // is always a safe limit for it
        return end - start > this.minSpeech ? this._pad(start, end, position) : null;
    }

    /**
     * Closes a segment still open at the end of the signal.
     *
     * @param {number} totalSamples
     * @returns {{start: number, end: number}|null}
     */
    flush(totalSamples) {
        if (!this.inSpeech) {
            return null;
        }

        const start = this.start;
        this.inSpeech = false;

        return totalSamples - start > this.minSpeech ? this._pad(start, totalSamples, totalSamples) : null;
    }

    _pad(start, end, limit) {
        const padded = {
            start: Math.max(this.previousEnd, start - this.pad),
            end: Math.min(limit, end + this.pad)
        };
        this.previousEnd = end;

        return padded;
    }
}

/**
 * Runs the segmenter over a whole set of probabilities.
 *
 * @param {Float32Array} probabilities
 * @param {number} totalSamples length of the signal, to clamp the padding
 * @returns {{start: number, end: number}[]} sample offsets
 */
export function findSpeechSegments(probabilities, totalSamples) {
    const segmenter = new SpeechSegmenter();
    const segments = [];

    for (const probability of probabilities) {
        const segment = segmenter.push(probability);
        if (segment) {
            segments.push(segment);
        }
    }

    const last = segmenter.flush(totalSamples);
    if (last) {
        segments.push(last);
    }

    return segments;
}

/**
 * @param {{start: number, end: number}[]} segments
 * @param {Float32Array} samples
 * @returns {Float32Array} the segments concatenated, with the rest dropped
 */
export function collectSegments(segments, samples) {
    const total = segments.reduce((sum, s) => sum + (s.end - s.start), 0);
    const collected = new Float32Array(total);

    let offset = 0;
    for (const segment of segments) {
        collected.set(samples.subarray(segment.start, segment.end), offset);
        offset += segment.end - segment.start;
    }

    return collected;
}
