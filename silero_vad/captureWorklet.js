// Forwards microphone audio to the main thread, replacing the deprecated
// ScriptProcessorNode. The audio thread hands over render quanta of 128 samples,
// which is too small to post one message at a time, so they are gathered into
// larger blocks first. Resampling and inference stay on the main thread, where
// the model lives.
const BLOCK_SAMPLES = 1024;

class CaptureProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.block = new Float32Array(BLOCK_SAMPLES);
        this.filled = 0;
    }

    process(inputs) {
        const channel = inputs[0] && inputs[0][0];

        // No input connected yet, or the track has gone away
        if (!channel) {
            return true;
        }

        for (let i = 0; i < channel.length; i++) {
            this.block[this.filled++] = channel[i];

            if (this.filled === BLOCK_SAMPLES) {
                // Hand over a copy, since this buffer is reused
                this.port.postMessage(this.block.slice());
                this.filled = 0;
            }
        }

        return true;
    }
}

registerProcessor('microphone-capture', CaptureProcessor);
