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
        this.calls = 0;
    }

    process(inputs) {
        const input = inputs[0];
        // Mix every channel down. Reading only the first would capture silence
        // from a device that puts audio on the right channel alone.
        const channels = (input || []).filter(channel => channel && channel.length);
        const channel = channels[0];
        this.calls++;

        // Report what the audio thread is actually being handed, once, so a
        // silent microphone can be told apart from one that never runs at all
        if (this.calls === 1) {
            this.port.postMessage({
                type: 'started',
                inputCount: inputs.length,
                channelCount: input ? input.length : 0,
                usableChannels: channels.length,
                quantum: channel ? channel.length : 0
            });
        }

        if (!channel) {
            // Keep the processor alive: an input can be absent for a few quanta
            // while the graph settles
            if (this.calls % 500 === 0) {
                this.port.postMessage({ type: 'noInput', calls: this.calls });
            }
            return true;
        }

        for (let i = 0; i < channel.length; i++) {
            let sum = 0;
            for (let c = 0; c < channels.length; c++) {
                sum += channels[c][i];
            }
            this.block[this.filled++] = sum / channels.length;

            if (this.filled === BLOCK_SAMPLES) {
                let peak = 0;
                for (let j = 0; j < BLOCK_SAMPLES; j++) {
                    const magnitude = Math.abs(this.block[j]);
                    if (magnitude > peak) {
                        peak = magnitude;
                    }
                }

                // Hand over a copy, since this buffer is reused
                this.port.postMessage({ type: 'block', samples: this.block.slice(), peak });
                this.filled = 0;
            }
        }

        return true;
    }
}

registerProcessor('microphone-capture', CaptureProcessor);
