/**
 * Streaming Audio Player with buffering support
 * Based on simple_client/index.html with enhanced buffering and TTFA measurement
 */

class StreamingAudioPlayer {
    constructor() {
        this.audioContext = null;
        this.nextPlayTime = 0;
        this.pendingBuffers = [];
        this.bufferDurationMs = 100; // Target buffer before first play (100ms)
        this.accumulatedDuration = 0;
        this.isPlaying = false;
        this.requestStartTime = null;
        this.firstChunkTime = null;
        this.playbackStartTime = null;

        // Callbacks
        this.onStatusChange = null;
        this.onTTFANetwork = null;
        this.onTTFAPlayback = null;

        // Streaming state
        this.abortController = null;
        this.wavInfo = null;
        this.headerParsed = false;
        this.leftoverData = new Uint8Array(0);
    }

    /**
     * Initialize the AudioContext (must be called on user interaction)
     */
    init() {
        if (!this.audioContext) {
            try {
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            } catch (e) {
                console.error('Web Audio API not supported:', e);
                throw new Error('Web Audio API is not supported by this browser');
            }
        }
        return this.audioContext;
    }

    /**
     * Reset player state for a new request
     */
    reset() {
        this.nextPlayTime = this.audioContext ? this.audioContext.currentTime : 0;
        this.pendingBuffers = [];
        this.accumulatedDuration = 0;
        this.isPlaying = false;
        this.requestStartTime = null;
        this.firstChunkTime = null;
        this.playbackStartTime = null;
        this.wavInfo = null;
        this.headerParsed = false;
        this.leftoverData = new Uint8Array(0);
    }

    /**
     * Parse WAV header (44 bytes standard)
     */
    parseWavHeader(wavBytes) {
        const dataView = new DataView(wavBytes.buffer);
        // Check for "RIFF" and "WAVE"
        if (dataView.getUint32(0, false) !== 0x52494646 ||
            dataView.getUint32(8, false) !== 0x57415645) {
            console.error("Not a valid WAV file.");
            return null;
        }
        return {
            numChannels: dataView.getUint16(22, true),
            sampleRate: dataView.getUint32(24, true),
            bitsPerSample: dataView.getUint16(34, true),
        };
    }

    /**
     * Convert PCM data to AudioBuffer
     */
    createAudioBuffer(chunk, wavInfo) {
        // Convert 16-bit PCM to 32-bit float samples
        const pcmData = new Int16Array(chunk.buffer, chunk.byteOffset, chunk.length / 2);
        const floatData = new Float32Array(pcmData.length);
        for (let i = 0; i < pcmData.length; i++) {
            floatData[i] = pcmData[i] / 32768.0;
        }

        // Create audio buffer
        const audioBuffer = this.audioContext.createBuffer(
            wavInfo.numChannels,
            floatData.length / wavInfo.numChannels,
            wavInfo.sampleRate
        );

        // Copy data to channel(s)
        for (let channel = 0; channel < wavInfo.numChannels; channel++) {
            const channelData = audioBuffer.getChannelData(channel);
            for (let i = 0; i < floatData.length / wavInfo.numChannels; i++) {
                channelData[i] = floatData[i * wavInfo.numChannels + channel];
            }
        }

        return audioBuffer;
    }

    /**
     * Schedule an audio buffer for playback
     */
    scheduleBuffer(audioBuffer) {
        const source = this.audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(this.audioContext.destination);

        if (this.nextPlayTime < this.audioContext.currentTime) {
            this.nextPlayTime = this.audioContext.currentTime;
        }
        source.start(this.nextPlayTime);
        this.nextPlayTime += audioBuffer.duration;
    }

    /**
     * Start playback of buffered audio
     */
    startPlayback() {
        if (this.isPlaying) return;

        this.isPlaying = true;
        this.playbackStartTime = performance.now();

        // Calculate playback TTFA
        if (this.requestStartTime && this.onTTFAPlayback) {
            const ttfa = this.playbackStartTime - this.requestStartTime;
            this.onTTFAPlayback(ttfa);
        }

        // Schedule all pending buffers
        for (const buffer of this.pendingBuffers) {
            this.scheduleBuffer(buffer);
        }
        this.pendingBuffers = [];

        if (this.onStatusChange) {
            this.onStatusChange('Playing...');
        }
    }

    /**
     * Process a raw audio chunk from the stream
     */
    processChunk(value) {
        // Record first chunk time for network TTFA
        if (this.firstChunkTime === null) {
            this.firstChunkTime = performance.now();
            if (this.requestStartTime && this.onTTFANetwork) {
                const ttfa = this.firstChunkTime - this.requestStartTime;
                this.onTTFANetwork(ttfa);
            }
        }

        // Append new data to leftover
        const currentData = new Uint8Array(this.leftoverData.length + value.length);
        currentData.set(this.leftoverData);
        currentData.set(value, this.leftoverData.length);

        if (!this.headerParsed) {
            // Wait for at least 44 bytes (WAV header)
            if (currentData.length >= 44) {
                this.wavInfo = this.parseWavHeader(currentData);
                if (!this.wavInfo) {
                    throw new Error("Failed to parse WAV header");
                }
                console.log("WAV Info:", this.wavInfo);
                this.headerParsed = true;

                // Process audio data after header
                const audioData = currentData.slice(44);
                if (audioData.length > 0) {
                    this.processAudioData(audioData);
                }
                this.leftoverData = new Uint8Array(0);
            } else {
                this.leftoverData = currentData;
            }
        } else {
            this.processAudioData(currentData);
        }
    }

    /**
     * Process raw audio data (after header parsing)
     */
    processAudioData(data) {
        // Align to sample boundaries
        const sampleSize = this.wavInfo.bitsPerSample / 8;
        const processableLength = Math.floor(data.length / sampleSize) * sampleSize;

        if (processableLength > 0) {
            const audioData = data.slice(0, processableLength);
            const audioBuffer = this.createAudioBuffer(audioData, this.wavInfo);
            const chunkDurationMs = audioBuffer.duration * 1000;

            if (!this.isPlaying) {
                // Buffer until we have enough
                this.pendingBuffers.push(audioBuffer);
                this.accumulatedDuration += chunkDurationMs;

                if (this.accumulatedDuration >= this.bufferDurationMs) {
                    this.startPlayback();
                }
            } else {
                // Already playing, schedule immediately
                this.scheduleBuffer(audioBuffer);
            }
        }

        // Store leftover bytes
        this.leftoverData = data.slice(processableLength);
    }

    /**
     * Start streaming audio from a URL
     */
    async streamFromUrl(url, formData) {
        // Initialize audio context
        this.init();
        this.reset();

        // Create abort controller for cancellation
        this.abortController = new AbortController();

        // Record request start time
        this.requestStartTime = performance.now();

        if (this.onStatusChange) {
            this.onStatusChange('Connecting...');
        }

        try {
            const response = await fetch(url, {
                method: 'POST',
                body: formData,
                signal: this.abortController.signal
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status} ${response.statusText}`);
            }

            if (this.onStatusChange) {
                this.onStatusChange('Streaming...');
            }

            const reader = response.body.getReader();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                this.processChunk(value);
            }

            // Process any remaining buffered audio
            if (!this.isPlaying && this.pendingBuffers.length > 0) {
                this.startPlayback();
            }

            if (this.onStatusChange) {
                this.onStatusChange('Playback finished');
            }

        } catch (error) {
            if (error.name === 'AbortError') {
                if (this.onStatusChange) {
                    this.onStatusChange('Stopped');
                }
            } else {
                console.error('Streaming failed:', error);
                if (this.onStatusChange) {
                    this.onStatusChange(`Error: ${error.message}`);
                }
                throw error;
            }
        }
    }

    /**
     * Stop current playback and streaming
     */
    stop() {
        if (this.abortController) {
            this.abortController.abort();
            this.abortController = null;
        }
        this.reset();
        if (this.onStatusChange) {
            this.onStatusChange('Idle');
        }
    }
}

// Export for use in main.js
window.StreamingAudioPlayer = StreamingAudioPlayer;
