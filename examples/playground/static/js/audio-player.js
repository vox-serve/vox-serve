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
        this.playbackStartTime = null;

        // Callbacks
        this.onStatusChange = null;
        this.onTTFANetwork = null;

        // Streaming state
        this.abortController = null;
        this.wavInfo = null;
        this.ttfaParsed = false;
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
        this.playbackStartTime = null;
        this.wavInfo = null;
        this.ttfaParsed = false;
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
            this.onStatusChange('▶️ Playing');
        }
    }

    /**
     * Process a raw audio chunk from the stream
     *
     * Stream format from server:
     * - First 4 bytes: TTFA in milliseconds (uint32 little-endian)
     * - Remaining: WAV data (44-byte header + audio samples)
     */
    processChunk(value) {
        // Append new data to leftover
        const currentData = new Uint8Array(this.leftoverData.length + value.length);
        currentData.set(this.leftoverData);
        currentData.set(value, this.leftoverData.length);

        let offset = 0;

        // First, parse TTFA prefix (4 bytes)
        if (!this.ttfaParsed) {
            if (currentData.length >= 4) {
                // Parse uint32 little-endian TTFA
                const ttfaMs = new DataView(currentData.buffer, currentData.byteOffset, 4).getUint32(0, true);
                if (this.onTTFANetwork) {
                    this.onTTFANetwork(ttfaMs);
                }
                this.ttfaParsed = true;
                offset = 4;
            } else {
                this.leftoverData = currentData;
                return;
            }
        }

        // Then, parse WAV header (44 bytes)
        if (!this.headerParsed) {
            const remaining = currentData.slice(offset);
            if (remaining.length >= 44) {
                this.wavInfo = this.parseWavHeader(remaining);
                if (!this.wavInfo) {
                    throw new Error("Failed to parse WAV header");
                }
                console.log("WAV Info:", this.wavInfo);
                this.headerParsed = true;

                // Process audio data after header
                const audioData = remaining.slice(44);
                if (audioData.length > 0) {
                    this.processAudioData(audioData);
                }
                // Note: processAudioData sets leftoverData for any partial samples,
                // so we should NOT reset it here
            } else {
                this.leftoverData = remaining;
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

        if (this.onStatusChange) {
            this.onStatusChange('⏳ Connecting');
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
                this.onStatusChange('⬇️ Streaming');
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

            // Wait for actual playback to finish
            if (this.isPlaying && this.audioContext) {
                const remainingTime = this.nextPlayTime - this.audioContext.currentTime;
                if (remainingTime > 0) {
                    if (this.onStatusChange) {
                        this.onStatusChange('▶️ Playing');
                    }
                    await new Promise(resolve => setTimeout(resolve, remainingTime * 1000));
                }
            }

            if (this.onStatusChange) {
                this.onStatusChange('✅ Finished');
            }

        } catch (error) {
            if (error.name === 'AbortError') {
                if (this.onStatusChange) {
                    this.onStatusChange('⏹️ Stopped');
                }
            } else {
                console.error('Streaming failed:', error);
                if (this.onStatusChange) {
                    this.onStatusChange('❌ Error');
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
            this.onStatusChange('⏸️ Idle');
        }
    }
}

// Export for use in main.js
window.StreamingAudioPlayer = StreamingAudioPlayer;
