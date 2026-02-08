/**
 * VoxServe Playground - Main Application Logic
 */

// Global state
let serverRunning = false;
let serverPort = 8000;
let serverSchedulerType = 'base';
let modelCapabilities = {};
let audioPlayer = null;
let llmAudioContext = null;
let logPollingInterval = null;
let lastLogCount = 0;

// LLM Chat state
let chatMessages = [];
let isLLMStreaming = false;
let llmAbortController = null;

// DOM Elements
const elements = {
    // Server controls - Basic
    modelSelect: document.getElementById('model-select'),
    portInput: document.getElementById('port-input'),
    gpuCheckboxes: document.getElementById('gpu-checkboxes'),
    batchSize: document.getElementById('batch-size'),
    startServerBtn: document.getElementById('start-server-btn'),
    stopServerBtn: document.getElementById('stop-server-btn'),
    serverStatus: document.getElementById('server-status'),
    statusMessage: document.getElementById('status-message'),
    serverIndicator: document.getElementById('server-indicator'),

    // Advanced - Scheduler
    schedulerType: document.getElementById('scheduler-type'),
    asyncScheduling: document.getElementById('async-scheduling'),
    dpSize: document.getElementById('dp-size'),

    // Advanced - Memory
    maxNumPages: document.getElementById('max-num-pages'),
    pageSize: document.getElementById('page-size'),

    // Advanced - Sampling
    topP: document.getElementById('top-p'),
    topK: document.getElementById('top-k'),
    minP: document.getElementById('min-p'),
    temperature: document.getElementById('temperature'),
    maxTokens: document.getElementById('max-tokens'),
    cfgScale: document.getElementById('cfg-scale'),
    repetitionPenalty: document.getElementById('repetition-penalty'),
    repetitionWindow: document.getElementById('repetition-window'),
    greedy: document.getElementById('greedy'),

    // Advanced - Performance
    cudaGraph: document.getElementById('cuda-graph'),
    enableDisaggregation: document.getElementById('enable-disaggregation'),
    enableTorchCompile: document.getElementById('enable-torch-compile'),
    enableNvtx: document.getElementById('enable-nvtx'),

    // Advanced - Other
    logLevel: document.getElementById('log-level'),
    detokenizeInterval: document.getElementById('detokenize-interval'),

    // Logs
    logsContainer: document.getElementById('logs-container'),
    autoScroll: document.getElementById('auto-scroll'),
    clearLogsBtn: document.getElementById('clear-logs-btn'),

    // TTS request
    textInput: document.getElementById('text-input'),
    audioInput: document.getElementById('audio-input'),
    audioUploadGroup: document.getElementById('audio-upload-group'),
    streamingCheckbox: document.getElementById('streaming-checkbox'),
    generateBtn: document.getElementById('generate-btn'),

    // Model-specific params
    modelParams: document.getElementById('model-params'),
    languageGroup: document.getElementById('language-group'),
    languageInput: document.getElementById('language-input'),
    speakerGroup: document.getElementById('speaker-group'),
    speakerInput: document.getElementById('speaker-input'),
    refTextGroup: document.getElementById('ref-text-group'),
    refTextInput: document.getElementById('ref-text-input'),
    instructGroup: document.getElementById('instruct-group'),
    instructInput: document.getElementById('instruct-input'),

    // Audio player
    playbackStatus: document.getElementById('playback-status'),
    ttfaNetwork: document.getElementById('ttfa-network'),
    stopAudioBtn: document.getElementById('stop-audio-btn'),

    // Tabs
    tabButtons: document.querySelectorAll('.tab-btn'),
    tabContents: document.querySelectorAll('.tab-content'),
    ttsSchedulerWarning: document.getElementById('tts-scheduler-warning'),

    // LLM Chat
    llmUrl: document.getElementById('llm-url'),
    llmModel: document.getElementById('llm-model'),
    llmApiKey: document.getElementById('llm-api-key'),
    llmModelParams: document.getElementById('llm-model-params'),
    llmLanguageGroup: document.getElementById('llm-language-group'),
    llmLanguageInput: document.getElementById('llm-language-input'),
    llmSpeakerGroup: document.getElementById('llm-speaker-group'),
    llmSpeakerInput: document.getElementById('llm-speaker-input'),
    chatContainer: document.getElementById('chat-container'),
    chatInput: document.getElementById('chat-input'),
    chatSendBtn: document.getElementById('chat-send-btn'),
    chatStopBtn: document.getElementById('chat-stop-btn'),
    clearChatBtn: document.getElementById('clear-chat-btn'),
    llmPlaybackStatus: document.getElementById('llm-playback-status'),
};

/**
 * Initialize the application
 */
async function init() {
    // Initialize audio player for TTS tab
    audioPlayer = new StreamingAudioPlayer();
    audioPlayer.onStatusChange = (status) => {
        elements.playbackStatus.textContent = status;
    };
    audioPlayer.onTTFANetwork = (ttfa) => {
        elements.ttfaNetwork.textContent = `${Math.round(ttfa)}ms`;
    };

    // Load GPUs
    await loadGPUs();

    // Load models and their capabilities
    await loadModels();

    // Setup event listeners
    setupEventListeners();

    // Start status polling
    startStatusPolling();

    // Initial UI update
    updateModelParams();
    updateLLMModelParams();
}

/**
 * Load available GPUs
 */
async function loadGPUs() {
    try {
        const response = await fetch('/api/gpus');
        const gpus = await response.json();

        if (gpus.length === 0) {
            elements.gpuCheckboxes.innerHTML = '<span class="loading-text">No GPUs detected</span>';
            return;
        }

        elements.gpuCheckboxes.innerHTML = gpus.map((gpu, idx) => `
            <label>
                <input type="checkbox" name="gpu" value="${gpu.index}" ${idx === 0 ? 'checked' : ''}>
                GPU ${gpu.index}: ${gpu.name}
            </label>
        `).join('');
    } catch (error) {
        console.error('Failed to load GPUs:', error);
        elements.gpuCheckboxes.innerHTML = '<span class="loading-text">Failed to load GPUs</span>';
    }
}

/**
 * Load supported models
 */
async function loadModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();

        // Store model capabilities
        for (const model of data.models) {
            modelCapabilities[model.id] = model;
        }

        // Update model dropdown
        elements.modelSelect.innerHTML = data.models.map(model =>
            `<option value="${model.id}">${model.name}</option>`
        ).join('');
    } catch (error) {
        console.error('Failed to load models:', error);
    }
}

/**
 * Setup event listeners
 */
function setupEventListeners() {
    // Server controls
    elements.startServerBtn.addEventListener('click', startServer);
    elements.stopServerBtn.addEventListener('click', stopServer);
    elements.modelSelect.addEventListener('change', () => {
        updateModelParams();
        updateLLMModelParams();
    });

    // Logs
    elements.clearLogsBtn.addEventListener('click', clearLogs);

    // TTS request
    elements.generateBtn.addEventListener('click', generateAudio);

    // Audio player
    elements.stopAudioBtn.addEventListener('click', () => {
        audioPlayer.stop();
        elements.stopAudioBtn.disabled = true;
    });

    // Tab switching
    elements.tabButtons.forEach(btn => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });

    // LLM Chat
    elements.chatSendBtn.addEventListener('click', sendChatMessage);
    elements.chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage();
        }
    });
    elements.chatStopBtn.addEventListener('click', stopLLMChat);
    elements.clearChatBtn.addEventListener('click', clearChatHistory);
}

/**
 * Update UI based on selected model's capabilities
 */
function updateModelParams() {
    const modelId = elements.modelSelect.value;
    const caps = modelCapabilities[modelId] || {};

    // Show/hide model params section (language, speaker, instruct)
    const hasParams = caps.supports_language || caps.supports_speaker || caps.supports_instruct;
    elements.modelParams.classList.toggle('hidden', !hasParams);

    // Show/hide individual params within model-params section
    elements.languageGroup.style.display = caps.supports_language ? '' : 'none';
    elements.speakerGroup.style.display = caps.supports_speaker ? '' : 'none';
    elements.instructGroup.style.display = caps.supports_instruct ? '' : 'none';

    // Show/hide voice cloning section (audio + ref_text together)
    elements.audioUploadGroup.classList.toggle('hidden', !caps.supports_audio_input);

    // Set detokenize interval to 1 for Qwen models, clear for others
    const isQwenModel = modelId.toLowerCase().includes('qwen');
    if (isQwenModel) {
        elements.detokenizeInterval.value = '1';
    } else {
        elements.detokenizeInterval.value = '';
    }
}

/**
 * Get selected GPU indices
 */
function getSelectedGPUs() {
    const checkboxes = elements.gpuCheckboxes.querySelectorAll('input[name="gpu"]:checked');
    return Array.from(checkboxes).map(cb => parseInt(cb.value));
}

/**
 * Start VoxServe server
 */
async function startServer() {
    const selectedGPUs = getSelectedGPUs();
    if (selectedGPUs.length === 0) {
        setStatus('Please select at least one GPU', 'error');
        return;
    }

    elements.startServerBtn.disabled = true;
    setStatus('Starting server...', '');
    updateIndicator('starting');
    clearLogs();

    const config = {
        // Basic
        model: elements.modelSelect.value,
        port: parseInt(elements.portInput.value),
        cuda_devices: selectedGPUs,
        max_batch_size: parseInt(elements.batchSize.value),

        // Scheduler
        scheduler_type: elements.schedulerType.value,
        async_scheduling: elements.asyncScheduling.checked,
        dp_size: parseInt(elements.dpSize.value),

        // Memory
        max_num_pages: parseInt(elements.maxNumPages.value),
        page_size: parseInt(elements.pageSize.value),

        // Sampling
        top_p: elements.topP.value ? parseFloat(elements.topP.value) : null,
        top_k: elements.topK.value ? parseInt(elements.topK.value) : null,
        min_p: elements.minP.value ? parseFloat(elements.minP.value) : null,
        temperature: elements.temperature.value ? parseFloat(elements.temperature.value) : null,
        max_tokens: elements.maxTokens.value ? parseInt(elements.maxTokens.value) : null,
        cfg_scale: elements.cfgScale.value ? parseFloat(elements.cfgScale.value) : null,
        repetition_penalty: elements.repetitionPenalty.value ? parseFloat(elements.repetitionPenalty.value) : null,
        repetition_window: elements.repetitionWindow.value ? parseInt(elements.repetitionWindow.value) : null,
        greedy: elements.greedy.checked,

        // Performance
        enable_cuda_graph: elements.cudaGraph.checked,
        enable_disaggregation: elements.enableDisaggregation.checked,
        enable_torch_compile: elements.enableTorchCompile.checked,
        enable_nvtx: elements.enableNvtx.checked,

        // Other
        log_level: elements.logLevel.value,
        detokenize_interval: elements.detokenizeInterval.value ? parseInt(elements.detokenizeInterval.value) : null,
    };

    try {
        const response = await fetch('/api/server/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config),
        });
        const result = await response.json();

        if (result.success) {
            serverPort = config.port;
            serverSchedulerType = config.scheduler_type;
            // Don't set serverRunning=true yet, wait for log confirmation
            setStatus('Preparing...', '');
            updateIndicator('starting');
            elements.startServerBtn.disabled = true;
            elements.stopServerBtn.disabled = false;
            startLogPolling();
        } else {
            setStatus(result.message, 'error');
            updateIndicator('stopped');
            elements.startServerBtn.disabled = false;
        }
    } catch (error) {
        console.error('Failed to start server:', error);
        setStatus(`Failed to start server: ${error.message}`, 'error');
        updateIndicator('stopped');
        elements.startServerBtn.disabled = false;
    }
}

/**
 * Stop VoxServe server
 */
async function stopServer() {
    elements.stopServerBtn.disabled = true;
    setStatus('Stopping server...', '');

    try {
        const response = await fetch('/api/server/stop', { method: 'POST' });
        const result = await response.json();

        serverRunning = false;
        setStatus(result.message, result.success ? 'success' : 'error');
        updateIndicator('stopped');
        elements.startServerBtn.disabled = false;
        elements.stopServerBtn.disabled = true;
        elements.generateBtn.disabled = true;
        elements.ttsSchedulerWarning.classList.add('hidden');
        stopLogPolling();
    } catch (error) {
        console.error('Failed to stop server:', error);
        setStatus(`Failed to stop server: ${error.message}`, 'error');
        elements.stopServerBtn.disabled = false;
    }
}

/**
 * Generate audio via the playground proxy
 */
async function generateAudio() {
    const text = elements.textInput.value.trim();
    if (!text) {
        setStatus('Please enter text to synthesize', 'error');
        return;
    }

    // Reset TTFA display
    elements.ttfaNetwork.textContent = '-';

    // Build form data
    const formData = new FormData();
    formData.append('text', text);
    formData.append('streaming', elements.streamingCheckbox.checked ? 'true' : 'false');

    // Add audio file if provided
    const audioFile = elements.audioInput.files[0];
    if (audioFile) {
        formData.append('audio', audioFile);
    }

    // Add model-specific params
    const modelId = elements.modelSelect.value;
    const caps = modelCapabilities[modelId] || {};

    if (caps.supports_language && elements.languageInput.value) {
        formData.append('language', elements.languageInput.value);
    }
    if (caps.supports_speaker && elements.speakerInput.value) {
        formData.append('speaker', elements.speakerInput.value);
    }
    if (caps.supports_ref_text && elements.refTextInput.value) {
        formData.append('ref_text', elements.refTextInput.value);
    }
    if (caps.supports_instruct && elements.instructInput.value) {
        formData.append('instruct', elements.instructInput.value);
    }

    // Disable generate button
    elements.generateBtn.disabled = true;
    elements.stopAudioBtn.disabled = false;

    // Use the playground proxy endpoint
    const proxyUrl = '/api/generate';

    try {
        await audioPlayer.streamFromUrl(proxyUrl, formData);
    } catch (error) {
        console.error('Audio generation failed:', error);
    } finally {
        elements.generateBtn.disabled = false;
        elements.stopAudioBtn.disabled = true;
    }
}

/**
 * Set status message
 */
function setStatus(message, type) {
    elements.statusMessage.textContent = message;
    elements.serverStatus.className = 'status-box';
    if (type) {
        elements.serverStatus.classList.add(type);
    }
}

/**
 * Update server indicator
 */
function updateIndicator(status) {
    elements.serverIndicator.className = 'status-indicator';
    const text = elements.serverIndicator.querySelector('.status-text');

    switch (status) {
        case 'running':
            elements.serverIndicator.classList.add('status-running');
            text.textContent = 'Server Running';
            break;
        case 'starting':
            elements.serverIndicator.classList.add('status-starting');
            text.textContent = 'Starting...';
            break;
        default:
            elements.serverIndicator.classList.add('status-stopped');
            text.textContent = 'Server Stopped';
    }
}

/**
 * Format uptime in human readable format
 */
function formatUptime(seconds) {
    if (!seconds) return '';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    if (mins > 0) {
        return `${mins}m ${secs}s`;
    }
    return `${secs}s`;
}

/**
 * Poll server status
 */
async function checkServerStatus() {
    try {
        const response = await fetch('/api/server/status');
        const status = await response.json();

        if (status.running && !serverRunning) {
            // Process is running but we haven't confirmed ready yet
            // Keep showing "Preparing..." - wait for log confirmation
            serverPort = status.port;
            elements.startServerBtn.disabled = true;
            elements.stopServerBtn.disabled = false;
            startLogPolling();
        } else if (!status.running && (serverRunning || elements.stopServerBtn.disabled === false)) {
            // Server stopped
            serverRunning = false;
            updateIndicator('stopped');
            elements.startServerBtn.disabled = false;
            elements.stopServerBtn.disabled = true;
            elements.generateBtn.disabled = true;
            elements.ttsSchedulerWarning.classList.add('hidden');
            setStatus('Server stopped', '');
            stopLogPolling();
        } else if (serverRunning && status.uptime_seconds) {
            // Server is confirmed running, update uptime
            const uptime = formatUptime(status.uptime_seconds);
            setStatus(`Running ${status.model} on port ${status.port} | uptime: ${uptime}`, 'success');
        }
    } catch (error) {
        // Ignore errors during polling
    }
}

/**
 * Start status polling
 */
function startStatusPolling() {
    // Initial check
    checkServerStatus();
    // Poll every 5 seconds
    setInterval(checkServerStatus, 5000);
}

/**
 * Classify log line for syntax highlighting
 */
function classifyLogLine(line) {
    const lower = line.toLowerCase();
    if (lower.includes('error') || lower.includes('exception') || lower.includes('failed')) {
        return 'error';
    }
    if (lower.includes('warning') || lower.includes('warn')) {
        return 'warning';
    }
    if (lower.includes('info')) {
        return 'info';
    }
    if (lower.includes('success') || lower.includes('started') || lower.includes('ready')) {
        return 'success';
    }
    return '';
}

/**
 * Check if log line indicates server is ready
 */
function checkServerReady(line) {
    // Pattern: "Scheduler (DP rank X/Y) started successfully with model: ..."
    return line.includes('Scheduler') && line.includes('started successfully');
}

/**
 * Append logs to the container
 */
function appendLogs(logs) {
    if (!logs || logs.length === 0) return;

    // Remove placeholder if present
    const placeholder = elements.logsContainer.querySelector('.logs-placeholder');
    if (placeholder) {
        placeholder.remove();
    }

    // Append new log lines
    for (const line of logs) {
        const lineEl = document.createElement('div');
        lineEl.className = 'log-line';
        const lineClass = classifyLogLine(line);
        if (lineClass) {
            lineEl.classList.add(lineClass);
        }
        lineEl.textContent = line;
        elements.logsContainer.appendChild(lineEl);

        // Check if server is ready based on log content
        if (checkServerReady(line)) {
            onServerReady();
        }
    }

    // Auto-scroll if enabled
    if (elements.autoScroll.checked) {
        elements.logsContainer.scrollTop = elements.logsContainer.scrollHeight;
    }
}

/**
 * Called when server ready message is detected in logs
 */
function onServerReady() {
    serverRunning = true;
    updateIndicator('running');
    elements.startServerBtn.disabled = true;
    elements.stopServerBtn.disabled = false;

    // Update TTS UI based on scheduler type
    updateSchedulerUI();

    setStatus('Server is ready!', 'success');
}

/**
 * Update UI based on current scheduler type
 */
function updateSchedulerUI() {
    const isInputStreaming = serverSchedulerType === 'input_streaming';

    // TTS mode: disabled when using input_streaming
    elements.generateBtn.disabled = !serverRunning || isInputStreaming;
    elements.ttsSchedulerWarning.classList.toggle('hidden', !serverRunning || !isInputStreaming);
}

/**
 * Clear logs display
 */
function clearLogs() {
    elements.logsContainer.innerHTML = '<div class="logs-placeholder">Server logs will appear here when the server is running...</div>';
    lastLogCount = 0;
}

/**
 * Fetch and update logs
 */
async function fetchLogs() {
    try {
        const response = await fetch('/api/server/logs?lines=100');
        const data = await response.json();

        if (data.logs && data.logs.length > lastLogCount) {
            // Only append new logs
            const newLogs = data.logs.slice(lastLogCount);
            appendLogs(newLogs);
            lastLogCount = data.logs.length;
        }
    } catch (error) {
        // Ignore errors during log polling
    }
}

/**
 * Start log polling
 */
function startLogPolling() {
    if (logPollingInterval) return;

    // Initial fetch
    fetchLogs();

    // Poll every 1 second
    logPollingInterval = setInterval(fetchLogs, 1000);
}

/**
 * Stop log polling
 */
function stopLogPolling() {
    if (logPollingInterval) {
        clearInterval(logPollingInterval);
        logPollingInterval = null;
    }
}

/**
 * Switch between tabs
 */
function switchTab(tabId) {
    // Update tab buttons
    elements.tabButtons.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tabId);
    });

    // Update tab contents
    elements.tabContents.forEach(content => {
        content.classList.toggle('active', content.id === `${tabId}-tab`);
    });

    // Auto-select appropriate scheduler type
    if (tabId === 'tts') {
        elements.schedulerType.value = 'base';
    } else if (tabId === 'llm-chat') {
        elements.schedulerType.value = 'input_streaming';
    }
}

/**
 * Update LLM Chat model parameters based on selected model
 */
function updateLLMModelParams() {
    const modelId = elements.modelSelect.value;
    const caps = modelCapabilities[modelId] || {};

    // Show/hide model params section
    const hasParams = caps.supports_language || caps.supports_speaker;
    elements.llmModelParams.classList.toggle('hidden', !hasParams);

    // Show/hide individual params
    elements.llmLanguageGroup.style.display = caps.supports_language ? '' : 'none';
    elements.llmSpeakerGroup.style.display = caps.supports_speaker ? '' : 'none';
}

/**
 * Add a message to the chat display
 */
function addChatMessage(role, content, isStreaming = false) {
    // Remove placeholder if present
    const placeholder = elements.chatContainer.querySelector('.chat-placeholder');
    if (placeholder) {
        placeholder.remove();
    }

    const messageEl = document.createElement('div');
    messageEl.className = `chat-message ${role}`;
    if (isStreaming) {
        messageEl.classList.add('streaming');
    }
    messageEl.innerHTML = `
        <div class="chat-message-role">${role === 'user' ? 'You' : 'Assistant'}</div>
        <div class="chat-message-content">${escapeHtml(content)}</div>
    `;
    elements.chatContainer.appendChild(messageEl);
    elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;
    return messageEl;
}

/**
 * Update the last assistant message content
 */
function updateLastAssistantMessage(content, isStreaming = false) {
    const messages = elements.chatContainer.querySelectorAll('.chat-message.assistant');
    const lastMessage = messages[messages.length - 1];
    if (lastMessage) {
        const contentEl = lastMessage.querySelector('.chat-message-content');
        contentEl.textContent = content;
        lastMessage.classList.toggle('streaming', isStreaming);
    }
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Stream audio from the /audio endpoint in the background
 * Audio chunks are played as they arrive, concurrent with text sending
 */
async function streamAudioInBackground(requestId, audioState, signal) {
    try {
        // Resume AudioContext if suspended (required by browsers)
        if (llmAudioContext.state === 'suspended') {
            await llmAudioContext.resume();
        }

        const response = await fetch(`/api/input-stream/${requestId}/audio`, {
            method: 'GET',
            signal: signal,
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Failed to start audio stream: ${response.status} - ${errorText}`);
        }

        const reader = response.body.getReader();
        // Minimum samples to accumulate before playing (avoid tiny buffers)
        const MIN_SAMPLES_TO_PLAY = 2400;  // 0.1 seconds at 24kHz

        while (true) {
            const { done, value } = await reader.read();
            if (done) {
                audioState.done = true;
                break;
            }

            // Append new data to buffer
            const newBuffer = new Uint8Array(audioState.buffer.length + value.length);
            newBuffer.set(audioState.buffer);
            newBuffer.set(value, audioState.buffer.length);
            audioState.buffer = newBuffer;

            // Parse TTFA prefix (first 4 bytes)
            if (!audioState.ttfaParsed && audioState.buffer.length >= 4) {
                audioState.ttfaParsed = true;
                audioState.buffer = audioState.buffer.slice(4);
            }

            // Parse WAV header (44 bytes)
            if (!audioState.wavHeaderParsed && audioState.buffer.length >= 44) {
                const dataView = new DataView(audioState.buffer.buffer, audioState.buffer.byteOffset);
                audioState.wavInfo = {
                    numChannels: dataView.getUint16(22, true),
                    sampleRate: dataView.getUint32(24, true),
                    bitsPerSample: dataView.getUint16(34, true),
                };
                audioState.wavHeaderParsed = true;
                audioState.buffer = audioState.buffer.slice(44);
                elements.llmPlaybackStatus.textContent = '▶️ Playing + Streaming';
            }

            // Process and play audio samples (with minimum buffer size)
            if (audioState.wavHeaderParsed && audioState.wavInfo) {
                const sampleSize = audioState.wavInfo.bitsPerSample / 8;
                const samplesInBuffer = audioState.buffer.length / sampleSize;

                // Only play when we have enough samples
                if (samplesInBuffer >= MIN_SAMPLES_TO_PLAY) {
                    const processableLength = Math.floor(audioState.buffer.length / sampleSize) * sampleSize;
                    const audioData = audioState.buffer.slice(0, processableLength);
                    audioState.buffer = audioState.buffer.slice(processableLength);

                    // Convert to AudioBuffer and play
                    const pcmData = new Int16Array(audioData.buffer, audioData.byteOffset, audioData.length / 2);
                    const floatData = new Float32Array(pcmData.length);
                    for (let i = 0; i < pcmData.length; i++) {
                        floatData[i] = pcmData[i] / 32768.0;
                    }

                    const numSamples = floatData.length / audioState.wavInfo.numChannels;
                    if (numSamples > 0) {
                        const audioBufferObj = llmAudioContext.createBuffer(
                            audioState.wavInfo.numChannels,
                            numSamples,
                            audioState.wavInfo.sampleRate
                        );

                        for (let channel = 0; channel < audioState.wavInfo.numChannels; channel++) {
                            const channelData = audioBufferObj.getChannelData(channel);
                            for (let i = 0; i < numSamples; i++) {
                                channelData[i] = floatData[i * audioState.wavInfo.numChannels + channel];
                            }
                        }

                        // Schedule playback
                        const source = llmAudioContext.createBufferSource();
                        source.buffer = audioBufferObj;
                        source.connect(llmAudioContext.destination);

                        // Update nextPlayTime based on current context time
                        if (audioState.nextPlayTime < llmAudioContext.currentTime) {
                            audioState.nextPlayTime = llmAudioContext.currentTime;
                        }
                        source.start(audioState.nextPlayTime);
                        audioState.nextPlayTime += audioBufferObj.duration;
                        audioState.isPlaying = true;
                    }
                }
            }
        }

        // Play any remaining audio in buffer
        if (audioState.wavHeaderParsed && audioState.wavInfo && audioState.buffer.length > 0) {
            const sampleSize = audioState.wavInfo.bitsPerSample / 8;
            const processableLength = Math.floor(audioState.buffer.length / sampleSize) * sampleSize;
            if (processableLength > 0) {
                const audioData = audioState.buffer.slice(0, processableLength);
                audioState.buffer = audioState.buffer.slice(processableLength);

                const pcmData = new Int16Array(audioData.buffer, audioData.byteOffset, audioData.length / 2);
                const floatData = new Float32Array(pcmData.length);
                for (let i = 0; i < pcmData.length; i++) {
                    floatData[i] = pcmData[i] / 32768.0;
                }

                const numSamples = floatData.length / audioState.wavInfo.numChannels;
                if (numSamples > 0) {
                    const audioBufferObj = llmAudioContext.createBuffer(
                        audioState.wavInfo.numChannels,
                        numSamples,
                        audioState.wavInfo.sampleRate
                    );

                    for (let channel = 0; channel < audioState.wavInfo.numChannels; channel++) {
                        const channelData = audioBufferObj.getChannelData(channel);
                        for (let i = 0; i < numSamples; i++) {
                            channelData[i] = floatData[i * audioState.wavInfo.numChannels + channel];
                        }
                    }

                    const source = llmAudioContext.createBufferSource();
                    source.buffer = audioBufferObj;
                    source.connect(llmAudioContext.destination);

                    if (audioState.nextPlayTime < llmAudioContext.currentTime) {
                        audioState.nextPlayTime = llmAudioContext.currentTime;
                    }
                    source.start(audioState.nextPlayTime);
                    audioState.nextPlayTime += audioBufferObj.duration;
                    audioState.isPlaying = true;
                }
            }
        }
    } catch (error) {
        if (error.name !== 'AbortError') {
            audioState.error = error.message;
            console.error('Audio streaming error:', error);
        }
    }
}

/**
 * Send a chat message to the LLM and stream response to TTS
 */
async function sendChatMessage() {
    const userMessage = elements.chatInput.value.trim();
    if (!userMessage) return;

    const llmUrl = elements.llmUrl.value.trim();
    if (!llmUrl) {
        alert('Please enter an LLM API URL');
        return;
    }

    // Enforce input_streaming scheduler for LLM chat
    if (!serverRunning) {
        alert('Please start the server first (with input_streaming scheduler)');
        return;
    }
    if (serverSchedulerType !== 'input_streaming') {
        alert('LLM Chat requires input_streaming scheduler. Please restart the server with scheduler type set to "input_streaming".');
        return;
    }

    // Disable input while processing
    elements.chatInput.disabled = true;
    elements.chatSendBtn.disabled = true;
    elements.chatStopBtn.disabled = false;
    isLLMStreaming = true;

    // Clear input and add user message to display
    elements.chatInput.value = '';
    addChatMessage('user', userMessage);

    // Add to chat history
    chatMessages.push({ role: 'user', content: userMessage });

    // Create abort controller
    llmAbortController = new AbortController();

    // Initialize audio context for streaming playback
    if (!llmAudioContext) {
        llmAudioContext = new (window.AudioContext || window.webkitAudioContext)();
    }

    // Audio streaming state (shared with audio receiver)
    const audioState = {
        buffer: new Uint8Array(0),
        ttfaParsed: false,
        wavHeaderParsed: false,
        wavInfo: null,
        nextPlayTime: llmAudioContext.currentTime,
        isPlaying: false,
        error: null,
        done: false,
    };

    try {
        // Get model-specific params
        const modelId = elements.modelSelect.value;
        const caps = modelCapabilities[modelId] || {};

        // Start TTS input streaming
        const startFormData = new FormData();
        if (caps.supports_speaker && elements.llmSpeakerInput.value) {
            startFormData.append('speaker', elements.llmSpeakerInput.value);
        }
        if (caps.supports_language && elements.llmLanguageInput.value) {
            startFormData.append('language', elements.llmLanguageInput.value);
        }

        const startResponse = await fetch('/api/input-stream/start', {
            method: 'POST',
            body: startFormData,
            signal: llmAbortController.signal,
        });

        if (!startResponse.ok) {
            throw new Error('Failed to start input stream');
        }

        const { request_id } = await startResponse.json();

        // Add streaming assistant message
        addChatMessage('assistant', '', true);
        elements.llmPlaybackStatus.textContent = '⬇️ Starting...';

        // Start audio streaming immediately (concurrent with text sending)
        const audioPromise = streamAudioInBackground(request_id, audioState, llmAbortController.signal);

        // Small delay to let audio stream start, then update status
        setTimeout(() => {
            if (!audioState.error && !audioState.done) {
                elements.llmPlaybackStatus.textContent = '⬇️ Streaming LLM + Audio';
            }
        }, 100);

        // Build LLM request headers
        const llmHeaders = { 'Content-Type': 'application/json' };
        if (elements.llmApiKey.value) {
            llmHeaders['Authorization'] = `Bearer ${elements.llmApiKey.value}`;
        }

        // Build LLM request body
        const llmBody = {
            messages: chatMessages,
            stream: true,
        };
        if (elements.llmModel.value) {
            llmBody.model = elements.llmModel.value;
        }

        // Stream LLM response directly from client
        const chatUrl = llmUrl.replace(/\/$/, '') + '/chat/completions';
        const llmResponse = await fetch(chatUrl, {
            method: 'POST',
            headers: llmHeaders,
            body: JSON.stringify(llmBody),
            signal: llmAbortController.signal,
        });

        if (!llmResponse.ok) {
            const errorText = await llmResponse.text();
            throw new Error(`LLM error: ${llmResponse.status} - ${errorText}`);
        }

        const reader = llmResponse.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = '';

        // Helper function to send text synchronously (matching Python input_streaming.py behavior)
        const sendText = async (text) => {
            const response = await fetch(`/api/input-stream/${request_id}/text`, {
                method: 'POST',
                body: new URLSearchParams({ text: text }),
                signal: llmAbortController.signal,
            });
            if (!response.ok) {
                throw new Error(`Failed to send text: ${response.status}`);
            }
        };

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;

                const data = line.slice(6);
                if (data === '[DONE]') break;

                try {
                    const parsed = JSON.parse(data);
                    const content = parsed.choices?.[0]?.delta?.content || '';

                    if (content) {
                        fullResponse += content;
                        updateLastAssistantMessage(fullResponse, true);

                        // Send each token directly to TTS (matching Python input_streaming.py behavior)
                        await sendText(content);
                    }
                } catch (e) {
                    // Ignore parse errors
                }
            }
        }

        // Finalize assistant message
        updateLastAssistantMessage(fullResponse, false);

        // Add to chat history
        if (fullResponse) {
            chatMessages.push({ role: 'assistant', content: fullResponse });
        }

        // Signal end of text input (audio continues streaming)
        await fetch(`/api/input-stream/${request_id}/end`, {
            method: 'POST',
            signal: llmAbortController.signal,
        });

        // Wait for audio streaming to complete
        await audioPromise;

        // Wait for playback to finish
        if (audioState.isPlaying && llmAudioContext) {
            const remainingTime = audioState.nextPlayTime - llmAudioContext.currentTime;
            if (remainingTime > 0) {
                elements.llmPlaybackStatus.textContent = '▶️ Playing';
                await new Promise(resolve => setTimeout(resolve, remainingTime * 1000));
            }
        }

        if (audioState.error) {
            throw new Error(audioState.error);
        }

        elements.llmPlaybackStatus.textContent = '✅ Finished';

    } catch (error) {
        if (error.name === 'AbortError') {
            elements.llmPlaybackStatus.textContent = '⏹️ Stopped';
        } else {
            console.error('LLM Chat error:', error);
            elements.llmPlaybackStatus.textContent = '❌ Error';

            // Remove the streaming message if it exists
            const streamingMsg = elements.chatContainer.querySelector('.chat-message.streaming');
            if (streamingMsg) {
                streamingMsg.remove();
            }

            // Show error in chat
            addChatMessage('assistant', `Error: ${error.message}`);
        }
    } finally {
        isLLMStreaming = false;
        llmAbortController = null;
        elements.chatInput.disabled = false;
        elements.chatSendBtn.disabled = false;
        elements.chatStopBtn.disabled = true;
    }
}

/**
 * Stop ongoing LLM chat
 */
function stopLLMChat() {
    if (llmAbortController) {
        llmAbortController.abort();
    }
    isLLMStreaming = false;
    elements.chatStopBtn.disabled = true;
}

/**
 * Clear chat history
 */
function clearChatHistory() {
    chatMessages = [];
    elements.chatContainer.innerHTML = '<div class="chat-placeholder">Chat messages will appear here. Type a message below to start.</div>';
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', init);
