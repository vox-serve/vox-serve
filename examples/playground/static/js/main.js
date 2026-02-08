/**
 * VoxServe Playground - Main Application Logic
 */

// Global state
let serverRunning = false;
let serverPort = 8000;
let modelCapabilities = {};
let audioPlayer = null;
let logPollingInterval = null;
let lastLogCount = 0;

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
    ttfaPlayback: document.getElementById('ttfa-playback'),
    stopAudioBtn: document.getElementById('stop-audio-btn'),
};

/**
 * Initialize the application
 */
async function init() {
    // Initialize audio player
    audioPlayer = new StreamingAudioPlayer();
    audioPlayer.onStatusChange = (status) => {
        elements.playbackStatus.textContent = status;
        elements.playbackStatus.classList.toggle('playing', status === 'Playing...');
    };
    audioPlayer.onTTFANetwork = (ttfa) => {
        elements.ttfaNetwork.textContent = `${Math.round(ttfa)}ms`;
    };
    audioPlayer.onTTFAPlayback = (ttfa) => {
        elements.ttfaPlayback.textContent = `${Math.round(ttfa)}ms`;
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
                GPU ${gpu.index}: ${gpu.name} (${gpu.memory_free_gb}/${gpu.memory_total_gb} GB)
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
    elements.modelSelect.addEventListener('change', updateModelParams);

    // Logs
    elements.clearLogsBtn.addEventListener('click', clearLogs);

    // TTS request
    elements.generateBtn.addEventListener('click', generateAudio);

    // Audio player
    elements.stopAudioBtn.addEventListener('click', () => {
        audioPlayer.stop();
        elements.stopAudioBtn.disabled = true;
    });
}

/**
 * Update UI based on selected model's capabilities
 */
function updateModelParams() {
    const modelId = elements.modelSelect.value;
    const caps = modelCapabilities[modelId] || {};

    // Show/hide model params section
    const hasParams = caps.supports_language || caps.supports_speaker ||
                      caps.supports_ref_text || caps.supports_instruct;
    elements.modelParams.classList.toggle('hidden', !hasParams);

    // Show/hide individual params
    elements.languageGroup.style.display = caps.supports_language ? '' : 'none';
    elements.speakerGroup.style.display = caps.supports_speaker ? '' : 'none';
    elements.refTextGroup.style.display = caps.supports_ref_text ? '' : 'none';
    elements.instructGroup.style.display = caps.supports_instruct ? '' : 'none';

    // Update audio upload hint
    if (caps.requires_audio) {
        elements.audioUploadGroup.querySelector('label').textContent = 'Reference Audio (required)';
    } else if (caps.supports_audio_input) {
        elements.audioUploadGroup.querySelector('label').textContent = 'Reference Audio (optional)';
    } else {
        elements.audioUploadGroup.style.display = 'none';
    }
    elements.audioUploadGroup.style.display = caps.supports_audio_input ? '' : 'none';
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
            serverRunning = true;
            serverPort = config.port;
            setStatus(result.message, 'success');
            updateIndicator('running');
            elements.startServerBtn.disabled = true;
            elements.stopServerBtn.disabled = false;
            elements.generateBtn.disabled = false;
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

    // Reset TTFA displays
    elements.ttfaNetwork.textContent = '-';
    elements.ttfaPlayback.textContent = '-';

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

        if (status.running !== serverRunning) {
            serverRunning = status.running;
            if (serverRunning) {
                serverPort = status.port;
                updateIndicator('running');
                elements.startServerBtn.disabled = true;
                elements.stopServerBtn.disabled = false;
                elements.generateBtn.disabled = false;
                const uptime = formatUptime(status.uptime_seconds);
                setStatus(`Running ${status.model} on port ${status.port} | uptime: ${uptime}`, 'success');
                startLogPolling();
            } else {
                updateIndicator('stopped');
                elements.startServerBtn.disabled = false;
                elements.stopServerBtn.disabled = true;
                elements.generateBtn.disabled = true;
                stopLogPolling();
            }
        } else if (serverRunning && status.uptime_seconds) {
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
    }

    // Auto-scroll if enabled
    if (elements.autoScroll.checked) {
        elements.logsContainer.scrollTop = elements.logsContainer.scrollHeight;
    }
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

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', init);
