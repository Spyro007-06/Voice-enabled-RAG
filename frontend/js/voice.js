/**
 * HH Goa 2026 Multilingual Voice RAG — Voice Recording & Web Audio Management
 */

import { state, APP_STATE } from './state.js';

const MAX_RECORDING_DURATION_MS = 60000; // 60 seconds hard limit
const MIN_RECORDING_DURATION_MS = 400;   // Minimum threshold to prevent accidental click

export class VoiceManager {
  constructor() {
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.audioStream = null;
    this.audioContext = null;
    this.analyser = null;
    this.timerInterval = null;
    this.maxDurationTimer = null;
    this.recordingStartTime = 0;
    this.animationFrameId = null;
    this.waveformCanvas = null;
    this.canvasContext = null;
    this._isCleaning = false;
  }

  initCanvas(canvasElement) {
    this.waveformCanvas = canvasElement;
    if (canvasElement) {
      this.canvasContext = canvasElement.getContext('2d');
    }
  }

  getSupportedMimeType() {
    const types = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/ogg;codecs=opus',
      'audio/mp4',
      'audio/wav',
    ];
    for (const type of types) {
      if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(type)) {
        return type;
      }
    }
    return '';
  }

  async checkMicrophoneAvailable() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return { available: false, reason: 'UNSUPPORTED' };
    }

    try {
      if (navigator.mediaDevices.enumerateDevices) {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const audioInputs = devices.filter((d) => d.kind === 'audioinput');
        if (devices.length > 0 && audioInputs.length === 0) {
          return { available: false, reason: 'NO_MIC_DETECTED' };
        }
      }
    } catch (_) {
      // Some browsers block enumeration prior to permission grant; proceed to getUserMedia
    }

    return { available: true };
  }

  async startRecording({ onTimerTick, onAutoStop, onError }) {
    if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
      return;
    }

    state.setState(APP_STATE.REQUESTING_MIC_PERMISSION);

    // Initial device availability check
    const check = await this.checkMicrophoneAvailable();
    if (!check.available && check.reason === 'NO_MIC_DETECTED') {
      const err = new Error('No microphone device was detected on this system.');
      err.code = 'NO_MIC_DETECTED';
      this.cleanup();
      state.setState(APP_STATE.ERROR);
      if (onError) onError(err);
      return;
    }

    try {
      this.audioStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      // Verify that at least one audio track exists and is active
      const audioTracks = this.audioStream.getAudioTracks();
      if (!audioTracks || audioTracks.length === 0) {
        throw Object.assign(new Error('No active audio track found in device stream.'), { code: 'NO_MIC_DETECTED' });
      }

      const mimeType = this.getSupportedMimeType();
      const options = mimeType ? { mimeType } : undefined;
      this.mediaRecorder = new MediaRecorder(this.audioStream, options);
      this.audioChunks = [];

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };

      // Initialize Web Audio Context and Analyser
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) {
        this.audioContext = new AudioCtx();
        if (this.audioContext.state === 'suspended') {
          await this.audioContext.resume();
        }
        const source = this.audioContext.createMediaStreamSource(this.audioStream);
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 256;
        this.analyser.smoothingTimeConstant = 0.8;
        source.connect(this.analyser);
      }

      this.mediaRecorder.start(100);
      this.recordingStartTime = Date.now();
      state.setState(APP_STATE.RECORDING);

      // Start Waveform Rendering
      this.startWaveformLoop();

      // Start Elapsed Timer
      if (this.timerInterval) clearInterval(this.timerInterval);
      this.timerInterval = setInterval(() => {
        const elapsedSeconds = Math.floor((Date.now() - this.recordingStartTime) / 1000);
        if (onTimerTick) onTimerTick(elapsedSeconds);
      }, 250);

      // Enforce 60s max duration auto-stop
      if (this.maxDurationTimer) clearTimeout(this.maxDurationTimer);
      this.maxDurationTimer = setTimeout(() => {
        if (this.isRecording()) {
          if (onAutoStop) onAutoStop();
        }
      }, MAX_RECORDING_DURATION_MS);

    } catch (err) {
      this.cleanup();
      state.setState(APP_STATE.ERROR);

      // Normalize error code
      if (
        err.name === 'NotFoundError' ||
        err.name === 'DevicesNotFoundError' ||
        err.code === 'NO_MIC_DETECTED' ||
        err.message?.toLowerCase().includes('device not found') ||
        err.message?.toLowerCase().includes('requested device not found')
      ) {
        err.code = 'NO_MIC_DETECTED';
      } else if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        err.code = 'PERMISSION_DENIED';
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        err.code = 'MIC_IN_USE';
      }

      if (onError) onError(err);
    }
  }

  isRecording() {
    return this.mediaRecorder && this.mediaRecorder.state === 'recording';
  }

  startWaveformLoop() {
    if (!this.analyser || !this.waveformCanvas || !this.canvasContext) return;

    const bufferLength = this.analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    const canvas = this.waveformCanvas;
    const ctx = this.canvasContext;

    const draw = () => {
      if (!this.isRecording()) return;

      this.animationFrameId = requestAnimationFrame(draw);
      this.analyser.getByteFrequencyData(dataArray);

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const barWidth = (canvas.width / bufferLength) * 2.2;
      let x = 0;

      // Create luminous linear gradient for audio bars
      const gradient = ctx.createLinearGradient(0, 0, canvas.width, 0);
      gradient.addColorStop(0, '#10b981');
      gradient.addColorStop(0.5, '#06b6d4');
      gradient.addColorStop(1, '#6366f1');

      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * (canvas.height * 0.85);

        ctx.fillStyle = gradient;
        const y = (canvas.height - barHeight) / 2;
        
        // Draw smooth rounded bar
        const bw = Math.max(barWidth - 2, 2);
        const bh = Math.max(barHeight, 3);
        ctx.beginPath();
        ctx.roundRect ? ctx.roundRect(x, y, bw, bh, 3) : ctx.rect(x, y, bw, bh);
        ctx.fill();

        x += barWidth + 2;
        if (x > canvas.width) break;
      }
    };

    draw();
  }

  async stopRecording() {
    if (!this.mediaRecorder || this.mediaRecorder.state !== 'recording') {
      return null;
    }

    state.setState(APP_STATE.STOPPING);

    if (this.timerInterval) clearInterval(this.timerInterval);
    if (this.maxDurationTimer) clearTimeout(this.maxDurationTimer);
    if (this.animationFrameId) cancelAnimationFrame(this.animationFrameId);

    const elapsedMs = Date.now() - this.recordingStartTime;

    const audioBlob = await new Promise((resolve) => {
      this.mediaRecorder.onstop = () => {
        const mimeType = this.mediaRecorder.mimeType || 'audio/webm';
        const blob = new Blob(this.audioChunks, { type: mimeType });
        resolve(blob);
      };
      this.mediaRecorder.stop();
    });

    this.cleanup();

    if (elapsedMs < MIN_RECORDING_DURATION_MS || audioBlob.size < 300) {
      state.setState(APP_STATE.STT_ERROR);
      throw new Error('Recording was too short or silent. Please try speaking again.');
    }

    state.lastRecordedBlob = audioBlob;
    return audioBlob;
  }

  cancelRecording() {
    if (this.timerInterval) clearInterval(this.timerInterval);
    if (this.maxDurationTimer) clearTimeout(this.maxDurationTimer);
    if (this.animationFrameId) cancelAnimationFrame(this.animationFrameId);

    if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
      try {
        this.mediaRecorder.stop();
      } catch (_) {}
    }

    this.cleanup();
    state.setState(APP_STATE.IDLE);
  }

  cleanup() {
    if (this._isCleaning) return;
    this._isCleaning = true;

    try {
      if (this.audioStream) {
        this.audioStream.getTracks().forEach((track) => {
          try {
            track.stop();
          } catch (_) {}
        });
        this.audioStream = null;
      }

      if (this.audioContext && this.audioContext.state !== 'closed') {
        this.audioContext.close().catch(() => {});
        this.audioContext = null;
      }

      if (this.waveformCanvas && this.canvasContext) {
        this.canvasContext.clearRect(0, 0, this.waveformCanvas.width, this.waveformCanvas.height);
      }
    } finally {
      this._isCleaning = false;
    }
  }
}

export const voiceManager = new VoiceManager();
