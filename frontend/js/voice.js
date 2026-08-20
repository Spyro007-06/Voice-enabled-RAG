/**
 * HH Goa 2026 Multilingual Voice RAG — Voice Recording & Web Audio Management
 * Phase 6.21: Max-duration enforcement, min-duration guard, page-visibility abort
 */

import { state } from './state.js';

const MAX_RECORDING_DURATION_MS = 60000; // 60 seconds hard limit
const MIN_RECORDING_DURATION_MS = 500;   // 500ms minimum to avoid accidental taps

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
    this.onWaveformData = null;
    this._cleaning = false; // double-cleanup guard
    this._onVisibilityChange = null; // page-visibility abort handler

    // Bind auto-stop callbacks once at construction time
    this._onMaxDurationReached = null;
  }

  /**
   * Determine best browser-supported audio mime type
   */
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

  /**
   * Start microphone recording.
   * @param {function} onWaveformCallback - receives Uint8Array frequency data per frame
   * @param {function} onMaxDuration - called when 60s limit is reached (auto-stop trigger)
   * @param {function} onRemainingTime - called every second with remaining seconds
   */
  async startRecording(onWaveformCallback = null, onMaxDuration = null, onRemainingTime = null) {
    if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
      return;
    }

    this.onWaveformData = onWaveformCallback;
    this._onMaxDurationReached = onMaxDuration;

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Microphone recording is not supported in this browser.');
    }

    try {
      this.audioStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
    } catch (err) {
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        throw new Error('Microphone permission was denied. Please allow microphone access.');
      }
      throw new Error(`Microphone access failed: ${err.message}`);
    }

    // Set up Web Audio API Analyser for live visualizer
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) {
        this.audioContext = new AudioCtx();
        const source = this.audioContext.createMediaStreamSource(this.audioStream);
        this.analyser = this.audioContext.createAnalyser();
        this.analyser.fftSize = 64;
        source.connect(this.analyser);
        this._startVisualizerLoop();
      }
    } catch (visErr) {
      console.warn('Visualizer AudioContext init skipped:', visErr);
    }

    const mimeType = this.getSupportedMimeType();
    const options = mimeType ? { mimeType } : {};

    this.audioChunks = [];
    this._cleaning = false;
    this.mediaRecorder = new MediaRecorder(this.audioStream, options);

    this.mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        this.audioChunks.push(event.data);
      }
    };

    this.recordingStartTime = Date.now();
    this.mediaRecorder.start(100); // collect in 100ms chunks

    // Start timer interval — updates state + optional remaining-time callback
    this.timerInterval = setInterval(() => {
      const elapsed = Date.now() - this.recordingStartTime;
      const durationSec = Math.floor(elapsed / 1000);
      state.setRecording(true, durationSec);

      if (onRemainingTime) {
        const remainingSec = Math.max(0, Math.ceil((MAX_RECORDING_DURATION_MS - elapsed) / 1000));
        onRemainingTime(remainingSec);
      }
    }, 500);

    state.setRecording(true, 0);

    // Max-duration hard stop at 60 seconds
    this.maxDurationTimer = setTimeout(() => {
      if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
        if (this._onMaxDurationReached) {
          this._onMaxDurationReached();
        }
      }
    }, MAX_RECORDING_DURATION_MS);

    // Page visibility abort — cancel recording on tab switch / navigation
    this._onVisibilityChange = () => {
      if (document.hidden && this.mediaRecorder && this.mediaRecorder.state === 'recording') {
        this.cancelRecording();
      }
    };
    document.addEventListener('visibilitychange', this._onVisibilityChange);
  }

  _startVisualizerLoop() {
    if (!this.analyser) return;
    const dataArray = new Uint8Array(this.analyser.frequencyBinCount);

    const update = () => {
      if (!this.analyser || state.getState().isRecording === false) return;
      this.analyser.getByteFrequencyData(dataArray);

      if (this.onWaveformData) {
        this.onWaveformData(dataArray);
      }
      this.animationFrameId = requestAnimationFrame(update);
    };
    update();
  }

  /**
   * Stop recording and resolve the audio blob.
   * Rejects if duration is below MIN_RECORDING_DURATION_MS.
   */
  async stopRecording() {
    return new Promise((resolve, reject) => {
      if (!this.mediaRecorder || this.mediaRecorder.state === 'inactive') {
        this._cleanup();
        reject(new Error('No active recording.'));
        return;
      }

      // Enforce minimum duration guard
      const elapsed = Date.now() - this.recordingStartTime;
      if (elapsed < MIN_RECORDING_DURATION_MS) {
        this._cleanup();
        reject(new Error('Recording too short. Please hold the button for at least half a second.'));
        return;
      }

      this.mediaRecorder.onstop = () => {
        const mimeType = this.mediaRecorder.mimeType || 'audio/webm';
        const audioBlob = new Blob(this.audioChunks, { type: mimeType });
        const chunksCount = this.audioChunks.length;
        this._cleanup();

        // Dev-only structured logging (no raw audio logged)
        if (typeof console !== 'undefined' && console.log) {
          console.log(`VOICE RECORDING mime=${mimeType} chunks=${chunksCount} bytes=${audioBlob.size} duration=${elapsed}ms`);
        }

        if (audioBlob.size === 0) {
          reject(new Error('Recording produced an empty audio payload.'));
        } else {
          let extension = 'webm';
          if (mimeType.includes('ogg')) extension = 'ogg';
          else if (mimeType.includes('wav')) extension = 'wav';
          else if (mimeType.includes('mp4') || mimeType.includes('m4a')) extension = 'mp4';

          resolve({
            blob: audioBlob,
            mimeType,
            filename: `query.${extension}`,
            durationMs: elapsed,
          });
        }
      };


      try {
        this.mediaRecorder.stop();
      } catch (err) {
        this._cleanup();
        reject(err);
      }
    });
  }

  /**
   * Cancel and discard recording
   */
  cancelRecording() {
    if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
      try {
        this.mediaRecorder.stop();
      } catch (_) {}
    }
    this._cleanup();
    state.setRecording(false, 0);
  }

  _cleanup() {
    // Double-cleanup guard: prevent race between stop + cancel
    if (this._cleaning) return;
    this._cleaning = true;

    if (this.maxDurationTimer) {
      clearTimeout(this.maxDurationTimer);
      this.maxDurationTimer = null;
    }
    if (this.timerInterval) {
      clearInterval(this.timerInterval);
      this.timerInterval = null;
    }
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    if (this.audioStream) {
      this.audioStream.getTracks().forEach((track) => track.stop());
      this.audioStream = null;
    }
    if (this.audioContext && this.audioContext.state !== 'closed') {
      try {
        this.audioContext.close();
      } catch (_) {}
      this.audioContext = null;
    }
    if (this._onVisibilityChange) {
      document.removeEventListener('visibilitychange', this._onVisibilityChange);
      this._onVisibilityChange = null;
    }
    this.analyser = null;
    state.setRecording(false, 0);
    this._cleaning = false;
  }

  /** Expose max duration for UI countdown display */
  get maxDurationSeconds() {
    return MAX_RECORDING_DURATION_MS / 1000;
  }
}

export const voiceManager = new VoiceManager();

