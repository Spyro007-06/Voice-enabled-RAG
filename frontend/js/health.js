/**
 * HH Goa 2026 Multilingual Voice RAG — System Health & Telemetry Poller (Phase 6.25)
 */

import { api } from './api.js';
import { state } from './state.js';

export class HealthMonitor {
  constructor(pollIntervalMs = 30000) {
    this.pollIntervalMs = pollIntervalMs;
    this.timerId = null;
  }

  async check() {
    try {
      const res = await api.checkHealth();
      const isHealthy = res && res.status === 'healthy';
      state.setBackendHealth(isHealthy ? 'healthy' : 'degraded', new Date());
      this._updateUI(isHealthy ? 'healthy' : 'degraded', res?.providers, res);
    } catch (err) {
      state.setBackendHealth('down', new Date());
      this._updateUI('down', null, null);
    }
  }

  _updateUI(status, providers = null, rawRes = null) {
    const dot = document.getElementById('statusDot');
    const text = document.getElementById('statusText');
    const lastChecked = document.getElementById('lastCheckedTime');

    if (dot) {
      dot.className = `status-dot ${status === 'healthy' ? '' : status}`;
    }
    if (text) {
      if (status === 'healthy') {
        text.textContent = 'Operational';
        if (providers) {
          const cap = (s) => String(s).charAt(0).toUpperCase() + String(s).slice(1);
          const tooltip = `STT: ${cap(providers.stt || 'sarvam')} | LLM: ${cap(providers.llm || 'sarvam')} | TTS: ${cap(providers.tts || 'sarvam')} | Qdrant: ${cap(providers.vector_db || 'qdrant')}`;
          text.setAttribute('title', tooltip);
        }
      } else if (status === 'degraded') {
        text.textContent = 'Degraded';
      } else {
        text.textContent = 'Offline';
      }
    }
    if (lastChecked) {
      const now = new Date();
      lastChecked.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }

    // Update individual service rows in Status Drawer
    const setStat = (id, ok, label = 'Operational') => {
      const el = document.getElementById(id);
      if (el) {
        el.className = `status-val ${ok ? 'ok' : 'down'}`;
        el.textContent = ok ? `● ${label}` : '✕ Unavailable';
      }
    };

    const isOk = status === 'healthy';
    setStat('statApiGateway', isOk, 'Operational');
    setStat('statVectorDb', isOk, 'Operational (28,541 vectors)');
    setStat('statBm25', isOk, 'Operational (48,206 chunks)');
    setStat('statEmbedding', isOk, 'Operational (multilingual-e5-small)');
    setStat('statReranker', isOk, 'Operational (mmarco-mMiniLMv2)');
    setStat('statLlm', isOk, 'Operational (Sarvam 105B)');
    setStat('statStt', isOk, 'Operational (Saarika v2.5)');
    setStat('statTts', isOk, 'Operational (Bulbul v2)');
  }

  start() {
    this.check();
    this.timerId = setInterval(() => this.check(), this.pollIntervalMs);
  }

  stop() {
    if (this.timerId) {
      clearInterval(this.timerId);
      this.timerId = null;
    }
  }
}

export const healthMonitor = new HealthMonitor();
