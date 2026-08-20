/**
 * HH Goa 2026 Multilingual Voice RAG — Chat Controller & Query Flow (Phase 6.25 Dataset-Centric)
 */

import { api, ApiError } from './api.js';
import { state } from './state.js';
import { renderMessage, showToast } from './ui.js';

export class ChatController {
  constructor(messagesContainerId = 'chatMessages', heroContainerId = 'welcomeState') {
    this.container = document.getElementById(messagesContainerId);
    this.hero = document.getElementById(heroContainerId) || document.getElementById('welcomeHero');
    this.activeStageElement = null;
  }

  _hideHero() {
    if (this.hero && this.hero.style.display !== 'none') {
      this.hero.style.display = 'none';
    }
  }

  _showHero() {
    if (this.hero) {
      this.hero.style.display = '';
    }
  }

  _scrollToBottom() {
    if (this.container) {
      this.container.scrollTop = this.container.scrollHeight;
    }
  }

  _showLoadingIndicator(stageText = 'Thinking...') {
    this._hideHero();
    this._removeLoadingIndicator();

    const row = document.createElement('div');
    row.className = 'message-row assistant animate-fade-in';
    row.id = 'loadingStageIndicator';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar assistant';
    avatar.textContent = 'AI';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble assistant';

    const stageWrap = document.createElement('div');
    stageWrap.className = 'loading-stage-indicator';

    const textSpan = document.createElement('span');
    textSpan.id = 'loadingStageText';
    textSpan.textContent = stageText;

    const dots = document.createElement('div');
    dots.className = 'loading-dots';
    dots.innerHTML = `
      <div class="loading-dot"></div>
      <div class="loading-dot"></div>
      <div class="loading-dot"></div>
    `;

    stageWrap.appendChild(textSpan);
    stageWrap.appendChild(dots);
    bubble.appendChild(stageWrap);
    row.appendChild(avatar);
    row.appendChild(bubble);

    this.container.appendChild(row);
    this.activeStageElement = row;
    this._scrollToBottom();
  }

  _updateLoadingText(stageText) {
    const textSpan = document.getElementById('loadingStageText');
    if (textSpan) {
      textSpan.textContent = stageText;
    }
  }

  _removeLoadingIndicator() {
    if (this.activeStageElement) {
      this.activeStageElement.remove();
      this.activeStageElement = null;
    }
    const elem = document.getElementById('loadingStageIndicator');
    if (elem) elem.remove();
  }

  /**
   * Reset the chat UI and state back to empty welcome screen
   */
  clear() {
    this._removeLoadingIndicator();
    if (this.container) {
      const rows = this.container.querySelectorAll('.message-row');
      rows.forEach((r) => r.remove());
    }
    this._showHero();
    state.setCurrentAudio(null, null);
    state.setState({ messages: [] });
  }

  /**
   * Handle text question submission — uses SSE streaming (/api/ask-stream) with
   * real pipeline-stage events and token streaming. Falls back to buffered /api/ask.
   */
  async submitTextQuery(query) {
    if (!query || !query.trim()) return;

    this._hideHero();

    const currentLang = state.getState().language;

    // 1. Add User Message
    const userMsg = state.addMessage({
      sender: 'user',
      text: query.trim(),
      language: currentLang,
    });
    this.container.appendChild(renderMessage(userMsg));
    this._scrollToBottom();

    // 2. Check if SSE / ReadableStream is supported
    const supportsStream = typeof ReadableStream !== 'undefined' && typeof fetch !== 'undefined';

    if (!supportsStream) {
      return this._submitTextQueryBuffered(query, currentLang);
    }

    // 3. SSE streaming path
    this._showLoadingIndicator('Retrieving sources...');
    state.setLoadingStage('retrieving');

    let streamController = null;
    let streamTokenBubble = null;
    let streamTokenContent = null;
    let accumulatedTokens = '';

    const _ensureTokenBubble = () => {
      if (streamTokenBubble) return;
      this._removeLoadingIndicator();

      const row = document.createElement('div');
      row.className = 'message-row assistant animate-fade-in';
      row.id = 'streamingResponseRow';

      const avatar = document.createElement('div');
      avatar.className = 'message-avatar assistant';
      avatar.textContent = 'AI';

      const bubble = document.createElement('div');
      bubble.className = 'message-bubble assistant';

      const contentDiv = document.createElement('div');
      contentDiv.className = 'message-content stream-typing';
      if (currentLang) contentDiv.setAttribute('lang', currentLang);
      contentDiv.textContent = '';

      bubble.appendChild(contentDiv);
      row.appendChild(avatar);
      row.appendChild(bubble);
      this.container.appendChild(row);

      streamTokenBubble = row;
      streamTokenContent = contentDiv;
      this._scrollToBottom();
    };

    return new Promise((resolve) => {
      streamController = api.askQuestionStream(
        query,
        currentLang,
        5,
        {
          onStage: (stage, message) => {
            state.setLoadingStage(stage);
            this._updateLoadingText(message || stage);
          },

          onToken: (token) => {
            _ensureTokenBubble();
            accumulatedTokens += token;
            if (streamTokenContent) {
              streamTokenContent.textContent = accumulatedTokens;
              this._scrollToBottom();
            }
          },

          onDone: (payload) => {
            const streamRow = document.getElementById('streamingResponseRow');
            if (streamRow) streamRow.remove();
            this._removeLoadingIndicator();

            const isRefusal = !payload.grounded || (payload.citations && payload.citations.length === 0);

            const assistantMsg = state.addMessage({
              sender: 'assistant',
              text: payload.answer || accumulatedTokens || "I don't have enough information in the retrieved context to answer that.",
              grounded: Boolean(payload.grounded),
              confidence: payload.confidence || 0,
              citations: payload.citations || [],
              citationProvenance: payload.citation_provenance || [],
              language: currentLang,
              latencyMs: typeof payload.latency_ms === 'number'
                ? { total: payload.latency_ms }
                : payload.latency_ms,
              error: null,
              showCrossLanguageRetry: isRefusal,
              originalQuery: query.trim(),
            });

            this.container.appendChild(
              renderMessage(assistantMsg, (q) => this.submitCrossLanguageQuery(q))
            );
            this._scrollToBottom();
            state.setLoadingStage(null);
            resolve();
          },

          onError: (message, code) => {
            const streamRow = document.getElementById('streamingResponseRow');
            if (streamRow) streamRow.remove();
            this._removeLoadingIndicator();

            if (!accumulatedTokens) {
              state.setLoadingStage(null);
              this._submitTextQueryBuffered(query, currentLang).then(resolve);
              return;
            }

            showToast(message, 'error');
            state.setLoadingStage(null);
            resolve();
          },
        }
      );
    });
  }

  /**
   * Buffered (non-streaming) text query — fallback
   */
  async _submitTextQueryBuffered(query, currentLang) {
    this._showLoadingIndicator('Retrieving sources...');
    state.setLoadingStage('retrieving');

    try {
      const responsePromise = api.askQuestion(query, currentLang);

      setTimeout(() => {
        if (state.getState().loadingStage) {
          this._updateLoadingText('Ranking context...');
          state.setLoadingStage('ranking');
        }
      }, 350);

      setTimeout(() => {
        if (state.getState().loadingStage) {
          this._updateLoadingText('Generating grounded answer...');
          state.setLoadingStage('generating');
        }
      }, 700);

      const response = await responsePromise;
      this._removeLoadingIndicator();

      const isRefusal = !response.grounded || (response.citations && response.citations.length === 0);

      const assistantMsg = state.addMessage({
        sender: 'assistant',
        text: response.answer || "I don't have enough information in the retrieved context to answer that.",
        grounded: Boolean(response.grounded),
        confidence: response.confidence,
        citations: response.citations || [],
        citationProvenance: response.citation_provenance || [],
        language: currentLang,
        latencyMs: response.latency_ms,
        error: response.error,
        showCrossLanguageRetry: isRefusal,
        originalQuery: query.trim(),
      });

      this.container.appendChild(
        renderMessage(assistantMsg, (q) => this.submitCrossLanguageQuery(q))
      );
      this._scrollToBottom();
    } catch (err) {
      this._removeLoadingIndicator();
      const errMsg = err instanceof ApiError ? err.message : 'Something went wrong. Please try again.';
      showToast(errMsg, 'error');

      const assistantErr = state.addMessage({
        sender: 'assistant',
        text: `⚠️ ${errMsg}`,
        grounded: false,
        citations: [],
        language: currentLang,
        error: errMsg,
      });
      this.container.appendChild(renderMessage(assistantErr));
      this._scrollToBottom();
    } finally {
      state.setLoadingStage(null);
    }
  }

  /**
   * Handle Cross-Language Retrieval Fallback (queries with language=None across full multilingual corpus)
   */
  async submitCrossLanguageQuery(originalQuery) {
    if (!originalQuery || !originalQuery.trim()) return;

    this._showLoadingIndicator('Searching across all 5 languages...');
    state.setLoadingStage('retrieving');

    try {
      const responsePromise = api.askQuestion(originalQuery, null);

      setTimeout(() => {
        if (state.getState().loadingStage) {
          this._updateLoadingText('Ranking cross-lingual candidates...');
          state.setLoadingStage('ranking');
        }
      }, 350);

      setTimeout(() => {
        if (state.getState().loadingStage) {
          this._updateLoadingText('Synthesizing cross-lingual answer...');
          state.setLoadingStage('generating');
        }
      }, 700);

      const response = await responsePromise;
      this._removeLoadingIndicator();

      const assistantMsg = state.addMessage({
        sender: 'assistant',
        text: response.answer || "I don't have enough information in the retrieved context to answer that.",
        grounded: Boolean(response.grounded),
        confidence: response.confidence,
        citations: response.citations || [],
        citationProvenance: response.citation_provenance || [],
        language: response.language || null,
        latencyMs: response.latency_ms,
        error: response.error,
        isCrossLanguage: true,
      });

      this.container.appendChild(renderMessage(assistantMsg));
      this._scrollToBottom();
    } catch (err) {
      this._removeLoadingIndicator();
      const errMsg = err instanceof ApiError ? err.message : 'Cross-language search failed.';
      showToast(errMsg, 'error');
    } finally {
      state.setLoadingStage(null);
    }
  }

  /**
   * Handle recorded audio blob submission
   */
  async submitVoiceQuery(audioBlob, filename = 'recording.webm', onRetryVoice = null) {
    if (!audioBlob) return;

    this._hideHero();

    const currentLang = state.getState().language;

    // 1. Show Uploading / Transcribing
    this._showLoadingIndicator('Transcribing voice (Saarika v2.5)...');
    state.setLoadingStage('transcribing');

    try {
      setTimeout(() => {
        if (state.getState().loadingStage) {
          this._updateLoadingText('Retrieving passage sources...');
          state.setLoadingStage('retrieving');
        }
      }, 500);

      setTimeout(() => {
        if (state.getState().loadingStage) {
          this._updateLoadingText('Ranking cross-encoder candidates...');
          state.setLoadingStage('ranking');
        }
      }, 1000);

      setTimeout(() => {
        if (state.getState().loadingStage) {
          this._updateLoadingText('Generating grounded answer (Sarvam 105B)...');
          state.setLoadingStage('generating');
        }
      }, 1500);

      setTimeout(() => {
        if (state.getState().loadingStage) {
          this._updateLoadingText('Synthesizing voice response (Bulbul v2)...');
          state.setLoadingStage('synthesizing');
        }
      }, 2000);

      const response = await api.askVoice(audioBlob, currentLang, 5, filename);

      this._removeLoadingIndicator();

      // 2. Render user's transcribed query explicitly
      if (response.transcript) {
        const userMsg = state.addMessage({
          sender: 'user',
          text: response.transcript,
          language: response.language || currentLang,
          isVoice: true,
        });
        this.container.appendChild(renderMessage(userMsg));
        this._scrollToBottom();
      }

      // 3. Handle Audio Payload & TTS Status
      let status = response.status || 'success';
      const audioB64 = (response.audio && response.audio.audio_base64) || response.audio_base64 || null;
      const hasAudio = Boolean(audioB64 && ((response.audio && response.audio.available !== false) || !response.audio));
      const audioFormat = (response.audio && response.audio.format) || 'wav';
      const ttsFailed = !hasAudio && (status === 'partial_success' || (response.audio && response.audio.available === false));

      if (ttsFailed && response.answer) {
        showToast('Text answer ready — voice synthesis is temporarily unavailable.', 'warning', 5000);
      }

      // 4. Render Assistant Response
      const assistantMsg = state.addMessage({
        sender: 'assistant',
        text: response.answer || "I don't have enough information in the retrieved context to answer that.",
        grounded: Boolean(response.grounded),
        confidence: response.confidence || 0,
        citations: response.citations || [],
        citationProvenance: response.citation_provenance || [],
        language: response.language || currentLang,
        status: status,
        ttsUnavailable: ttsFailed,
        audioBase64: hasAudio ? audioB64 : null,
        audioFormat: audioFormat,
        latencyMs: response.latency_ms,
        error: response.error,
        originalQuery: response.transcript || '',
      });

      this.container.appendChild(
        renderMessage(assistantMsg, (q) => this.submitCrossLanguageQuery(q), null, onRetryVoice)
      );
      this._scrollToBottom();
    } catch (err) {
      this._removeLoadingIndicator();
      const errMsg = err instanceof ApiError ? err.message : 'Voice request failed. Please try again.';
      showToast(errMsg, 'error');

      const isSTTErr = errMsg.toLowerCase().includes('speech') || errMsg.toLowerCase().includes('transcrib') || errMsg.toLowerCase().includes('audio');

      const assistantErr = state.addMessage({
        sender: 'assistant',
        text: `⚠️ ${errMsg}`,
        grounded: false,
        citations: [],
        language: currentLang,
        error: errMsg,
        isSTTFailure: isSTTErr,
      });
      this.container.appendChild(renderMessage(assistantErr, null, null, onRetryVoice));
      this._scrollToBottom();
    } finally {
      state.setLoadingStage(null);
    }
  }
}
