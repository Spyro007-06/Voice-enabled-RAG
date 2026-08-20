/**
 * HH Goa 2026 Multilingual Voice RAG — Centralized API Client
 *
 * Handles HTTP requests, timeouts, AbortController cancellation,
 * 429 rate-limiting with Retry-After, 502 provider failures, and payload encoding.
 */

const DEFAULT_TIMEOUT_MS = 45000;

export class ApiError extends Error {
  constructor(message, status = 500, retryAfter = null, details = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.retryAfter = retryAfter;
    this.details = details;
  }
}

export class ApiClient {
  constructor(baseUrl = '') {
    // If running standalone (e.g. localhost:3000), default fallback to localhost:8000
    if (!baseUrl && typeof window !== 'undefined') {
      const isStandaloneDev = window.location.port !== '8000';
      this.baseUrl = window.API_BASE_URL || (isStandaloneDev ? 'http://localhost:8000' : '');
    } else {
      this.baseUrl = baseUrl || '';
    }
  }

  /**
   * Helper to execute a fetch request with timeout & error normalization.
   */
  async _request(endpoint, options = {}, timeoutMs = DEFAULT_TIMEOUT_MS) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    const headers = {
      ...(options.headers || {}),
    };

    const config = {
      ...options,
      headers,
      signal: options.signal || controller.signal,
    };

    const url = `${this.baseUrl}${endpoint}`;

    try {
      const response = await fetch(url, config);
      clearTimeout(timeoutId);

      // Handle 429 Rate Limiting
      if (response.status === 429) {
        const retryAfter = response.headers.get('Retry-After') || '60';
        throw new ApiError(
          `Too many requests. Please wait ${retryAfter} seconds and try again.`,
          429,
          parseInt(retryAfter, 10) || 60
        );
      }

      // Handle 502 Upstream/Provider Failures
      if (response.status === 502) {
        let errDetail = 'The voice or generation service is temporarily unavailable. Please try again.';
        try {
          const errJson = await response.json();
          if (errJson.detail && typeof errJson.detail === 'string') {
            errDetail = errJson.detail;
          }
        } catch (_) {}
        throw new ApiError(errDetail, 502);
      }

      // Parse JSON
      let data = null;
      const contentType = response.headers.get('content-type') || '';
      if (contentType.includes('application/json')) {
        data = await response.json();
      } else {
        const text = await response.text();
        data = { text };
      }

      if (!response.ok) {
        let message = 'Request failed. Please try again.';
        if (data && data.detail) {
          if (typeof data.detail === 'string') {
            message = data.detail;
          } else if (Array.isArray(data.detail)) {
            message = data.detail.map((d) => d.msg || d).join(', ');
          }
        }
        throw new ApiError(message, response.status, null, data);
      }

      return data;
    } catch (err) {
      clearTimeout(timeoutId);

      if (err.name === 'AbortError') {
        throw new ApiError('Request timed out. The server took too long to respond.', 408);
      }
      if (err instanceof ApiError) {
        throw err;
      }
      // Network or connection errors
      throw new ApiError('Unable to connect to the Voice RAG backend server.', 0, null, err.message);
    }
  }

  /**
   * Check backend health status via GET /health.
   */
  async checkHealth() {
    return this._request('/health', { method: 'GET' }, 8000);
  }

  /**
   * Submit multilingual text query via POST /api/ask.
   */
  async askQuestion(query, language = null, topK = 5, signal = null) {
    const payload = {
      query: query.trim(),
      language: language || null,
      top_k: topK || 5,
    };

    return this._request(
      '/api/ask',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
        signal,
      },
      DEFAULT_TIMEOUT_MS
    );
  }

  /**
   * Submit audio query via POST /api/voice-ask.
   */
  async askVoice(audioBlob, language = null, topK = 5, filename = 'recording.webm', signal = null) {
    const formData = new FormData();
    formData.append('audio', audioBlob, filename);

    if (language) {
      formData.append('language', language);
    }
    formData.append('top_k', String(topK || 5));
    formData.append('synthesize_speech', 'true');

    return this._request(
      '/api/voice-ask',
      {
        method: 'POST',
        body: formData,
        signal,
      },
      60000 // Voice pipeline may take slightly longer for STT + Retrieval + LLM + TTS
    );
  }

  /**
   * Stream text query via GET /api/ask-stream (Server-Sent Events).
   *
   * Calls onStage(stage, message) as pipeline stages progress.
   * Calls onToken(token) for each LLM token chunk.
   * Calls onDone(payload) with the full final result.
   * Calls onError(message, code) on failure.
   *
   * Returns an AbortController so caller can cancel the stream.
   *
   * STREAMING STATUS: LLM token streaming supported via Sarvam SSE.
   * STREAMING STATUS: TTS DOES NOT SUPPORT STREAMING (use askVoice for audio).
   */
  askQuestionStream(query, language = null, topK = 5, { onStage, onToken, onDone, onError } = {}) {
    const controller = new AbortController();
    const params = new URLSearchParams({ query: query.trim(), top_k: String(topK || 5) });
    if (language) params.set('language', language);

    const url = `${this.baseUrl}/api/ask-stream?${params.toString()}`;

    (async () => {
      let response;
      try {
        response = await fetch(url, {
          method: 'GET',
          headers: { Accept: 'text/event-stream' },
          signal: controller.signal,
        });
      } catch (fetchErr) {
        if (fetchErr.name === 'AbortError') return;
        if (onError) onError('Unable to connect to the streaming endpoint.', 0);
        return;
      }

      if (!response.ok) {
        if (onError) onError(`Stream request failed (HTTP ${response.status}).`, response.status);
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      const parseEvents = (raw) => {
        // SSE frame parser: split on double newline
        const frames = (buffer + raw).split('\n\n');
        buffer = frames.pop(); // last incomplete frame stays in buffer
        for (const frame of frames) {
          let eventType = 'message';
          let dataStr = '';
          for (const line of frame.split('\n')) {
            if (line.startsWith('event:')) eventType = line.slice(6).trim();
            if (line.startsWith('data:')) dataStr = line.slice(5).trim();
          }
          if (!dataStr) continue;
          let payload;
          try { payload = JSON.parse(dataStr); } catch { continue; }

          if (eventType === 'stage' && onStage) onStage(payload.stage, payload.message);
          else if (eventType === 'token' && onToken) onToken(payload.token);
          else if (eventType === 'done' && onDone) onDone(payload);
          else if (eventType === 'error' && onError) onError(payload.message, payload.code);
        }
      };

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          parseEvents(decoder.decode(value, { stream: true }));
        }
      } catch (readErr) {
        if (readErr.name !== 'AbortError' && onError) {
          onError('Stream connection was interrupted.', 0);
        }
      }
    })();

    return controller;
  }
}

export const api = new ApiClient();
