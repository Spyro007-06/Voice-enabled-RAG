/**
 * HH Goa 2026 Multilingual Voice RAG — API Integration Service
 */

export class ApiService {
  /**
   * Submit audio blob for end-to-end voice RAG pipeline
   */
  static async submitVoiceAsk(audioBlob, language = 'en', topK = 5) {
    const formData = new FormData();
    const filename = audioBlob.type.includes('ogg') ? 'query.ogg' : 'query.webm';
    formData.append('audio', audioBlob, filename);
    formData.append('language', language);
    formData.append('top_k', String(topK));
    formData.append('synthesize_speech', 'true');

    const response = await fetch('/api/voice-ask', {
      method: 'POST',
      body: formData,
    });

    let data;
    try {
      data = await response.json();
    } catch (_) {
      throw new Error(`Server returned status ${response.status}`);
    }

    if (!response.ok) {
      const errorMsg = data?.detail || data?.error || 'Voice RAG request failed';
      throw new Error(errorMsg);
    }

    return data;
  }

  /**
   * Stream text query via Server-Sent Events (/api/ask-stream)
   */
  static async streamAsk({ query, language = 'en', topK = 5, onStage, onToken, onDone, onError }) {
    const params = new URLSearchParams({
      query: query.trim(),
      language,
      top_k: String(topK),
    });

    try {
      const response = await fetch(`/api/ask-stream?${params.toString()}`, {
        headers: { Accept: 'text/event-stream' },
      });

      if (!response.ok) {
        let errDetail = 'Streaming request failed';
        try {
          const errData = await response.json();
          errDetail = errData.detail || errDetail;
        } catch (_) {}
        throw new Error(errDetail);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      let accumulatedAnswer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split('\n\n');
        buffer = frames.pop();

        for (const frame of frames) {
          if (!frame.trim()) continue;

          const eventMatch = frame.match(/event:\s*([^\n\r]+)/);
          const dataMatch = frame.match(/data:\s*([^\n\r]+)/);

          const eventType = eventMatch ? eventMatch[1].trim() : 'message';
          const rawData = dataMatch ? dataMatch[1].trim() : '';

          if (!rawData) continue;

          try {
            const parsed = JSON.parse(rawData);

            if (eventType === 'stage') {
              if (onStage) onStage(parsed);
            } else if (eventType === 'token') {
              accumulatedAnswer += parsed.token || '';
              if (onToken) onToken(parsed.token || '', accumulatedAnswer);
            } else if (eventType === 'done') {
              if (onDone) onDone({ ...parsed, answer: parsed.answer || accumulatedAnswer });
              return;
            } else if (eventType === 'error') {
              throw new Error(parsed.message || 'Stream error');
            }
          } catch (jsonErr) {
            console.warn('SSE frame parse error:', jsonErr, frame);
          }
        }
      }
    } catch (error) {
      if (onError) onError(error);
    }
  }

  /**
   * Health check endpoint
   */
  static async checkHealth() {
    try {
      const res = await fetch('/health');
      if (res.ok) {
        return await res.json();
      }
      return { status: 'degraded' };
    } catch (_) {
      return { status: 'offline' };
    }
  }
}
