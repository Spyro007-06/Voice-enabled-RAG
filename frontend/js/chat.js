/**
 * HH Goa 2026 Multilingual Voice RAG — Conversation & Orchestration Manager
 */

import { state, APP_STATE, MULTILINGUAL_CONFIG } from './state.js';
import { ApiService } from './api.js';
import { UIManager } from './ui.js';

export class ChatManager {
  /**
   * Handle Voice Audio Submission (E2E Pipeline)
   */
  static async handleVoiceSubmission(audioBlob) {
    const lang = state.language;
    UIManager.announce('Audio recording stopped. Transcribing speech with Saarika v2.5…');

    // Display initial pipeline progress
    let progressBox = UIManager.renderProgress(0);
    state.setState(APP_STATE.TRANSCRIBING);

    try {
      // Progress to transcribing
      progressBox = UIManager.updateProgress(progressBox, 1);

      const response = await ApiService.submitVoiceAsk(audioBlob, lang, 5);

      // Verify transcript
      if (!response.transcript || !response.transcript.trim()) {
        if (progressBox) progressBox.remove();
        state.setState(APP_STATE.STT_ERROR);
        UIManager.renderErrorCard(
          "We couldn't understand the recording. Please try speaking again.",
          () => {
            const centralMic = document.getElementById('centralVoiceBtn') || document.getElementById('micBtn');
            if (centralMic) centralMic.click();
          }
        );
        return;
      }

      // 1. Display Transcript FIRST
      state.setState(APP_STATE.TRANSCRIPT_READY);
      UIManager.renderUserTurn({
        text: response.transcript,
        isVoice: true,
        languageCode: response.detected_language || response.language || lang,
      });

      // 2. Display retrieval & generation progress
      state.setState(APP_STATE.RETRIEVING);
      progressBox = UIManager.updateProgress(progressBox, 2);

      // Small tick to show progress transitioning
      await new Promise((r) => setTimeout(r, 120));
      progressBox = UIManager.updateProgress(progressBox, 3);

      await new Promise((r) => setTimeout(r, 100));
      progressBox = UIManager.updateProgress(progressBox, 4, response.citations?.length || 0);

      await new Promise((r) => setTimeout(r, 80));
      progressBox = UIManager.updateProgress(progressBox, 5);

      // 3. Remove progress and render final answer
      if (progressBox) progressBox.remove();

      state.setState(APP_STATE.ANSWER_READY);
      UIManager.renderAnswer({
        answer: response.answer,
        grounded: response.grounded,
        citations: response.citations,
        citation_provenance: response.citation_provenance,
        audio: response.audio,
        isVoice: true,
        model: 'Google Gemini (gemini-2.5-flash)',
        onRetryVoice: () => {
          if (state.lastRecordedBlob) {
            ChatManager.handleVoiceSubmission(state.lastRecordedBlob);
          }
        },
      });

      UIManager.announce('Answer ready. Voice response loaded.');

    } catch (error) {
      if (progressBox) progressBox.remove();
      state.setState(APP_STATE.ERROR);

      const msg = error.message || 'Voice RAG execution failed.';
      UIManager.renderErrorCard(
        `Error: ${msg}`,
        () => {
          if (state.lastRecordedBlob) {
            ChatManager.handleVoiceSubmission(state.lastRecordedBlob);
          }
        }
      );
      UIManager.showToast(msg, true);
    }
  }

  /**
   * Handle Text Query Submission (Streamed)
   */
  static async handleTextSubmission(queryText) {
    if (!queryText || !queryText.trim()) return;

    const lang = state.language;
    const cleanQuery = queryText.trim();

    // 1. Render User Turn
    UIManager.renderUserTurn({
      text: cleanQuery,
      isVoice: false,
      languageCode: lang,
    });

    // 2. Initial Progress Tracker
    let progressBox = UIManager.renderProgress(2);
    state.setState(APP_STATE.RETRIEVING);

    let accumulatedTokens = '';

    await ApiService.streamAsk({
      query: cleanQuery,
      language: lang,
      topK: 5,
      onStage: (stagePayload) => {
        const stageName = stagePayload.stage;
        if (stageName === 'retrieving') {
          state.setState(APP_STATE.RETRIEVING);
          progressBox = UIManager.updateProgress(progressBox, 2);
        } else if (stageName === 'ranking') {
          state.setState(APP_STATE.RERANKING);
          progressBox = UIManager.updateProgress(progressBox, 3);
        } else if (stageName === 'generating') {
          state.setState(APP_STATE.GENERATING);
          progressBox = UIManager.updateProgress(progressBox, 4);
        }
      },
      onToken: (tokenDelta) => {
        accumulatedTokens += tokenDelta;
      },
      onDone: (donePayload) => {
        if (progressBox) progressBox.remove();
        state.setState(APP_STATE.ANSWER_READY);

        UIManager.renderAnswer({
          answer: donePayload.answer || accumulatedTokens,
          grounded: donePayload.grounded !== false,
          citations: donePayload.citations || [],
          citation_provenance: donePayload.citation_provenance || [],
          audio: null,
          isVoice: false,
          model: 'Google Gemini (gemini-2.5-flash)',
        });

        UIManager.announce('Answer received.');
      },
      onError: (err) => {
        if (progressBox) progressBox.remove();
        state.setState(APP_STATE.ERROR);
        const msg = err.message || 'Failed to generate answer.';
        UIManager.renderErrorCard(msg);
        UIManager.showToast(msg, true);
      },
    });
  }
}
