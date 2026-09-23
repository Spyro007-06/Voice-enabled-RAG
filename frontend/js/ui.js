/**
 * HH Goa 2026 Multilingual Voice RAG — UI Components & Safe DOM Renderer
 */

import { MULTILINGUAL_CONFIG, state, APP_STATE } from './state.js';

export class UIManager {
  static announce(text) {
    const el = document.getElementById('srAnnouncement');
    if (el) el.textContent = text;
  }

  static showToast(message, isError = false) {
    const container = document.getElementById('toastContainer');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast${isError ? ' error' : ''}`;
    toast.textContent = message;

    container.replaceChildren(toast);
    setTimeout(() => {
      if (toast.isConnected) toast.remove();
    }, 4500);
  }

  /**
   * Open Accessible Modal Dialog with Focus Trap
   */
  static _activeModalId = null;
  static _previousFocusedElem = null;
  static _trapFocusHandler = null;

  static openModal(modalId) {
    const modal = document.getElementById(modalId);
    const backdrop = document.getElementById('modalBackdrop');
    if (!modal) return;

    this._previousFocusedElem = document.activeElement;
    this._activeModalId = modalId;

    modal.hidden = false;
    if (backdrop) backdrop.hidden = false;
    document.body.classList.add('modal-open');

    // Find first focusable element inside the modal
    const focusableSelectors = 'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])';
    const focusables = Array.from(modal.querySelectorAll(focusableSelectors)).filter(
      (el) => !el.hasAttribute('disabled') && !el.getAttribute('aria-hidden')
    );

    const firstFocusable = focusables[0];
    const lastFocusable = focusables[focusables.length - 1];

    if (firstFocusable) {
      setTimeout(() => firstFocusable.focus(), 50);
    }

    // Attach trap focus listener
    if (this._trapFocusHandler) {
      document.removeEventListener('keydown', this._trapFocusHandler);
    }

    this._trapFocusHandler = (e) => {
      if (e.key !== 'Tab' || !this._activeModalId) return;

      if (e.shiftKey) {
        if (document.activeElement === firstFocusable) {
          e.preventDefault();
          if (lastFocusable) lastFocusable.focus();
        }
      } else {
        if (document.activeElement === lastFocusable) {
          e.preventDefault();
          if (firstFocusable) firstFocusable.focus();
        }
      }
    };

    document.addEventListener('keydown', this._trapFocusHandler);
  }

  /**
   * Close Modal Dialog & Restore Focus
   */
  static closeModal(modalId) {
    const modal = document.getElementById(modalId);
    const backdrop = document.getElementById('modalBackdrop');
    if (modal) modal.hidden = true;
    if (backdrop) backdrop.hidden = true;
    document.body.classList.remove('modal-open');

    if (this._trapFocusHandler) {
      document.removeEventListener('keydown', this._trapFocusHandler);
      this._trapFocusHandler = null;
    }

    this._activeModalId = null;

    if (this._previousFocusedElem && typeof this._previousFocusedElem.focus === 'function') {
      try {
        this._previousFocusedElem.focus();
      } catch (_) {}
      this._previousFocusedElem = null;
    }
  }

  /**
   * Show "No Microphone Detected" Modal
   */
  static showNoMicModal({ onTypeInstead, onRetry } = {}) {
    const modal = document.getElementById('noMicModal');
    if (!modal) {
      this.showToast('No microphone detected on this device. Please connect a microphone or type your question.', true);
      return;
    }

    this.openModal('noMicModal');
    this.announce('Alert: No microphone detected on this device.');

    const typeBtn = document.getElementById('noMicTypeBtn');
    if (typeBtn) {
      typeBtn.onclick = () => {
        this.closeModal('noMicModal');
        if (onTypeInstead) {
          onTypeInstead();
        } else {
          const input = document.getElementById('queryInput');
          if (input) {
            input.focus();
            input.scrollIntoView({ behavior: 'smooth', block: 'center' });
          }
        }
      };
    }

    const retryBtn = document.getElementById('noMicRetryBtn');
    if (retryBtn) {
      retryBtn.onclick = () => {
        this.closeModal('noMicModal');
        if (onRetry) onRetry();
      };
    }

    const closeBtn = document.getElementById('closeNoMicBtn');
    if (closeBtn) {
      closeBtn.onclick = () => this.closeModal('noMicModal');
    }
  }

  /**
   * Show "Microphone Permission Required" Modal
   */
  static showMicPermissionModal({ onRetry } = {}) {
    const modal = document.getElementById('micPermModal');
    if (!modal) {
      this.showToast('Microphone access was denied. Please allow microphone permissions in your browser settings.', true);
      return;
    }

    this.openModal('micPermModal');
    this.announce('Alert: Microphone permission required.');

    const retryBtn = document.getElementById('micPermRetryBtn');
    if (retryBtn) {
      retryBtn.onclick = () => {
        this.closeModal('micPermModal');
        if (onRetry) onRetry();
      };
    }

    const closeBtn = document.getElementById('closeMicPermBtn');
    if (closeBtn) {
      closeBtn.onclick = () => this.closeModal('micPermModal');
    }
  }

  /**
   * Render User Transcript Turn
   */
  static renderUserTurn({ text, isVoice = false, languageCode = 'en' }) {
    const welcome = document.getElementById('welcomeState');
    if (welcome) welcome.hidden = true;

    const stream = document.getElementById('chatMessages');
    if (!stream) return;

    const config = MULTILINGUAL_CONFIG[languageCode] || MULTILINGUAL_CONFIG.en;

    const turn = document.createElement('article');
    turn.className = 'turn-user';

    const header = document.createElement('div');
    header.className = 'turn-header';

    const tag = document.createElement('span');
    tag.className = 'turn-tag';
    tag.textContent = isVoice ? config.userPromptLabel : 'YOU';

    const meta = document.createElement('span');
    meta.className = `turn-meta-badge${isVoice ? ' voice' : ''}`;
    meta.textContent = isVoice ? config.sttModelLabel : `${config.nativeName} · Text Query`;

    header.append(tag, meta);

    const quote = document.createElement('p');
    quote.className = 'transcript-quote';
    quote.textContent = `“${text}”`;

    turn.append(header, quote);
    stream.append(turn);
    turn.scrollIntoView({ block: 'end', behavior: 'smooth' });
  }

  /**
   * Pipeline Progress Tracker
   */
  static renderProgress(activeStageIndex = 0, retrievedCount = null) {
    const stream = document.getElementById('chatMessages');
    if (!stream) return null;

    const STAGES = [
      'VOICE RECEIVED',
      'TRANSCRIPT CREATED',
      'SEARCHING MULTILINGUAL CORPUS',
      'RANKING EVIDENCE',
      'GENERATING GROUNDED ANSWER',
      'PREPARING VOICE RESPONSE',
    ];

    const box = document.createElement('section');
    box.id = 'pipelineProgressBox';
    box.className = 'pipeline-progress-box';
    box.setAttribute('aria-live', 'polite');

    const header = document.createElement('div');
    header.className = 'pipeline-progress-header';
    header.textContent = 'PIPELINE EXECUTION';
    if (retrievedCount !== null) {
      const badge = document.createElement('span');
      badge.textContent = `${retrievedCount} evidence passages retrieved`;
      header.append(badge);
    }
    box.append(header);

    const list = document.createElement('div');
    list.className = 'progress-stages-list';

    STAGES.forEach((stageName, idx) => {
      const row = document.createElement('div');
      const isDone = idx < activeStageIndex;
      const isActive = idx === activeStageIndex;

      row.className = `stage-row ${isDone ? 'done' : isActive ? 'active' : 'pending'}`;

      const icon = document.createElement('span');
      icon.className = 'stage-icon';
      icon.textContent = isDone ? '✓' : isActive ? '●' : '○';

      const label = document.createElement('span');
      label.textContent = stageName;

      row.append(icon, label);
      list.append(row);
    });

    box.append(list);
    stream.append(box);
    box.scrollIntoView({ block: 'end', behavior: 'smooth' });
    return box;
  }

  static updateProgress(boxElement, activeStageIndex, retrievedCount = null) {
    if (!boxElement || !boxElement.isConnected) return this.renderProgress(activeStageIndex, retrievedCount);

    const newBox = this.renderProgress(activeStageIndex, retrievedCount);
    boxElement.replaceWith(newBox);
    return newBox;
  }

  /**
   * Render Grounded Answer Turn
   */
  static renderAnswer({
    answer,
    grounded = true,
    citations = [],
    citation_provenance = [],
    audio = null,
    isVoice = false,
    model = 'Google Gemini (gemini-2.5-flash)',
    onRetryVoice = null,
  }) {
    const stream = document.getElementById('chatMessages');
    if (!stream) return;

    const turn = document.createElement('article');
    turn.className = 'turn-answer';

    // 1. Header & Grounding Status
    const header = document.createElement('div');
    header.className = 'answer-header';

    const statusTag = document.createElement('span');
    statusTag.className = `grounding-status-tag ${grounded ? 'grounded' : 'refusal'}`;
    statusTag.textContent = grounded ? '✓ GROUNDED' : '! INSUFFICIENT EVIDENCE';

    const modelMeta = document.createElement('span');
    modelMeta.className = 'answer-model-meta';
    modelMeta.textContent = grounded
      ? 'Gemini 2.5 Flash'
      : 'Corpus Guardrails';

    header.append(statusTag, modelMeta);
    turn.append(header);

    // 2. Answer Body
    const body = document.createElement('div');
    body.className = 'answer-body';
    
    // Split by newlines into clean paragraph nodes
    const paragraphs = (answer || "I couldn't find enough relevant information in the indexed multilingual corpus to answer this reliably.")
      .split('\n\n')
      .filter((p) => p.trim());

    if (paragraphs.length === 0) {
      const p = document.createElement('p');
      p.textContent = answer;
      body.append(p);
    } else {
      paragraphs.forEach((pText) => {
        const p = document.createElement('p');
        p.textContent = pText;
        body.append(p);
      });
    }
    turn.append(body);

    // If refusal (insufficient evidence), add quick action buttons
    if (!grounded) {
      const refusalActions = document.createElement('div');
      refusalActions.style.display = 'flex';
      refusalActions.style.gap = '10px';
      refusalActions.style.marginTop = '14px';
      refusalActions.style.flexWrap = 'wrap';

      const tryAnotherBtn = document.createElement('button');
      tryAnotherBtn.className = 'btn btn-ghost';
      tryAnotherBtn.textContent = 'Try another question';
      tryAnotherBtn.onclick = () => {
        const input = document.getElementById('queryInput');
        if (input) input.focus();
      };

      const searchAllBtn = document.createElement('button');
      searchAllBtn.className = 'btn btn-ghost';
      searchAllBtn.textContent = '🌐 Search all languages';
      searchAllBtn.onclick = () => {
        state.setLanguage('en');
        const input = document.getElementById('queryInput');
        if (input) input.focus();
      };

      refusalActions.append(tryAnotherBtn, searchAllBtn);
      turn.append(refusalActions);
    }

    // 3. Sources Accordion
    if (citations && citations.length > 0) {
      const sourcesDetails = document.createElement('details');
      sourcesDetails.className = 'sources-details';

      const summary = document.createElement('summary');
      summary.className = 'sources-summary';
      summary.textContent = `Retrieved Evidence (${citations.length} sources)`;
      sourcesDetails.append(summary);

      const list = document.createElement('ul');
      list.className = 'sources-list';

      citations.forEach((citId, i) => {
        const prov = citation_provenance && citation_provenance[i] ? citation_provenance[i] : null;
        const li = document.createElement('li');
        li.className = 'source-item';

        const itemHeader = document.createElement('div');
        itemHeader.className = 'source-item-header';

        const rank = document.createElement('span');
        rank.className = 'source-rank';
        rank.textContent = `#${i + 1} Document ${prov?.document_id || citId}`;

        const langBadge = document.createElement('span');
        langBadge.textContent = prov?.language ? `${prov.language.toUpperCase()} · Score ${prov.score ? Number(prov.score).toFixed(2) : '0.90'}` : 'MSMARCO-XI';

        itemHeader.append(rank, langBadge);

        const snippet = document.createElement('p');
        snippet.className = 'source-snippet';
        snippet.textContent = prov?.snippet || prov?.text || `Retrieved corpus passage chunk ${citId}`;

        li.append(itemHeader, snippet);
        list.append(li);
      });

      sourcesDetails.append(list);
      turn.append(sourcesDetails);
    }

    // 4. Voice Player (Sarvam Bulbul v2)
    if (isVoice) {
      if (audio && audio.available && audio.audio_base64) {
        const player = this.createAudioPlayer(audio);
        turn.append(player);
      } else {
        const fallback = document.createElement('div');
        fallback.className = 'voice-response-card';
        
        const msg = document.createElement('p');
        msg.className = 'voice-fallback-msg';
        msg.textContent = 'Voice response unavailable. You can read the grounded answer above.';

        if (onRetryVoice) {
          const retryBtn = document.createElement('button');
          retryBtn.className = 'btn btn-ghost';
          retryBtn.textContent = 'Try voice again';
          retryBtn.onclick = onRetryVoice;
          fallback.append(msg, retryBtn);
        } else {
          fallback.append(msg);
        }
        turn.append(fallback);
      }
    }

    stream.append(turn);
    turn.scrollIntoView({ block: 'end', behavior: 'smooth' });
  }

  /**
   * Create Sarvam Bulbul v2 Custom Audio Player
   */
  static createAudioPlayer(audio) {
    const card = document.createElement('div');
    card.className = 'voice-response-card';

    const header = document.createElement('div');
    header.className = 'voice-response-header';

    const tag = document.createElement('span');
    tag.className = 'voice-tag';
    tag.innerHTML = '<span aria-hidden="true">🔊</span><span>Voice Response</span>';

    const speaker = document.createElement('span');
    speaker.className = 'voice-speaker-badge';
    speaker.textContent = `Sarvam Bulbul v2 · ${audio.speaker || 'Anushka'}`;

    header.append(tag, speaker);
    card.append(header);

    // Audio Element
    const mime = `audio/${audio.format || 'wav'}`;
    const audioUrl = `data:${mime};base64,${audio.audio_base64}`;
    const media = new Audio(audioUrl);

    // Stop previous playing audio
    if (state.activeAudio) {
      try {
        state.activeAudio.pause();
        state.activeAudio.currentTime = 0;
      } catch (_) {}
    }
    state.activeAudio = media;

    const controls = document.createElement('div');
    controls.className = 'audio-controls-row';

    const playBtn = document.createElement('button');
    playBtn.className = 'audio-play-btn';
    playBtn.setAttribute('aria-label', 'Play voice response');
    playBtn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>';

    const scrubberWrap = document.createElement('div');
    scrubberWrap.className = 'audio-scrubber-wrapper';

    const scrubber = document.createElement('input');
    scrubber.type = 'range';
    scrubber.className = 'audio-scrubber';
    scrubber.min = '0';
    scrubber.max = '100';
    scrubber.value = '0';
    scrubber.setAttribute('aria-label', 'Playback scrubber');

    const timeRow = document.createElement('div');
    timeRow.className = 'audio-time-row';

    const timeElapsed = document.createElement('span');
    timeElapsed.textContent = '0:00';

    const timeTotal = document.createElement('span');
    timeTotal.textContent = audio.duration_s ? `0:${String(Math.round(audio.duration_s)).padStart(2, '0')}` : '0:00';

    timeRow.append(timeElapsed, timeTotal);
    scrubberWrap.append(scrubber, timeRow);

    const volumeWrap = document.createElement('div');
    volumeWrap.className = 'audio-volume-wrapper';
    volumeWrap.setAttribute('title', 'Volume');

    const volumeIcon = document.createElement('span');
    volumeIcon.textContent = '🔈';
    volumeIcon.setAttribute('aria-hidden', 'true');

    const volumeSlider = document.createElement('input');
    volumeSlider.type = 'range';
    volumeSlider.className = 'volume-slider';
    volumeSlider.min = '0';
    volumeSlider.max = '1';
    volumeSlider.step = '0.05';
    volumeSlider.value = '1';
    volumeSlider.setAttribute('aria-label', 'Voice volume');

    volumeWrap.append(volumeIcon, volumeSlider);

    controls.append(playBtn, scrubberWrap, volumeWrap);
    card.append(controls);

    // Audio Event Wiring
    const updatePlayIcon = (isPlaying) => {
      if (isPlaying) {
        playBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>';
        playBtn.setAttribute('aria-label', 'Pause voice response');
      } else {
        playBtn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>';
        playBtn.setAttribute('aria-label', 'Play voice response');
      }
    };

    playBtn.onclick = () => {
      if (media.paused) {
        media.play().catch(() => {
          this.showToast('Tap play to start listening.', false);
        });
      } else {
        media.pause();
      }
    };

    media.onplay = () => {
      updatePlayIcon(true);
      state.setState(APP_STATE.PLAYING);
    };

    media.onpause = () => {
      updatePlayIcon(false);
      if (!media.ended) state.setState(APP_STATE.ANSWER_READY);
    };

    media.onended = () => {
      updatePlayIcon(false);
      scrubber.value = '0';
      timeElapsed.textContent = '0:00';
      state.setState(APP_STATE.ANSWER_READY);
    };

    media.ontimeupdate = () => {
      if (media.duration) {
        const percent = (media.currentTime / media.duration) * 100;
        scrubber.value = String(percent);
        const mins = Math.floor(media.currentTime / 60);
        const secs = String(Math.floor(media.currentTime % 60)).padStart(2, '0');
        timeElapsed.textContent = `${mins}:${secs}`;
      }
    };

    media.onloadedmetadata = () => {
      if (media.duration) {
        const mins = Math.floor(media.duration / 60);
        const secs = String(Math.floor(media.duration % 60)).padStart(2, '0');
        timeTotal.textContent = `${mins}:${secs}`;
      }
    };

    scrubber.oninput = () => {
      if (media.duration) {
        media.currentTime = (Number(scrubber.value) / 100) * media.duration;
      }
    };

    volumeSlider.oninput = () => {
      media.volume = Number(volumeSlider.value);
    };

    // Autoplay attempt with polite fallback
    media.play().catch(() => {
      // Browser autoplay prevented — non-blocking
    });

    return card;
  }

  /**
   * Render Error Notice
   */
  static renderErrorCard(message, onRetry = null) {
    const stream = document.getElementById('chatMessages');
    if (!stream) return;

    const card = document.createElement('article');
    card.className = 'turn-answer';
    card.style.borderLeft = '4px solid var(--danger)';

    const header = document.createElement('div');
    header.className = 'answer-header';

    const statusTag = document.createElement('span');
    statusTag.className = 'grounding-status-tag refusal';
    statusTag.style.color = 'var(--danger)';
    statusTag.style.background = 'var(--danger-soft)';
    statusTag.textContent = 'ERROR';

    header.append(statusTag);
    card.append(header);

    const body = document.createElement('div');
    body.className = 'answer-body';
    const p = document.createElement('p');
    p.textContent = message;
    body.append(p);
    card.append(body);

    if (onRetry) {
      const actions = document.createElement('div');
      actions.style.marginTop = '12px';
      const btn = document.createElement('button');
      btn.className = 'btn btn-ghost';
      btn.textContent = 'Try again';
      btn.onclick = () => {
        card.remove();
        onRetry();
      };
      actions.append(btn);
      card.append(actions);
    }

    stream.append(card);
    card.scrollIntoView({ block: 'end', behavior: 'smooth' });
  }
}
