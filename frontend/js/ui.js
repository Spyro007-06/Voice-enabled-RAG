/**
 * HH Goa 2026 Multilingual Voice RAG — UI Rendering & DOM Interactions (Phase 6.25 Dataset-Centric)
 */

import { state } from './state.js';

export function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

export function formatLanguageName(code) {
  const map = {
    en: 'English',
    hi: 'हिन्दी — Hindi',
    ta: 'தமிழ் — Tamil',
    te: 'తెలుగు — Telugu',
    ml: 'മലയാളം — Malayalam',
  };
  return map[code] || (code ? code.toUpperCase() : 'All Languages');
}

export function showToast(message, type = 'info', durationMs = 4000) {
  const container = document.getElementById('toastContainer');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${escapeHtml(message)}</span>`;

  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(8px)';
    toast.style.transition = 'all 0.25s ease';
    setTimeout(() => toast.remove(), 250);
  }, durationMs);
}

/**
 * Render message element into the chat container
 */
export function renderMessage(msg, onCrossLanguageRetry = null, onTryAnother = null, onRetryVoice = null) {
  const isUser = msg.sender === 'user';
  const row = document.createElement('div');
  row.className = `message-row ${isUser ? 'user' : 'assistant'} animate-fade-in`;
  row.id = msg.id;

  const avatar = document.createElement('div');
  avatar.className = `message-avatar ${isUser ? 'user' : 'assistant'}`;
  avatar.textContent = isUser ? 'U' : 'AI';

  const bubble = document.createElement('div');
  bubble.className = `message-bubble ${isUser ? 'user' : 'assistant'}`;

  // -------------------------------------------------------------------------
  // USER MESSAGE RENDERING
  // -------------------------------------------------------------------------
  if (isUser) {
    if (msg.isVoice) {
      const voiceHeader = document.createElement('div');
      voiceHeader.className = 'voice-transcript-header';
      const langLabel = formatLanguageName(msg.language);
      voiceHeader.innerHTML = `
        <div class="stt-banner">
          <span>🎤 Speech Recognition (Saarika v2.5)</span>
          <span class="stt-meta-tag">${escapeHtml(langLabel)}</span>
        </div>
      `;
      bubble.appendChild(voiceHeader);
    }

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    if (msg.language) {
      contentDiv.setAttribute('lang', msg.language);
    }
    contentDiv.textContent = msg.text || '';
    bubble.appendChild(contentDiv);
  }

  // -------------------------------------------------------------------------
  // ASSISTANT MESSAGE RENDERING
  // -------------------------------------------------------------------------
  if (!isUser) {
    const isRefusal = !msg.grounded || (msg.text && (
      msg.text.toLowerCase().includes("don't have enough information") ||
      msg.text.toLowerCase().includes("not enough information") ||
      msg.text.toLowerCase().includes("not present in the provided context") ||
      msg.text.toLowerCase().includes("does not contain information") ||
      msg.text.toLowerCase().includes("cannot be found") ||
      msg.text.toLowerCase().includes("not available") ||
      msg.text.toLowerCase().includes("insufficient")
    ));

    // STT Failure State
    if (msg.isSTTFailure) {
      const errBox = document.createElement('div');
      errBox.className = 'insufficient-evidence-box';
      errBox.innerHTML = `
        <div style="font-weight: 700; color: var(--danger); font-size: 0.88rem; display: flex; align-items: center; gap: 6px;">
          <span>⚠️ Speech Recognition Failed</span>
        </div>
        <div class="insufficient-text">${escapeHtml(msg.text || 'Could not transcribe speech audio.')}</div>
      `;

      const actionsRow = document.createElement('div');
      actionsRow.className = 'refusal-actions-row';

      const retryMicBtn = document.createElement('button');
      retryMicBtn.className = 'refusal-action-btn';
      retryMicBtn.setAttribute('type', 'button');
      retryMicBtn.innerHTML = `<span>🎤 Try Recording Again</span>`;
      retryMicBtn.onclick = () => {
        if (onRetryVoice) onRetryVoice();
      };

      actionsRow.appendChild(retryMicBtn);
      errBox.appendChild(actionsRow);
      bubble.appendChild(errBox);
    }
    // Standard Refusal / Insufficient Evidence State
    else if (isRefusal && !msg.error) {
      const refusalBox = document.createElement('div');
      refusalBox.className = 'insufficient-evidence-box';

      const heading = document.createElement('div');
      heading.style.cssText = 'font-weight: 700; color: var(--warning); font-size: 0.88rem; display: flex; align-items: center; gap: 6px;';
      heading.innerHTML = `<span>! Insufficient Evidence in MSMARCO-XI Corpus</span>`;

      const explText = document.createElement('div');
      explText.className = 'insufficient-text';
      explText.textContent = "I couldn't find enough relevant information in the indexed multilingual corpus to answer this reliably.";

      const actionsRow = document.createElement('div');
      actionsRow.className = 'refusal-actions-row';

      // 1. Try another question button
      const tryAnotherBtn = document.createElement('button');
      tryAnotherBtn.className = 'refusal-action-btn';
      tryAnotherBtn.setAttribute('type', 'button');
      tryAnotherBtn.innerHTML = `<span>Try another question</span>`;
      tryAnotherBtn.onclick = () => {
        const inp = document.getElementById('queryInput');
        if (inp) {
          inp.focus();
          inp.scrollIntoView({ behavior: 'smooth' });
        }
      };
      actionsRow.appendChild(tryAnotherBtn);

      // 2. Search all languages button
      if (msg.originalQuery) {
        const crossLangBtn = document.createElement('button');
        crossLangBtn.className = 'refusal-action-btn cross-lang';
        crossLangBtn.setAttribute('type', 'button');
        crossLangBtn.innerHTML = `<span>🌐 Search all languages</span>`;
        crossLangBtn.onclick = () => {
          crossLangBtn.disabled = true;
          crossLangBtn.textContent = 'Searching all languages...';
          if (onCrossLanguageRetry) {
            onCrossLanguageRetry(msg.originalQuery);
          }
        };
        actionsRow.appendChild(crossLangBtn);
      }

      refusalBox.appendChild(heading);
      refusalBox.appendChild(explText);
      refusalBox.appendChild(actionsRow);
      bubble.appendChild(refusalBox);
    }
    // Successful Grounded or Partial Answer
    else {
      const contentDiv = document.createElement('div');
      contentDiv.className = 'message-content';
      if (msg.language) {
        contentDiv.setAttribute('lang', msg.language);
      }
      contentDiv.textContent = msg.text || '';
      bubble.appendChild(contentDiv);
    }

    // Audio Player Card
    if (msg.audioBase64) {
      const audioPlayerCard = createAudioPlayer(msg.audioBase64, msg.audioFormat || 'wav', true);
      bubble.appendChild(audioPlayerCard);
    }

    // Sources & Citations Accordion
    if (msg.citations && msg.citations.length > 0) {
      const sourcesAccordion = createSourcesAccordion(msg.citations, msg.citationProvenance, msg.language);
      bubble.appendChild(sourcesAccordion);
    }

    // Metadata Badges Row
    const metaDiv = document.createElement('div');
    metaDiv.className = 'message-meta';

    // Grounding Status Badge
    if (msg.grounded === true && !isRefusal) {
      const groundBadge = document.createElement('span');
      groundBadge.className = 'meta-badge grounded';
      groundBadge.setAttribute('title', 'Answer grounded in retrieved MSMARCO-XI passages.');
      groundBadge.textContent = '✓ Grounded';
      metaDiv.appendChild(groundBadge);
    } else if (isRefusal || msg.isSTTFailure) {
      const errBadge = document.createElement('span');
      errBadge.className = 'meta-badge refusal';
      errBadge.setAttribute('title', 'Insufficient context found in indexed corpus.');
      errBadge.textContent = '! Insufficient evidence';
      metaDiv.appendChild(errBadge);
    } else {
      const limBadge = document.createElement('span');
      limBadge.className = 'meta-badge limited';
      limBadge.setAttribute('title', 'Answer generated from limited retrieved information.');
      limBadge.textContent = '○ Limited context';
      metaDiv.appendChild(limBadge);
    }

    // Retrieval Confidence Badge
    if (typeof msg.confidence === 'number' && msg.confidence > 0) {
      const confBadge = document.createElement('span');
      confBadge.className = 'meta-badge';
      const pct = Math.round(msg.confidence * 100);
      confBadge.textContent = `Confidence: ${pct}%`;
      metaDiv.appendChild(confBadge);
    }

    // Voice response available / unavailable badges
    if (msg.audioBase64) {
      const voiceBadge = document.createElement('span');
      voiceBadge.className = 'meta-badge grounded';
      voiceBadge.textContent = '🔊 Voice available';
      metaDiv.appendChild(voiceBadge);
    } else if (msg.ttsUnavailable || msg.status === 'partial_success') {
      const partBadge = document.createElement('span');
      partBadge.className = 'meta-badge limited';
      partBadge.textContent = '🔊 Voice unavailable';
      metaDiv.appendChild(partBadge);
    }

    // Cross-Language Retrieval Indicator
    if (msg.isCrossLanguage) {
      const crossBadge = document.createElement('span');
      crossBadge.className = 'meta-badge cross-lang';
      crossBadge.textContent = '🌐 Cross-Language';
      metaDiv.appendChild(crossBadge);
    }

    // Language Badge
    if (msg.language) {
      const langBadge = document.createElement('span');
      langBadge.className = 'meta-badge';
      langBadge.textContent = formatLanguageName(msg.language);
      metaDiv.appendChild(langBadge);
    }

    // Clickable Latency Breakdown Badge
    if (msg.latencyMs) {
      const totalLatency = typeof msg.latencyMs === 'object' ? msg.latencyMs.total : msg.latencyMs;
      if (totalLatency && totalLatency > 0) {
        const latBadge = document.createElement('button');
        latBadge.className = 'meta-badge latency';
        latBadge.setAttribute('type', 'button');
        latBadge.setAttribute('title', 'Click to view latency telemetry breakdown');
        latBadge.textContent = `⚡ ${Math.round(totalLatency)} ms`;

        if (typeof msg.latencyMs === 'object') {
          let breakdownDiv = null;
          latBadge.onclick = () => {
            if (breakdownDiv) {
              breakdownDiv.remove();
              breakdownDiv = null;
            } else {
              breakdownDiv = document.createElement('div');
              breakdownDiv.className = 'latency-breakdown animate-fade-in';
              const l = msg.latencyMs;
              const items = [
                l.stt !== undefined ? `<span>STT:</span> <b>${Math.round(l.stt)}ms</b>` : '',
                l.embedding !== undefined ? `<span>Embed:</span> <b>${Math.round(l.embedding)}ms</b>` : '',
                l.retrieval !== undefined ? `<span>Retrieval:</span> <b>${Math.round(l.retrieval)}ms</b>` : '',
                l.reranking !== undefined ? `<span>Rerank:</span> <b>${Math.round(l.reranking)}ms</b>` : '',
                l.generation !== undefined ? `<span>Sarvam 105B:</span> <b>${Math.round(l.generation)}ms</b>` : '',
                l.tts !== undefined ? `<span>Bulbul TTS:</span> <b>${Math.round(l.tts)}ms</b>` : '',
                l.total !== undefined ? `<span>Total:</span> <b>${Math.round(l.total)}ms</b>` : '',
              ].filter(Boolean);

              breakdownDiv.innerHTML = items.map((it) => `<div class="latency-item">${it}</div>`).join('');
              bubble.appendChild(breakdownDiv);
            }
          };
        }
        metaDiv.appendChild(latBadge);
      }
    }

    bubble.appendChild(metaDiv);
  }

  if (isUser) {
    row.appendChild(bubble);
    row.appendChild(avatar);
  } else {
    row.appendChild(avatar);
    row.appendChild(bubble);
  }

  return row;
}

/**
 * Creates custom Base64 Audio Player
 */
function createAudioPlayer(base64Data, format = 'wav', autoPlay = false) {
  const card = document.createElement('div');
  card.className = 'audio-player-card';

  if (window._lastVoiceAudioUrl) {
    try {
      URL.revokeObjectURL(window._lastVoiceAudioUrl);
    } catch (_) {}
  }

  const byteCharacters = atob(base64Data);
  const byteNumbers = new Array(byteCharacters.length);
  for (let i = 0; i < byteCharacters.length; i++) {
    byteNumbers[i] = byteCharacters.charCodeAt(i);
  }
  const byteArray = new Uint8Array(byteNumbers);
  const blob = new Blob([byteArray], { type: `audio/${format}` });
  const audioUrl = URL.createObjectURL(blob);

  window._lastVoiceAudioUrl = audioUrl;

  const audio = new Audio(audioUrl);
  window._lastVoiceAudio = audio;

  const playBtn = document.createElement('button');
  playBtn.className = 'play-pause-btn';
  playBtn.setAttribute('type', 'button');
  playBtn.setAttribute('aria-label', 'Play voice response');
  playBtn.innerHTML = `
    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
      <polygon points="5 3 19 12 5 21 5 3"></polygon>
    </svg>
  `;

  const track = document.createElement('div');
  track.className = 'audio-track';

  const progressWrap = document.createElement('div');
  progressWrap.className = 'audio-progress-bar';
  progressWrap.style.cursor = 'pointer';
  const progressFill = document.createElement('div');
  progressFill.className = 'audio-progress-fill';
  progressWrap.appendChild(progressFill);

  const timestamps = document.createElement('div');
  timestamps.className = 'audio-timestamps';
  const curTime = document.createElement('span');
  curTime.textContent = '00:00';
  const durTime = document.createElement('span');
  durTime.textContent = '00:00';
  timestamps.appendChild(curTime);
  timestamps.appendChild(durTime);

  const volumeWrap = document.createElement('div');
  volumeWrap.className = 'audio-volume-wrap';
  const volIcon = document.createElement('span');
  volIcon.className = 'audio-vol-icon';
  volIcon.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path></svg>`;
  const volSlider = document.createElement('input');
  volSlider.type = 'range';
  volSlider.className = 'audio-volume-slider';
  volSlider.min = '0';
  volSlider.max = '1';
  volSlider.step = '0.05';
  volSlider.value = '1';
  volSlider.setAttribute('aria-label', 'Volume');
  volSlider.addEventListener('input', () => { audio.volume = parseFloat(volSlider.value); });
  volumeWrap.appendChild(volIcon);
  volumeWrap.appendChild(volSlider);

  track.appendChild(progressWrap);
  track.appendChild(timestamps);

  function formatTime(s) {
    const mins = Math.floor(s / 60);
    const secs = Math.floor(s % 60);
    return `${mins < 10 ? '0' : ''}${mins}:${secs < 10 ? '0' : ''}${secs}`;
  }

  const setPlayIcon = () => {
    playBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>`;
    playBtn.setAttribute('aria-label', 'Play voice response');
  };
  const setPauseIcon = () => {
    playBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>`;
    playBtn.setAttribute('aria-label', 'Pause voice response');
  };

  audio.onloadedmetadata = () => {
    durTime.textContent = formatTime(audio.duration || 0);
  };

  audio.ontimeupdate = () => {
    if (audio.duration) {
      const pct = (audio.currentTime / audio.duration) * 100;
      progressFill.style.width = `${pct}%`;
      curTime.textContent = formatTime(audio.currentTime);
    }
  };

  audio.onplay = () => setPauseIcon();
  audio.onpause = () => setPlayIcon();

  audio.onended = () => {
    setPlayIcon();
    progressFill.style.width = '0%';
    curTime.textContent = '00:00';
  };

  playBtn.onclick = () => {
    if (audio.paused) {
      state.setCurrentAudio(audioUrl, audio);
      audio.play().catch(() => {});
    } else {
      audio.pause();
    }
  };

  const scrubTo = (e, elem) => {
    const rect = elem.getBoundingClientRect();
    const clickPos = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    if (audio.duration) {
      audio.currentTime = clickPos * audio.duration;
    }
  };

  progressWrap.addEventListener('click', (e) => scrubTo(e, progressWrap));

  card.appendChild(playBtn);
  card.appendChild(track);
  card.appendChild(volumeWrap);

  if (autoPlay) {
    audio.play().catch(() => {});
  }

  return card;
}

/**
 * Creates Sources (N) Accordion with clean ranking, language, and snippets
 */
function createSourcesAccordion(citations, provenance = [], defaultLang = null) {
  const container = document.createElement('div');

  const toggleBtn = document.createElement('button');
  toggleBtn.className = 'sources-toggle-btn';
  toggleBtn.setAttribute('type', 'button');
  toggleBtn.setAttribute('aria-expanded', 'false');
  toggleBtn.innerHTML = `
    <span>Sources (${citations.length})</span>
    <svg class="chevron" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <polyline points="6 9 12 15 18 9"></polyline>
    </svg>
  `;

  const sourcesList = document.createElement('div');
  sourcesList.className = 'sources-container';

  citations.forEach((chunkId, idx) => {
    const prov = Array.isArray(provenance)
      ? provenance.find((p) => (p.chunk_id || p) === chunkId)
      : null;

    const card = document.createElement('div');
    card.className = 'citation-card';

    const header = document.createElement('div');
    header.className = 'citation-header';

    const rankLabel = `#${idx + 1}`;
    const scoreVal = prov && typeof prov.score === 'number' ? `Relevance ${(prov.score).toFixed(2)}` : 'Relevance high';
    const langVal = prov && prov.language ? formatLanguageName(prov.language) : (defaultLang ? formatLanguageName(defaultLang) : 'MSMARCO-XI');
    const stratVal = prov && prov.chunk_type ? prov.chunk_type : 'Semantic';

    header.innerHTML = `
      <span>${escapeHtml(rankLabel)}</span>
      <div class="citation-badges">
        <span class="citation-tag">${escapeHtml(scoreVal)}</span>
        <span class="citation-tag">${escapeHtml(langVal)}</span>
        <span class="citation-tag">${escapeHtml(stratVal)}</span>
      </div>
    `;
    card.appendChild(header);

    // Clean snippet
    if (prov && prov.snippet) {
      const snippet = document.createElement('div');
      snippet.className = 'citation-snippet';
      snippet.textContent = `“${prov.snippet}”`;
      card.appendChild(snippet);
    }

    // Expandable technical details (hiding raw internal Qdrant ID by default)
    const techToggle = document.createElement('button');
    techToggle.className = 'citation-tech-toggle';
    techToggle.setAttribute('type', 'button');
    techToggle.textContent = 'Technical details';

    const techBody = document.createElement('div');
    techBody.className = 'citation-tech-body';
    techBody.hidden = true;
    techBody.innerHTML = `
      <div><b>Chunk ID:</b> <code>${escapeHtml(chunkId)}</code></div>
      ${prov && prov.doc_id ? `<div><b>Document ID:</b> <code>${escapeHtml(prov.doc_id)}</code></div>` : ''}
    `;

    techToggle.onclick = () => {
      techBody.hidden = !techBody.hidden;
      techToggle.textContent = techBody.hidden ? 'Technical details' : 'Hide technical details';
    };

    card.appendChild(techToggle);
    card.appendChild(techBody);

    sourcesList.appendChild(card);
  });

  toggleBtn.onclick = () => {
    const isExpanded = sourcesList.classList.toggle('visible');
    toggleBtn.classList.toggle('expanded', isExpanded);
    toggleBtn.setAttribute('aria-expanded', isExpanded ? 'true' : 'false');
  };

  container.appendChild(toggleBtn);
  container.appendChild(sourcesList);
  return container;
}

// ---------------------------------------------------------------------------
// Keyboard shortcut: Space / P to toggle most recent voice audio
// ---------------------------------------------------------------------------
if (typeof document !== 'undefined') {
  document.addEventListener('keydown', (e) => {
    const tag = document.activeElement && document.activeElement.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    if (e.key === ' ' || e.key === 'p' || e.key === 'P') {
      const audio = window._lastVoiceAudio;
      if (!audio) return;
      e.preventDefault();
      if (audio.paused) {
        audio.play().catch(() => {});
      } else {
        audio.pause();
      }
    }
  });
}
