/**
 * HH Goa 2026 Multilingual Voice RAG — Main Application Entry Point (Phase 6.25 Dataset-Centric)
 */

import { state } from './state.js';
import { voiceManager } from './voice.js';
import { ChatController } from './chat.js';
import { healthMonitor } from './health.js';
import { showToast } from './ui.js';

document.addEventListener('DOMContentLoaded', () => {
  const chat = new ChatController('chatMessages', 'welcomeState');

  // DOM Elements — Inputs & Actions
  const queryInput = document.getElementById('queryInput');
  const sendBtn = document.getElementById('sendBtn');
  const micBtn = document.getElementById('micBtn');
  const welcomeMicBtn = document.getElementById('welcomeMicBtn');
  const charCounter = document.getElementById('charCounter');
  const langSelect = document.getElementById('langSelect');
  const suggestionChips = document.querySelectorAll('.suggestion-chip');
  const langTabs = document.querySelectorAll('.lang-tab');

  // DOM Elements — Recording Bar
  const recordingBar = document.getElementById('recordingBar');
  const recordingStatusLabel = document.getElementById('recordingStatusLabel');
  const recordingTimer = document.getElementById('recordingTimer');
  const recordingRemaining = document.getElementById('recordingRemaining');
  const stopRecordingBtn = document.getElementById('stopRecordingBtn');
  const cancelRecordingBtn = document.getElementById('cancelRecordingBtn');
  const waveformBars = document.querySelectorAll('.waveform-bar');

  // DOM Elements — Drawers & Backdrops
  const drawerBackdrop = document.getElementById('drawerBackdrop');

  // Menu Drawer
  const menuBtn = document.getElementById('menuBtn');
  const menuDrawer = document.getElementById('menuDrawer');
  const closeMenuBtn = document.getElementById('closeMenuBtn');

  // Dataset Drawer
  const headerDatasetBtn = document.getElementById('headerDatasetBtn');
  const footerDatasetBtn = document.getElementById('footerDatasetBtn');
  const openDatasetFromMenu = document.getElementById('openDatasetFromMenu');
  const datasetDrawer = document.getElementById('datasetDrawer');
  const closeDatasetBtn = document.getElementById('closeDatasetBtn');

  // Pipeline Drawer
  const headerPipelineBtn = document.getElementById('headerPipelineBtn');
  const footerPipelineBtn = document.getElementById('footerPipelineBtn');
  const openPipelineFromMenu = document.getElementById('openPipelineFromMenu');
  const pipelineDrawer = document.getElementById('pipelineDrawer');
  const closePipelineBtn = document.getElementById('closePipelineBtn');

  // Status Drawer
  const statusBtn = document.getElementById('statusBtn');
  const footerStatusBtn = document.getElementById('footerStatusBtn');
  const openStatusFromMenu = document.getElementById('openStatusFromMenu');
  const statusDrawer = document.getElementById('statusDrawer');
  const closeStatusBtn = document.getElementById('closeStatusBtn');
  const refreshHealthBtn = document.getElementById('refreshHealthBtn');

  // About Drawer
  const openAboutBtn = document.getElementById('openAboutBtn');
  const aboutDrawer = document.getElementById('aboutDrawer');
  const closeAboutBtn = document.getElementById('closeAboutBtn');

  // Chat Actions
  const newChatBtn = document.getElementById('newChatBtn');
  const clearChatBtn = document.getElementById('clearChatBtn');

  // Start background health polling
  healthMonitor.start();

  // --------------------------------------------------------------------------
  // 1. Language Selector with Session Synchronization
  // --------------------------------------------------------------------------
  if (langSelect) {
    langSelect.addEventListener('change', (e) => {
      const selectedLang = e.target.value;
      state.setLanguage(selectedLang);
      syncLangTab(selectedLang);
      const text = e.target.options[e.target.selectedIndex].text;
      showToast(`Language set to ${text}`, 'info', 2000);
    });
  }

  // --------------------------------------------------------------------------
  // 2. Suggestion Language Tabs
  // --------------------------------------------------------------------------
  function syncLangTab(langCode) {
    langTabs.forEach((tab) => {
      const isMatch = tab.getAttribute('data-lang') === langCode;
      tab.classList.toggle('active', isMatch);
      tab.setAttribute('aria-selected', isMatch ? 'true' : 'false');
    });

    const grids = document.querySelectorAll('.suggestion-grid');
    grids.forEach((grid) => {
      const isMatch = grid.getAttribute('data-lang-group') === langCode;
      grid.classList.toggle('active', isMatch);
      grid.hidden = !isMatch;
    });
  }

  langTabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      const lang = tab.getAttribute('data-lang');
      if (lang) {
        state.setLanguage(lang);
        if (langSelect) langSelect.value = lang;
        syncLangTab(lang);
      }
    });
  });

  // --------------------------------------------------------------------------
  // 3. Textarea Auto-Resize & Character Counter
  // --------------------------------------------------------------------------
  if (queryInput) {
    queryInput.addEventListener('input', () => {
      queryInput.style.height = 'auto';
      const newHeight = Math.max(48, Math.min(queryInput.scrollHeight, 160));
      queryInput.style.height = `${newHeight}px`;

      const len = queryInput.value.length;
      if (charCounter) {
        charCounter.textContent = `${len} / 2000`;
        if (len > 2000) {
          charCounter.className = 'char-counter over';
        } else if (len > 1850) {
          charCounter.className = 'char-counter warn';
        } else {
          charCounter.className = 'char-counter';
        }
      }

      if (sendBtn) {
        sendBtn.disabled = len === 0 || len > 2000 || state.getState().loadingStage !== null;
      }
    });

    queryInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        const text = queryInput.value.trim();
        if (text && (!sendBtn || !sendBtn.disabled)) {
          handleTextSubmit();
        }
      }
    });
  }

  // --------------------------------------------------------------------------
  // 4. Text Submit Handler
  // --------------------------------------------------------------------------
  if (sendBtn) {
    sendBtn.addEventListener('click', handleTextSubmit);
  }

  function handleTextSubmit() {
    if (!queryInput) return;
    const text = queryInput.value.trim();
    if (!text) return;

    queryInput.value = '';
    queryInput.style.height = '48px';
    if (charCounter) {
      charCounter.textContent = '0 / 2000';
      charCounter.className = 'char-counter';
    }
    if (sendBtn) sendBtn.disabled = true;

    chat.submitTextQuery(text);
  }

  // --------------------------------------------------------------------------
  // 5. Suggested Questions Click
  // --------------------------------------------------------------------------
  suggestionChips.forEach((chip) => {
    chip.addEventListener('click', () => {
      const q = chip.getAttribute('data-query') || chip.textContent.trim();
      if (queryInput) {
        queryInput.value = q;
        queryInput.dispatchEvent(new Event('input'));
        handleTextSubmit();
      }
    });
  });

  // --------------------------------------------------------------------------
  // 6. Voice Interaction Flow (Inline Recording Bar with Waveform)
  // --------------------------------------------------------------------------
  if (micBtn) {
    micBtn.addEventListener('click', () => {
      if (state.getState().isRecording) {
        handleStopRecording();
      } else {
        handleStartRecording();
      }
    });
  }

  if (welcomeMicBtn) {
    welcomeMicBtn.addEventListener('click', () => {
      handleStartRecording();
    });
  }

  if (stopRecordingBtn) {
    stopRecordingBtn.addEventListener('click', handleStopRecording);
  }

  if (cancelRecordingBtn) {
    cancelRecordingBtn.addEventListener('click', handleCancelRecording);
  }

  async function handleStartRecording() {
    try {
      if (micBtn) {
        micBtn.classList.add('recording');
        micBtn.setAttribute('aria-label', 'Stop voice recording');
      }
      if (recordingBar) {
        recordingBar.hidden = false;
      }
      if (recordingStatusLabel) {
        recordingStatusLabel.textContent = 'Listening...';
      }

      await voiceManager.startRecording(
        // onWaveform callback
        (frequencyData) => {
          if (waveformBars && waveformBars.length > 0) {
            for (let i = 0; i < waveformBars.length; i++) {
              const val = frequencyData[i * 2] || 0;
              const barHeight = Math.max(4, Math.min(24, (val / 255) * 28));
              waveformBars[i].style.height = `${barHeight}px`;
            }
          }
        },
        // onMaxDuration callback — auto-stop at 60s
        () => {
          showToast('Maximum recording length (60s) reached — submitting automatically.', 'info', 3000);
          handleStopRecording();
        },
        // onRemainingTime callback
        (remainingSec) => {
          if (recordingRemaining) {
            if (remainingSec <= 10) {
              recordingRemaining.textContent = `${remainingSec}s left`;
              recordingRemaining.hidden = false;
            } else {
              recordingRemaining.hidden = true;
            }
          }
        }
      );
    } catch (err) {
      resetRecordingUI();
      showToast(err.message, 'error');
    }
  }

  async function handleStopRecording() {
    if (recordingStatusLabel) {
      recordingStatusLabel.textContent = 'Processing voice...';
    }
    try {
      const { blob, filename } = await voiceManager.stopRecording();
      resetRecordingUI();
      chat.submitVoiceQuery(blob, filename, () => handleStartRecording());
    } catch (err) {
      resetRecordingUI();
      if (err.message && err.message !== 'No active recording.') {
        showToast(err.message, 'warning');
      }
    }
  }

  function handleCancelRecording() {
    voiceManager.cancelRecording();
    resetRecordingUI();
    showToast('Recording cancelled', 'info', 1500);
  }

  function resetRecordingUI() {
    if (micBtn) {
      micBtn.classList.remove('recording');
      micBtn.setAttribute('aria-label', 'Start voice recording');
    }
    if (recordingBar) {
      recordingBar.hidden = true;
    }
    if (recordingRemaining) {
      recordingRemaining.hidden = true;
    }
    if (waveformBars) {
      waveformBars.forEach((bar) => {
        bar.style.height = '6px';
      });
    }
  }

  // --------------------------------------------------------------------------
  // 7. Drawers Management
  // --------------------------------------------------------------------------
  function openDrawer(drawerElem, triggerBtn) {
    closeAllDrawers();
    if (!drawerElem) return;
    drawerElem.hidden = false;
    if (drawerBackdrop) drawerBackdrop.hidden = false;
    if (triggerBtn) triggerBtn.setAttribute('aria-expanded', 'true');
    const isLeft = drawerElem.classList.contains('drawer-left');
    drawerElem.classList.add(isLeft ? 'animating-in-left' : 'animating-in-right');
  }

  function closeAllDrawers() {
    [menuDrawer, datasetDrawer, pipelineDrawer, statusDrawer, aboutDrawer].forEach((d) => {
      if (d) {
        d.hidden = true;
        d.classList.remove('animating-in-left', 'animating-in-right');
      }
    });
    if (drawerBackdrop) drawerBackdrop.hidden = true;
    if (menuBtn) menuBtn.setAttribute('aria-expanded', 'false');
    if (statusBtn) statusBtn.setAttribute('aria-expanded', 'false');
  }

  // Menu Drawer
  if (menuBtn) menuBtn.addEventListener('click', () => openDrawer(menuDrawer, menuBtn));
  if (closeMenuBtn) closeMenuBtn.addEventListener('click', closeAllDrawers);

  // Dataset Drawer
  if (headerDatasetBtn) headerDatasetBtn.addEventListener('click', () => openDrawer(datasetDrawer, headerDatasetBtn));
  if (footerDatasetBtn) footerDatasetBtn.addEventListener('click', () => openDrawer(datasetDrawer, footerDatasetBtn));
  if (openDatasetFromMenu) openDatasetFromMenu.addEventListener('click', () => openDrawer(datasetDrawer, null));
  if (closeDatasetBtn) closeDatasetBtn.addEventListener('click', closeAllDrawers);

  // Pipeline Drawer
  if (headerPipelineBtn) headerPipelineBtn.addEventListener('click', () => openDrawer(pipelineDrawer, headerPipelineBtn));
  if (footerPipelineBtn) footerPipelineBtn.addEventListener('click', () => openDrawer(pipelineDrawer, footerPipelineBtn));
  if (openPipelineFromMenu) openPipelineFromMenu.addEventListener('click', () => openDrawer(pipelineDrawer, null));
  if (closePipelineBtn) closePipelineBtn.addEventListener('click', closeAllDrawers);

  // Status Drawer
  if (statusBtn) statusBtn.addEventListener('click', () => openDrawer(statusDrawer, statusBtn));
  if (footerStatusBtn) footerStatusBtn.addEventListener('click', () => openDrawer(statusDrawer, footerStatusBtn));
  if (openStatusFromMenu) openStatusFromMenu.addEventListener('click', () => openDrawer(statusDrawer, statusBtn));
  if (closeStatusBtn) closeStatusBtn.addEventListener('click', closeAllDrawers);

  // About Drawer
  if (openAboutBtn) openAboutBtn.addEventListener('click', () => openDrawer(aboutDrawer, null));
  if (closeAboutBtn) closeAboutBtn.addEventListener('click', closeAllDrawers);

  // Backdrop click & Escape key
  if (drawerBackdrop) drawerBackdrop.addEventListener('click', closeAllDrawers);

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeAllDrawers();
      if (queryInput) queryInput.blur();
    }
  });

  // --------------------------------------------------------------------------
  // 8. New Session / Reset Chat
  // --------------------------------------------------------------------------
  function handleResetChat() {
    chat.clear();
    closeAllDrawers();
    if (queryInput) {
      queryInput.value = '';
      queryInput.style.height = '48px';
    }
    showToast('New query session started', 'info', 2000);
  }

  if (newChatBtn) newChatBtn.addEventListener('click', handleResetChat);
  if (clearChatBtn) clearChatBtn.addEventListener('click', handleResetChat);

  // --------------------------------------------------------------------------
  // 9. Manual Health Refresh
  // --------------------------------------------------------------------------
  if (refreshHealthBtn) {
    refreshHealthBtn.addEventListener('click', async () => {
      showToast('Refreshing system status...', 'info', 1500);
      await healthMonitor.check();
    });
  }

  // --------------------------------------------------------------------------
  // 10. Reactive State Subscriptions
  // --------------------------------------------------------------------------
  state.subscribe((s) => {
    if (recordingTimer && s.isRecording) {
      const m = Math.floor(s.recordingDuration / 60);
      const sec = s.recordingDuration % 60;
      recordingTimer.textContent = `${m < 10 ? '0' : ''}${m}:${sec < 10 ? '0' : ''}${sec}`;
    }

    const isBusy = s.loadingStage !== null;
    if (sendBtn && !s.isRecording) {
      sendBtn.disabled = isBusy || (queryInput ? queryInput.value.trim().length === 0 : true);
    }
    if (micBtn) {
      micBtn.disabled = isBusy && !s.isRecording;
    }
  });
});
