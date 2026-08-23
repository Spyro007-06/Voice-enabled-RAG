/**
 * HH Goa 2026 Multilingual Voice RAG — Main Application Entry Point
 */

import { state, APP_STATE, MULTILINGUAL_CONFIG } from './state.js';
import { voiceManager } from './voice.js';
import { ChatManager } from './chat.js';
import { UIManager } from './ui.js';
import { HealthInspector } from './health.js';

// DOM Elements
const elements = {
  menuBtn: document.getElementById('menuBtn'),
  closeMenuBtn: document.getElementById('closeMenuBtn'),
  menuDrawer: document.getElementById('menuDrawer'),
  drawerBackdrop: document.getElementById('drawerBackdrop'),
  langSelect: document.getElementById('langSelect'),
  languageTabs: document.getElementById('languageTabs'),
  suggestionsContainer: document.getElementById('suggestionsContainer'),
  welcomeState: document.getElementById('welcomeState'),
  centralVoiceBtn: document.getElementById('centralVoiceBtn'),
  welcomeMicBtn: document.getElementById('welcomeMicBtn'),
  micBtn: document.getElementById('micBtn'),
  micLabel: document.getElementById('micLabel'),
  queryInput: document.getElementById('queryInput'),
  sendBtn: document.getElementById('sendBtn'),
  charCounter: document.getElementById('charCounter'),
  recordingBar: document.getElementById('recordingBar'),
  waveform: document.getElementById('waveform'),
  recordingTimer: document.getElementById('recordingTimer'),
  recordingLangBadge: document.getElementById('recordingLangBadge'),
  stopRecordingBtn: document.getElementById('stopRecordingBtn'),
  cancelRecordingBtn: document.getElementById('cancelRecordingBtn'),
  chatMessages: document.getElementById('chatMessages'),
  newChatBtn: document.getElementById('newChatBtn'),
  clearChatBtn: document.getElementById('clearChatBtn'),
  openDatasetBtn: document.getElementById('openDatasetFromMenu'),
  openPipelineBtn: document.getElementById('openPipelineFromMenu'),
  openStatusBtn: document.getElementById('openStatusFromMenu'),
  openAboutBtn: document.getElementById('openAboutBtn'),
  datasetDrawer: document.getElementById('datasetDrawer'),
  pipelineDrawer: document.getElementById('pipelineDrawer'),
  statusDrawer: document.getElementById('statusDrawer'),
  aboutDrawer: document.getElementById('aboutDrawer'),
};

/**
 * Open or close specified drawer
 */
function setDrawerOpen(drawerElement, isOpen) {
  if (!drawerElement) return;

  // Close all other info drawers first
  [
    elements.menuDrawer,
    elements.datasetDrawer,
    elements.pipelineDrawer,
    elements.statusDrawer,
    elements.aboutDrawer,
  ].forEach((d) => {
    if (d && d !== drawerElement) d.hidden = true;
  });

  drawerElement.hidden = !isOpen;
  if (elements.drawerBackdrop) elements.drawerBackdrop.hidden = !isOpen;

  if (elements.menuBtn) {
    elements.menuBtn.setAttribute('aria-expanded', String(isOpen && drawerElement === elements.menuDrawer));
  }

  if (isOpen && drawerElement === elements.statusDrawer) {
    HealthInspector.updateSystemStatus();
  }
}

function closeAllDrawers() {
  [
    elements.menuDrawer,
    elements.datasetDrawer,
    elements.pipelineDrawer,
    elements.statusDrawer,
    elements.aboutDrawer,
  ].forEach((d) => {
    if (d) d.hidden = true;
  });
  if (elements.drawerBackdrop) elements.drawerBackdrop.hidden = true;
  if (elements.menuBtn) elements.menuBtn.setAttribute('aria-expanded', 'false');
}

/**
 * Render Quick Language Chips & Select Box
 */
function updateLanguageUI(langCode) {
  const config = MULTILINGUAL_CONFIG[langCode] || MULTILINGUAL_CONFIG.en;

  // Update Select Element
  if (elements.langSelect) elements.langSelect.value = langCode;

  // Update Recording Lang Badge
  if (elements.recordingLangBadge) {
    elements.recordingLangBadge.textContent = `${config.nativeName} · Saarika v2.5`;
  }

  // Update Language Chips
  if (elements.languageTabs) {
    elements.languageTabs.replaceChildren(
      ...Object.entries(MULTILINGUAL_CONFIG).map(([code, item]) => {
        const btn = document.createElement('button');
        btn.className = 'lang-chip';
        btn.textContent = item.nativeName;
        btn.setAttribute('aria-pressed', String(code === langCode));
        btn.onclick = () => {
          state.setLanguage(code);
        };
        return btn;
      })
    );
  }

  // Update Suggested Corpus Questions
  if (elements.suggestionsContainer) {
    elements.suggestionsContainer.replaceChildren(
      ...config.questions.map((question) => {
        const btn = document.createElement('button');
        btn.className = 'suggestion-btn';
        btn.textContent = `“${question}”`;
        btn.onclick = () => {
          if (elements.queryInput) elements.queryInput.value = '';
          ChatManager.handleTextSubmission(question);
        };
        return btn;
      })
    );
  }

  UIManager.announce(`Language set to ${config.fullName}`);
}

/**
 * Reactively apply app state to the DOM: mic button class/label, welcome screen
 */
function applyAppState(appState) {
  const micBtns = [
    elements.centralVoiceBtn,
    elements.micBtn,
    elements.welcomeMicBtn,
  ].filter(Boolean);

  const isRecording = appState === APP_STATE.RECORDING;
  const isBusy = [
    APP_STATE.REQUESTING_MIC_PERMISSION,
    APP_STATE.STOPPING,
    APP_STATE.TRANSCRIBING,
    APP_STATE.RETRIEVING,
    APP_STATE.RERANKING,
    APP_STATE.GENERATING,
    APP_STATE.SYNTHESIZING,
  ].includes(appState);

  // Toggle .recording class on hero mic button
  micBtns.forEach((btn) => {
    btn.classList.toggle('recording', isRecording);
    btn.classList.toggle('processing', isBusy);
    btn.disabled = isBusy;
  });

  // Update the mic rings wrapper
  const rings = document.querySelector('.mic-rings');
  if (rings) {
    rings.classList.toggle('is-recording', isRecording);
    rings.classList.toggle('is-processing', isBusy);
  }

  // Update hero label
  if (elements.micLabel) {
    if (isRecording) {
      elements.micLabel.textContent = 'Listening… tap to stop';
    } else if (isBusy) {
      elements.micLabel.textContent = 'Processing…';
    } else {
      elements.micLabel.textContent = 'Tap to speak';
    }
  }

  // Hide welcome screen as soon as user starts interacting
  if (isRecording || isBusy) {
    if (elements.welcomeState) elements.welcomeState.hidden = true;
  }
}

/**
 * Voice Recording Controls
 */
async function startVoiceRecording() {
  const config = MULTILINGUAL_CONFIG[state.language] || MULTILINGUAL_CONFIG.en;
  if (elements.recordingLangBadge) {
    elements.recordingLangBadge.textContent = `${config.nativeName} · Saarika v2.5`;
  }

  if (elements.recordingTimer) elements.recordingTimer.textContent = '00:00';
  if (elements.recordingBar) elements.recordingBar.hidden = false;

  [elements.centralVoiceBtn, elements.micBtn, elements.welcomeMicBtn].forEach((btn) => {
    if (btn) btn.setAttribute('aria-label', 'Stop voice recording');
  });

  UIManager.announce(`Listening. Recording in ${config.name} started.`);

  await voiceManager.startRecording({
    onTimerTick: (seconds) => {
      const mins = Math.floor(seconds / 60);
      const secs = String(seconds % 60).padStart(2, '0');
      if (elements.recordingTimer) {
        elements.recordingTimer.textContent = `0${mins}:${secs}`;
      }
    },
    onAutoStop: async () => {
      UIManager.showToast('Reached 60-second limit. Processing voice input…');
      await stopVoiceRecording();
    },
    onError: (err) => {
      if (elements.recordingBar) elements.recordingBar.hidden = true;
      [elements.centralVoiceBtn, elements.micBtn, elements.welcomeMicBtn].forEach((btn) => {
        if (btn) btn.setAttribute('aria-label', 'Start voice recording');
      });

      if (err.code === 'NO_MIC_DETECTED') {
        UIManager.showNoMicModal({
          onRetry: () => startVoiceRecording(),
          onTypeInstead: () => {
            if (elements.queryInput) {
              elements.queryInput.focus();
              elements.queryInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
          },
        });
      } else if (err.code === 'PERMISSION_DENIED') {
        UIManager.showMicPermissionModal({
          onRetry: () => startVoiceRecording(),
        });
      } else {
        const msg = `Microphone error: ${err.message || 'Unable to access audio device'}`;
        UIManager.showToast(msg, true);
        UIManager.announce('Microphone access denied or unavailable.');
      }
    },
  });
}

async function stopVoiceRecording() {
  if (elements.recordingBar) elements.recordingBar.hidden = true;

  [elements.centralVoiceBtn, elements.micBtn, elements.welcomeMicBtn].forEach((btn) => {
    if (btn) btn.setAttribute('aria-label', 'Start voice recording');
  });

  try {
    const audioBlob = await voiceManager.stopRecording();
    if (audioBlob) {
      await ChatManager.handleVoiceSubmission(audioBlob);
    }
  } catch (err) {
    UIManager.showToast(err.message || 'Failed to capture audio', true);
  }
}

function cancelVoiceRecording() {
  voiceManager.cancelRecording();
  if (elements.recordingBar) elements.recordingBar.hidden = true;

  [elements.centralVoiceBtn, elements.micBtn, elements.welcomeMicBtn].forEach((btn) => {
    if (btn) btn.setAttribute('aria-label', 'Start voice recording');
  });

  UIManager.announce('Recording cancelled.');
}

/**
 * Handle Send Text Button / Enter Key
 */
function submitComposerText() {
  if (!elements.queryInput) return;
  const text = elements.queryInput.value.trim();
  if (!text) return;

  elements.queryInput.value = '';
  if (elements.charCounter) elements.charCounter.textContent = '0 / 2000';
  if (elements.sendBtn) elements.sendBtn.disabled = true;

  ChatManager.handleTextSubmission(text);
}

/**
 * Initialize Application
 */
document.addEventListener('DOMContentLoaded', () => {
  // 1. Initialize Waveform Canvas
  voiceManager.initCanvas(elements.waveform);

  // 2. State Subscriber — drives mic button + label + welcome visibility reactively
  state.subscribe((type, data) => {
    if (type === 'language') {
      updateLanguageUI(data);
    }

    if (type === 'state') {
      applyAppState(data);
    }
  });
  state.setLanguage('en');

  // 3. Language Selector Dropdown
  if (elements.langSelect) {
    elements.langSelect.onchange = (e) => {
      state.setLanguage(e.target.value);
    };
  }

  // 4. Voice Mic Trigger Buttons
  const toggleRecording = () => {
    if (voiceManager.isRecording()) {
      stopVoiceRecording();
    } else {
      startVoiceRecording();
    }
  };

  if (elements.centralVoiceBtn) elements.centralVoiceBtn.onclick = toggleRecording;
  if (elements.welcomeMicBtn) elements.welcomeMicBtn.onclick = toggleRecording;
  if (elements.micBtn) elements.micBtn.onclick = toggleRecording;

  if (elements.stopRecordingBtn) elements.stopRecordingBtn.onclick = stopVoiceRecording;
  if (elements.cancelRecordingBtn) elements.cancelRecordingBtn.onclick = cancelVoiceRecording;

  // 5. Composer Textarea & Send Button
  if (elements.queryInput) {
    elements.queryInput.oninput = (e) => {
      const len = e.target.value.length;
      if (elements.charCounter) elements.charCounter.textContent = `${len} / 2000`;
      if (elements.sendBtn) elements.sendBtn.disabled = !e.target.value.trim();

      // Auto-grow
      e.target.style.height = 'auto';
      e.target.style.height = `${Math.min(e.target.scrollHeight, 140)}px`;
    };

    elements.queryInput.onkeydown = (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        submitComposerText();
      }
    };
  }

  if (elements.sendBtn) {
    elements.sendBtn.onclick = submitComposerText;
  }

  // 6. Drawer Navigation Controls
  if (elements.menuBtn) elements.menuBtn.onclick = () => setDrawerOpen(elements.menuDrawer, true);
  if (elements.closeMenuBtn) elements.closeMenuBtn.onclick = () => setDrawerOpen(elements.menuDrawer, false);
  if (elements.drawerBackdrop) elements.drawerBackdrop.onclick = closeAllDrawers;

  // Info Drawers
  if (elements.openDatasetBtn) elements.openDatasetBtn.onclick = () => setDrawerOpen(elements.datasetDrawer, true);
  if (elements.openPipelineBtn) elements.openPipelineBtn.onclick = () => setDrawerOpen(elements.pipelineDrawer, true);
  if (elements.openStatusBtn) elements.openStatusBtn.onclick = () => setDrawerOpen(elements.statusDrawer, true);
  if (elements.openAboutBtn) elements.openAboutBtn.onclick = () => setDrawerOpen(elements.aboutDrawer, true);

  // Close buttons on all info drawers
  document.querySelectorAll('.close-info-drawer').forEach((btn) => {
    btn.onclick = closeAllDrawers;
  });

  // 7. Global Keyboard Shortcuts (Escape to close drawers/recording/modals)
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      if (voiceManager.isRecording()) {
        cancelVoiceRecording();
      }
      closeAllDrawers();
      UIManager.closeModal('noMicModal');
      UIManager.closeModal('micPermModal');
    }
  });

  const modalBackdrop = document.getElementById('modalBackdrop');
  if (modalBackdrop) {
    modalBackdrop.onclick = () => {
      UIManager.closeModal('noMicModal');
      UIManager.closeModal('micPermModal');
    };
  }

  // 8. New / Clear Chat
  const resetConversation = () => {
    if (elements.chatMessages) elements.chatMessages.replaceChildren();
    if (elements.welcomeState) elements.welcomeState.hidden = false;
    state.setState(APP_STATE.IDLE);
    closeAllDrawers();
    UIManager.announce('New conversation started.');
  };

  if (elements.newChatBtn) elements.newChatBtn.onclick = resetConversation;
  if (elements.clearChatBtn) elements.clearChatBtn.onclick = resetConversation;

  // 9. Initial Health Check
  HealthInspector.updateSystemStatus();
});
