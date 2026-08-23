/**
 * HH Goa 2026 Multilingual Voice RAG — Finite State Management & Multilingual Dictionary
 */

export const APP_STATE = Object.freeze({
  IDLE: 'IDLE',
  REQUESTING_MIC_PERMISSION: 'REQUESTING_MIC_PERMISSION',
  RECORDING: 'RECORDING',
  STOPPING: 'STOPPING',
  TRANSCRIBING: 'TRANSCRIBING',
  TRANSCRIPT_READY: 'TRANSCRIPT_READY',
  RETRIEVING: 'RETRIEVING',
  RERANKING: 'RERANKING',
  GENERATING: 'GENERATING',
  SYNTHESIZING: 'SYNTHESIZING',
  ANSWER_READY: 'ANSWER_READY',
  PLAYING: 'PLAYING',
  STT_ERROR: 'STT_ERROR',
  RAG_REFUSAL: 'RAG_REFUSAL',
  TTS_ERROR: 'TTS_ERROR',
  ERROR: 'ERROR',
});

export const MULTILINGUAL_CONFIG = Object.freeze({
  en: {
    code: 'en',
    name: 'English',
    nativeName: 'English',
    fullName: 'English',
    userPromptLabel: 'YOU SAID',
    sttModelLabel: 'English · Saarika v2.5 · ✓ Speech recognized',
    questions: [
      'What is a computer?',
      'How does machine learning work?',
      'What is artificial intelligence?',
      'What is the internet?',
      'What is a programming language?',
    ],
  },
  hi: {
    code: 'hi',
    name: 'Hindi',
    nativeName: 'हिन्दी',
    fullName: 'हिन्दी — Hindi',
    userPromptLabel: 'आपने पूछा',
    sttModelLabel: 'हिन्दी · Saarika v2.5 · ✓ आवाज़ पहचानी गई',
    questions: [
      'कंप्यूटर क्या है?',
      'मशीन लर्निंग क्या है?',
      'कृत्रिम बुद्धिमत्ता क्या है?',
      'इंटरनेट क्या है?',
      'प्रोग्रामिंग भाषा क्या है?',
    ],
  },
  ta: {
    code: 'ta',
    name: 'Tamil',
    nativeName: 'தமிழ்',
    fullName: 'தமிழ் — Tamil',
    userPromptLabel: 'நீங்கள் கேட்டது',
    sttModelLabel: 'தமிழ் · Saarika v2.5 · ✓ Speech recognized',
    questions: [
      'கணினி என்றால் என்ன?',
      'இயந்திர கற்றல் என்றால் என்ன?',
      'செயற்கை நுண்ணறிவு என்றால் என்ன?',
      'இணையம் என்றால் என்ன?',
      'நிரலாக்க மொழி என்றால் என்ன?',
    ],
  },
  te: {
    code: 'te',
    name: 'Telugu',
    nativeName: 'తెలుగు',
    fullName: 'తెలుగు — Telugu',
    userPromptLabel: 'మీరు అడిగారు',
    sttModelLabel: 'తెలుగు · Saarika v2.5 · ✓ Speech recognized',
    questions: [
      'కంప్యూటర్ అంటే ఏమిటి?',
      'మెషిన్ లెర్నింగ్ అంటే ఏమిటి?',
      'కృత్రిమ మేధస్సు అంటే ఏమిటి?',
      'ఇంటర్నెట్ అంటే ఏమిటి?',
      'ప్రోగ్రామింగ్ భాష అంటే ఏమిటి?',
    ],
  },
  ml: {
    code: 'ml',
    name: 'Malayalam',
    nativeName: 'മലയാളം',
    fullName: 'മലയാളം — Malayalam',
    userPromptLabel: 'നിങ്ങൾ ചോദിച്ചത്',
    sttModelLabel: 'മലയാളം · Saarika v2.5 · ✓ Speech recognized',
    questions: [
      'കമ്പ്യൂട്ടർ എന്താണ്?',
      'മെഷീൻ ലേണിംഗ് എന്താണ്?',
      'കൃത്രിമ ബുദ്ധി എന്താണ്?',
      'ഇന്റർനെറ്റ് എന്താണ്?',
      'പ്രോഗ്രാമിംഗ് ഭാഷ എന്താണ്?',
    ],
  },
});

class ApplicationState {
  constructor() {
    this.currentLanguage = 'en';
    this.currentState = APP_STATE.IDLE;
    this.activeAudio = null;
    this.lastRecordedBlob = null;
    this.subscribers = new Set();
  }

  get language() {
    return this.currentLanguage;
  }

  setLanguage(langCode) {
    if (MULTILINGUAL_CONFIG[langCode]) {
      this.currentLanguage = langCode;
      this.notify('language', this.currentLanguage);
    }
  }

  get state() {
    return this.currentState;
  }

  setState(nextState) {
    this.currentState = nextState;
    document.body.dataset.state = nextState;
    const voiceStateElem = document.getElementById('voiceState');
    if (voiceStateElem) {
      voiceStateElem.textContent = nextState.replace(/_/g, ' ').toLowerCase();
    }
    this.notify('state', this.currentState);
  }

  subscribe(listener) {
    this.subscribers.add(listener);
    return () => this.subscribers.delete(listener);
  }

  notify(type, data) {
    for (const sub of this.subscribers) {
      try {
        sub(type, data);
      } catch (err) {
        console.error('State subscriber error:', err);
      }
    }
  }
}

export const state = new ApplicationState();
