/**
 * HH Goa 2026 Multilingual Voice RAG — Reactive State Management
 */

class AppState {
  constructor() {
    this.state = {
      language: 'en', // 'en', 'hi', 'ta', 'te', 'ml'
      messages: [],
      isRecording: false,
      recordingDuration: 0,
      loadingStage: null, // null | 'recording' | 'uploading' | 'transcribing' | 'retrieving' | 'generating' | 'synthesizing'
      backendHealth: {
        status: 'checking', // 'healthy' | 'degraded' | 'down' | 'checking'
        lastChecked: null,
      },
      currentAudioUrl: null,
      activeAudioElement: null,
      sourcesSidebarOpen: false,
      leftSidebarOpen: false,
    };

    this.listeners = new Set();
  }

  getState() {
    return this.state;
  }

  setState(partialState) {
    this.state = { ...this.state, ...partialState };
    this.notify();
  }

  subscribe(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  notify() {
    for (const listener of this.listeners) {
      try {
        listener(this.state);
      } catch (err) {
        console.error('Error in state listener:', err);
      }
    }
  }

  setLanguage(lang) {
    this.setState({ language: lang });
  }

  addMessage(message) {
    const newMessage = {
      id: 'msg-' + Date.now() + '-' + Math.random().toString(36).substr(2, 5),
      timestamp: new Date(),
      ...message,
    };
    this.setState({
      messages: [...this.state.messages, newMessage],
    });
    return newMessage;
  }

  updateMessage(id, partial) {
    const updated = this.state.messages.map((m) =>
      m.id === id ? { ...m, ...partial } : m
    );
    this.setState({ messages: updated });
  }

  setLoadingStage(stage) {
    this.setState({ loadingStage: stage });
  }

  setRecording(isRecording, duration = 0) {
    this.setState({ isRecording, recordingDuration: duration });
  }

  setBackendHealth(status, lastChecked = new Date()) {
    this.setState({
      backendHealth: { status, lastChecked },
    });
  }

  /**
   * Safe audio playback lifecycle cleanup
   */
  setCurrentAudio(audioUrl, audioElement) {
    // Revoke previous blob URL to prevent memory leaks
    if (this.state.currentAudioUrl && this.state.currentAudioUrl.startsWith('blob:')) {
      try {
        URL.revokeObjectURL(this.state.currentAudioUrl);
      } catch (_) {}
    }
    // Pause any currently playing audio
    if (this.state.activeAudioElement && !this.state.activeAudioElement.paused) {
      this.state.activeAudioElement.pause();
    }
    this.setState({
      currentAudioUrl: audioUrl,
      activeAudioElement: audioElement,
    });
  }
}

export const state = new AppState();
