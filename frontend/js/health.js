/**
 * HH Goa 2026 Multilingual Voice RAG — Health & Observability Inspector
 */

import { ApiService } from './api.js';

export class HealthInspector {
  static async updateSystemStatus() {
    const statusText = document.getElementById('healthStatusText');
    const statusCard = document.getElementById('healthStatusCard');

    if (!statusText || !statusCard) return;

    try {
      const data = await ApiService.checkHealth();
      if (data.status === 'healthy') {
        statusCard.className = 'status-badge-card healthy';
        statusText.textContent = 'System Healthy & All Providers Connected';
      } else {
        statusCard.className = 'status-badge-card';
        statusCard.style.background = 'var(--warning-soft)';
        statusCard.style.color = 'var(--warning)';
        statusText.textContent = `Status: ${data.status || 'Degraded'}`;
      }
    } catch (_) {
      statusCard.className = 'status-badge-card';
      statusCard.style.background = 'var(--danger-soft)';
      statusCard.style.color = 'var(--danger)';
      statusText.textContent = 'Backend Offline';
    }
  }
}
