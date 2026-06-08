async function createNewSession() {
  if (!state.scenarioId && state.scenarios.length) {
    state.scenarioId = state.scenarios[0].scenario_id;
  }
  const query = state.scenarioId ? `?scenario_id=${encodeURIComponent(state.scenarioId)}` : '';
  const payload = await fetchJson(`/api/create_session${query}`, { method: 'POST' });
  state.sessionId = payload.session_id;
  els.sessionInfo.textContent = `会话 ${shortText(state.sessionId, 28)}`;
  renderWelcome();
  await loadHistory();
  updateSideStats();
}

async function loadHistory() {
  if (!state.sessionId) return;
  try {
    const payload = await fetchJson(`/api/history/${encodeURIComponent(state.sessionId)}`);
    const history = payload.history || [];
    if (!history.length) {
      els.historyList.innerHTML = '<div class="empty-state">暂无历史记录</div>';
      return;
    }
    els.historyList.innerHTML = history.slice(-8).reverse().map(item => `
      <div class="history-item" data-question="${escapeAttribute(item.question)}">
        <div class="history-question">${escapeHtml(item.question)}</div>
        <div class="history-answer">${escapeHtml(shortText(item.answer, 44))}</div>
      </div>
    `).join('');
    els.historyList.querySelectorAll('.history-item').forEach(item => {
      item.addEventListener('click', () => {
        els.chatInput.value = item.dataset.question || '';
        autoResizeInput();
        els.chatInput.focus();
      });
    });
  } catch {
    els.historyList.innerHTML = '<div class="empty-state">历史暂不可用</div>';
  }
}

async function clearHistory() {
  if (!state.sessionId) return;
  try {
    await fetchJson(`/api/history/${encodeURIComponent(state.sessionId)}`, { method: 'DELETE' });
  } catch {
    // 历史清理失败不影响页面继续使用。
  }
  renderWelcome('历史已清空。');
  await loadHistory();
}
