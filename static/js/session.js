const SESSION_CARD_STORAGE_KEY = 'knowforge_session_cards_v1';

function loadSessionCards() {
  try {
    const raw = localStorage.getItem(SESSION_CARD_STORAGE_KEY);
    state.sessionCards = raw ? JSON.parse(raw) : [];
  } catch {
    state.sessionCards = [];
  }
}

function saveSessionCards() {
  try {
    localStorage.setItem(SESSION_CARD_STORAGE_KEY, JSON.stringify(state.sessionCards.slice(0, 30)));
  } catch {
    // 本地存储不可用时，会话列表只在当前页面生命周期内保留。
  }
}

function upsertSessionCard(sessionId, patch = {}) {
  if (!sessionId) return;
  const now = Date.now();
  const existing = state.sessionCards.find(item => item.sessionId === sessionId);
  if (existing) {
    Object.assign(existing, patch, { updatedAt: now });
  } else {
    state.sessionCards.unshift({
      sessionId,
      title: '新会话',
      summary: currentScenario()?.display_name || '等待提问',
      scenarioId: state.scenarioId || '-',
      createdAt: now,
      updatedAt: now,
      ...patch
    });
  }
  state.sessionCards.sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
  saveSessionCards();
  renderSessionCards();
}

function renderSessionCards() {
  const query = (els.sessionSearchInput?.value || '').trim().toLowerCase();
  const cards = (state.sessionCards || [])
    .filter(item => {
      const text = `${item.title || ''} ${item.summary || ''} ${item.sessionId || ''}`.toLowerCase();
      return !query || text.includes(query);
    })
    .slice(0, 20);
  if (!cards.length) {
    els.historyList.innerHTML = `<div class="empty-state">${query ? '暂无匹配会话' : '暂无历史会话'}</div>`;
    return;
  }
  els.historyList.innerHTML = cards.map(item => {
    const active = item.sessionId === state.sessionId ? ' active' : '';
    return `
      <button class="history-item session-card${active}" data-session-id="${escapeAttribute(item.sessionId)}" type="button">
        <div class="history-question">${escapeHtml(shortText(item.title || '新会话', 34))}</div>
        <div class="history-answer">${escapeHtml(shortText(item.summary || '等待提问', 54))}</div>
        <div class="history-session-id">会话 ${escapeHtml(shortText(item.sessionId || '-', 22))}</div>
      </button>
    `;
  }).join('');
  els.historyList.querySelectorAll('.session-card').forEach(item => {
    item.addEventListener('click', () => switchSession(item.dataset.sessionId));
  });
}

async function createNewSession() {
  if (!state.scenarioId && state.scenarios.length) {
    state.scenarioId = state.scenarios[0].scenario_id;
  }
  if (els.sessionSearchInput) {
    els.sessionSearchInput.value = '';
  }
  const query = state.scenarioId ? `?scenario_id=${encodeURIComponent(state.scenarioId)}` : '';
  const payload = await fetchJson(`/api/create_session${query}`, { method: 'POST' });
  state.sessionId = payload.session_id;
  state.historyItems = [];
  els.sessionInfo.textContent = `当前会话 ${shortText(state.sessionId, 28)}`;
  upsertSessionCard(state.sessionId, {
    title: '新会话',
    summary: currentScenario()?.display_name || '等待提问',
    scenarioId: state.scenarioId || '-'
  });
  renderWelcome();
  updateSideStats();
}

async function switchSession(sessionId) {
  if (!sessionId || sessionId === state.sessionId) return;
  if (state.inProgress) cancelStream();
  state.sessionId = sessionId;
  els.sessionInfo.textContent = `当前会话 ${shortText(state.sessionId, 28)}`;
  renderSessionCards();
  await loadHistory(true);
  updateSideStats();
}

async function loadHistory(renderChat = false) {
  if (!state.sessionId) return;
  try {
    const payload = await fetchJson(`/api/history/${encodeURIComponent(state.sessionId)}`);
    state.historyItems = payload.history || [];
    if (state.historyItems.length) {
      const latest = state.historyItems[state.historyItems.length - 1];
      upsertSessionCard(state.sessionId, {
        title: latest.question || '新会话',
        summary: latest.answer || '等待回答'
      });
    } else {
      renderSessionCards();
    }
    if (renderChat) {
      renderChatHistoryFromItems(state.historyItems);
    }
  } catch {
    renderSessionCards();
    if (renderChat) {
      renderWelcome('历史暂不可用。');
    }
  }
}

function renderChatHistoryFromItems(items) {
  els.chatHistory.innerHTML = '';
  if (!items || !items.length) {
    renderWelcome();
    return;
  }
  items.forEach(item => {
    if (item.question) appendMessage('user', item.question, '你');
    if (item.answer) appendMessage('assistant', item.answer, '助手');
  });
}

async function clearHistory() {
  if (!state.sessionId) return;
  try {
    await fetchJson(`/api/history/${encodeURIComponent(state.sessionId)}`, { method: 'DELETE' });
  } catch {
    // 历史清理失败不影响页面继续使用。
  }
  renderWelcome('历史已清空。');
  state.historyItems = [];
  upsertSessionCard(state.sessionId, {
    title: '新会话',
    summary: '历史已清空'
  });
}

function filterHistory() {
  renderSessionCards();
}
