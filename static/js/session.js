const SESSION_CARD_STORAGE_KEY = 'knowforge_session_cards_v1';

function loadSessionCards() {
  try {
    const raw = localStorage.getItem(SESSION_CARD_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    state.sessionCards = Array.isArray(parsed) ? parsed : [];
    state.sessionCards = state.sessionCards.filter(item => item && item.sessionId && item.summary !== '历史已清空');
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

function displaySessionId(sessionId) {
  return String(sessionId || '-').replace(/^会话\s+/, '');
}

function upsertSessionCard(sessionId, patch = {}) {
  if (!sessionId) return;
  const now = Date.now();
  const scenarioId = patch.scenarioId || state.scenarioId || '-';
  const existing = state.sessionCards.find(item => item.sessionId === sessionId);
  if (existing) {
    Object.assign(existing, patch, { scenarioId: existing.scenarioId || scenarioId, updatedAt: now });
  } else {
    state.sessionCards.unshift({
      sessionId,
      title: '新会话',
      summary: currentScenario()?.display_name || '等待提问',
      scenarioId,
      createdAt: now,
      updatedAt: now,
      ...patch
    });
  }
  state.sessionCards.sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
  saveSessionCards();
  renderSessionCards();
}

function currentScenarioSessionCards() {
  const scenarioId = state.scenarioId || '-';
  return (state.sessionCards || [])
    .filter(item => (item.scenarioId || '-') === scenarioId)
    .sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
}

function renderSessionCards() {
  const query = (els.sessionSearchInput?.value || '').trim().toLowerCase();
  const cards = currentScenarioSessionCards()
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
      <div class="history-item session-card${active}" data-session-id="${escapeAttribute(item.sessionId)}" role="button" tabindex="0">
        <div class="history-question">${escapeHtml(shortText(item.title || '新会话', 34))}</div>
        <div class="history-answer">${escapeHtml(shortText(item.summary || '等待提问', 54))}</div>
        <div class="history-footer">
          <div class="history-session-id">${escapeHtml(shortText(displaySessionId(item.sessionId), 27))}</div>
          <button class="session-delete-btn" data-session-id="${escapeAttribute(item.sessionId)}" type="button" title="删除对话" aria-label="删除对话">
            <i data-lucide="x"></i>
          </button>
        </div>
      </div>
    `;
  }).join('');
  els.historyList.querySelectorAll('.session-card').forEach(item => {
    item.addEventListener('click', () => switchSession(item.dataset.sessionId));
    item.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        switchSession(item.dataset.sessionId);
      }
    });
  });
  els.historyList.querySelectorAll('.session-delete-btn').forEach(item => {
    item.addEventListener('click', event => {
      event.stopPropagation();
      deleteSession(item.dataset.sessionId);
    });
  });
  refreshIcons();
}

async function restoreLatestSessionForScenario() {
  const latest = currentScenarioSessionCards()[0];
  if (!latest) return false;
  await switchSession(latest.sessionId);
  return true;
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
  const card = state.sessionCards.find(item => item.sessionId === sessionId);
  if (card?.scenarioId && card.scenarioId !== state.scenarioId) {
    state.scenarioId = card.scenarioId;
    els.scenarioSelect.value = card.scenarioId;
    updateScenarioActive();
  }
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

function removeSessionCard(sessionId) {
  if (!sessionId) return;
  state.sessionCards = (state.sessionCards || []).filter(item => item.sessionId !== sessionId);
  saveSessionCards();
  renderSessionCards();
}

async function deleteSession(sessionId) {
  if (!sessionId) return;
  try {
    await fetchJson(`/api/history/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
  } catch {
    // 后端删除失败时仍移除本地会话卡片，避免前端历史列表继续堆积无效对话。
  }
  const deletingActiveSession = sessionId === state.sessionId;
  removeSessionCard(sessionId);
  if (!deletingActiveSession) return;
  state.historyItems = [];
  const next = currentScenarioSessionCards()[0];
  if (next) {
    await switchSession(next.sessionId);
    return;
  }
  await createNewSession();
}

async function clearHistory() {
  if (!state.sessionId) return;
  try {
    await fetchJson(`/api/history/${encodeURIComponent(state.sessionId)}`, { method: 'DELETE' });
  } catch {
    // 历史清理失败不影响页面继续使用。
  }
  state.historyItems = [];
  removeSessionCard(state.sessionId);
  await createNewSession();
}

function filterHistory() {
  renderSessionCards();
}
