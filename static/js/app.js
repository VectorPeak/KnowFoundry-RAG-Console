document.addEventListener('DOMContentLoaded', async () => {
  bindEvents();
  bindInsightCardToggles();
  loadSessionCards();
  renderSessionCards();
  await loadScenarios();
  const restored = await restoreLatestSessionForScenario();
  if (!restored) {
    await createNewSession();
  }
});

function bindEvents() {
  els.newSessionBtn?.addEventListener('click', createNewSession);
  els.sidebarNewSessionBtn?.addEventListener('click', createNewSession);
  els.clearHistoryBtn?.addEventListener('click', clearHistory);
  els.sendBtn.addEventListener('click', () => state.inProgress ? cancelStream() : sendMessage());
  els.scenarioSelect?.addEventListener('change', async () => {
    if (state.inProgress) cancelStream();
    await applyScenario(els.scenarioSelect.value, true);
  });
  [els.sourceFilter, els.tenantInput, els.datasetInput, els.visibilitySelect, els.roleSelect].forEach(item => {
    item?.addEventListener('change', updateScopeDisplay);
    item?.addEventListener('input', updateScopeDisplay);
  });
  els.sourceFilter?.addEventListener('change', updateCategoryActive);
  els.sessionSearchInput?.addEventListener('input', filterHistory);
  els.chatInput.addEventListener('input', autoResizeInput);
  els.chatInput.addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      if (!state.inProgress) sendMessage();
    }
  });
}

function bindInsightCardToggles() {
  document.querySelectorAll('.right-panel .insight-card').forEach(card => {
    const title = card.querySelector('.side-section-title');
    const icon = card.querySelector('.card-toggle-icon');
    if (!title || !icon) return;
    title.setAttribute('role', 'button');
    title.setAttribute('tabindex', '0');
    title.setAttribute('aria-expanded', 'true');
    icon.setAttribute('aria-hidden', 'true');
    const toggle = () => {
      const collapsed = card.classList.toggle('is-collapsed');
      title.setAttribute('aria-expanded', String(!collapsed));
    };
    title.addEventListener('click', toggle);
    title.addEventListener('keydown', event => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        toggle();
      }
    });
  });
}
