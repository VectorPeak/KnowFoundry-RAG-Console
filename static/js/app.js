document.addEventListener('DOMContentLoaded', async () => {
  bindEvents();
  await loadScenarios();
  await createNewSession();
  renderWelcome();
});

function bindEvents() {
  els.newSessionBtn?.addEventListener('click', createNewSession);
  els.sidebarNewSessionBtn?.addEventListener('click', createNewSession);
  els.clearHistoryBtn?.addEventListener('click', clearHistory);
  els.sendBtn.addEventListener('click', () => state.inProgress ? cancelStream() : sendMessage());
  els.scenarioSelect.addEventListener('change', async () => {
    if (state.inProgress) cancelStream();
    await applyScenario(els.scenarioSelect.value, true);
  });
  [els.sourceFilter, els.tenantInput, els.datasetInput, els.visibilitySelect, els.roleSelect].forEach(item => {
    item.addEventListener('change', updateScopeDisplay);
    item.addEventListener('input', updateScopeDisplay);
  });
  els.sourceFilter.addEventListener('change', updateCategoryActive);
  els.sessionSearchInput?.addEventListener('input', filterHistory);
  els.chatInput.addEventListener('input', autoResizeInput);
  els.chatInput.addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      if (!state.inProgress) sendMessage();
    }
  });
}
