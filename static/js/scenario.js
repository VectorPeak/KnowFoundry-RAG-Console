async function loadScenarios() {
  const payload = await fetchJson('/api/scenarios');
  state.scenarios = payload.scenarios || [];
  els.scenarioSelect.innerHTML = '';
  for (const scenario of state.scenarios) {
    const option = document.createElement('option');
    option.value = scenario.scenario_id;
    option.textContent = `${scenario.display_name} (${scenario.industry})`;
    els.scenarioSelect.appendChild(option);
  }
  const saved = localStorage.getItem('activeScenarioId');
  const active = state.scenarios.find(item => item.scenario_id === saved)
    || state.scenarios.find(item => item.scenario_id === payload.active_scenario_id)
    || state.scenarios[0];
  if (active) {
    await applyScenario(active.scenario_id, false);
  }
}

async function applyScenario(scenarioId, resetSession) {
  state.scenarioId = scenarioId;
  localStorage.setItem('activeScenarioId', scenarioId);
  els.scenarioSelect.value = scenarioId;
  const scenario = currentScenario();
  state.lastDiagnostics = null;
  state.lastStreamStatus = '等待提问';
  state.lastHitType = '-';
  state.lastSourceCount = 0;
  state.lastTraceId = null;
  els.scenarioDescription.textContent = scenario ? `${scenario.industry}｜${scenario.description}` : '未选择场景';
  els.contextTitle.textContent = scenario ? scenario.display_name : '知识问答';
  els.contextSubtitle.textContent = scenario ? scenario.business_domain : '等待场景配置';
  await Promise.all([loadSources(), loadKbVersion()]);
  renderSamples();
  updateScopeDisplay();
  updateSideStats();
  if (resetSession) {
    await createNewSession();
  }
}

async function loadSources() {
  const query = state.scenarioId ? `?scenario_id=${encodeURIComponent(state.scenarioId)}` : '';
  const payload = await fetchJson(`/api/sources${query}`);
  const options = payload.source_options || (payload.sources || []).map(item => ({ value: item, label: item }));
  els.sourceFilter.innerHTML = '<option value="">全部</option>';
  for (const source of options) {
    const option = document.createElement('option');
    option.value = source.value;
    option.textContent = source.label;
    els.sourceFilter.appendChild(option);
  }
}

async function loadKbVersion() {
  if (!state.scenarioId) return;
  try {
    const payload = await fetchJson(`/api/kb_versions?scenario_id=${encodeURIComponent(state.scenarioId)}`);
    state.kbVersion = payload.effective_active_version || payload.active_version || null;
  } catch {
    state.kbVersion = null;
  }
  els.kbPill.innerHTML = `<i class="fas fa-code-branch"></i><span>${escapeHtml(shortText(state.kbVersion || '未激活', 24))}</span>`;
}

function currentScenario() {
  return state.scenarios.find(item => item.scenario_id === state.scenarioId);
}
