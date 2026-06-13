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
  renderScenarioList(state.scenarios);
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
  updateScenarioActive();
  const scenario = currentScenario();
  state.lastDiagnostics = null;
  state.lastStreamStatus = '等待提问';
  state.lastHitType = '-';
  state.lastSourceCount = 0;
  state.lastTraceId = null;
  // The sidebar describes the platform capability, while the selected scene remains visible in the main header.
  els.scenarioDescription.textContent = '支持多场景、多专有知识库下的知识助手。';
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

function renderScenarioList(scenarios) {
  if (!els.scenarioList) return;
  if (!scenarios.length) {
    els.scenarioList.innerHTML = '<div class="empty-state compact">暂无业务场景</div>';
    return;
  }
  els.scenarioList.innerHTML = scenarios.map(scenario => `
    <button class="scenario-item" type="button" data-scenario="${escapeAttribute(scenario.scenario_id)}">
      <i data-lucide="${escapeAttribute(scenarioIcon(scenario))}"></i>
      <span>${escapeHtml(scenario.display_name)}</span>
    </button>
  `).join('');
  els.scenarioList.querySelectorAll('.scenario-item').forEach(button => {
    button.addEventListener('click', async () => {
      if (state.inProgress) cancelStream();
      await applyScenario(button.dataset.scenario, true);
    });
  });
  updateScenarioActive();
  refreshIcons();
}

function updateScenarioActive() {
  if (!els.scenarioList) return;
  const activeValue = state.scenarioId || els.scenarioSelect.value || '';
  els.scenarioList.querySelectorAll('.scenario-item').forEach(button => {
    button.classList.toggle('active', button.dataset.scenario === activeValue);
  });
}

function scenarioIcon(scenario) {
  const text = `${scenario.display_name || ''} ${scenario.industry || ''} ${scenario.business_domain || ''}`;
  if (text.includes('合规') || text.includes('法务')) return 'scale';
  if (text.includes('贸易') || text.includes('跨境')) return 'ship';
  if (text.includes('工程') || text.includes('施工')) return 'hard-hat';
  if (text.includes('企业') || text.includes('内部')) return 'building-2';
  if (text.includes('设备') || text.includes('运维')) return 'settings-2';
  if (text.includes('保险') || text.includes('理赔')) return 'badge-dollar-sign';
  if (text.includes('客服') || text.includes('SaaS')) return 'headphones';
  if (text.includes('招投标') || text.includes('合同')) return 'file-signature';
  return 'layers-3';
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
  renderCategoryList(options);
}

function renderCategoryList(options) {
  if (!els.categoryList) return;
  const normalized = [{ value: '', label: '全部分类' }, ...options];
  els.categoryList.innerHTML = normalized.map((source, index) => `
    <button class="taxonomy-item${index === 0 ? ' active' : ''}" type="button" data-source="${escapeAttribute(source.value)}">
      <i data-lucide="${escapeAttribute(categoryIcon(source.label, index))}"></i>
      <span>${escapeHtml(source.label)}</span>
    </button>
  `).join('');
  els.categoryList.querySelectorAll('.taxonomy-item').forEach(button => {
    button.addEventListener('click', () => {
      els.sourceFilter.value = button.dataset.source || '';
      updateCategoryActive();
      updateScopeDisplay();
    });
  });
  refreshIcons();
}

function updateCategoryActive() {
  if (!els.categoryList) return;
  const activeValue = els.sourceFilter.value || '';
  els.categoryList.querySelectorAll('.taxonomy-item').forEach(button => {
    button.classList.toggle('active', (button.dataset.source || '') === activeValue);
  });
}

function categoryIcon(label, index) {
  const text = String(label || '');
  if (text.includes('安全') || text.includes('风控') || text.includes('合规')) return 'shield-check';
  if (text.includes('质量') || text.includes('验收') || text.includes('审核')) return 'badge-check';
  if (text.includes('图纸') || text.includes('变更') || text.includes('合同')) return 'file-diff';
  if (text.includes('进度') || text.includes('计划') || text.includes('流程')) return 'calendar-days';
  if (text.includes('设备') || text.includes('巡检') || text.includes('运维')) return 'settings-2';
  if (text.includes('客服') || text.includes('问答')) return 'messages-square';
  return ['folder-tree', 'clipboard-list', 'database', 'file-text'][index % 4];
}

async function loadKbVersion() {
  if (!state.scenarioId) return;
  try {
    const payload = await fetchJson(`/api/kb_versions?scenario_id=${encodeURIComponent(state.scenarioId)}`);
    state.kbVersion = payload.effective_active_version || payload.active_version || null;
  } catch {
    state.kbVersion = null;
  }
  els.kbPill.innerHTML = `<i data-lucide="git-branch"></i><span>${escapeHtml(shortText(state.kbVersion || '未激活', 24))}</span>`;
  refreshIcons();
}

function currentScenario() {
  return state.scenarios.find(item => item.scenario_id === state.scenarioId);
}
