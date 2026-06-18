(function () {
  const tokenInput = document.getElementById('adminTokenInput');
  const scenarioSelect = document.getElementById('scenarioSelect');
  const refreshBtn = document.getElementById('refreshBtn');

  tokenInput.value = localStorage.getItem('qa_admin_token') || '';

  function adminHeaders() {
    const token = tokenInput.value.trim();
    return token ? { 'X-Admin-Token': token } : {};
  }

  async function fetchJson(url, withToken = true) {
    const response = await fetch(url, { headers: withToken ? adminHeaders() : {} });
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
    return response.json();
  }

  function escapeHtml(value) {
    return String(value ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function text(value) {
    return value === undefined || value === null || value === '' ? '-' : String(value);
  }

  function dateText(value) {
    if (!value) return '-';
    if (typeof value === 'number') return new Date(value * 1000).toLocaleString();
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
  }

  function shortText(value, max = 48) {
    const raw = text(value);
    return raw.length > max ? `${raw.slice(0, max - 1)}...` : raw;
  }

  function showBanner(message, type = 'warn') {
    const banner = document.getElementById('adminBanner');
    banner.textContent = message;
    banner.className = `admin-banner is-${type}`;
    banner.style.display = '';
  }

  function hideBanner() {
    const banner = document.getElementById('adminBanner');
    banner.style.display = 'none';
  }

  function iconSvg(name = 'info') {
    const icons = new Set([
      'activity',
      'alert',
      'database',
      'file-check',
      'folder',
      'info',
      'key',
      'link',
      'shield',
    ]);
    const safeName = icons.has(name) ? name : 'info';
    return `<svg class="ui-icon" aria-hidden="true"><use href="#icon-${safeName}"></use></svg>`;
  }

  function item(label, value, icon = 'info', tone = '') {
    return `
      <div class="quality-item ${tone ? `is-${escapeHtml(tone)}` : ''}">
        <div class="quality-icon">${iconSvg(icon)}</div>
        <div>
          <div class="quality-title">${escapeHtml(label)}</div>
          <div class="quality-desc">${escapeHtml(text(value))}</div>
        </div>
      </div>
    `;
  }

  function fileSummary(label, summary) {
    const available = summary && summary.available;
    const value = available
      ? `${summary.file || '-'} | ${dateText(summary.updated_at)}`
      : '暂无报告';
    return item(label, value, available ? 'file-check' : 'alert', available ? 'ok' : 'warn');
  }

  function statusBadge(status) {
    const raw = text(status);
    const normalized = raw.toLowerCase();
    const tone = normalized === 'active' ? 'is-active' : normalized === 'staged' ? 'is-staged' : '';
    return `<span class="status-badge ${tone}">${escapeHtml(raw)}</span>`;
  }

  function statsSummary(stats = {}) {
    const faq = stats.last_faq_count ?? stats.total_faq_written ?? 0;
    const docs = stats.last_doc_count ?? stats.total_doc_written ?? 0;
    const runs = (stats.doc_ingest_runs ?? 0) + (stats.faq_ingest_runs ?? 0);
    return `FAQ ${faq} 文档 ${docs} 入库 ${runs}`;
  }

  function statsCell(stats = {}) {
    const json = JSON.stringify(stats || {}, null, 2);
    return `
      <button class="stats-chip" type="button" data-stats-json="${escapeHtml(json)}" aria-label="复制统计 JSON">
        <span>${escapeHtml(statsSummary(stats))}</span>
        <pre>${escapeHtml(json)}</pre>
      </button>
    `;
  }

  function renderLangSmith(status) {
    document.getElementById('langsmithEnabledValue').textContent = status.enabled ? '已开启' : '未开启';
    document.getElementById('langsmithProjectValue').textContent = status.project || '-';
    document.getElementById('langsmithStatus').innerHTML = [
      item('Tracing', status.enabled ? 'LANGSMITH_TRACING=true' : 'LANGSMITH_TRACING=false', 'activity', status.enabled ? 'ok' : 'warn'),
      item('Project', status.project || '-', 'folder'),
      item('Endpoint', status.endpoint || '-', 'link'),
      item('API Key', status.has_api_key ? '已配置' : '未配置', status.has_api_key ? 'key' : 'alert', status.has_api_key ? 'ok' : 'warn'),
    ].join('');
  }

  function renderVersions(payload) {
    const versions = payload.versions || [];
    document.getElementById('kbValue').textContent = shortText(payload.effective_active_version || '-', 24);
    document.getElementById('versionSource').textContent = `来源：${text(payload.active_version_source)}`;
    if (!versions.length) {
      document.getElementById('versionRows').innerHTML = '<tr><td colspan="5">暂无版本清单</td></tr>';
      return;
    }
    document.getElementById('versionRows').innerHTML = versions.map(version => `
      <tr>
        <td class="mono">${escapeHtml(version.kb_version || '-')}</td>
        <td>${statusBadge(version.status || '-')}</td>
        <td>${escapeHtml(dateText(version.created_at))}</td>
        <td>${escapeHtml(shortText(version.description || version.notes || '-', 70))}</td>
        <td>${statsCell(version.stats || {})}</td>
      </tr>
    `).join('');
  }

  function renderIngestion(rows) {
    if (!rows.length) {
      document.getElementById('ingestionList').innerHTML = item('入库质量报告', '暂无报告', 'alert', 'warn');
      return;
    }
    document.getElementById('ingestionList').innerHTML = rows.map(row => {
      const summary = row.summary || {};
      const value = [
        `文件 ${summary.files_loaded_count ?? 0}/${summary.files_scanned ?? 0}`,
        `FAQ 冲突 ${summary.faq_document_conflicts?.count ?? 0}`,
        `表格 ${summary.table_files_count ?? 0}`,
        `OCR 风险 ${summary.ocr_risk_files_count ?? 0}`,
      ].join(' | ');
      return item(row.file_name || row.path || '入库报告', value, summary.ok === false ? 'alert' : 'file-check', summary.ok === false ? 'warn' : 'ok');
    }).join('');
  }

  function renderGates(gates, performance) {
    document.getElementById('gateList').innerHTML = [
      ...((gates.reports || []).map(report => fileSummary('质量回归', report))),
      ...((performance.reports || []).map(report => fileSummary('性能回归', report))),
    ].join('') || item('回归报告', '暂无报告', 'alert', 'warn');
  }

  function renderGovernance(payload) {
    document.getElementById('enterpriseGovernanceList').innerHTML = [
      fileSummary('资料真实度', payload.data_realism),
      fileSummary('增强包预检', payload.overlay_readiness),
    ].join('');
  }

  async function loadScenarios() {
    const payload = await fetchJson('/api/scenarios', false);
    const scenarios = payload.scenarios || [];
    scenarioSelect.innerHTML = scenarios.map(scenario => `
      <option value="${escapeHtml(scenario.scenario_id)}">${escapeHtml(scenario.display_name || scenario.scenario_id)}</option>
    `).join('');
    scenarioSelect.value = payload.active_scenario_id || scenarios[0]?.scenario_id || '';
    document.getElementById('scenarioCountValue').textContent = `${scenarios.length} 个场景`;
  }

  async function loadDashboard() {
    hideBanner();
    localStorage.setItem('qa_admin_token', tokenInput.value.trim());
    const scenarioId = scenarioSelect.value;
    try {
      const [
        adminStatus,
        langsmith,
        versions,
        ingestion,
        gates,
        performance,
        governance,
      ] = await Promise.all([
        fetchJson('/api/admin/status'),
        fetchJson('/api/admin/langsmith'),
        fetchJson(`/api/kb_versions?scenario_id=${encodeURIComponent(scenarioId)}`),
        fetchJson(`/api/admin/ingestion_reports?scenario_id=${encodeURIComponent(scenarioId)}&limit=10`),
        fetchJson('/api/admin/gate_reports'),
        fetchJson('/api/admin/performance_reports'),
        fetchJson('/api/admin/enterprise_governance'),
      ]);

      document.getElementById('updatedAt').textContent = new Date().toLocaleString();
      const activeScenario = (adminStatus.scenarios || []).includes(scenarioId) ? scenarioSelect.selectedOptions[0]?.textContent : scenarioId;
      document.getElementById('scenarioValue').textContent = activeScenario || '-';
      renderLangSmith(langsmith);
      renderVersions(versions);
      renderIngestion(ingestion.reports || []);
      renderGates(gates || {}, performance || {});
      renderGovernance(governance || {});
    } catch (error) {
      showBanner(`加载失败：${error.message}。请确认 ADMIN_API_TOKEN 是否正确。`, 'warn');
    }
  }

  refreshBtn.addEventListener('click', loadDashboard);
  tokenInput.addEventListener('change', loadDashboard);
  scenarioSelect.addEventListener('change', loadDashboard);
  document.addEventListener('click', event => {
    const statsButton = event.target.closest('.stats-chip');
    if (!statsButton) return;
    const value = statsButton.dataset.statsJson || '{}';
    navigator.clipboard?.writeText(value).catch(() => {});
  });

  loadScenarios()
    .then(loadDashboard)
    .catch(error => showBanner(`场景加载失败：${error.message}`, 'warn'));
})();
