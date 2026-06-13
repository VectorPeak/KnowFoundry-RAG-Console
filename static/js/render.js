function renderWelcome(prefix = '') {
  const scenario = currentScenario();
  els.chatHistory.innerHTML = '';
  let content;
  if (scenario) {
    const samples = (scenario.sample_questions || []).slice(0, 3);
    content = [
      prefix,
      prefix ? '' : '',
      `### ${scenario.display_name}`,
      '',
      scenario.description || scenario.business_domain || '',
      '',
      samples.length ? '**试着提问：**' : '',
      samples.map(q => `- ${q}`).join('\n'),
    ].filter(line => line !== undefined).join('\n');
  } else {
    content = '### 欢迎使用 KnowFoundry RAG Console\n\n请在上方选择一个业务场景后开始提问。';
  }
  const welcome = appendMessage('assistant', content, '系统');
  welcome.wrapper.classList.add('welcome-message');
}

function renderSamples() {
  const scenario = currentScenario();
  const questions = (scenario && scenario.sample_questions) || [];
  if (!questions.length) {
    els.sampleQuestions.innerHTML = '<div class="empty-state"><div class="empty-state-icon"><i data-lucide="message-circle-question"></i></div>当前场景暂无示例问题</div>';
    return;
  }
  els.sampleQuestions.innerHTML = questions.slice(0, 6).map(question => `
    <button class="sample-question" data-question="${escapeAttribute(question)}">${escapeHtml(question)}</button>
  `).join('');
  refreshIcons();
  els.sampleQuestions.querySelectorAll('.sample-question').forEach(button => {
    button.addEventListener('click', () => {
      els.chatInput.value = button.dataset.question || '';
      autoResizeInput();
      els.chatInput.focus();
    });
  });
}

function updateScopeDisplay() {
  const sourceLabel = els.sourceFilter.selectedOptions[0]?.textContent || '全部';
  const tenant = els.tenantInput.value.trim() || 'default';
  const dataset = els.datasetInput.value.trim() || 'default';
  const visibility = els.visibilitySelect.value || 'public';
  const role = els.roleSelect.value || 'public';
  els.composerScope.textContent = `${sourceLabel}｜${tenant}/${dataset}｜${visibility}/${role}`;
  updateSideStats();
}

function updateSideStats() {
  const diagnostics = state.lastDiagnostics || {};
  const promptProfile = diagnostics.promptProfile || '-';
  const intentName = diagnostics.intentName || '-';
  const questionCategory = diagnostics.questionCategory || '-';
  const performance = performanceStatus(diagnostics);
  const firstToken = diagnostics.firstTokenMs ? `${diagnostics.firstTokenMs} ms` : '-';
  const totalElapsed = diagnostics.totalElapsedMs ? `${diagnostics.totalElapsedMs} ms` : '-';
  const slowestStage = diagnostics.slowestStageName
    ? `${diagnostics.slowestStageName} ${diagnostics.slowestStageMs || 0} ms`
    : '-';
  els.sideStats.innerHTML = `
    <div class="diagnostic-panel">
      <div class="diagnostic-section-title">运行上下文</div>
      <div class="side-stat"><span>场景</span><strong>${escapeHtml(currentScenario()?.scenario_id || '-')}</strong></div>
      <div class="side-stat"><span>业务分类</span><strong>${escapeHtml(els.sourceFilter.selectedOptions[0]?.textContent || '全部')}</strong></div>
      <div class="side-stat"><span>数据域</span><strong>${escapeHtml(scopeLabel())}</strong></div>
      <div class="side-stat"><span>知识库版本</span><strong title="${escapeAttribute(state.kbVersion || '-')}">${escapeHtml(shortText(state.kbVersion || '-', 22))}</strong></div>
    </div>
    <div class="diagnostic-panel">
      <div class="diagnostic-section-title">最近一次回答</div>
      <div class="performance-badge ${escapeAttribute(performance.level)}"><i data-lucide="${escapeAttribute(performance.icon)}"></i><span>${escapeHtml(performance.label)}</span></div>
      <div class="side-stat"><span>流式状态</span><strong>${escapeHtml(state.lastStreamStatus)}</strong></div>
      <div class="side-stat"><span>命中路径</span><strong>${escapeHtml(state.lastHitType)}</strong></div>
      <div class="side-stat"><span>Prompt</span><strong title="${escapeAttribute(promptProfile)}">${escapeHtml(shortText(promptProfile, 22))}</strong></div>
      <div class="side-stat"><span>意图/类别</span><strong title="${escapeAttribute(`${intentName} / ${questionCategory}`)}">${escapeHtml(shortText(`${intentName} / ${questionCategory}`, 22))}</strong></div>
      <div class="side-stat"><span>来源数量</span><strong>${escapeHtml(String(state.lastSourceCount))}</strong></div>
      <div class="side-stat"><span>首 token</span><strong>${escapeHtml(firstToken)}</strong></div>
      <div class="side-stat"><span>总耗时</span><strong>${escapeHtml(totalElapsed)}</strong></div>
      <div class="side-stat"><span>最慢阶段</span><strong title="${escapeAttribute(slowestStage)}">${escapeHtml(shortText(slowestStage, 22))}</strong></div>
      <div class="side-stat"><span>Trace</span><strong title="${escapeAttribute(state.lastTraceId || '-')}">${escapeHtml(shortText(state.lastTraceId || '-', 22))}</strong></div>
    </div>
    ${renderSideSourceList(diagnostics.sources || [])}
  `;
  refreshIcons();
}

function appendMessage(role, content, meta, rawHtml = false) {
  const wrapper = document.createElement('div');
  wrapper.className = `message ${role === 'user' ? 'user' : 'assistant'}`;
  const metaElement = document.createElement('div');
  metaElement.className = 'message-meta';
  metaElement.textContent = meta || (role === 'user' ? '你' : '助手');
  const contentElement = document.createElement('div');
  contentElement.className = 'message-content';
  contentElement.innerHTML = rawHtml ? content : renderMarkdown(content);
  wrapper.appendChild(metaElement);
  wrapper.appendChild(contentElement);
  els.chatHistory.appendChild(wrapper);
  refreshIcons();
  scrollToBottom();
  return { wrapper, content: contentElement };
}

function renderSources(sources) {
  const wrapper = document.createElement('div');
  wrapper.className = 'answer-sources';
  const title = document.createElement('div');
  title.className = 'message-meta';
  title.textContent = '参考来源';
  wrapper.appendChild(title);
  sources.slice(0, 5).forEach((source, index) => {
    const metadata = source.metadata || {};
    const label = source.citation || metadata.file_name || metadata.standard_question || metadata.source || `来源 ${index + 1}`;
    const table = source.table || {};
    const tableMeta = table.row_number
      ? `表格行 · ${table.sheet_name || '工作表'} · 第 ${table.row_number} 行`
      : '';
    const scoreValue = Number(source.score);
    const score = Number.isFinite(scoreValue) ? scoreValue.toFixed(3) : '-';
    const item = document.createElement('div');
    item.className = `source-item${tableMeta ? ' table-source' : ''}`;
    item.innerHTML = `
      <span>
        ${index + 1}. ${escapeHtml(label)}
        ${tableMeta ? `<small>${escapeHtml(tableMeta)}</small>` : ''}
      </span>
      <span>${score}</span>
    `;
    wrapper.appendChild(item);
  });
  return wrapper;
}

function scopeLabel() {
  const tenant = els.tenantInput.value.trim() || 'default';
  const dataset = els.datasetInput.value.trim() || 'default';
  const visibility = els.visibilitySelect.value || 'public';
  const role = els.roleSelect.value || 'public';
  return `${tenant}/${dataset}/${visibility}/${role}`;
}

function numberOrNull(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function formatMs(value) {
  const number = numberOrNull(value);
  return number === null ? '-' : `${Math.round(number)} ms`;
}

function sourceLabel(source, index) {
  const metadata = source.metadata || {};
  return source.citation
    || metadata.file_name
    || metadata.standard_question
    || metadata.h1
    || metadata.source
    || `来源 ${index + 1}`;
}

function buildDiagnosticsSnapshot(event, sources) {
  const retrieval = event.retrieval || {};
  const plan = retrieval.plan || {};
  const promptProfile = retrieval.prompt_profile || {};
  const intent = event.intent || {};
  const slowest = event.slowest_stage || retrieval.slowest_stage || {};
  return {
    traceId: event.trace_id || state.lastTraceId || '-',
    hitType: event.hit_type || '-',
    scenarioId: retrieval.scenario_id || event.scenario_id || state.scenarioId || '-',
    scenarioName: retrieval.scenario_name || currentScenario()?.display_name || '-',
    kbVersion: retrieval.kb_version || event.kb_version || state.kbVersion || '-',
    sourceFilter: retrieval.source_filter || els.sourceFilter.value || '全部',
    promptProfile: promptProfile.name || '-',
    promptReason: promptProfile.reason || '',
    intentName: intent.intent || '-',
    intentConfidence: numberOrNull(intent.confidence),
    intentReason: intent.reason || '',
    classification: retrieval.classification || null,
    questionCategory: plan.question_category || '-',
    planReason: plan.reason || '-',
    queryVariants: retrieval.query_variants || plan.query_variants || [],
    faqTopScore: numberOrNull(retrieval.faq_top_score),
    firstTokenMs: numberOrNull(event.first_token_ms || retrieval.first_token_ms),
    totalElapsedMs: numberOrNull(retrieval.total_elapsed_ms || event.processing_time * 1000),
    slowestStageName: slowest.name || '',
    slowestStageMs: numberOrNull(slowest.elapsed_ms),
    stageTimings: event.stage_timings_ms || retrieval.stage_timings_ms || {},
    sources: sources || []
  };
}

function renderClassificationResult(classification) {
  if (!els.classificationResult) return;
  const candidates = Array.isArray(classification?.candidates)
    ? classification.candidates.filter(item => item && item.label)
    : [];
  if (!candidates.length) {
    els.classificationResult.innerHTML = `
      <div><span>建议分类</span><strong class="classification-badge is-waiting">等待分类</strong></div>
      <ol></ol>
      <p>分类结果会随最近一次回答更新，辅助判断问题归属与资料适用范围</p>
    `;
    return;
  }

  const suggestedLabel = classification.suggested_label || candidates[0].label;
  const rows = candidates.slice(0, 4).map(item => {
    const score = Number(item.score);
    const normalized = Number.isFinite(score) ? Math.max(0, Math.min(1, score)) : 0;
    return `
      <li style="--score: ${(normalized * 100).toFixed(0)}%" title="${escapeAttribute(`${item.source || item.label} · raw ${item.raw_score ?? '-'}`)}">
        <span>${escapeHtml(item.label)}</span>
        <strong>${normalized.toFixed(2)}</strong>
      </li>
    `;
  }).join('');

  els.classificationResult.innerHTML = `
    <div><span>建议分类</span><strong class="classification-badge">${escapeHtml(suggestedLabel)}</strong></div>
    <ol>${rows}</ol>
    <p>分类结果来自最近一次问题的业务资料分类信号，按匹配强度归一化展示。</p>
  `;
}

function renderAnswerDiagnostics(diagnostics) {
  const wrapper = document.createElement('div');
  wrapper.className = 'answer-diagnostics';
  const stageEntries = Object.entries(diagnostics.stageTimings || {})
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .slice(0, 4);
  const variants = (diagnostics.queryVariants || []).slice(0, 3);
  wrapper.innerHTML = `
    <div class="answer-diagnostics-title">
      <i data-lucide="chart-line"></i>
      <span>检索诊断</span>
    </div>
    <div class="diagnostic-chips">
      <span>路径：${escapeHtml(diagnostics.hitType)}</span>
      <span>模板：${escapeHtml(diagnostics.promptProfile)}</span>
      <span>类别：${escapeHtml(diagnostics.questionCategory)}</span>
      <span>性能：${escapeHtml(performanceStatus(diagnostics).label)}</span>
      <span>首 token：${escapeHtml(formatMs(diagnostics.firstTokenMs))}</span>
    </div>
    <div class="diagnostic-grid">
      <div><span>意图</span><strong>${escapeHtml(diagnostics.intentName)}${diagnostics.intentConfidence === null ? '' : ` · ${diagnostics.intentConfidence}`}</strong></div>
      <div><span>FAQ 最高分</span><strong>${diagnostics.faqTopScore === null ? '-' : diagnostics.faqTopScore.toFixed(3)}</strong></div>
      <div><span>知识库版本</span><strong title="${escapeAttribute(diagnostics.kbVersion)}">${escapeHtml(shortText(diagnostics.kbVersion, 28))}</strong></div>
      <div><span>最慢阶段</span><strong>${escapeHtml(diagnostics.slowestStageName || '-')}${diagnostics.slowestStageMs === null ? '' : ` · ${formatMs(diagnostics.slowestStageMs)}`}</strong></div>
    </div>
    ${variants.length ? `<div class="diagnostic-mini-list"><span>查询变体</span>${variants.map(item => `<code>${escapeHtml(item)}</code>`).join('')}</div>` : ''}
    ${stageEntries.length ? `<div class="stage-bars">${stageEntries.map(([name, value]) => renderStageBar(name, value, stageEntries[0][1])).join('')}</div>` : ''}
  `;
  return wrapper;
}

function performanceStatus(diagnostics) {
  const firstToken = numberOrNull(diagnostics.firstTokenMs);
  const total = numberOrNull(diagnostics.totalElapsedMs);
  if (firstToken === null && total === null) {
    return { level: 'idle', label: '等待数据', icon: 'circle' };
  }
  if ((firstToken !== null && firstToken > 8000) || (total !== null && total > 15000)) {
    return { level: 'slow', label: '较慢', icon: 'triangle-alert' };
  }
  if ((firstToken !== null && firstToken > 4000) || (total !== null && total > 8000)) {
    return { level: 'warn', label: '偏慢', icon: 'clock' };
  }
  return { level: 'ok', label: '正常', icon: 'circle-check' };
}

function renderStageBar(name, value, maxValue) {
  const current = Number(value) || 0;
  const max = Math.max(Number(maxValue) || 1, 1);
  const width = Math.max(6, Math.min(100, (current / max) * 100));
  return `
    <div class="stage-bar">
      <div class="stage-bar-label"><span>${escapeHtml(name)}</span><strong>${escapeHtml(formatMs(current))}</strong></div>
      <div class="stage-bar-track"><span style="width:${width.toFixed(1)}%"></span></div>
    </div>
  `;
}

function renderSideSourceList(sources) {
  if (!sources.length) {
    return '<div class="diagnostic-panel"><div class="diagnostic-section-title">命中来源</div><div class="empty-state compact">暂无来源</div></div>';
  }
  return `
    <div class="diagnostic-panel">
      <div class="diagnostic-section-title">命中来源</div>
      <div class="side-source-list">
        ${sources.slice(0, 4).map((source, index) => {
          const score = numberOrNull(source.score);
          const metadata = source.metadata || {};
          const sourceType = source.source_type || metadata.source_type || '-';
          return `
            <div class="side-source-item">
              <div title="${escapeAttribute(sourceLabel(source, index))}">${index + 1}. ${escapeHtml(shortText(sourceLabel(source, index), 24))}</div>
              <span>${escapeHtml(sourceType)}${score === null ? '' : ` · ${score.toFixed(3)}`}</span>
            </div>
          `;
        }).join('')}
      </div>
    </div>
  `;
}

function renderFeedbackActions(question, answer, sources) {
  const wrapper = document.createElement('div');
  wrapper.className = 'answer-actions';
  const label = document.createElement('span');
  label.textContent = '反馈';
  const useful = document.createElement('button');
  useful.title = '有用';
  useful.innerHTML = '<i data-lucide="thumbs-up"></i>'; refreshIcons();
  const notUseful = document.createElement('button');
  notUseful.title = '无用';
  notUseful.innerHTML = '<i data-lucide="thumbs-down"></i>'; refreshIcons();

  const submit = async rating => {
    useful.disabled = true;
    notUseful.disabled = true;
    try {
      await fetchJson('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: state.sessionId,
          scenario_id: state.scenarioId,
          tenant_id: els.tenantInput.value.trim() || 'default',
          dataset_id: els.datasetInput.value.trim() || 'default',
          question,
          answer,
          rating,
          sources
        })
      });
      wrapper.textContent = '反馈已记录';
    } catch {
      wrapper.textContent = '反馈暂未保存';
    }
  };

  useful.addEventListener('click', () => submit('useful'));
  notUseful.addEventListener('click', () => submit('not_useful'));
  wrapper.append(label, useful, notUseful);
  refreshIcons();
  return wrapper;
}

function setConnectionState(type, text) {
  const map = { ready: 'ok', error: 'error', connecting: 'warn', disconnected: '' };
  const state = map[type] || '';
  els.connectionPill.className = `pill ${state}`;
  els.connectionPill.innerHTML = `<span class="status-dot"></span><span>${escapeHtml(text)}</span>`;
  refreshIcons();
}

function setWebSocketHealth(type, text) {
  const healthClass = {
    ok: 'status-ok',
    working: 'status-working',
    pending: 'status-pending',
    error: 'status-error'
  }[type] || 'status-pending';
  if (els.websocketHealth) {
    els.websocketHealth.className = healthClass;
    els.websocketHealth.textContent = text;
  }
  if (els.servicePillText) {
    els.servicePillText.textContent = type === 'error' ? '异常 6/7' : type === 'pending' ? '待检测' : type === 'working' ? '检测中' : '正常 7/7';
  }
}

function updateSendState() {
  els.sendBtn.classList.toggle('is-stopping', state.inProgress);
  els.sendBtn.title = state.inProgress ? '停止' : '发送';
  els.sendBtn.innerHTML = state.inProgress ? '<i data-lucide="square"></i><span>停止</span>' : '<i data-lucide="send-horizontal"></i><span>发送</span>';
  refreshIcons();
}

function autoResizeInput() {
  els.chatInput.style.height = 'auto';
  els.chatInput.style.height = `${Math.min(els.chatInput.scrollHeight, 160)}px`;
}

function scrollToBottom() {
  els.chatHistory.scrollTop = els.chatHistory.scrollHeight;
}

/* ---- Toast notification system ---- */
function showToast(message, type = 'info', duration = 3500) {
  let container = document.querySelector('.toast-container');
  if (!container) {
    container = document.createElement('div');
    container.className = 'toast-container';
    document.body.appendChild(container);
  }

  const icons = { success: 'circle-check', error: 'circle-alert', warning: 'triangle-alert', info: 'circle-info' };
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <i data-lucide="${icons[type] || icons.info}" class="toast-icon"></i>
    <span class="toast-message">${escapeHtml(message)}</span>
    <button class="toast-close" aria-label="关闭"><i data-lucide="x"></i></button>
  `;
  toast.querySelector('.toast-close').addEventListener('click', () => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(20px)';
    setTimeout(() => toast.remove(), 200);
  });

  container.appendChild(toast);
  refreshIcons();

  setTimeout(() => {
    if (toast.parentNode) {
      toast.style.opacity = '0';
      toast.style.transform = 'translateX(20px)';
      setTimeout(() => toast.remove(), 200);
    }
  }, duration);
}


// Lucide replaces SVG icons after DOM fragments are inserted dynamically.
// This keeps rendered diagnostics, feedback buttons, and toast icons aligned with the static shell.
function refreshIcons() {
  if (window.lucide && typeof window.lucide.createIcons === 'function') {
    window.lucide.createIcons();
  }
}
