async function fetchJson(url, options = {}) {
  const response = await fetch(url, { cache: 'no-store', ...options });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

function streamAnswer(query, contentElement) {
  if (state.socket && state.socket.readyState === WebSocket.OPEN) {
    state.socket.close();
  }

  return new Promise((resolve, reject) => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    state.socket = new WebSocket(`${protocol}//${window.location.host}${API_BASE_URL}/api/stream`);
    let answer = '';
    let sources = [];
    let completed = false;
    let settled = false;
    let socketError = false;

    const finish = result => {
      if (settled) return;
      settled = true;
      resolve(result);
    };

    setWebSocketHealth('working', '检测中');

    state.socket.onopen = () => {
      setWebSocketHealth('ok', '正常');
      state.lastStreamStatus = '已连接，正在提交问题';
      updateSideStats();
      setConnectionState('working', '生成中');
      state.socket.send(JSON.stringify({
        query,
        source_filter: els.sourceFilter.value,
        session_id: state.sessionId,
        scenario_id: state.scenarioId,
        tenant_id: els.tenantInput.value.trim() || 'default',
        dataset_id: els.datasetInput.value.trim() || 'default',
        visibility: els.visibilitySelect.value || 'public',
        user_role: els.roleSelect.value || 'public'
      }));
    };

    state.socket.onmessage = event => {
      const data = JSON.parse(event.data);
      if (data.trace_id) {
        state.lastTraceId = data.trace_id;
      }
      if (data.type === 'start') {
        state.kbVersion = data.kb_version || state.kbVersion;
        state.lastStreamStatus = '请求已接收';
        updateSideStats();
      }
      if (data.type === 'status') {
        state.lastStreamStatus = data.message || '正在处理';
        updateSideStats();
        if (!answer) {
          contentElement.classList.add('stream-status');
          contentElement.textContent = data.message || '正在处理...';
        }
      } else if (data.type === 'token') {
        answer += data.token || '';
        contentElement.classList.remove('stream-status');
        contentElement.innerHTML = renderMarkdown(answer);
      } else if (data.type === 'end') {
        completed = true;
        sources = data.sources || [];
        state.lastHitType = data.hit_type || '-';
        state.lastSourceCount = sources.length;
        state.lastTraceId = data.trace_id || state.lastTraceId;
        state.lastStreamStatus = '回答完成';
        state.lastDiagnostics = buildDiagnosticsSnapshot(data, sources);
        if (sources.length) {
          contentElement.appendChild(renderSources(sources));
        }
        contentElement.appendChild(renderAnswerDiagnostics(state.lastDiagnostics));
        setWebSocketHealth('ok', '正常');
        updateSideStats();
        setConnectionState('ready', '就绪');
        state.socket.close();
        finish({ answer, sources });
      } else if (data.type === 'error') {
        completed = true;
        socketError = true;
        setWebSocketHealth('error', '异常');
        state.lastStreamStatus = '处理异常';
        updateSideStats();
        setConnectionState('error', '异常');
        reject(new Error(data.error || '流式响应失败'));
      }
      scrollToBottom();
    };

    state.socket.onerror = () => {
      socketError = true;
      setWebSocketHealth('error', '异常');
      state.lastStreamStatus = 'WebSocket 连接失败';
      updateSideStats();
      setConnectionState('error', '异常');
      reject(new Error('WebSocket 连接失败'));
    };

    state.socket.onclose = () => {
      if (!completed) {
        if (!socketError) {
          setConnectionState('ready', '就绪');
        }
        if (state.cancelled) {
          state.lastStreamStatus = '已停止生成';
          setWebSocketHealth('pending', '已停止');
          updateSideStats();
          contentElement.appendChild(document.createTextNode('\n\n[已停止生成]'));
        }
        finish({ answer, sources });
      }
    };
  });
}
