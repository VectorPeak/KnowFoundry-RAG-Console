async function sendMessage() {
  const query = els.chatInput.value.trim();
  if (!query || state.inProgress) return;
  els.chatInput.value = '';
  autoResizeInput();
  els.chatHistory.querySelector('.welcome-message')?.remove();
  appendMessage('user', query, '你');
  const assistant = appendMessage('assistant', '<div class="typing-row"><span>正在处理</span><span class="typing-dots"><span></span><span></span><span></span></span></div>', '助手', true);
  state.inProgress = true;
  state.cancelled = false;
  updateSendState();

  try {
    const result = await streamAnswer(query, assistant.content);
    upsertSessionCard(state.sessionId, {
      title: query,
      summary: result.answer || '回答完成'
    });
    if (result.answer) {
      assistant.content.appendChild(renderFeedbackActions(query, result.answer, result.sources));
    }
    await loadHistory();
  } catch (error) {
    upsertSessionCard(state.sessionId, {
      title: query,
      summary: '处理失败'
    });
    assistant.content.classList.remove('stream-status');
    assistant.content.innerHTML = renderMarkdown(`抱歉，处理失败：${error.message || error}`);
  } finally {
    state.inProgress = false;
    updateSendState();
    scrollToBottom();
  }
}

function cancelStream() {
  state.cancelled = true;
  if (state.socket && state.socket.readyState === WebSocket.OPEN) {
    state.socket.close();
  }
  state.inProgress = false;
  updateSendState();
}
