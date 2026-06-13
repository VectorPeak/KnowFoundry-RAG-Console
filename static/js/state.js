const API_BASE_URL = '/projects/knowforge-rag-platform-vp';

const state = {
  sessionId: null,
  scenarioId: null,
  scenarios: [],
  socket: null,
  inProgress: false,
  cancelled: false,
  kbVersion: null,
  lastTraceId: null,
  lastHitType: '-',
  lastSourceCount: 0,
  lastStreamStatus: '等待提问',
  lastDiagnostics: null,
  historyItems: [],
  sessionCards: []
};

const els = {
  scenarioSelect: document.getElementById('scenarioSelect'),
  scenarioList: document.getElementById('scenarioList'),
  scenarioDescription: document.getElementById('scenarioDescription'),
  sourceFilter: document.getElementById('sourceFilter'),
  categoryList: document.getElementById('categoryList'),
  tenantInput: document.getElementById('tenantInput'),
  datasetInput: document.getElementById('datasetInput'),
  visibilitySelect: document.getElementById('visibilitySelect'),
  roleSelect: document.getElementById('roleSelect'),
  newSessionBtn: document.getElementById('newSessionBtn'),
  sidebarNewSessionBtn: document.getElementById('sidebarNewSessionBtn'),
  clearHistoryBtn: document.getElementById('clearHistoryBtn'),
  sessionInfo: document.getElementById('sessionInfo'),
  contextTitle: document.getElementById('contextTitle'),
  contextSubtitle: document.getElementById('contextSubtitle'),
  connectionPill: document.getElementById('connectionPill'),
  servicePillText: document.getElementById('servicePillText'),
  websocketHealth: document.getElementById('websocketHealth'),
  kbPill: document.getElementById('kbPill'),
  chatHistory: document.getElementById('chatHistory'),
  chatInput: document.getElementById('chatInput'),
  sendBtn: document.getElementById('sendBtn'),
  composerScope: document.getElementById('composerScope'),
  sessionSearchInput: document.getElementById('sessionSearchInput'),
  sampleQuestions: document.getElementById('sampleQuestions'),
  historyList: document.getElementById('historyList'),
  sideStats: document.getElementById('sideStats')
};
