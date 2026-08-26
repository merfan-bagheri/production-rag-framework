// Xilinx LogiCORE IP RAG Assistant - Frontend Controller

// Default Settings
const defaultSettings = {
  provider: 'gemini',
  model: 'gemini-3.5-flash-lite',
  chat_width: 'wide',
  temperature: 0.1,
  top_k_rerank: 20,
  auto_k: false,
  fast_rerank: false,
  use_chat_history: true
};

// Global State
let state = {
  settings: Object.assign({}, defaultSettings, JSON.parse(localStorage.getItem('xilinx_rag_settings')) || {}),
  chats: JSON.parse(localStorage.getItem('xilinx_rag_chats')) || {},
  currentChatId: null,
  isGenerating: false,
};

// DOM Elements
const elements = {
  sidebar: document.getElementById('sidebar'),
  toggleSidebarBtn: document.getElementById('toggleSidebarBtn'),
  newChatBtn: document.getElementById('newChatBtn'),
  chatList: document.getElementById('chatList'),
  welcomeHero: document.getElementById('welcomeHero'),
  messagesList: document.getElementById('messagesList'),
  messagesContainer: document.getElementById('messagesContainer'),
  chatFormContainer: document.getElementById('chatFormContainer'),
  chatForm: document.getElementById('chatForm'),
  queryInput: document.getElementById('queryInput'),
  sendBtn: document.getElementById('sendBtn'),
  currentModelLabel: document.getElementById('currentModelLabel'),
  modelSelectorBtn: document.getElementById('modelSelectorBtn'),
  modelDropdown: document.getElementById('modelDropdown'),
  widthSelectorBtn: document.getElementById('widthSelectorBtn'),
  currentWidthLabel: document.getElementById('currentWidthLabel'),
  widthDropdown: document.getElementById('widthDropdown'),
  fastModeBadge: document.getElementById('fastModeBadge'),
  memoryBadge: document.getElementById('memoryBadge'),
  autoKBadge: document.getElementById('autoKBadge'),
  clearChatBtn: document.getElementById('clearChatBtn'),
  exportChatBtn: document.getElementById('exportChatBtn'),
  openSettingsBtn: document.getElementById('openSettingsBtn'),
  closeSettingsBtn: document.getElementById('closeSettingsBtn'),
  settingsModal: document.getElementById('settingsModal'),
  saveSettingsBtn: document.getElementById('saveSettingsBtn'),
  settingsModel: document.getElementById('settings_model'),
  settingsTemperature: document.getElementById('settings_temperature'),
  settingsTopK: document.getElementById('settings_top_k'),
  settingsAutoK: document.getElementById('settings_auto_k'),
  manualTopKSection: document.getElementById('manualTopKSection'),
  settingsHistory: document.getElementById('settings_history'),
  settingsFast: document.getElementById('settings_fast'),
  tempValue: document.getElementById('tempValue'),
  topKValue: document.getElementById('topKValue'),
  indexedChunksBadge: document.getElementById('indexedChunksBadge')
};

// Load Backend Application Configuration
async function loadAppConfig() {
  try {
    const res = await fetch('/api/config');
    if (res.ok) {
      const cfg = await res.json();
      
      const serverTopK = cfg.final_top_k || 20;
      defaultSettings.top_k_rerank = serverTopK;

      // If user hasn't explicitly customized top_k or it is at default, update to server config value
      const saved = JSON.parse(localStorage.getItem('xilinx_rag_settings') || '{}');
      if (!saved.top_k_rerank || saved.top_k_rerank === 5 || saved.top_k_rerank === 11) {
        state.settings.top_k_rerank = serverTopK;
      }

      if (elements.settingsTopK) {
        elements.settingsTopK.max = Math.max(40, serverTopK + 10);
        elements.settingsTopK.value = state.settings.top_k_rerank;
      }
      if (elements.topKValue) {
        elements.topKValue.textContent = state.settings.top_k_rerank;
      }

      if (cfg.title) {
        document.title = `${cfg.title} - AI Studio`;
      }

      syncSettingsUI();
    }
  } catch (err) {
    console.warn('Could not fetch app configuration', err);
  }
}

// Initialize Application
async function init() {
  lucide.createIcons();
  setupEventListeners();
  await loadAppConfig();
  loadSystemHealth();
  syncSettingsUI();

  // Restore sidebar state
  if (localStorage.getItem('xilinx_rag_sidebar_collapsed') === 'true') {
    elements.sidebar.classList.add('sidebar-collapsed');
  }

  // Apply response layout width
  applyChatWidth(state.settings.chat_width || 'wide');

  // Load existing or create first chat
  const chatIds = Object.keys(state.chats);
  if (chatIds.length > 0) {
    switchChat(chatIds[0]);
  } else {
    createNewChat();
  }
}

// System Health Check
async function loadSystemHealth() {
  try {
    const res = await fetch('/api/health');
    if (res.ok) {
      const data = await res.json();
      if (elements.indexedChunksBadge) {
        elements.indexedChunksBadge.textContent = `${data.indexed_chunks} chunks`;
      }
    }
  } catch (err) {
    console.warn('Could not fetch health status', err);
  }
}

// Setup Event Listeners
function setupEventListeners() {
  // Toggle Sidebar
  elements.toggleSidebarBtn.addEventListener('click', () => {
    elements.sidebar.classList.toggle('sidebar-collapsed');
    const isCollapsed = elements.sidebar.classList.contains('sidebar-collapsed');
    localStorage.setItem('xilinx_rag_sidebar_collapsed', isCollapsed ? 'true' : 'false');
  });

  // Model Dropdown
  elements.modelSelectorBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    if (elements.widthDropdown) elements.widthDropdown.classList.add('hidden');
    elements.modelDropdown.classList.toggle('hidden');
  });

  // Width Selector Dropdown
  if (elements.widthSelectorBtn) {
    elements.widthSelectorBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      elements.modelDropdown.classList.add('hidden');
      elements.widthDropdown.classList.toggle('hidden');
    });
  }

  document.querySelectorAll('.width-option').forEach(btn => {
    btn.addEventListener('click', () => {
      const width = btn.dataset.width;
      applyChatWidth(width);
      if (elements.widthDropdown) elements.widthDropdown.classList.add('hidden');
    });
  });

  document.addEventListener('click', () => {
    elements.modelDropdown.classList.add('hidden');
    if (elements.widthDropdown) elements.widthDropdown.classList.add('hidden');
  });

  document.querySelectorAll('.model-option').forEach(btn => {
    btn.addEventListener('click', () => {
      const provider = btn.dataset.provider;
      const model = btn.dataset.model;
      state.settings.provider = provider;
      state.settings.model = model;
      saveSettings();
      syncSettingsUI();
      elements.modelDropdown.classList.add('hidden');
    });
  });


  // New Chat
  elements.newChatBtn.addEventListener('click', () => createNewChat());

  // Clear Chat
  elements.clearChatBtn.addEventListener('click', () => {
    if (confirm('Clear messages in this conversation?')) {
      if (state.currentChatId && state.chats[state.currentChatId]) {
        state.chats[state.currentChatId].messages = [];
        saveChats();
        renderCurrentChat();
      }
    }
  });

  // Export Chat
  elements.exportChatBtn.addEventListener('click', () => exportConversation());

  // Settings Modal
  elements.openSettingsBtn.addEventListener('click', () => openSettings());
  elements.closeSettingsBtn.addEventListener('click', () => closeSettings());
  elements.settingsModal.addEventListener('click', (e) => {
    if (e.target === elements.settingsModal) closeSettings();
  });

  // Settings Sliders & Auto-K Toggle
  elements.settingsTemperature.addEventListener('input', (e) => {
    elements.tempValue.textContent = e.target.value;
  });
  elements.settingsTopK.addEventListener('input', (e) => {
    elements.topKValue.textContent = e.target.value;
  });
  elements.settingsAutoK.addEventListener('change', (e) => {
    if (e.target.checked) {
      elements.manualTopKSection.classList.add('opacity-40', 'pointer-events-none');
    } else {
      elements.manualTopKSection.classList.remove('opacity-40', 'pointer-events-none');
    }
  });

  elements.saveSettingsBtn.addEventListener('click', () => {
    const selectedProvider = document.querySelector('input[name="settings_provider"]:checked').value;
    const selectedWidthRadio = document.querySelector('input[name="settings_chat_width"]:checked');
    const selectedWidth = selectedWidthRadio ? selectedWidthRadio.value : (state.settings.chat_width || 'wide');

    state.settings.provider = selectedProvider;
    state.settings.model = elements.settingsModel.value.trim() || 'gemini-3.5-flash-lite';
    state.settings.temperature = parseFloat(elements.settingsTemperature.value);
    state.settings.auto_k = elements.settingsAutoK.checked;
    state.settings.top_k_rerank = parseInt(elements.settingsTopK.value);
    state.settings.use_chat_history = elements.settingsHistory.checked;
    state.settings.fast_rerank = elements.settingsFast.checked;
    state.settings.chat_width = selectedWidth;

    applyChatWidth(selectedWidth);
    saveSettings();
    syncSettingsUI();
    closeSettings();
  });


  // Auto-expanding textarea & Submit
  elements.queryInput.addEventListener('input', () => {
    elements.queryInput.style.height = 'auto';
    elements.queryInput.style.height = Math.min(elements.queryInput.scrollHeight, 180) + 'px';
  });

  elements.queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      elements.chatForm.requestSubmit();
    }
  });

  elements.chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    handleSubmit();
  });

  // Quick Prompt Chips
  document.querySelectorAll('.prompt-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const title = chip.querySelector('div:first-child').innerText.trim();
      let prompt = '';
      if (title.includes('35x35')) {
        prompt = 'How is a fully pipelined 35x35-bit multiplier constructed using cascaded DSP48A1 slices, and why must the 17-bit partial product shift route through FPGA fabric rather than internal PCIN cascade?';
      } else if (title.includes('Hard vs Soft ECC')) {
        prompt = 'Detail the differences between BuiltIn_ECC (Hard ECC) and Soft ECC in Block Memory Generator v7.3 regarding primitive size, data width boundaries, supported device architectures, and algorithm selection.';
      } else if (title.includes('Spread Spectrum')) {
        prompt = 'When Spread Spectrum (SS) is enabled on an MMCME2 primitive in the Clocking Wizard, which specific output clock ports are disabled, what are their divide registers repurposed for, and what are the restrictions on input frequency, bandwidth, and secondary input clocks?';
      } else if (title.includes('Distributed vs Block RAM')) {
        prompt = 'Compare Distributed RAM (PG036) against Block RAM (PG058) regarding physical storage elements, read latencies, port operating modes, and write synchronicity.';
      }
      elements.queryInput.value = prompt;
      elements.queryInput.focus();
      elements.chatForm.requestSubmit();
    });
  });
}


// Chat Session Management
function createNewChat() {
  const id = 'chat_' + Date.now();
  state.chats[id] = {
    id: id,
    title: 'New Hardware Inquiry',
    createdAt: new Date().toISOString(),
    messages: []
  };
  saveChats();
  switchChat(id);
}

function switchChat(chatId) {
  if (!state.chats[chatId]) return;
  state.currentChatId = chatId;
  renderChatList();
  renderCurrentChat();
}

function deleteChat(chatId, e) {
  e.stopPropagation();
  if (confirm('Delete this conversation?')) {
    delete state.chats[chatId];
    saveChats();
    const remaining = Object.keys(state.chats);
    if (remaining.length > 0) {
      switchChat(remaining[0]);
    } else {
      createNewChat();
    }
  }
}

function saveChats() {
  localStorage.setItem('xilinx_rag_chats', JSON.stringify(state.chats));
}

function saveSettings() {
  localStorage.setItem('xilinx_rag_settings', JSON.stringify(state.settings));
}

function applyChatWidth(widthKey) {
  if (!widthKey) widthKey = 'wide';
  state.settings.chat_width = widthKey;

  const widthClasses = {
    standard: 'chat-width-standard',
    medium: 'chat-width-medium',
    wide: 'chat-width-wide',
    full: 'chat-width-full'
  };

  const labelNames = {
    standard: 'Standard',
    medium: 'Medium',
    wide: 'Wide',
    full: 'Full Width'
  };

  const targetClass = widthClasses[widthKey] || 'chat-width-wide';

  const targets = [
    elements.welcomeHero,
    elements.messagesList,
    elements.chatFormContainer
  ];

  targets.forEach(el => {
    if (!el) return;
    Object.values(widthClasses).forEach(cls => el.classList.remove(cls));
    el.classList.add(targetClass);
  });

  if (elements.currentWidthLabel) {
    elements.currentWidthLabel.textContent = labelNames[widthKey] || 'Wide';
  }

  // Update checkmarks in width dropdown
  document.querySelectorAll('.width-option').forEach(btn => {
    const isSelected = btn.dataset.width === widthKey;
    const checkIcon = btn.querySelector('.check-icon');
    if (checkIcon) {
      if (isSelected) {
        checkIcon.classList.remove('hidden');
        btn.classList.add('text-emerald-400', 'font-semibold');
      } else {
        checkIcon.classList.add('hidden');
        btn.classList.remove('text-emerald-400', 'font-semibold');
      }
    }
  });

  // Update modal radio if present
  const radio = document.querySelector(`input[name="settings_chat_width"][value="${widthKey}"]`);
  if (radio) radio.checked = true;

  saveSettings();
}

function syncSettingsUI() {
  elements.currentModelLabel.textContent = state.settings.model;

  if (state.settings.auto_k) {
    elements.autoKBadge.classList.remove('hidden');
    elements.autoKBadge.classList.add('flex');
  } else {
    elements.autoKBadge.classList.add('hidden');
    elements.autoKBadge.classList.remove('flex');
  }

  if (state.settings.fast_rerank) {
    elements.fastModeBadge.classList.remove('hidden');
    elements.fastModeBadge.classList.add('flex');
  } else {
    elements.fastModeBadge.classList.add('hidden');
    elements.fastModeBadge.classList.remove('flex');
  }

  if (state.settings.use_chat_history) {
    elements.memoryBadge.classList.remove('hidden');
    elements.memoryBadge.classList.add('flex');
  } else {
    elements.memoryBadge.classList.add('hidden');
    elements.memoryBadge.classList.remove('flex');
  }
}

function openSettings() {
  const providerRadio = document.querySelector(`input[name="settings_provider"][value="${state.settings.provider}"]`);
  if (providerRadio) providerRadio.checked = true;

  const widthRadio = document.querySelector(`input[name="settings_chat_width"][value="${state.settings.chat_width || 'wide'}"]`);
  if (widthRadio) widthRadio.checked = true;

  elements.settingsModel.value = state.settings.model;
  elements.settingsTemperature.value = state.settings.temperature;
  elements.tempValue.textContent = state.settings.temperature;
  elements.settingsAutoK.checked = state.settings.auto_k;
  elements.settingsTopK.value = state.settings.top_k_rerank;
  elements.topKValue.textContent = state.settings.top_k_rerank;

  if (state.settings.auto_k) {
    elements.manualTopKSection.classList.add('opacity-40', 'pointer-events-none');
  } else {
    elements.manualTopKSection.classList.remove('opacity-40', 'pointer-events-none');
  }

  elements.settingsHistory.checked = state.settings.use_chat_history;
  elements.settingsFast.checked = state.settings.fast_rerank;

  elements.settingsModal.classList.remove('hidden');
}

function closeSettings() {
  elements.settingsModal.classList.add('hidden');
}


// Rendering
function renderChatList() {
  elements.chatList.innerHTML = '';
  const sortedIds = Object.keys(state.chats).sort((a, b) => {
    return new Date(state.chats[b].createdAt) - new Date(state.chats[a].createdAt);
  });

  sortedIds.forEach(id => {
    const chat = state.chats[id];
    const isActive = id === state.currentChatId;

    const div = document.createElement('div');
    div.className = `group flex items-center justify-between px-3 py-2 rounded-xl text-xs cursor-pointer transition ${
      isActive ? 'bg-dark-card border border-dark-border text-white font-medium' : 'text-slate-400 hover:bg-dark-hover hover:text-slate-200'
    }`;
    div.onclick = () => switchChat(id);

    div.innerHTML = `
      <div class="flex items-center gap-2 truncate pr-2">
        <i data-lucide="message-square" class="w-3.5 h-3.5 flex-shrink-0 ${isActive ? 'text-emerald-400' : 'text-slate-500'}"></i>
        <span class="truncate">${escapeHtml(chat.title)}</span>
      </div>
      <button class="delete-chat-btn opacity-0 group-hover:opacity-100 p-1 hover:text-red-400 transition" title="Delete">
        <i data-lucide="trash" class="w-3 h-3"></i>
      </button>
    `;

    div.querySelector('.delete-chat-btn').onclick = (e) => deleteChat(id, e);
    elements.chatList.appendChild(div);
  });

  lucide.createIcons();
}

function renderCurrentChat() {
  const chat = state.chats[state.currentChatId];
  if (!chat || chat.messages.length === 0) {
    elements.welcomeHero.classList.remove('hidden');
    elements.messagesList.innerHTML = '';
    return;
  }

  elements.welcomeHero.classList.add('hidden');
  elements.messagesList.innerHTML = '';

  chat.messages.forEach(msg => {
    appendMessageToDOM(msg);
  });

  scrollToBottom();
}

function appendMessageToDOM(msg) {
  const isUser = msg.role === 'user';
  const container = document.createElement('div');
  container.className = `flex gap-3.5 ${isUser ? 'justify-end' : 'justify-start'} animate-scale-up`;

  if (isUser) {
    container.innerHTML = `
      <div class="max-w-[85%] sm:max-w-[75%] rounded-2xl px-4 py-3 bg-emerald-600 text-white shadow-md text-sm leading-relaxed whitespace-pre-wrap">
        ${escapeHtml(msg.content)}
      </div>
      <div class="w-8 h-8 rounded-xl bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center flex-shrink-0">
        <i data-lucide="user" class="w-4 h-4 text-emerald-400"></i>
      </div>
    `;
  } else {
    // Render assistant markdown with KaTeX math, pre-parsed citation pills, and responsive tables
    function renderContent(rawContent) {
      if (!rawContent) return '';

      const mathPlaceholders = [];
      function storeMath(mathStr, isBlock) {
        const id = `@@@MATH_${isBlock ? 'BLOCK' : 'INLINE'}_${mathPlaceholders.length}@@@`;
        mathPlaceholders.push({ id, math: mathStr, isBlock });
        return id;
      }

      // 1. Protect code blocks so math, citations, and pseudo-tags inside code are untouched
      const codeBlocks = [];
      let text = rawContent.replace(/(```[\s\S]*?```|`[^`\n]+`)/g, (match) => {
        const id = `@@@CODE_${codeBlocks.length}@@@`;
        codeBlocks.push({ id, code: match });
        return id;
      });

      // 2. Extract Display Math: $$ ... $$ and \[ ... \]
      text = text.replace(/\$\$([\s\S]+?)\$\$/g, (match, math) => storeMath(math.trim(), true));
      text = text.replace(/\\\[([\s\S]+?)\\\]/g, (match, math) => storeMath(math.trim(), true));

      // 3. Extract Inline Math: $ ... $ and \( ... \)
      text = text.replace(/(?<!\\)\$([^\$\n]+?)(?<!\\)\$/g, (match, math) => storeMath(math.trim(), false));
      text = text.replace(/\\\(([\s\S]+?)\\\)/g, (match, math) => storeMath(math.trim(), false));

      // 4. Pre-process and extract ALL Citations BEFORE Markdown parsing
      // This prevents unescaped pipes '|' inside citations from corrupting Markdown table columns!
      const citationPlaceholders = [];
      function createCitationPill(docName, pageNum, secPath) {
        let shortDoc = '';
        let fullDoc = docName ? docName.trim() : '';
        if (fullDoc) {
          const docMatch = fullDoc.match(/\b(UG\d+|PG\d+)\b/i);
          if (docMatch) {
            shortDoc = docMatch[1].toUpperCase();
          } else if (fullDoc.length <= 10) {
            shortDoc = fullDoc;
          } else {
            shortDoc = fullDoc.slice(0, 8) + '...';
          }
        }

        const page = pageNum ? String(pageNum).trim() : '';
        const section = secPath ? secPath.trim() : '';

        const tooltipParts = [];
        if (fullDoc) tooltipParts.push(fullDoc);
        if (page) tooltipParts.push(`Page ${page}`);
        if (section) tooltipParts.push(`Section: ${section}`);
        const tooltip = escapeHtml(tooltipParts.join(' • '));

        let pillBody = '';
        if (shortDoc && page) {
          pillBody = `<span class="doc-tag">${escapeHtml(shortDoc)}</span><span class="page-tag">p.${page}</span>`;
        } else if (shortDoc) {
          pillBody = `<span class="doc-tag">${escapeHtml(shortDoc)}</span>`;
        } else if (page) {
          pillBody = `<span class="doc-tag">DOC</span><span class="page-tag">p.${page}</span>`;
        } else if (section) {
          pillBody = `<span class="doc-tag">SEC</span><span class="sec-preview">${escapeHtml(section)}</span>`;
        } else {
          pillBody = `<span class="doc-tag">REF</span>`;
        }

        const pillHtml = `<span class="citation-pill" data-doc="${escapeHtml(shortDoc)}" data-page="${page}" data-tooltip="${tooltip}"><i data-lucide="book-open" class="w-2.5 h-2.5 inline text-emerald-400"></i>${pillBody}</span>`;
        const id = `@@@CITE_${citationPlaceholders.length}@@@`;
        citationPlaceholders.push({ id, html: pillHtml });
        return id;
      }

      // Pattern 1: [Doc: <doc> | Page: <page> | Section: <sec>]
      const multiDocRegex = /\[(?:Doc:\s*([^|\]]+))?(?:\s*\|\s*)?(?:Page:\s*(\d+))?(?:\s*\|\s*)?(?:Section:\s*([^\]]+))?\]/gi;
      text = text.replace(multiDocRegex, (match, docName, pageNum, secPath) => {
        if (!docName && !pageNum && !secPath) return match;
        return createCitationPill(docName, pageNum, secPath);
      });

      // Pattern 2: [Page X • Section Y] or [Page X, Section Y]
      const pageRegex = /\[Page\s*(\d+)(?:\s*[•,]\s*(?:Section:?\s*)?([^\]]+))?\]/gi;
      text = text.replace(pageRegex, (match, pageNum, sec) => {
        return createCitationPill('', pageNum, sec);
      });

      // Pattern 3: [UG380 p. 89] or [PG058, p. 55]
      const shortRefRegex = /\[(UG\d+|PG\d+)[,\s]+p(?:age|\.)?\s*(\d+)(?:[,\s]+Section:?\s*([^\]]+))?\]/gi;
      text = text.replace(shortRefRegex, (match, docName, pageNum, sec) => {
        return createCitationPill(docName, pageNum, sec);
      });

      // 5. Escape pseudo-HTML tags like <source>, <phase>, <custom_type> that are not standard HTML tags
      const allowedTags = new Set(['br', 'b', 'i', 'strong', 'em', 'code', 'pre', 'p', 'ul', 'ol', 'li', 'table', 'thead', 'tbody', 'tr', 'th', 'td', 'span', 'div', 'a', 'hr', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']);
      text = text.replace(/<(\/?[a-zA-Z][a-zA-Z0-9_-]*(?:\s+[^>]*)?)>/g, (match, tagContent) => {
        const tagName = tagContent.replace(/^\//, '').split(/\s+/)[0].toLowerCase();
        if (allowedTags.has(tagName)) {
          return match;
        }
        return `&lt;${tagContent}&gt;`;
      });

      // 6. Restore code blocks before marked
      codeBlocks.forEach(item => {
        text = text.replace(item.id, item.code);
      });

      // 7. Parse Markdown with Marked.js
      let html = marked.parse(text);

      // 8. Wrap Markdown tables in responsive scrolling wrapper
      html = html.replace(/<table>([\s\S]*?)<\/table>/g, '<div class="table-wrapper"><table>$1</table></div>');

      // 9. Restore Citations
      citationPlaceholders.forEach(item => {
        html = html.replace(new RegExp(item.id, 'g'), item.html);
      });

      // 10. Render LaTeX math placeholders via KaTeX
      mathPlaceholders.forEach(item => {
        let renderedMath = item.math;
        if (typeof katex !== 'undefined') {
          try {
            renderedMath = katex.renderToString(item.math, {
              displayMode: item.isBlock,
              throwOnError: false,
              output: 'htmlAndMathml'
            });
          } catch (err) {
            console.warn('KaTeX rendering error:', item.math, err);
            renderedMath = item.isBlock ? `$$${escapeHtml(item.math)}$$` : `$${escapeHtml(item.math)}$`;
          }
        }
        html = html.replace(new RegExp(item.id, 'g'), renderedMath);
      });

      return html;
    }

    let parsedMarkdown = renderContent(msg.content || '');


    let sourcesHtml = '';
    if (msg.sources && msg.sources.length > 0) {
      sourcesHtml = `
        <div class="mt-4 pt-3 border-t border-dark-border/60">
          <button class="toggle-sources-btn flex items-center gap-1.5 text-xs font-semibold text-slate-400 hover:text-emerald-400 transition">
            <i data-lucide="file-text" class="w-3.5 h-3.5 text-emerald-400"></i>
            <span>Verified Hardware Sources (${msg.sources.length} Chunks)</span>
            <i data-lucide="chevron-down" class="w-3.5 h-3.5 ml-1 transition-transform"></i>
          </button>
          <div class="sources-drawer hidden mt-2.5 space-y-2">
            ${msg.sources.map((s, idx) => `
              <div class="source-card p-2.5 rounded-xl bg-slate-900/90 border border-dark-border text-xs space-y-1 transition duration-200" data-doc="${escapeHtml(s.doc_id || '')}" data-page="${s.page_number || ''}">
                <div class="flex items-center justify-between text-slate-300">
                  <span class="font-bold text-emerald-400 flex items-center gap-1.5">
                    <span class="w-4 h-4 rounded bg-emerald-500/10 text-emerald-400 flex items-center justify-center text-[10px] font-mono">${idx+1}</span>
                    <span class="px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-mono text-[10px] border border-emerald-500/30 font-bold">${escapeHtml(s.doc_id || 'DOC')}</span>
                    Page ${s.page_number || '?'} • ${escapeHtml(s.breadcrumb || s.section_title || 'General')}
                  </span>
                  <span class="text-[10px] font-mono bg-dark-card px-1.5 py-0.5 rounded border border-dark-border text-slate-400">
                    Rerank: ${s.rerank_score || s.rrf_score || 'N/A'}${s.adaptive_boost ? ` (+${s.adaptive_boost} Boost)` : ''}
                  </span>
                </div>
                <div class="text-slate-400 text-[11px] leading-relaxed line-clamp-3 hover:line-clamp-none font-mono bg-dark-bg/60 p-1.5 rounded border border-dark-border/40">
                  ${escapeHtml(s.content || '')}
                </div>
              </div>
            `).join('')}
          </div>
        </div>
      `;
    }

    let metaHtml = '';
    if (msg.timings_ms) {
      const t = msg.timings_ms;
      const strat = msg.adaptive_strategy || {};
      metaHtml = `
        <div class="flex flex-wrap items-center gap-2 pt-2 text-[10px] text-slate-500 font-mono">
          <span class="px-1.5 py-0.5 rounded bg-dark-card border border-dark-border text-emerald-400">
            ${msg.provider ? msg.provider.toUpperCase() : 'LLM'} (${msg.model || ''})
          </span>
          ${strat.auto_k_applied ? `
            <span class="px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" title="${escapeHtml(strat.rationale || '')}">
              Auto-K: ${strat.selected_k} Chunks
            </span>
          ` : ''}
          <span>•</span>
          <span>Pre-Inf: ${t.pre_inference_total || t.retrieval || 0}ms</span>
          <span>•</span>
          <span>Gen: ${t.generation || 0}ms</span>
          <span>•</span>
          <span>Total: ${t.total || 0}ms</span>
          ${msg.reformulated_query ? `
            <span class="text-blue-400 font-sans" title="Reformulated: ${escapeHtml(msg.reformulated_query)}">
              [Context Aware: "${escapeHtml(msg.reformulated_query.slice(0, 30))}..."]
            </span>
          ` : ''}
        </div>
      `;
    }

    container.innerHTML = `
      <div class="w-8 h-8 rounded-xl bg-dark-card border border-dark-border flex items-center justify-center flex-shrink-0 shadow-md">
        <i data-lucide="cpu" class="w-4 h-4 text-emerald-400"></i>
      </div>
      <div class="flex-1 min-w-0 max-w-full rounded-2xl px-5 py-4 bg-dark-card/90 border border-dark-border shadow-xl space-y-2">
        <div class="markdown-body">
          ${parsedMarkdown}
        </div>
        ${sourcesHtml}
        ${metaHtml}
        <div class="pt-2 flex justify-end">
          <button class="copy-btn text-slate-400 hover:text-white text-xs flex items-center gap-1 px-2 py-1 rounded hover:bg-dark-hover transition">
            <i data-lucide="copy" class="w-3.5 h-3.5"></i> Copy
          </button>
        </div>
      </div>
    `;


    // Setup sources toggle
    const toggleBtn = container.querySelector('.toggle-sources-btn');
    if (toggleBtn) {
      toggleBtn.onclick = () => {
        const drawer = container.querySelector('.sources-drawer');
        const icon = toggleBtn.querySelector('i[data-lucide="chevron-down"]');
        drawer.classList.toggle('hidden');
        if (drawer.classList.contains('hidden')) {
          icon.style.transform = 'rotate(0deg)';
        } else {
          icon.style.transform = 'rotate(180deg)';
        }
      };
    }

    // Setup interactive citation pill click -> highlight matching source card
    container.querySelectorAll('.citation-pill').forEach(pill => {
      pill.onclick = () => {
        const doc = pill.dataset.doc;
        const page = pill.dataset.page;
        const drawer = container.querySelector('.sources-drawer');
        const toggleBtn = container.querySelector('.toggle-sources-btn');

        if (drawer && drawer.classList.contains('hidden') && toggleBtn) {
          toggleBtn.click();
        }

        if (drawer) {
          const cards = drawer.querySelectorAll('.source-card');
          let matchedCard = null;
          cards.forEach(card => {
            card.classList.remove('source-card-highlight');
            if (page && card.dataset.page === page) {
              matchedCard = card;
            } else if (!matchedCard && doc && card.dataset.doc && card.dataset.doc.toUpperCase().includes(doc.toUpperCase())) {
              matchedCard = card;
            }
          });

          if (matchedCard) {
            matchedCard.classList.add('source-card-highlight');
            matchedCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          }
        }
      };
    });

    // Setup copy button
    const copyBtn = container.querySelector('.copy-btn');
    if (copyBtn) {
      copyBtn.onclick = () => {
        navigator.clipboard.writeText(msg.content);
        copyBtn.innerHTML = `<i data-lucide="check" class="w-3.5 h-3.5 text-emerald-400"></i> Copied`;
        lucide.createIcons();
        setTimeout(() => {
          copyBtn.innerHTML = `<i data-lucide="copy" class="w-3.5 h-3.5"></i> Copy`;
          lucide.createIcons();
        }, 2000);
      };
    }
  }

  elements.messagesList.appendChild(container);

  lucide.createIcons();
  hljs.highlightAll();
}

// Handle Query Submission
async function handleSubmit() {
  const query = elements.queryInput.value.trim();
  if (!query || state.isGenerating) return;

  const activeChat = state.chats[state.currentChatId];
  if (!activeChat) return;

  // Append user message
  const userMsg = { role: 'user', content: query };
  activeChat.messages.push(userMsg);

  // Set chat title if first message
  if (activeChat.messages.length === 1) {
    activeChat.title = query.slice(0, 32) + (query.length > 32 ? '...' : '');
    renderChatList();
  }

  saveChats();
  renderCurrentChat();

  // Reset input
  elements.queryInput.value = '';
  elements.queryInput.style.height = 'auto';
  state.isGenerating = true;
  elements.sendBtn.disabled = true;

  // Show Loading indicator
  const loadingIndicator = document.createElement('div');
  loadingIndicator.id = 'loadingIndicator';
  loadingIndicator.className = 'flex gap-3.5 justify-start animate-pulse';
  loadingIndicator.innerHTML = `
    <div class="w-8 h-8 rounded-xl bg-dark-card border border-dark-border flex items-center justify-center">
      <i data-lucide="cpu" class="w-4 h-4 text-emerald-400"></i>
    </div>
    <div class="rounded-2xl px-5 py-3.5 bg-dark-card/90 border border-dark-border text-xs text-slate-400 flex items-center gap-2">
      <span class="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
      Hybrid retrieving, neural reranking & generating zero-hallucination hardware analysis...
    </div>
  `;
  elements.messagesList.appendChild(loadingIndicator);
  lucide.createIcons();
  scrollToBottom();

  try {
    const historyPayload = state.settings.use_chat_history
      ? activeChat.messages.slice(-6).map(m => ({ role: m.role, content: m.content }))
      : [];

    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        provider: state.settings.provider,
        model: state.settings.model,
        temperature: state.settings.temperature,
        top_k_rerank: state.settings.top_k_rerank,
        auto_k: state.settings.auto_k,
        fast_rerank: state.settings.fast_rerank,
        use_chat_history: state.settings.use_chat_history,
        history: historyPayload
      })
    });

    if (!response.ok) {
      const errData = await response.json();
      throw new Error(errData.detail || 'Failed to generate response');
    }

    const data = await response.json();

    const assistantMsg = {
      role: 'assistant',
      content: data.answer,
      provider: data.provider,
      model: data.model,
      effective_query: data.effective_query,
      reformulated_query: data.reformulated_query,
      adaptive_strategy: data.adaptive_strategy,
      sources: data.sources,
      timings_ms: data.timings_ms,
      eval_tokens: data.eval_tokens
    };

    activeChat.messages.push(assistantMsg);
    saveChats();
  } catch (err) {
    activeChat.messages.push({
      role: 'assistant',
      content: `**Error:** ${err.message}\n\nPlease check server logs or Google API / Ollama status.`
    });
    saveChats();
  } finally {
    const loader = document.getElementById('loadingIndicator');
    if (loader) loader.remove();
    state.isGenerating = false;
    elements.sendBtn.disabled = false;
    renderCurrentChat();
    elements.queryInput.focus();
  }
}

// Helpers
function scrollToBottom() {
  elements.messagesContainer.scrollTop = elements.messagesContainer.scrollHeight;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function exportConversation() {
  const chat = state.chats[state.currentChatId];
  if (!chat || chat.messages.length === 0) return;

  let md = `# Conversation Export: ${chat.title}\n\n*Created: ${new Date(chat.createdAt).toLocaleString()}*\n*Target Spec: Xilinx PG036 Distributed Memory Generator*\n\n---\n\n`;
  chat.messages.forEach(m => {
    const role = m.role === 'user' ? '### 👤 User' : '### 🤖 Assistant';
    md += `${role}\n\n${m.content}\n\n`;
    if (m.sources && m.sources.length > 0) {
      md += `**Sources:**\n`;
      m.sources.forEach(s => {
        md += `- Page ${s.page_number}: ${s.breadcrumb || s.section_title} (Rerank: ${s.rerank_score})\n`;
      });
      md += `\n`;
    }
    md += `---\n\n`;
  });

  const blob = new Blob([md], { type: 'text/markdown;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `xilinx_rag_chat_${Date.now()}.md`;
  a.click();
}

// Start application
window.addEventListener('DOMContentLoaded', init);
