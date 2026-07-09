/**
 * app.js — Chat UI with multi-session management
 *
 * Session management overview:
 *   - Each "session" is an independent conversation stored on the server.
 *   - The sidebar lists all sessions; clicking one switches the active session.
 *   - The "＋" button creates a new empty session.
 *   - Each session has its own message history displayed in the chat panel.
 *   - On page load, we fetch existing sessions from the server and restore them.
 *
 * Client-side state:
 *   sessions     — map of session_id -> { id, title, messages[] }
 *                  messages[] holds the DOM-ready history for rendering
 *   activeId     — the currently displayed session's id
 */

// ── DOM references ────────────────────────────────────────────────────────────

const messagesEl    = document.getElementById('messages');
const inputEl       = document.getElementById('input');
const sendBtn       = document.getElementById('sendBtn');
const settingsBtn   = document.getElementById('settingsBtn');
const settingsPanel = document.getElementById('settingsPanel');
const sessionListEl = document.getElementById('sessionList');
const newChatBtn    = document.getElementById('newChatBtn');
const chatTitleEl   = document.getElementById('chatTitle');

const tempSlider     = document.getElementById('temperature');
const topkSlider     = document.getElementById('topK');
const toppSlider     = document.getElementById('topP');
const maxTokensInput = document.getElementById('maxTokens');
const tempVal  = document.getElementById('tempVal');
const topkVal  = document.getElementById('topkVal');
const toppVal  = document.getElementById('toppVal');

tempSlider.addEventListener('input', () => tempVal.textContent = tempSlider.value);
topkSlider.addEventListener('input', () => topkVal.textContent = topkSlider.value);
toppSlider.addEventListener('input', () => toppVal.textContent = toppSlider.value);
settingsBtn.addEventListener('click', () => { settingsPanel.hidden = !settingsPanel.hidden; });

// ── Client-side session state ─────────────────────────────────────────────────

/**
 * sessions: { [id]: { id, title, history } }
 *   history is the array of {role, content} sent to the server on each request.
 *   We maintain it client-side so switching sessions is instant (no server round-trip).
 */
let sessions = {};
let activeId = null;  // currently displayed session id

// ── Session API helpers ───────────────────────────────────────────────────────

async function fetchSessions() {
  const res = await fetch('/v1/sessions');
  return res.json();  // [{id, title}, ...]
}

async function createSessionOnServer() {
  const res = await fetch('/v1/sessions', { method: 'POST' });
  return res.json();  // {id, title}
}

async function deleteSessionOnServer(id) {
  await fetch(`/v1/sessions/${id}`, { method: 'DELETE' });
}

// ── Sidebar rendering ─────────────────────────────────────────────────────────

function renderSidebar() {
  sessionListEl.innerHTML = '';
  for (const s of Object.values(sessions)) {
    const li = document.createElement('li');
    li.className = 'session-item' + (s.id === activeId ? ' active' : '');
    li.dataset.id = s.id;

    const titleSpan = document.createElement('span');
    titleSpan.className = 'session-item-title';
    titleSpan.textContent = s.title;

    const delBtn = document.createElement('button');
    delBtn.className = 'session-delete';
    delBtn.textContent = '✕';
    delBtn.title = 'Delete';
    delBtn.addEventListener('click', async (e) => {
      e.stopPropagation();  // Don't trigger the session switch
      await deleteSessionOnServer(s.id);
      delete sessions[s.id];
      // If we deleted the active session, switch to another or create new
      if (activeId === s.id) {
        const remaining = Object.keys(sessions);
        if (remaining.length > 0) {
          switchSession(remaining[0]);
        } else {
          await newChat();
        }
      } else {
        renderSidebar();
      }
    });

    li.appendChild(titleSpan);
    li.appendChild(delBtn);
    li.addEventListener('click', () => switchSession(s.id));
    sessionListEl.appendChild(li);
  }
}

// ── Session switching ─────────────────────────────────────────────────────────

function switchSession(id) {
  activeId = id;
  const session = sessions[id];

  // Update header title
  chatTitleEl.textContent = session.title;

  // Re-render the message list for this session
  messagesEl.innerHTML = '';
  for (const msg of session.history) {
    addMessage(msg.role, msg.content);
  }

  renderSidebar();
  inputEl.focus();
}

async function newChat() {
  // Create session on server so it gets a persistent id
  const s = await createSessionOnServer();
  sessions[s.id] = { id: s.id, title: s.title, history: [] };
  switchSession(s.id);
}

// ── UI helpers ────────────────────────────────────────────────────────────────

function addMessage(role, content) {
  const msg = document.createElement('div');
  msg.className = `message ${role}`;

  const label = document.createElement('div');
  label.className = 'role-label';
  label.textContent = role === 'user' ? 'You' : 'Assistant';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = content;

  msg.appendChild(label);
  msg.appendChild(bubble);
  messagesEl.appendChild(msg);
  scrollToBottom();
  return bubble;
}

function addError(text) {
  const el = document.createElement('div');
  el.className = 'error-msg';
  el.textContent = text;
  messagesEl.appendChild(el);
  scrollToBottom();
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function cleanModelOutput(text) {
  return text
    .replace(/<think>[\s\S]*?<\/think>/g, '')
    .replace(/^[\s\S]*?<\/think>/g, '')
    .replace(/<[|｜][^|｜]*[|｜]>/g, '')
    .trim();
}

// ── Send message ──────────────────────────────────────────────────────────────

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || !activeId) return;

  inputEl.value = '';
  inputEl.style.height = 'auto';
  sendBtn.disabled = true;

  const session = sessions[activeId];

  // Add user message to local history and display it
  session.history.push({ role: 'user', content: text });
  addMessage('user', text);

  // Update session title from first user message
  if (session.title === 'New chat') {
    session.title = text.slice(0, 30);
    chatTitleEl.textContent = session.title;
    renderSidebar();
  }

  // Create empty assistant bubble with typing indicator
  const assistantMsg = document.createElement('div');
  assistantMsg.className = 'message assistant typing';
  const label = document.createElement('div');
  label.className = 'role-label';
  label.textContent = 'Assistant';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  assistantMsg.appendChild(label);
  assistantMsg.appendChild(bubble);
  messagesEl.appendChild(assistantMsg);
  scrollToBottom();

  const body = {
    session_id: activeId,
    messages: session.history,
    temperature: parseFloat(tempSlider.value),
    top_k: parseInt(topkSlider.value),
    top_p: parseFloat(toppSlider.value),
    max_tokens: parseInt(maxTokensInput.value),
    stream: true,
  };

  let fullText = '';
  try {
    const resp = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || resp.statusText);
    }

    // Read SSE stream
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      const lines = buf.split('\n');
      buf = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6).trim();
        if (payload === '[DONE]') break;
        try {
          const chunk = JSON.parse(payload);
          if (chunk.error) throw new Error(chunk.error.message);
          const delta = chunk.choices?.[0]?.delta?.content;
          if (delta) {
            fullText += delta;
            bubble.textContent = fullText;
            scrollToBottom();
          }
        } catch (e) { /* skip malformed chunks */ }
      }
    }

    assistantMsg.classList.remove('typing');
    const cleanText = cleanModelOutput(fullText);
    bubble.textContent = cleanText;

    // Persist assistant reply in local session history
    session.history.push({ role: 'assistant', content: cleanText });

  } catch (err) {
    assistantMsg.remove();
    addError(`Error: ${err.message}`);
    session.history.pop();  // Remove the user message added optimistically
  } finally {
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

// ── Event listeners ───────────────────────────────────────────────────────────

newChatBtn.addEventListener('click', newChat);
sendBtn.addEventListener('click', sendMessage);

inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
});

// ── Initialization ────────────────────────────────────────────────────────────

async function fetchSessionHistory(id) {
  const res = await fetch(`/v1/sessions/${id}`);
  if (!res.ok) return [];
  const data = await res.json();
  return data.history || [];
}

async function init() {
  // Load any sessions that already exist on the server (e.g. after page refresh)
  const existing = await fetchSessions();
  for (const s of existing) {
    const history = await fetchSessionHistory(s.id);
    sessions[s.id] = { id: s.id, title: s.title, history };
  }

  if (Object.keys(sessions).length > 0) {
    // Restore the first session
    switchSession(Object.keys(sessions)[0]);
  } else {
    // No sessions yet — create the first one automatically
    await newChat();
  }
}

init();
