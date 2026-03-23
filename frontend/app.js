let activeId = "";
const conversations = [];

const chat = document.getElementById("chat");
const form = document.getElementById("chat-form");
const promptInput = document.getElementById("prompt");
const endpointInput = document.getElementById("endpoint");
const maxTokensInput = document.getElementById("max-tokens");
const samplingModeInput = document.getElementById("sampling-mode");
const temperatureInput = document.getElementById("temperature");
const topKInput = document.getElementById("top-k");
const topPInput = document.getElementById("top-p");
const seedInput = document.getElementById("seed");
const editHint = document.getElementById("edit-hint");
const sendButton = document.getElementById("send");
const stopButton = document.getElementById("stop");
const sessionList = document.getElementById("session-list");
const newChatButton = document.getElementById("new-chat");
let activeStreamController = null;
let pendingEdit = null;

const createLocalId = () => {
  if (crypto && crypto.randomUUID) return crypto.randomUUID();
  return `local-${Date.now()}-${Math.random().toString(16).slice(2)}`;
};

const dedupeAdjacentParagraphs = (text) => {
  const parts = text
    .split(/\n{2,}/)
    .map((p) => p.trim())
    .filter(Boolean);
  const deduped = [];
  for (const p of parts) {
    if (deduped.length > 0 && deduped[deduped.length - 1] === p) continue;
    deduped.push(p);
  }
  return deduped.join("\n\n");
};

const parseAssistantSections = (rawText) => {
  const normalized = String(rawText || "").replaceAll("<|end_of_sentence|>", "");
  const openTag = "<think>";
  const closeTag = "</think>";
  const start = normalized.indexOf(openTag);
  const closeOnly = normalized.indexOf(closeTag);

  // Tolerate outputs containing only a closing tag.
  if (start < 0 && closeOnly >= 0) {
    const thinking = normalized.slice(0, closeOnly).trim();
    const answer = normalized.slice(closeOnly + closeTag.length).trim();
    return {
      thinking: dedupeAdjacentParagraphs(thinking.replaceAll(closeTag, "")),
      answer: dedupeAdjacentParagraphs(answer.replaceAll(closeTag, "")),
    };
  }

  if (start < 0) {
    return { thinking: "", answer: dedupeAdjacentParagraphs(normalized.replaceAll(closeTag, "")) };
  }
  const afterOpen = start + openTag.length;
  const end = normalized.indexOf(closeTag, afterOpen);
  if (end < 0) {
    return {
      thinking: dedupeAdjacentParagraphs(normalized.slice(afterOpen).replaceAll(openTag, "")),
      answer: "",
    };
  }
  const thinking = normalized.slice(afterOpen, end).replaceAll(openTag, "").replaceAll(closeTag, "");
  const answer = normalized.slice(end + closeTag.length).replaceAll(openTag, "").replaceAll(closeTag, "");
  return {
    thinking: dedupeAdjacentParagraphs(thinking),
    answer: dedupeAdjacentParagraphs(answer),
  };
};

const renderAssistantBubble = (bubble, rawText) => {
  bubble.dataset.raw = rawText;
  const { thinking, answer } = parseAssistantSections(rawText);
  const thinkingSection = bubble.querySelector(".assistant-thinking");
  const answerSection = bubble.querySelector(".assistant-answer");
  const normalizedThinking = thinking.replace(/\s+/g, " ").trim();
  const normalizedAnswer = answer.replace(/\s+/g, " ").trim();
  const isRedundantThinking =
    normalizedThinking &&
    normalizedAnswer &&
    (normalizedThinking === normalizedAnswer ||
      normalizedAnswer.includes(normalizedThinking));

  if (thinking && thinking.trim() && !isRedundantThinking) {
    thinkingSection.style.display = "block";
    thinkingSection.querySelector(".assistant-thinking-content").textContent = thinking.trim();
  } else {
    thinkingSection.style.display = "none";
    thinkingSection.querySelector(".assistant-thinking-content").textContent = "";
  }
  answerSection.textContent = answer.trimStart();
};

const clearPendingEdit = () => {
  pendingEdit = null;
  sendButton.textContent = "发送";
  editHint.style.display = "none";
  editHint.textContent = "";
};

const setPendingEdit = (state) => {
  pendingEdit = state;
  sendButton.textContent = "分叉发送";
  const round = Number(state.editMessageIndex) + 1;
  editHint.textContent = `正在编辑第 ${round} 轮用户消息，发送后将创建分叉会话（Esc 可取消）`;
  editHint.style.display = "block";
};

const appendBubble = (text, role, options = {}) => {
  const div = document.createElement("div");
  div.className = `bubble ${role}`;
  if (role === "assistant") {
    div.innerHTML = `
      <div class="assistant-thinking" style="display:none">
        <div class="assistant-thinking-label">思考中</div>
        <div class="assistant-thinking-content"></div>
      </div>
      <div class="assistant-answer"></div>
    `;
    renderAssistantBubble(div, text || "");
  } else {
    const content = document.createElement("div");
    content.className = "user-content";
    content.textContent = text;
    div.appendChild(content);
    if (options.canEdit) {
      const editButton = document.createElement("button");
      editButton.type = "button";
      editButton.className = "bubble-edit";
      editButton.textContent = "编辑";
      editButton.addEventListener("click", options.onEdit);
      div.appendChild(editButton);
    }
  }
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
};

const renderChat = (conversation) => {
  chat.innerHTML = "";
  for (let i = 0; i < conversation.messages.length; i += 1) {
    const message = conversation.messages[i];
    const canEdit = message.role === "user" && Boolean(conversation.serverId);
    appendBubble(message.text, message.role, {
      canEdit,
      onEdit: () => {
        if (!conversation.serverId) return;
        setPendingEdit({
          sourceLocalId: conversation.id,
          sourceServerId: conversation.serverId,
          editMessageIndex: i,
        });
        promptInput.value = message.text || "";
        promptInput.focus();
      },
    });
  }
};

const renderSessions = () => {
  sessionList.innerHTML = "";
  for (const convo of conversations) {
    const item = document.createElement("div");
    item.className = `session-item${convo.id === activeId ? " active" : ""}`;
    item.textContent = convo.title || "新对话";
    item.addEventListener("click", () => {
      activeId = convo.id;
      clearPendingEdit();
      renderSessions();
      renderChat(convo);
    });
    sessionList.appendChild(item);
  }
};

const createConversation = () => {
  const convo = {
    id: createLocalId(),
    serverId: "",
    title: "新对话",
    messages: [],
  };
  conversations.unshift(convo);
  activeId = convo.id;
  clearPendingEdit();
  renderSessions();
  renderChat(convo);
  return convo;
};

const getActiveConversation = () => {
  let convo = conversations.find((c) => c.id === activeId);
  if (!convo) {
    convo = createConversation();
  }
  return convo;
};

const streamChat = async (payload, bubble, convo, controller) => {
  const res = await fetch(`${endpointInput.value}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...payload, stream: true }),
    signal: controller.signal,
  });

  if (!res.ok || !res.body) {
    throw new Error(`请求失败：${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const part of parts) {
      if (!part.startsWith("data: ")) continue;
      const payload_str = part.slice(6).trim();
      if (payload_str === "[DONE]") return;
      const data = JSON.parse(payload_str);
      if (data.session_id && !convo.serverId) {
        convo.serverId = data.session_id;
      }
      const delta = data.choices && data.choices[0] && data.choices[0].delta;
      if (delta && delta.content) {
        const raw = (bubble.dataset.raw || "") + delta.content;
        renderAssistantBubble(bubble, raw);
      }
      if (data.choices && data.choices[0] && data.choices[0].finish_reason) {
        return;
      }
    }
  }
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const prompt = promptInput.value.trim();
  if (!prompt) return;

  const activeConvo = getActiveConversation();
  let convo = activeConvo;
  let payloadSessionId = activeConvo.serverId;
  let payloadEditFrom = "";
  let payloadEditIndex = -1;
  const editState = pendingEdit;
  const isForkEdit = Boolean(editState && editState.sourceServerId);

  if (isForkEdit) {
    const sourceConvo = conversations.find((c) => c.id === editState.sourceLocalId);
    if (!sourceConvo || !sourceConvo.serverId) {
      clearPendingEdit();
      return;
    }
    convo = createConversation();
    convo.title = `${sourceConvo.title || "新对话"} (分叉)`;
    const prefix = sourceConvo.messages.slice(0, editState.editMessageIndex + 1).map((m) => ({ ...m }));
    if (
      prefix.length === 0 ||
      prefix[prefix.length - 1].role !== "user"
    ) {
      clearPendingEdit();
      return;
    }
    prefix[prefix.length - 1].text = prompt;
    convo.messages = prefix;
    renderSessions();
    renderChat(convo);
    payloadEditFrom = sourceConvo.serverId;
    payloadEditIndex = editState.editMessageIndex;
    payloadSessionId = "";
  }

  if (!isForkEdit) {
    convo.messages.push({ role: "user", text: prompt });
    appendBubble(prompt, "user", { canEdit: false });
  }
  promptInput.value = "";
  const assistantBubble = appendBubble("", "assistant");
  convo.messages.push({ role: "assistant", text: "" });

  sendButton.disabled = true;
  stopButton.disabled = false;
  activeStreamController = new AbortController();
  const payload = {
    prompt,
    max_tokens: Number(maxTokensInput.value) || 128,
    temperature: Number(temperatureInput.value) || 0,
    top_k: Number(topKInput.value) || 1,
    top_p: Number(topPInput.value) || 0,
    seed: Number(seedInput.value) || 0,
  };
  if (samplingModeInput.value) {
    payload.sampling = samplingModeInput.value;
  }
  if (payloadSessionId) {
    payload.session_id = payloadSessionId;
  }
  if (payloadEditFrom && payloadEditIndex >= 0) {
    payload.edit_from_session_id = payloadEditFrom;
    payload.edit_message_index = payloadEditIndex;
  }

  try {
    await streamChat(payload, assistantBubble, convo, activeStreamController);
    convo.messages[convo.messages.length - 1].text = assistantBubble.dataset.raw || "";
    if (convo.title === "新对话") {
      convo.title = prompt.slice(0, 12);
      renderSessions();
    }
  } catch (err) {
    if (err && err.name === "AbortError") {
      convo.messages[convo.messages.length - 1].text = assistantBubble.dataset.raw || "";
      return;
    }
    renderAssistantBubble(assistantBubble, `请求失败：${err.message}`);
    convo.messages[convo.messages.length - 1].text = assistantBubble.dataset.raw || "";
  } finally {
    clearPendingEdit();
    activeStreamController = null;
    stopButton.disabled = true;
    sendButton.disabled = false;
  }
});

newChatButton.addEventListener("click", () => {
  createConversation();
});

stopButton.addEventListener("click", async () => {
  if (activeStreamController) {
    activeStreamController.abort();
  }
  const convo = getActiveConversation();
  if (!convo.serverId) {
    return;
  }
  try {
    await fetch(`${endpointInput.value}/chat/stop`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: convo.serverId }),
    });
  } catch (_) {
    // no-op
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && pendingEdit) {
    clearPendingEdit();
  }
});

createConversation();
