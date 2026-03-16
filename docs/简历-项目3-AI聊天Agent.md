# 简历描述：LLAISYS AI 聊天 Agent 项目

以下内容可直接或稍作修改后写入简历「项目经历」一栏，突出 **Agent / 对话式 AI** 相关能力。

---

## 项目名称（建议）

**基于自研推理引擎的对话式 AI Agent 系统**  
或：**LLM 推理引擎与对话 Agent 服务端到端实现**

---

## 项目描述（一段话版）

参与教育型 AI 系统 LLAISYS 的推理与服务层开发，**独立完成对话式 AI Agent 的完整链路**：从底层随机采样算子（Temperature / Top-K / Top-P）实现与 API 打通，到 HTTP 聊天服务（OpenAI chat-completion 兼容）、再到多轮对话 CLI 与 Web 前端。Agent 基于自研 C++ 推理引擎，支持单用户实时多轮对话与流式输出，为后续多用户推理服务与连续批处理奠定基础。

---

##  bullet 点（任选 4～6 条）

- **对话 Agent 采样与推理**：设计并实现随机采样算子（Temperature / Top-K / Top-P），替代原有 argmax，使 Agent 回复更自然、可调；在 C++ 推理管线中集成采样逻辑，经 C API 与 Python 封装贯通至上层调用。
- **Agent 服务端**：使用 FastAPI 实现 OpenAI chat-completion 风格的 HTTP 接口，支持非流式与 SSE 流式响应；单用户阻塞式处理，保证对话上下文一致，便于后续扩展为多 Agent/多用户服务。
- **多轮对话与上下文管理**：在 CLI 与 Web 客户端维护完整 `messages` 历史，每次请求将整段对话上下文发给服务端，实现**多轮连续对话**，体现 Agent 的会话记忆与上下文理解能力。
- **端到端对话体验**：实现命令行聊天客户端（Python + requests）与内嵌 Web 聊天页（HTML/JS + Fetch），用户可连续发消息、收回复，形成完整「人机对话 Agent」闭环。
- **技术栈**：C++ 推理引擎（自研算子、张量、设备抽象）、Python 模型封装与 HTTP 服务、transformers 分词与对话模板，接口设计兼容 OpenAI，便于与现有 Agent 框架对接。
- **工程实践**：完成从算子实现、C/Python API 打通、服务端到前端的全链路开发；编写项目总结文档与运行说明，便于复现与后续迭代（多用户、KV-Cache 池等）。

---

## 关键词（便于简历筛选）

对话式 AI · Agent · 多轮对话 · LLM 推理 · 随机采样（Temperature / Top-K / Top-P）· FastAPI · OpenAI API 兼容 · 流式输出（SSE）· C++ / Python · 自研推理引擎

---

## 简短版（空间有限时用）

**LLAISYS 对话 Agent**：实现随机采样算子（Temperature / Top-K / Top-P）并打通 C/Python API；基于 FastAPI 提供 OpenAI 兼容的 chat-completion 服务，支持流式输出；开发 CLI 与 Web 多轮对话前端，完成单用户对话式 AI Agent 的端到端链路。

---

按需选用「一段话版」+ 部分 bullet，或仅用「简短版」即可突出 Agent 与对话式 AI 能力。
