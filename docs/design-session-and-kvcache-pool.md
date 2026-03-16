# 会话管理 + KV-Cache 池：接口与流程设计

本文档描述 Project #3 可选部分「会话管理」与「支持前缀匹配的 KV-Cache 池」的接口与流程设计，不涉及具体实现代码。

---

## 一、目标与约束

### 1.1 目标

- **会话管理**：用户可创建多个对话、在对话间切换；可在某条历史用户消息上编辑并「从此处重新生成」，后续助手回复被替换。
- **KV-Cache 池**：对同一会话内「前缀一致」的多次请求（含编辑后重生成、连续多轮），尽量复用已计算过的 KV cache，只对新增 token 做 prefill，减少重复计算。

### 1.2 当前架构约束

- **C++ 侧**：`LlaisysQwen2Model` 内嵌单份 KV cache（`k_caches` / `v_caches` + `cache_len`）；每次请求从「整段 token 序列」进入 `llaisysQwen2ModelInfer`，内部根据 `cache_len==0` 判断 prefill 或 decode，**无跨请求的 cache 复用**。
- **Python 侧**：`Qwen2.generate()` / `next_token()` 每次传入**完整 token 序列**；decode 时 C 侧只取最后一个 token，依赖模型内部已填好的 cache。
- **服务端**：单进程、单模型实例；`/v1/chat/completions` 无会话概念，请求体仅 `messages`，每次调用即一次完整 generate。

---

## 二、会话管理设计

### 2.1 数据模型

- **会话 (Session)**  
  - 唯一标识：`session_id`（UUID 或服务端自增 ID）。  
  - 内容：有序消息列表 `messages: List[{ role, content }]`，与现有 OpenAI 风格一致。  
  - 元数据（可选）：`title`（如首条用户消息摘要）、`created_at` / `updated_at`。

- **分支 / 重生成**  
  - 一次「编辑第 k 条用户消息并重新生成」视为：从「前 k 条消息」为前缀，重新生成第 k+1 条（助手）及之后。  
  - 为简化，可约定：同一会话内只保留「当前线性历史」；编辑即截断到该条并替换该条内容，再重生成后续。不要求多分支并存（分支可留作后续扩展）。

### 2.2 HTTP API 设计

在现有 `POST /v1/chat/completions` 基础上，增加会话维度的 CRUD 与「带会话的补全」：

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/v1/sessions` | 列出当前用户（单用户时可省略鉴权）的会话列表：`[{ session_id, title?, updated_at }]`。 |
| POST | `/v1/sessions` | 创建新会话，body 可选 `{ title? }`，返回 `{ session_id, ... }`。 |
| GET | `/v1/sessions/{session_id}` | 获取会话详情：`{ session_id, messages, title?, ... }`。 |
| PATCH | `/v1/sessions/{session_id}` | 更新会话（如重命名 title、或服务端用于「截断 + 替换某条」）。 |
| DELETE | `/v1/sessions/{session_id}` | 删除会话。 |
| POST | `/v1/chat/completions` | **扩展**：请求体增加可选 `session_id`。若带 `session_id`，则：先根据 `session_id` 取会话的 `messages`，再与 body 中的 `messages` 合并（或约定 body 中 `messages` 仅表示「本轮的增量」）；生成完成后，将本轮 user + assistant 追加到该会话并落库/落内存。 |
| POST | `/v1/sessions/{session_id}/regenerate` | **可选**：从某条消息之后重新生成。body：`{ from_message_index: int }`（0-based 的用户消息序号，表示「该条及之前的消息保留，该条之后全部删除并重新生成」）。服务端截断会话到该条，可选地允许 body 带新的 `content` 替换该条用户消息，然后对该会话调用一次「带前缀复用的」generate，结果写回会话。 |

**简化方案**：若暂不做服务端会话存储，可仅在前端维护「多会话」：每个会话一个 `session_id`（前端 UUID），`messages` 仅存在前端；请求仍发 `POST /v1/chat/completions`，body 中带 `session_id`（或仅作前端路由用），服务端仍按「单次请求的 messages」处理，但可结合 `session_id` 做 KV-Cache 池的 key 一部分（见下）。

### 2.3 前端 UI 行为

- **会话列表**：侧栏或顶部 Tab 展示会话列表；点击切换当前会话；支持「新建会话」、删除会话。
- **当前会话**：展示线性消息列表；每条用户消息可提供「编辑」入口；编辑后触发「从此处重新生成」。
- **重新生成**：调用 `regenerate` 或带「截断后的 messages」的 `chat/completions`；UI 上移除该条之后的助手回复，再流式追加新回复。

---

## 三、KV-Cache 池设计

### 3.1 复用语义

- 将「当前请求的 prompt」对应为 token 序列 `P = [t_0, ..., t_{n-1}]`。  
- 若池中存在「前缀等于 `P[0:k]`」的 KV 状态（即曾对长度为 k 的 token 序列做过 prefill），则本次只需对 `P[k:n]` 做 prefill（或 k==n 则仅 decode），并将新产生的 KV 写回池中、与前缀 k 对应的条目合并或替换。

- **前缀匹配**：用「前缀 token 序列」的某种**指纹**作为 key（见下），value 为「该前缀长度下的 KV 状态」；请求时对当前 P 找「最长匹配前缀」，再只对后缀做 prefill。

### 3.2 Key 设计

- **Key**：能唯一对应「一段 token 序列前缀」的标识。可选方案：
  - **方案 A**：对前缀 `token_ids[0:k]` 做哈希（如 xxHash / SHA256 取前 8B），记为 `prefix_hash(k)`；池中存 `(prefix_hash(k), k)` → KV。查找时对当前 P 的每个前缀长度 k 查表（从长到短），先命中者即为「最长匹配前缀」。
  - **方案 B**：`(session_id, message_index)` 表示「该会话、到第 message_index 条消息为止的 prompt 对应前缀」。查找时：当前请求若带 `session_id` 且对应会话的 messages 已存在，则前缀由「该会话的 messages 转成的 token 序列」决定；用 `(session_id, index)` 直接查池。编辑/重生成会改变后续消息，故前缀只到「某条用户消息为止」，index 为该条对应的逻辑位置（如「第几条 user 消息」）。
- **推荐**：方案 A 与实现无关、可跨会话复用；方案 B 更贴合「会话 + 编辑」语义，实现简单。可先做 B，后续再引入 A 做跨会话复用。

### 3.3 Value 设计

- **Value**：与「前缀长度」对应的 KV 状态。即每层的 `K`、`V` 在「该前缀长度」下的张量数据（形状与当前 C++ 实现一致，如每层 `[maxseq, nkvh, dh]`，有效长度为前缀长度）。
- 存储形式：要么在 **C++ 侧** 提供「从外部写入/读出 KV 的接口」；要么在 **Python 侧** 维护多份「模型实例 + 其内部 cache」，由 Python 决定把哪一份「绑定」到当前请求（内存占用大，仅适合极小规模）。**推荐在 C++ 侧扩展**：见 3.5。

### 3.4 池的容量与淘汰

- 池中条目数上限：`max_entries`（如 16 或 32）；超过时需淘汰。
- **淘汰策略**：LRU（最近最少使用）；或按「前缀长度」优先保留较长前缀（因长前缀复用收益大）。每条条目可带 `last_used_at` 或引用计数。
- 单条条目体积：与模型层数、maxseq、nkvh、dh、dtype 相关；可估算单条约数十 MB 量级，总池大小需可配置。

### 3.5 C++ 侧扩展（推荐）

当前 C 接口仅支持「整段 token 进、单步出下一个 token」，且 cache 完全内置于模型。要支持「前缀复用」，需下列之一或组合：

- **方案 I：导出/导入 KV**  
  - 新增：`llaisysQwen2ModelExportKVCache(model, ptr_out)`：将当前 `model->k_caches / v_caches` 中有效长度 `cache_len` 的数据拷贝到 `ptr_out`（或写入到某块由调用方管理的内存）。  
  - 新增：`llaisysQwen2ModelImportKVCache(model, ptr_in, prefix_len)`：从 `ptr_in` 读入前缀长度为 `prefix_len` 的 KV，写入 `model->k_caches/v_caches`，并设置 `model->cache_len = prefix_len`。  
  - 之后调用方再调用 `llaisysQwen2ModelInfer(model, suffix_tokens, n_suffix, ...)` 时，C 侧应支持「仅对 suffix 做 prefill」（即 cache_start = prefix_len，输入仅为 suffix 的 token）；**当前实现**是 prefill 时输入整段 token，需改为：当「已导入 cache 且 prefix_len>0」时，本次输入仅 suffix，prefill 只写 cache 的 [prefix_len, prefix_len+len(suffix)) 段。

- **方案 II：显式 prefill / decode 两步 API**  
  - `llaisysQwen2ModelPrefill(model, token_ids, ntoken)`：对整段做 prefill，写满 cache，不返回 next token。  
  - `llaisysQwen2ModelDecodeStep(model, temperature, top_k, top_p, seed)`：仅用当前 cache 做一步 decode，返回 next token；内部 cache_len += 1。  
  - 池中存「prefill 后的 KV 快照」；复用前先 `ImportKVCache` 再多次 `DecodeStep`；若需「对后缀 prefill」，则需支持 `PrefillFrom(model, start_pos, token_ids, ntoken)`（从 start_pos 起写 cache），与方案 I 等价。

- **方案 III：池在 C++ 内**  
  - 模型侧增加「多个 cache slot」或「cache 池句柄」；API 形如 `InferWithCachePool(pool, session_id, prefix_key, token_ids, ntoken, ...)`，C++ 内查池、命中则只对后缀 prefill、未命中则全量 prefill 并写入池。  
  - 对现有 Python/服务端侵入最小，但 C++ 侧改动最大，且与「会话」语义耦合。

**推荐**：先做 **方案 I**（Export/Import + 支持「带 prefix_len 的 suffix-only prefill」），池与 key 管理放在 **Python 服务端**；这样 C++ 只做「无状态」的 cache 读写与 infer 语义扩展，会话与淘汰策略全部在 Python 中实现。

### 3.6 Python 服务端与池的交互流程

- **请求进入**：body 含 `messages`（及可选 `session_id`、`regenerate_from_index`）。
- **构造 prompt**：根据 messages（及是否 regenerate、截断到哪一条）得到最终用于生成的 `messages'`，再 `tokenizer.apply_chat_template(..., tokenize=False)` 得到字符串，再 `tokenizer.encode(...)` 得到 `input_ids = P`（长度 n）。
- **查池**：  
  - 若使用 `(session_id, message_index)` 为 key：则 key = (session_id, 当前会话中「最后一条包含进 prompt 的用户消息」的 index)。  
  - 若使用 prefix hash：对 P 的每个前缀 P[0:k] 计算 hash，从 k=n-1 往下查池，首次命中即得到最长匹配前缀长度 `k_star` 和对应的 KV 句柄。
- **命中**：  
  - 从池中取出 KV 数据，调用 `llaisysQwen2ModelImportKVCache(model, ptr, k_star)`；  
  - 对 `P[k_star : n]` 做 prefill（需 C 侧支持「仅输入 suffix」）；  
  - 然后对 `P` 的 last token 做 decode 得到 next token，再自回归直到 EOS 或 max_new_tokens；  
  - 将新产生的 KV（长度从 k_star 到当前 cache_len）写回池（覆盖或新条目），并更新 LRU。
- **未命中**：  
  - 全量 prefill P（与现有行为一致），decode 循环；  
  - 将本次完整 KV（长度 n, n+1, ...）在每次 decode 后或最终按「若干前缀长度」写入池（例如仅存 n、n+1、… 的 snapshot，或只存最终长度）；更新 LRU。
- **淘汰**：在「写入新条目前」若 `len(pool) >= max_entries`，按 LRU 删掉一条，再写入。

---

## 四、端到端流程小结

### 4.1 用户发送新消息（当前会话）

1. 前端将当前会话的 `messages` 追加本条 user，调用 `POST /v1/chat/completions`（带 `session_id` 与完整 `messages`）。
2. 服务端根据 `session_id` 取会话（或直接用 body 的 messages），转成 `input_ids` = P。
3. KV 池查前缀（如用 session_id + 上一条消息的 index 或 prefix hash）。
4. 命中则 ImportKV + 仅对「本条 user 对应的后缀」prefill + decode 循环；未命中则全量 prefill + decode。
5. 流式/非流式返回；将 assistant 回复追加到会话并落库/落内存；可选地将新 KV 写入池。

### 4.2 用户编辑某条并「从此处重新生成」

1. 前端截断会话到该条（含），可选地替换该条内容，调用 `POST /v1/sessions/{id}/regenerate` 或带「截断后的 messages」的 `POST /v1/chat/completions`。
2. 服务端截断会话，得到新的 `messages'`，转成 `input_ids` = P。
3. 此前缀可能与「编辑前」不同，池中可能仍能命中「更短的前缀」（例如该条之前的对话未变）。查池得到最长匹配前缀 k_star。
4. ImportKV(k_star)；对 P[k_star:n] prefill；decode 循环；写回会话并可选写回池。
5. 前端移除该条之后的旧回复，流式展示新回复。

### 4.3 用户切换会话

- 前端切换当前 `session_id`，拉取该会话的 `messages`（GET `/v1/sessions/{id}` 或本地状态），展示历史。
- 下次发送或重生成时，用该 `session_id` 参与池的 key；池中若曾有该会话的更长前缀，可复用。

---

## 五、实现顺序建议

1. **Phase 1：会话管理（无池）**  
   - 服务端：实现 `/v1/sessions` CRUD 与内存存储（或简单文件/ SQLite）；`POST /v1/chat/completions` 支持 `session_id`，自动追加回复到会话。  
   - 前端：多会话列表、切换、新建/删除；编辑某条 + 「从此处重新生成」调用「截断后的 messages」的 chat/completions。  
   - 不实现 KV 池，每次请求仍全量 prefill。

2. **Phase 2：C++ KV 导出/导入与 suffix prefill**  
   - 在 C 侧实现 ExportKVCache / ImportKVCache，以及「当 cache_len>0 时，Infer 可仅接受 suffix token 做 prefill」的语义（或拆成 PrefillSuffix + DecodeStep）。  
   - Python 侧封装：`model.import_kv_cache(buf, prefix_len)`，`model.prefill_suffix(suffix_ids)`（若有独立 API），再 `next_token()` 循环。

3. **Phase 3：Python 侧 KV-Cache 池**  
   - 池结构：key（如 (session_id, index) 或 prefix_hash）、value（KV 二进制 + prefix_len）、LRU。  
   - 请求路径中：查池 → 命中则 import + prefill_suffix + decode 循环；未命中则全量 prefill + decode，并写回池。  
   - 淘汰策略与 `max_entries` 可配置。

4. **Phase 4（可选）**  
   - 前缀 key 改为 hash(prefix_token_ids)，支持跨会话复用；  
   - 池持久化（如落盘），重启后部分热前缀可加载。

---

## 六、与现有代码的对接点

- **app.py**：新增 `/v1/sessions` 路由；`chat_completions` 中读取 `session_id`、`regenerate_from_index`，调用「会话存储」与「带池的 generate」封装。
- **qwen2.py**：若 C 侧提供 Import/Export 与 suffix prefill，此处增加 `import_kv_cache`、`prefill_suffix`（或通过修改 `generate` 的入参语义实现）。
- **qwen2.cc / qwen2.h**：新增 Export/Import 接口；修改 Infer 或拆成 Prefill + DecodeStep，支持「已有 cache 时仅对 suffix 做 prefill」。

以上为会话管理 + KV-Cache 池的接口与流程设计，可按 Phase 1 → 2 → 3 的顺序分步实现。
