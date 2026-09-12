# Changelog

本项目两个模块独立演进：`rag_service` 已到 v1（接口稳定），`agent_service` 为首版。

## [Unreleased]

### Fixed — 首轮质检（pre-release）

从「克隆下来能不能用」的角度复检，修复以下问题：

- **`.gitignore` 吞掉了测试夹具**（发布阻断）：裸 `data/` 在 git 中匹配任意深度，导致
  `tests/data/retrieval_queries.jsonl` 从未入库。新克隆上 `test_evaluate_labels.py` 直接
  `FileNotFoundError`，且文档里所有评测流程都无法运行——而本地因为文件恰好在磁盘上，测试全绿。
  规则改为锚定仓库根 `/data/`。
- **降级标记粘在线程上**：`_reset_search_state()` 定义了但从未调用，嵌入提供方短暂中断后，
  同一线程上后续**成功**的查询仍被标为 `degraded=lexical-only`，并在工具文本里加
  `DEGRADED`，让调用方不要相信正确的余弦分数。改为每次 `search()` 开始重置。
- **浏览模式跳过 sidecar 时效校验**：`_browse_ranges` 只核对行数，不比对源文件指纹，
  于是索引重建但未重跑 `build_cosine` 时，语料索引/来源列表/文档分页仍返回**旧索引**的文档，
  而查询路径正确报错。三种浏览路径现在与查询同样的校验、同样的错误。
- **CLI 把范围错误抛成 pydantic 回溯**：`--top-k 0`、`--filters year=1800` 等打印内部堆栈。
  现在输出字段级错误并返回退出码 `2`；索引未就绪返回 `3`。其他入口（MCP/OpenAI/HTTP）
  本来就这么做，只剩 CLI 没转换。
- **`create_langchain_tool` 的 ImportError 不可操作**：`langchain-core` 只在 `[index]` extra 中，
  基础安装下只报裸 `ModuleNotFoundError`。现在指明要装的 extra。
- **空字符串布尔环境变量反转默认值**：`AGENT_RAG_ENABLED=`（写在 shell profile 里）会把检索
  关掉，而语义应是「未配置」。数字/布尔值无法解析时现在报出变量名。
- **`/api/health` 把可用端点报成不可用**：不实现 `GET /models` 的 OpenAI 兼容网关会被标红，
  尽管对话正常。404/405 时回退到一次最小补全请求，结论确定。
- **health 每次调用重开索引**：大语料上一次约 1 GB 读取，而 UI 每次加载都会轮询。成功结果改为
  记忆化；失败仍每次重查（索引可能只是还没建）。
- **Agent 丢失「检索过但没命中」与「DEGRADED」信号**：检索服务在自己的工具文本里会写这两件事，
  Agent 路径上却被丢掉，于是模型可能把自己的知识当成语料证据回答。现在两者都进 prompt，
  并在 UI 上显示降级提示。
- **`--check` 在检索不可用时仍返回 0**，部署门禁无法发现坏索引。现在 `1` = 检索不可用、`2` = 无可用模型。
- 文档：README 写了代码不读的 `expected_sources` 键名；重建命令缺 `-ServerRoot`；
  MCP.md 的 `mcp==1.12.0` 与锁定的 1.12.4 不一致；测试基线从 271 更新到实际值；
  `untrusted_evidence` 被写成请求参数（实际是响应字段，传入会 422）。

测试从 280 增至 303：新增 `tests/test_backend_state.py`（降级标记、sidecar 时效）与
`tests/test_agent_llm.py`（provider 探测、SSE 流解析），并补充配置与证据渲染的回归用例。
每个修复都在回退后验证过「测试确实失败」。

### Added — `agent_service` v0.1.0rc1

离线优先的本地 Web 问答。新增模块，之前不存在。

- **双 provider**：本地 Ollama 为默认路径（离线可用）；云端 OpenAI 兼容端点仅在显式配置
  `AGENT_CLOUD_BASE_URL` + `AGENT_CLOUD_MODELS` 后出现。API Key 通过
  `AGENT_CLOUD_API_KEY_ENV` 间接引用环境变量，不落盘、不进日志、`/api/config` 脱敏。
- **流式对话**：`POST /api/chat` 返回 SSE，事件为 `start` / `retrieval` / `delta` /
  `warning` / `done` / `error`。错误事件带可执行的 `hint`。
- **可选检索前置**：`use_rag` 打开时先检索再把证据放进 prompt，`retrieval` 事件回传
  `source` / `chunk_id` / `score` 供 UI 展示引用。
- **检索失败降级**：索引不可用或 Ollama 未起时只发 `warning`，继续走纯对话，不让索引问题
  阻断断网现场的使用。
- **单页 UI**：`agent_service/static/index.html`，无外部依赖（无 CDN、无构建步骤），
  离线可加载。含 provider/模型选择、检索开关、引用列表、Token 输入、健康状态灯。
- **自检命令**：`python -m agent_service --check`（provider 连通性 + 索引状态）、
  `--print-config`（脱敏配置）。
- **鉴权**：`AGENT_API_TOKEN` 设置后 `/api/chat` 需要 Bearer；页面与 `/api/health`、
  `/api/config` 保持开放，便于排障。

### Changed — 仓库结构

从上游 `LangGraph-Chatchat` 的 `chatchat-server/` 下拆出，成为独立仓库：

- `rag_service/`、`agent_service/`、`tests/` 移到仓库根目录；`pyproject.toml` 与
  `requirements.txt` 重新生成，不再依赖上游的 editable 安装。
- **移除 `agent_service/rag.py` 对 `chatchat.settings` 的兜底导入**：in-process 检索改为
  直接要求 `RAG_KB_ROOT` / `KB_ROOT_PATH`。独立仓库里没有 `chatchat`，原兜底路径不可达。
- `faiss.py` / `evaluate.py` 里指向旧 venv 的提示文案改为「仓库虚拟环境」。
- `rag_service/README.md`、`rag_service/MCP.md` 中的路径示例改为 `<repo>` 占位。
- 新增 `[index]` extra：`build_cosine` 需要反序列化上游写的 `index.pkl`，因此依赖
  `langchain-community` / `langchain-core`；只查询已有索引不需要。

### Removed

- `tests/test_cybersec_docs.py`：断言 `README-cybersec.md` 存在并包含指定命令。该文档已删除，
  且这类「文档里必须出现某字符串」的测试只锁文案、不验证行为，故删除而非改指向。

## [v1] — `rag_service`

只读本地检索服务，接口稳定。以下为进入 v1 的能力集合。

### 接口

- **MCP stdio server**，工具名 `ctf_rag`。支持查询与四种免 embedding 浏览模式：语料索引、
  来源列表分页、单篇文档分页回取、按 `chunk_id` 精确回取单块。
- **HTTP API**：`/v1/rag/health`、`/v1/rag/categories`、`/v1/rag/search`，可选 Bearer。
- **CLI**：`python -m rag_service search ...`（含 `--filters`、`--json`、`--cursor`）。
- **库调用**：`RagService` / `RagHttpClient` / `create_langchain_tool` /
  `create_openai_tool_schema` / `dispatch_openai_tool_call`。
- 后端可替换：实现 `RetrievalBackend.search(request)` 即可换成 Milvus、pgvector、
  Elasticsearch 或远程检索 API。

### 检索质量

- 余弦 memmap 后端 + int8 SQ8 全量扫描（不走 `index.faiss` 全量加载）；faiss 缺失时降级到
  numpy 并在启动日志与 health 中显式报告 `dense_path`。
- 词法重排（IDF + 路径字段加权）与 dense 分数融合，默认 `lexical_weight=0.35`。
- 离线 postings 精确标识符召回，解决「只给一个 CVE 编号」时正确文档落在候选池外的问题。
- 同源相邻 chunk 合并、跨源镜像去重（8-gram minhash ≥0.85）、低信息块过滤。
- 路径过滤器在打分前收窄候选，窄过滤不会返回空。
- 版本/架构/CVE 事实上内联到结果头部，便于比对语料与目标环境。
- 分数语义与 agent 使用约定写入文档：score 只用于排序，不是相关概率。

### 运维

- `build_cosine` 生成 sidecar 产物（向量/文档/偏移/SQ8/行区间/清单），原子替换，
  记录源文件指纹，索引重建后过期即报错而不是静默用旧数据。
- `RAG_LEXICAL_FALLBACK=1` 时 Ollama 不可用仍可返回词法证据，并标注 `DEGRADED`。
- 文档行缓存与 store 缓存都有上限，长驻服务不会无界增长。
- `rag_service.evaluate` 用标注好的问句集测 top-1 / MRR。
