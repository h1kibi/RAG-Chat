# Changelog

本项目两个模块独立演进：`rag_service` 已到 v1（接口稳定），`agent_service` 为首版。

## [Unreleased]

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
