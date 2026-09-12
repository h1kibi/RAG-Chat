# 本地资料库智能问答

把一批本地资料（Markdown / CTF writeup / 漏洞分析 / 手册）变成一个**离线可用**的问答系统。

主场景是 **CTF 线下赛断网环境**：机器上没有外网，模型、向量、检索全在本机跑。联网时也可以切换成云端 API 获得更强的回答质量，但这不是默认路径。

---

## 两个模块

```
<repo>/
├── rag_service/        RAG 检索模块 —— 只读，可独立交付
│                       MCP 工具 / HTTP API / CLI / LangChain / OpenAI schema
├── agent_service/      Agent 模块 —— 本地 Web 对话
│                       离线 Ollama，或填 API Key 走云端
├── knowledge-base/     知识库模板（Git 内，8 篇示例文档）
├── examples/demo-kb/   演示索引（93 KB），开箱即可查询
├── scripts/            建库与导入脚本
└── tests/              401 个测试
```

两者**解耦**：`rag_service` 不 import `agent_service`，也不 import 任何 Agent 框架；`agent_service` 通过 `agent_service/rag.py` 这一个桥接点消费检索能力。所以你可以只用 RAG 工具接自己的 Agent，完全不需要 Agent 模块。

| | RAG 模块 | Agent 模块 |
|---|---|---|
| 定位 | 检索与证据核验 | 人机问答界面 |
| 状态 | **v1 已完成** | 首版，功能克制 |
| 依赖 | faiss + Ollama embedding | 可用任意 OpenAI 兼容模型 |
| 入口 | MCP / HTTP / CLI / 库调用 | Web UI `http://127.0.0.1:8801` |
| 写操作 | 无（只读） | 无 |

---

## 快速开始（离线）

### 0. 前置：Python 3.12 + Ollama

```powershell
ollama pull qwen2.5:7b      # 对话模型
ollama pull bge-m3          # 检索用 embedding
ollama list
```

### 1. 安装

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt   # 完全固定版本
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

`requirements.txt` 是实测通过的完整锁定版本，方便在联网机器上先下好 wheel，再搬到隔离环境。

### 2. 准备知识库

RAG 模块读的是 **转换后的索引产物**，不是原始的 `index.faiss`：

```powershell
$env:RAG_KB_ROOT = 'C:\RAG-Agent-Data\data\knowledge_base'
.\.venv\Scripts\python.exe -m rag_service.build_cosine --kb-root $env:RAG_KB_ROOT
```

这一步把上游索引转成 memmap + int8 形式（原文件不动）。**每次重建索引后都要重跑**，否则服务会报索引过期。

> 构建索引本身由 `LangGraph-Chatchat` 等上游工具完成，本项目只负责转换、检索和服务。`build_cosine` 需要反序列化上游写的 `index.pkl`，因此要装 `.[index]`（即 `langchain-community`）；只查询已有索引则不需要。

**本仓库不含索引。** 已装好的 `C:\RAG-Agent-Data` 属于运行数据，不进 Git；克隆下来的仓库里只有 `knowledge-base/` 模板（8 篇手写示例文档），没有 `vector_store/`。要先用一会儿再准备自己的语料，可以直接用随仓库附带的演示索引：

```powershell
$env:RAG_KB_ROOT = 'examples\demo-kb'
$env:RAG_ALLOWED_KNOWLEDGE_BASES = 'cybersec'
$env:RAG_DEFAULT_SCORE_THRESHOLD = '0.35'   # 见下方说明
.\.venv\Scripts\python.exe -m rag_service search "授权渗透测试开始前要确认什么" --top-k 2
```

`examples/demo-kb`（93 KB，11 行）由上面 8 篇模板文档生成，用来验证安装是否完整。它由 `scripts/build-demo-index.py` 一步重建（需要 Ollama 与 `bge-m3`）：

```powershell
.\.venv\Scripts\python.exe scripts/build-demo-index.py
```

它是 `build_cosine --prebuilt` 发布的产物：**只含转换后的 sidecar，不含 `index.faiss` / `index.pkl`**。原因是 manifest 的源指纹记录 `size + mtime_ns`，而 **Git 无法保留 mtime**——提交进仓库的索引在任何一次 `git clone` 后都会被判为过期（实测：克隆后 `dense_path=unavailable`，检索直接不可用）。`--prebuilt` 让产物集显式声明「没有源索引」，服务据此直接读取自包含的 sidecar；**普通知识库不受影响**，源索引一旦存在仍走严格的 `size+mtime` 校验（已覆盖测试）。用真实语料时不要照搬这个脚本的切块规则。

> **演示索引要把 `RAG_DEFAULT_SCORE_THRESHOLD` 降到 `0.35`。** 默认 `0.45` 是按百万行语料标定的：小语料里几乎每个词都"稀有"，词法项贡献接近 0，融合分基本等于 `0.65 × 余弦`，于是 11 行里 6 条正常提问会掉到 0.45 以下而返回 `no_match`。降到 0.35 后 6/6 命中正确文档，域外提问（如"今天晚饭吃什么"）仍然返回空。阈值是**按语料标定**的，换语料就该重跑 `rag_service.evaluate` / `scripts/rag_threshold_band.py`。

### 3. 起 Agent

```powershell
$env:AGENT_RAG_KNOWLEDGE_BASE = 'cybersec'
.\.venv\Scripts\python.exe -m agent_service
```

打开 `http://127.0.0.1:8801`。加载时会做自检，告诉你模型通不通、索引是否就绪。

先自检再启动（排障第一步）：

```powershell
.\.venv\Scripts\python.exe -m agent_service --check
.\.venv\Scripts\python.exe -m agent_service --print-config   # 脱敏，不含 API Key
```

`--check` 的退出码可以直接给部署脚本用：`0` 全部就绪；`1` 模型可用但检索不可用（问答降级为纯对话）；`2` 没有可用模型（完全无法回答）。

### 4. 起 RAG 工具（给第三方 Agent 用）

```powershell
$env:RAG_ALLOWED_KNOWLEDGE_BASES = 'cybersec'
.\.venv\Scripts\python.exe -m rag_service.mcp_server        # stdio MCP，工具名 ctf_rag
```

MCP client 配置（`cwd` 必须是本仓库根目录）：

```json
{
  "mcpServers": {
    "ctf-rag": {
      "command": "<repo>\\.venv\\Scripts\\python.exe",
      "args": ["-m", "rag_service.mcp_server"],
      "cwd": "<repo>",
      "env": {
        "RAG_KB_ROOT": "C:\\RAG-Agent-Data\\data\\knowledge_base",
        "RAG_ALLOWED_KNOWLEDGE_BASES": "cybersec",
        "RAG_EMBEDDING_MODEL": "bge-m3",
        "RAG_OLLAMA_BASE_URL": "http://127.0.0.1:11434",
        "RAG_LEXICAL_FALLBACK": "1"
      }
    }
  }
}
```

`RAG_LEXICAL_FALLBACK=1`：Ollama 挂掉时降级为纯词法检索，结果会明确标注 `DEGRADED (lexical-only)`，而不是假装正常。

---

## RAG 模块

只读检索服务。详细运维说明见 [`rag_service/README.md`](rag_service/README.md)，MCP 接入细节见 [`rag_service/MCP.md`](rag_service/MCP.md)。

### 四种接入方式

```python
# 1) 进程内
from rag_service import RagConfig, RagService, RetrievalRequest
from rag_service.backends.faiss import FaissBackend

config = RagConfig.from_environment()
service = RagService(config, FaissBackend(config))
response = service.search(RetrievalRequest(query="CVE-2021-43798 Grafana 任意文件读取", top_k=5))

# 2) 远程 HTTP（让 FAISS 留在别的进程）
from rag_service.http_client import RagHttpClient
client = RagHttpClient("http://127.0.0.1:8791", api_token="...")
client.search_dict(query="...", top_k=5)

# 3) LangChain StructuredTool
from rag_service.adapters import create_langchain_tool
tool = create_langchain_tool(service)

# 4) OpenAI function/tool schema
from rag_service.adapters import create_openai_tool_schema, dispatch_openai_tool_call
```

换后端只需实现 `RetrievalBackend.search(request) -> list[SearchResult]` 注入 `RagService`，即可换成 Milvus、pgvector、Elasticsearch 或远程检索 API。

### CLI（不启动服务也能查）

```powershell
# 查询
.\.venv\Scripts\python.exe -m rag_service search "glibc 2.31 tcache double free safe-linking" --top-k 5

# 浏览语料：空 query，不触发 embedding
.\.venv\Scripts\python.exe -m rag_service search "" --filters category=15_butian --limit 20

# 过滤 + 原始 JSON
.\.venv\Scripts\python.exe -m rag_service search "JNDI" --filters year=2021 --json
```

`--filters` 可重复：`category`、`source_prefix`、`source`、`chunk_id`、`year`、`exclude_source_prefix`。

**抽取模式**（`extract=code|payload`）：直接返回命中文档里的**围栏代码段**并标注语言，不再把
query 窗口切片丢给 agent 自行拼接。`code` 收全部围栏，`payload` 只收可执行的（语言标注为
shell/编程语言，或正文形如命令/exploit）。该模式下不做窗口裁剪，因为调用方要的是完整片段。
文档内没有匹配片段时会明确说明，而不是返回空。CRLF 文档同样可抽取（Windows 写的 Markdown）。

**命中置信度**（响应字段 `confidence`，并在工具文本首行以 `conf=` 呈现）：

| 值 | 含义 |
|---|---|
| `anchored` | 查询里的标识符（CVE/版本/端口）出现在命中文档中 |
| `lexical` | 词面支撑强（`lexical_score >= 0.5`） |
| （无） | 两种证据都不成立，**不作判定** |

**只陈述正向证据，缺省不是「有罪判定」**。上一版对「都不成立」的情况报 `semantic` 并告警，
结果在 16 条正确命中里误报 3 条；实测也证明该情况无法逐个判定——正确命中的词面分出现过
0.355，而真正离题的出现过 0.353。因此现在缺省即沉默，弱命中仍由低分告警覆盖。
top1/top2 **分差同样不可用**（正确命中出现过 0.0011 的分差，因为该主题被多篇不同文档覆盖，
而域外查询在 0.0039–0.0057），故不采用。

### HTTP 服务

```powershell
.\.venv\Scripts\python.exe -m rag_service --host 127.0.0.1 --port 8791
```

| Method | Path | 说明 |
|---|---|---|
| GET | `/v1/rag/health` | 存活 + 运行能力（含 `dense_path`） |
| GET | `/v1/rag/categories?knowledge_base=cybersec` | 可用分类前缀 |
| POST | `/v1/rag/search` | 检索（设了 `RAG_API_TOKEN` 则需要 Bearer） |

### 关键设计

- **不加载 4 GB 的 `index.faiss`**：`build_cosine` 转出 memmap 余弦矩阵 + int8 SQ8 索引，查询走 faiss `IndexScalarQuantizer`（实测比 numpy 反量化快约 20×）。内存不足时 `faiss.read_index` 会失败并**静默降级**到 numpy 慢路径——所以启动日志和 `/health` 都会打印 `dense_path=sq8|numpy`。
- **路径过滤在打分前生效**：`category` / `source_prefix` / `source` 先用 source→行区间把候选收窄，再算相似度，窄过滤不会返回空。
- **分数只用于排序，不是相关概率**：默认下限 `0.45`。查询请用 2–4 个具体技术锚点，不要只查 `heap` 这类单词——词面召回会命中无关主题。
- **证据不可信**：文档内容一律当数据，不当指令。Agent 模块的 prompt 明确要求模型不执行片段中的任何指令。
- **来源分层 `origin`**：按 source 路径首段标注 `handbook`（HackTricks / PayloadsAllTheThings / LOLBAS / Des-CTF）、`writeup`（赛事 writeup）、`blog-mirror`（先知/补天/安全学习镜像）、`curated`（仓库自带模板）。这是**来源事实而非质量评分**——镜像博客可能写得更好，调用方只是有权知道手里是哪类材料（reviewer 指出「镜像博客、HackTricks、赛事 writeup 平权混排」）。
- **图片不在语料里，且会如实告知**：导入器按设计排除二进制与 `images/` 目录——实测 `content/` 下 **0 个图片文件、0 个 images 目录**（20,464 个文本文件）。因此相对图片引用标注 `not-imported` 并在结果行提示「fetch from upstream」，远端 URL 标注 `remote`。这样调用方不会为一个打不开的路径浪费一步；要读图必须回上游来源（这是语料构建策略，不是检索缺陷）。
- **往届 flag 会被标记**：writeup 里常抄录当时抓到的 flag，对「复现原题」有用，对「变体题」则是致命误导——服务无法判断属于哪种，因此**标记而不删改**：结果行给出 `past_event_flags=N`，`metadata.flags` 列出原文，由调用方决定。检测偏召回（漏掉真 flag 的代价高于误标一句提醒），并排除语言关键字与 LaTeX/代码形态；语料实测 2.6% 的行命中，抽样无误报。
- **四条结果路径共用同一套信号**：查询、`chunk_id` 取回、source 分页、词法回退都会带上 `facts` / `flags` / `has_screenshots` / `image_refs`。此前只有查询路径完整，`chunk_id` 取回（正是核对引用用的那条）缺失——已用单一助手收敛。
- **截图地址随计数一起返回**：`shots=N` 之外还给出 `images=<地址,…>`，有视觉能力的 agent 可以自己去读那张图，而不是只知道"证据在图片里"。
- **结果自带事实**：命中行会内联文档自述的 `glibc=` / `arch=` / `cve=`，方便机械比对目标环境与语料的版本差异。

### 主要环境变量

`RAG_KB_ROOT`（必填）、`RAG_ALLOWED_KNOWLEDGE_BASES`、`RAG_DEFAULT_KNOWLEDGE_BASE`、`RAG_EMBEDDING_MODEL`、`RAG_OLLAMA_BASE_URL`、`RAG_DEFAULT_SCORE_THRESHOLD`、`RAG_LEXICAL_WEIGHT`、`RAG_LEXICAL_FALLBACK`、`RAG_SNIPPET_CHARS`、`RAG_API_TOKEN`、`RAG_SERVICE_URL`、`RAG_HTTP_TIMEOUT`。

其余可直接调整的旋钮（默认值即实测最优，改动前请先跑评测）：

| 变量 | 默认 | 作用 |
|---|---|---|
| `RAG_HOST` / `RAG_PORT` | `127.0.0.1` / `8791` | HTTP 服务监听；`--host` / `--port` 覆盖 |
| `RAG_DEFAULT_TOP_K` | `5` | 未传 `top_k` 时的条数 |
| `RAG_MAX_TOP_K` | `50` | `top_k` / `limit` 的上限 |
| `RAG_MAX_QUERY_LENGTH` | `8000` | 查询字符上限（1..8000） |
| `RAG_MAX_CONTENT_CHARS` | `8000` | 单条结果正文硬上限 |
| `RAG_CANDIDATE_POOL` | `600` | 无过滤查询的重排候选池 |
| `RAG_FILTERED_CANDIDATE_LIMIT` | `1200` | 带过滤查询的候选池 |
| `RAG_PATH_RECALL_LIMIT` / `RAG_PATH_RECALL_ROWS_PER_PATH` | `400` / `2` | 路径命中的强制召回上限 |
| `RAG_IDENTIFIER_RECALL_LIMIT` | `40` | 每个标识符 token 的 postings 召回行数（`0` 关闭） |
| `RAG_STORE_CACHE_LIMIT` / `RAG_DOCUMENT_CACHE_LIMIT` | `3` / `2048` | 常驻索引代数 / 文档行缓存 |
| `RAG_EMBEDDING_FAILURE_TTL` | `30` | embedding 失败的记忆秒数 |
| `RAG_LOW_SCORE_WARN` | `0.55` | 低分警告阈值（`0` 关闭） |
| `RAG_LOW_INFO_FILTER` | `1` | 低信息 chunk 过滤 |
| `RAG_MERGE_NEIGHBOR_LIMIT` | `2` | 相邻 chunk 合并的每侧上限 |
| `RAG_FILTER_FLAG_ATTACHMENTS` / `RAG_STRIP_IMAGES` / `RAG_STRIP_PROVENANCE` | `1` | 附件、图片语法、provenance 头处理 |

完整列表（含取值范围校验）见 `rag_service/config.py`。

---

## Agent 模块

问答界面 + 一个可选的检索前置步骤。跑在本机，只监听 `127.0.0.1`。

### 两种模型来源

| provider | 何时可用 | 说明 |
|---|---|---|
| `ollama` | 永远（离线主路径） | 本地 Ollama，默认 `qwen2.5:7b` |
| `cloud` | 配了 endpoint + 模型列表 | 任意 OpenAI 兼容端点；未配 Key 时会出现但明确报错，不会静默失败 |

云端 provider **只在显式配置后才出现**，默认配置里根本没有它——断网时不会因为找不到 Key 而失败。API Key 只从环境变量读，不落盘、不进日志、不回传给浏览器（`/api/config` 做了脱敏）。

```powershell
# 云端（可选）
$env:AGENT_CLOUD_BASE_URL = 'https://open.bigmodel.cn/api/paas/v4'
$env:AGENT_CLOUD_MODELS   = 'glm-5.3-flash'
$env:AGENT_CLOUD_API_KEY_ENV = 'ZAI_API_KEY'   # 间接引用，避免 Key 出现在配置里
$env:ZAI_API_KEY = '...'
```

### 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `AGENT_HOST` / `AGENT_PORT` | `127.0.0.1` / `8801` | 监听地址 |
| `AGENT_API_TOKEN` | 空 | 设置后 `/api/chat` 需要 Bearer |
| `AGENT_DEFAULT_PROVIDER` | `ollama` | `ollama` \| `cloud` |
| `AGENT_DEFAULT_MODEL` | 首个可用模型 | |
| `AGENT_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | 自动补 `/v1` |
| `AGENT_OLLAMA_MODELS` | `qwen2.5:7b` | 逗号分隔 |
| `AGENT_SYSTEM_PROMPT` | 内置 | |
| `AGENT_HISTORY_LEN` | `10` | 送入模型的最近消息条数 |
| `AGENT_TEMPERATURE` / `AGENT_MAX_TOKENS` | `0.2` / `1024` | **推理模型要给够**：思考内容与实际回答共用这个预算 |
| `AGENT_REQUEST_TIMEOUT` | `300` | 秒。离线 7B 首 token 较慢 |
| `AGENT_RAG_ENABLED` | `1` | 关掉就是纯对话 |
| `AGENT_RAG_KNOWLEDGE_BASE` | 空 | 空则用 RAG 侧默认 |
| `AGENT_RAG_TOP_K` | `5` | |
| `AGENT_RAG_SCORE_THRESHOLD` | 空 | 空则用 RAG 侧默认 |
| `AGENT_RAG_EVIDENCE_CHARS` | `4000` | 放入 prompt 的证据字符上限 |

**推理模型（如 `deepseek-flash`）需要更大的 `AGENT_MAX_TOKENS`。** 它的思考写在
`reasoning_content`（界面不显示），并**与最终回答共用同一个 token 预算**：预算太小会导致长时间无输出、
然后回答还没开始就被截断。实测 `max_tokens=1024` 时，一个正常问题先花掉约 1.3 秒/百字的思考，
13 秒后才出现第一个可见字符。服务会识别这种情况并给出可执行的提示；界面在等待期间显示
`生成中… Ns` 计时，避免误判为卡死。建议推理模型用 `4096`。
可选：`AGENT_CLOUD_API_KEY_ENV`/`AGENT_CLOUD_LABEL`（见下）。

云端 provider 相关：`AGENT_CLOUD_BASE_URL`、`AGENT_CLOUD_MODELS`（两者必须同时设置）、`AGENT_CLOUD_API_KEY`（直接给 Key）或 `AGENT_CLOUD_API_KEY_ENV`（指向存放 Key 的环境变量名，默认 `ZAI_API_KEY`）、`AGENT_CLOUD_LABEL`（UI 显示名）。未配置 Key 时云端 provider 仍会出现，但探测与对话都会明确报「未配置 API Key」，不会静默失败。

`RAG_KB_ROOT` 仍需设置，否则检索不可用（Agent 会警告并降级为纯对话，不会报错退出）。

**问答模型和检索 embedding 是两套配置**，把其中一个指向远端不会带动另一个：

| 用途 | 变量 | 归谁管 |
|---|---|---|
| 对话模型 | `AGENT_OLLAMA_BASE_URL` | Agent |
| 检索 embedding | `RAG_OLLAMA_BASE_URL` | RAG（索引是用它建的） |
| 检索哪个知识库 | `AGENT_RAG_KNOWLEDGE_BASE` | Agent 选择，RAG 用 `RAG_ALLOWED_KNOWLEDGE_BASES` 放行 |

知识库名不在 allow-list 时，健康检查和对话都会明确告诉你该改哪个变量。

### HTTP 接口

| Method | Path | 说明 |
|---|---|---|
| GET | `/` | 单页聊天 UI（Markdown 渲染；依赖本地内置，离线可加载） |
| GET | `/api/config` | 脱敏配置：provider、模型、RAG 开关 |
| GET | `/api/health` | 每个 provider 的连通性 + 索引状态 |
| POST | `/api/chat` | SSE 流式问答 |

`/api/chat` 的事件类型：`start` → `retrieval`（含 `source` / `chunk_id` / `score`，UI 用来显示引用）→ `delta`* → `done`；失败时 `warning`（降级）或 `error`（终止，带 `hint`）。

### 行为约定

- **回答按 Markdown 渲染**：助手输出经 `marked` 解析、`DOMPurify` 清洗后显示（标题/列表/代码块/表格/引用）。两个库都**内置在仓库里**（`agent_service/static/vendor/`，共 62 KB），不引用 CDN——断网环境必须能加载。
- **清洗不可省略**：模型输出受检索片段影响，而语料**刻意包含 prompt-injection payload**；把这类文本解析成 HTML 而不清洗，等于让被投毒的文章在操作者浏览器里执行脚本。允许列表剔除 `script`/`iframe`/`style`/`form`，URL 仅放行 `http(s)`/`mailto`/锚点/同源路径（`javascript:` 与 `data:` 被拒）。若清洗确实移除了内容，页面会明示「部分内容已被安全过滤」，而不是静默给出残缺回答。
- **检索失败不中断对话**：索引坏了或 Ollama 没起，只发一条 `warning` 然后走纯对话。断网现场不该因为索引问题让人没法用。
- **不编造**：有证据时 system prompt 要求「片段没覆盖就说没有足够依据」并给出 `source + chunk_id`。
- **错误可执行**：连不上本地模型时给的是「先 `ollama serve`，确认模型已 `ollama pull`」，不是裸的 `ConnectionError`。
- **历史长度有上限**，避免 6 GB 显存环境被长上下文拖死。

---

## 知识库

`knowledge-base/` 是 Git 内的**模板**，运行时数据在 `C:\RAG-Agent-Data`（不进 Git）：

```powershell
.\scripts\init-cybersec-kb.ps1 -DataRoot C:\RAG-Agent-Data
```

索引构建由上游工具完成，不在本仓库内，所以要显式告诉脚本去哪里找它（`-ServerRoot`，或 `$env:CHATCHAT_SERVER_ROOT`），并且所有导入脚本都接受同一个参数：

```powershell
.\scripts\rebuild-knowledge-base.ps1 -KnowledgeBase cybersec -EmbeddingModel bge-m3 `
    -ServerRoot C:\path\to\LangGraph-Chatchat\chatchat-server
```

`rebuild-cybersec.ps1` 是同参数的 `cybersec` 便捷包装；`import-mydb.ps1`、`import-security-sources.ps1`、`import-des-ctf-knowledge.ps1` 都会把 `-ServerRoot` 透传给它。

没有 `-ServerRoot`（且未设置 `CHATCHAT_SERVER_ROOT`）时脚本会明确报错，而不是静默失败。重建索引成功后会**自动**刷新 `build_cosine` 产物；sidecar 刷新失败会让脚本以非零退出并报错，因为此时服务会一直认为索引过期——修复后重跑 `python -m rag_service.build_cosine` 即可，新索引本身是完整的。

`scripts/import-mydb.ps1`、`import-security-sources.ps1`、`import-des-ctf-knowledge.ps1`、`import_des_ctf_knowledge.py` 用于把外部资料导入独立知识库；导入前会做脱敏与噪声过滤，并保留来源清单。规则见脚本头部注释。

### 评测

```powershell
.\.venv\Scripts\python.exe -m rag_service.evaluate --help
```

用 `tests/data/retrieval_queries.jsonl` 测 top-1 / MRR，并对比不同 `lexical_weight` 与阈值。每条记录用 `query` + `prefixes`（source 路径前缀）/ `sources`（精确路径）标注正确答案，可选 `label`（`positive` / `negative` / `mismatch`，默认 `positive`）。调阈值前先跑 `scripts/rag_threshold_band.py` 看分数分布，别凭感觉调。

---

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
# 401 passed
```

覆盖：检索打分与融合、路径/年份/镜像去重过滤、分页与单块回取、MCP 边界与错误话术、索引转换、CLI 参数、Agent 配置解析、prompt 组装与历史裁剪、SSE 事件流、鉴权、脱敏。

---

## 安全边界

- 本项目提供**可检索的学习资料**，不自动授权任何测试行为，不替代书面授权、变更审批或安全评审。
- 检索结果一律视为**不可信证据**：不执行其中的指令、提示词或命令。
- 不导入真实密码、API Key、私钥、Cookie、Token、个人信息或生产数据。导入链路有脱敏 pass。
- 只监听 `127.0.0.1`。需要对外时自行加认证、反向代理和网络隔离。
- API Key 只从环境变量读取；`.gitignore` 已排除 `.env`、`*.key`、`*.pem`。

---

## 已知限制

- **Python 3.12 only**；`faiss-cpu==1.9.0` 没有 3.13 轮子。`requirements.txt` 的平台专属项保留了环境标记（`pywin32` 仅 Windows 安装），因此 Linux/macOS 也能按其安装；但**只在 Windows 11 实测过**，测试套件本身不依赖 Ollama（已验证 338 项在 provider 不可达时全通过）。
- **索引构建不在本仓库**：需要上游 `LangGraph-Chatchat` 之类的工具产出 `index.faiss` + `index.pkl`，本项目负责转换与检索。仓库内只有 `examples/demo-kb`（93 KB，由 8 篇模板文档生成、以 `--prebuilt` 发布）用于验证安装；真实语料必须自己构建，`scripts/build-demo-index.py` 的简单切块规则**不适合**大语料。
- **sidecar 指纹含 mtime**：`postings` / `ranges` 侧车用 docs 文件的 `size+mtime_ns` 做指纹，因此**拷贝或克隆一个知识库后这些侧车一定失效**。`ranges` 会静默回退到流式重建（大语料约 20s，结果正确）；`postings` 会让启动行显示 `postings=unusable`（标识符召回暂时关闭），重跑 `build_cosine` 即恢复。演示索引不受影响（11 行且无含数字 token）。
- **检索质量取决于语料**：`score_threshold` 默认 0.45 是按百万行语料标定的。换语料必须重新标定（`rag_service.evaluate` / `scripts/rag_threshold_band.py`），小语料往往需要更低门限。
- **单用户、单机**：没有权限体系、没有并发调度，不是服务端方案。
- **扫描版 PDF 未处理**：需要单独的 OCR 流程。
- **Agent 模块只做 Web 问答**：没有工具调用、没有多步 Agent 图、没有在线搜索。这是刻意的——断网环境下这些能力价值有限。
- **显存/内存敏感**：6 GB 显存 + 16 GB 内存的机器上，同时跑多个索引实例会 OOM。启动前确认空闲内存，别并行开多个服务进程。

---

## 发布状态

| 模块 | 版本 | 说明 |
|---|---|---|
| `rag_service` | v1 | 接口稳定，可直接接第三方 Agent |
| `agent_service` | v0.1.0rc1 | 首版 Web 问答，接口可能调整 |

变更记录见 [CHANGELOG.md](CHANGELOG.md)。

## 许可

Apache-2.0，见 [LICENSE](LICENSE)。
