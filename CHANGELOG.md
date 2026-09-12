# Changelog

本项目两个模块独立演进：`rag_service` 已到 v1（接口稳定），`agent_service` 为首版。

## [Unreleased]

### Fixed — 演示索引在真实克隆上不可用（第九轮续）

上一版把演示索引连同 `index.faiss` / `index.pkl` 一起提交，从 GitHub 克隆后**它自己就是坏的**：

```
clone → 安装 → pytest
  FAILED test_the_demo_index_is_not_reported_as_stale
  {'dense_path': 'unavailable', 'error': 'RagIndexNotReadyError'}
```

根因不是换行符（文件字节数逐一比对完全一致），而是 **manifest 的源指纹记录 `size + mtime_ns`**。
`git clone` 必然重写 mtime，所以**任何提交进仓库的索引都不可能通过校验**——这是设计层面的
事实，不是配置问题。上一轮我只在本地验证了演示索引，没做真克隆，所以漏掉了。

修法是让产物集显式声明自己的形态，而不是放宽校验：

- `build_cosine --prebuilt`：发布转换产物时把 manifest 的 `source` 记为 `null`，表示
  「这份产物集自包含，不绑定它不携带的源索引」。
- 读取侧据此直接加载 sidecar；**普通知识库完全不受影响**——只要 `source` 非空就仍然执行
  严格的 `size+mtime` 比对（`test_faiss_backend_rejects_stale_cosine_files` 继续通过）。
- 反过来也守住：manifest 声明 prebuilt 却出现了 `index.faiss`/`index.pkl` 时**拒绝加载**，
  避免拿来源不明的产物去服务。四种组合均已实测（缺失→加载；源索引/源 pickle/两者出现→拒绝）。
- `scripts/build-demo-index.py` 改为一步产出可提交状态（转换 + prebuilt + 删除源索引对），
  演示产物从 147 KB 降到 93 KB，且不再携带 LangChain pickle。

`tests/test_demo_index.py` 增加用例，断言产物集**不得**包含源索引、且 manifest 必须声明
`source: null`，防止有人「顺手」把它加回去。

测试 334 → 335。

### Added — 演示索引（让克隆后 RAG 立即可用）

之前克隆下来的仓库**无法直接做检索**：`rag_service` 只读，`build_cosine` 只转换，两者都需要
上游工具产出的 `index.faiss` + `index.pkl`，而仓库里没有索引。第一次实测确认了这一点
（全新克隆 + 模板，`search` 退出码 3），也就是「git clone → 安装 → MCP 可用」的最后一环
是断的，新用户只能看到 `RAG 不可用`。

- 新增 `examples/demo-kb`：由 `knowledge-base/cybersec` 的 8 篇模板文档生成的小索引
  （11 行 × 1024 维，147 KB）。`RAG_KB_ROOT=examples/demo-kb` 后 CLI / MCP / HTTP / Agent
  立即可查，无需任何外部工具。
- 新增 `scripts/build-demo-index.py` 生成它，使这份随仓库发布的二进制产物**可复现**，
  而不是一个没人知道怎么重建的黑盒。
- 新增 `tests/test_demo_index.py`：校验产物完整、manifest 未过期、可浏览、且索引覆盖
  全部模板文档。用例不依赖 Ollama（走 browse 模式），并已验证产物损坏或缺失时确实失败。

演示语料暴露了一个真实标定问题并写进文档：**`score_threshold` 是按语料标定的**。默认 0.45
面向百万行语料；11 行的小语料里几乎每个词都"稀有"，词法项贡献趋近 0，融合分≈`0.65 × 余弦`，
于是 6 条正常提问里 3 条掉到 0.45 以下返回 `no_match`——而 top-1 其实都是正确文档。降到 0.35
后 6/6 命中，域外提问仍返回空。演示用法据此在 README 中显式给出。

测试从 330 增至 334。

### Fixed — 第八轮质检（文档化入口点执行、测量数据复核）

本轮把「文档里写了、但从没被执行过」的路径全部跑了一遍，两个文档化入口点有问题：

- **`rag_service/mcp_adapter.py` 在真实 MCP server 上注册即崩溃**（发布阻断）。该模块用了
  `from __future__ import annotations`，于是所有注解都是字符串；MCP SDK 构造工具时会
  `issubclass(param.annotation, Context)`，直接抛
  `TypeError: issubclass() arg 1 must be a class`——宿主 server 根本起不来。
  MCP.md §12 把它列为通用适配器，却**没有任何测试覆盖**。移除该 future import 后注册、schema、
  调用全部正常（已在真实 `FastMCP` 上验证），并补上三处回归用例；把 future import 加回去会
  立刻让用例失败。
- **同一适配器泄漏 pydantic 原始 dump**：`top_k=0` 返回
  `1 validation error for RetrievalRequest ... https://errors.pydantic.dev/...`，违反本仓库
  自己写下的契约（"Every entry point must fail with a usable message, not pydantic's dump"）。
  这是最后一个未转换的边界，因为它是唯一没有测试的入口。
- **`RAG_HTTP_TIMEOUT` 非法值不报变量名**：`float()` 的裸错误经 Agent 降级链路原样转述为
  「检索不可用，已降级为纯对话：could not convert string to float」，把配置笔误说成检索故障。
  该变量读在 `agent_service/rag.py` 而非 config 模块，是上一轮「数值变量必须具名」修复的漏网项。

文档与实测数据复核（跑完两个文档指定的工具后逐项核对）：

- `rag_service/README.md` 说后端不加载 **2.75 GB** 的 `index.faiss`，实测 **4.16 GB**；同段
  「转换出的三个文件」也已过时（现在还会产出 int8/scales/sq8/ranges/postings）。
- `scripts/rag_threshold_band.py` docstring 的域外语料 dense 区间写 0.480，实测 0.446；
  改为直接引用该脚本当前输出，并说明 0.45 相对 0.50/0.55 的取舍。
- `rag_service/README.md` 阈值带表：域外样本数 10 → 12，0.45 的放行数 6 → 7（正例行与
  分布行复核后与实测完全一致，只有这两处漂移）；随之修正「换 4 条域外噪声」为 5 条。

复核通过、未改动的事实：`CVE-2021-3490` top1 = 0.48696、`machine learning overfitting
regularization` = 0.54674、dense 行数 1,016,721、postings tokens 2,612,681 / 条数 4,277,333 /
150 MB / stride 256、embedding 维度 1024、`mcp==1.12.4`、`pydantic==2.9.2`、`faiss-cpu==1.9.0`、
sidecar 体积（f32 4.16 GB、sq8/int8 1.04 GB、jsonl 930 MB）。

另外端到端验证：模板初始化器与导入器（含脱敏、噪声剥离、附件排除、幂等性）、
`evaluate --threshold-scan`、`rag_threshold_band.py`、HTTP API 的鉴权与字段级错误、
以及 Agent 远端模式 + 真实浏览器问答。

测试从 325 增至 330。

### Fixed — 第七轮质检（健康字段自相矛盾）

- **`/v1/rag/health` 在完全健康时报 `sq8_state=unreadable`**：`sq8_state` 回答的是
  「为什么走了 numpy 回退」，但被无条件写进 capabilities，于是同一份 JSON 里同时出现
  `dense_path=sq8` 与 `sq8_state=unreadable`。实测（真实 1,016,721 行索引）确实如此。
  该端点文档写明「供监控探活」，所以这是一个机器可读的假告警：任何读 `sq8_state` 的探活、
  看板或调用方都会把健康服务判成产物损坏。现在该字段在回退发生时报告 `missing` /
  `unreadable`，正常走 sq8 时报告 `loaded`；两种回退原因的区分（以及「不可读不要报成缺失、
  要指向 `--force`」）由既有用例继续守住。
- 文档：云端 provider 的「何时可用」改为「配了 endpoint + 模型列表」，与代码一致
  （Key 缺失时 provider 仍出现，但探测与对话都明确报错）。

测试从 324 增至 325：新增健康字段一致性用例，回退到旧实现后确认失败。

### Fixed — 第六轮质检（CLI 配置诊断、脚本契约、环境变量文档）

- **Agent CLI 把配置错误抛成回溯**：`AGENT_PORT=not-a-port` 运行
  `python -m agent_service --print-config` 会打印十行回溯并以 1 退出。上一轮只给 RAG CLI
  补了字段级契约，Agent CLI 漏了。现在同样输出 `error: AGENT_PORT must be an integer,
  got 'not-a-port'` 并返回 2。
- **RAG CLI 端口解析器吞掉命令行值**：`--port` 在 argparse 构建期用 `int()` 解析
  `RAG_PORT`，`RAG_PORT=not-a-port` 直接抛 `ValueError` 回溯（退出 1）。改为端口专用解析器，
  并把错误话术指向变量名。
- **`--kb-root` 放在子命令之后会被静默忽略**：子解析器用 `default=None` 覆盖了全局值，
  于是 `python -m rag_service --kb-root X search ...` 仍然去读环境变量。子解析器改用
  `argparse.SUPPRESS`，两种位置都生效。
- **`import-security-sources.ps1` 调用有必填参数的 `rebuild-knowledge-base.ps1` 时漏传
  `-KnowledgeBase`**：PowerShell 不会报错，而是**停在交互式参数提示**上等待输入——
  无人值守的导入脚本会静默挂起。`import-mydb.ps1` 与 `import-des-ctf-knowledge.ps1`
  更糟：它们给 `rebuild-cybersec.ps1` 传了它并不存在的 `-KnowledgeBase`。
- **三个导入脚本无法指定索引构建器**：新仓库要求显式 `-ServerRoot`（或
  `CHATCHAT_SERVER_ROOT`），但包装脚本不转发该参数，于是导入完成后重建必然报
  「index builder was not specified」。现在三处都接受并转发 `-ServerRoot`。
- **sidecar 刷新失败仍返回成功**：PowerShell 脚本以非终止错误结束时退出码仍是 0，
  因此 `rebuild-knowledge-base.ps1` 只会打印 `Write-Warning` 就「成功」返回，而服务此后一直
  认为索引过期。改为 `throw`。实测：sidecar 构建失败时退出码 0 → 1。
- **`rag_service/evaluate.py` 的注解引用了未导入的 `Any`**：`from __future__ import
  annotations` 让它不至于报错，但注解无法求值。补上 `typing.Any`。
- **环境变量文档补齐**：README 现在覆盖 RAG 全部可直接调整的旋钮（`RAG_HOST`/`RAG_PORT`、
  候选池与召回上限、各类缓存与 TTL、`RAG_LOW_SCORE_WARN` 等）与 Agent 云端变量
  （`AGENT_CLOUD_API_KEY`、`AGENT_CLOUD_LABEL`、以及「未配 Key 时云端 provider 仍会出现但会
  明确报错」的行为）。`knowledge-base/cybersec/README.md` 的重建命令补上必需的 `-ServerRoot`。

测试从 318 增至 324：新增 `tests/test_script_contracts.py`（跨脚本参数契约、sidecar 失败必须
终止）与两个 CLI 诊断用例。参数契约用例在回退到真实缺陷形态后确认失败。

### Fixed — 第五轮质检（错误话术与诊断真实性）

- **CLI 把配置错误抛成回溯**：`RAG_KB_ROOT` 未设置（第一次运行最常犯的错）时打印十行回溯并
  以 1 退出，而文档承诺的是字段级错误 + 退出码 2。配置与 backend 构造现在也在错误处理内。
  顺带给 `rag_service` 的 ~18 个数值环境变量加上具名解析：`RAG_DEFAULT_TOP_K=abc` 现在报
  `RAG_DEFAULT_TOP_K must be an integer, got 'abc'`，而不是 `invalid literal for int()`（后者
  不说是哪个变量）。
- **sq8 降级原因说错**：任何 numpy 回退都被解释成「sq8 产物缺失」，但当产物**存在却读不出来**
  （文件损坏，或进程内存不足无法映射——这正是实测中触发 numpy 慢路径的情形）时，
  建议的 `build_cosine` 会回答「产物已是最新」而什么都不修。现在区分三种状态：缺失
  （指向 build_cosine）、存在但不可读（指向 `--force` 并提示检查内存）、以及正常走 sq8。

测试从 311 增至 318：新增 `tests/test_cli_diagnostics.py`（CLI 错误契约、sq8 三态诊断），
两个修复都在回退后确认测试确实失败。

### Fixed — 第四轮质检（对抗性 + 全流程复检）

**健康检查是本轮重灾区**（上一轮我自己刚改过的代码）：

- `_rag_status` 拿**默认**知识库去探测，却用 Agent **配置**的知识库名标注结果。于是
  `AGENT_RAG_KNOWLEDGE_BASE=samples`、allow-list 只放行 `cybersec` 时，`--check` 打印
  `[OK] 知识库 samples: rows=1016721`（那是 cybersec 的行数）并以 0 退出，而每一轮问答
  其实都在静默降级。现在探测的就是运行时真正会用的那个知识库。
- **完全不存在索引也被报成 OK**：`backend.status()` 把失败包成值
  （`dense_path="unavailable"`）而不抛异常，`describe_status()` 又丢掉 `error` 字段，于是
  「没有任何索引」变成绿色的就绪信号、`--check` 退出 0 —— 上一轮刚加的退出码契约形同虚设。
  现在把这种结果当失败处理，且**失败结果不缓存**。
- 缓存是模块级单例、不区分配置：同一进程里第二个 `create_app`（或改环境变量后重新构造
  config）会读到另一个部署的知识库与状态。现在按部署（知识库/根目录/模型/远端 URL）分键。
- 配置了 `RAG_SERVICE_URL`（索引故意放在别的进程）时，健康检查仍去开本地索引，于是
  `--check` 报「RAG 不可用」退出 1，而 HTTP 检索其实完全正常。现在改为探远端 `/v1/rag/health`。
- 健康检查不看 embedding 提供方：`RAG_OLLAMA_BASE_URL` 指向死端口时页面仍是绿色的
  「检索就绪」，而每次查询都返回 `degraded=lexical-only`。现在健康检查包含 embedding 预检。

**并发与资源生命周期**：

- `_release_store` 显式 `memmap._mmap.close()` 并在并发读时把数组置 None。实测：带活视图关闭
  映射**会成功**，随后读取是**访问违例（进程直接死，退出码 5）**，不是可捕获的异常。上一次
  审计把它评为「不可达的 P3」，但 `store_cache_limit` 淘汰路径同样会触发它。改为**读者租约**：
  读路径先登记，释放时若有在飞读者则推迟；空闲时仍然立即解映射（保留原有语义与测试）。
  真实索引下 6 线程检索中途 close，0 错误。
- 文档缓存的淘汰循环用裸 `del`，两个并发读者可能选中同一批 key 而抛 `KeyError`（逃逸成 500）。
  改为 `pop(key, None)`（`_cache_put` 早就这么做了）。
- `RAG_MAX_QUERY_LENGTH` 未校验：`0` 让每个查询都报错，`>8000` 无法兑现（请求模型硬上限 8000）。
  现在限定 `1..8000`。

**知识库门禁**：`AGENT_RAG_KNOWLEDGE_BASE` 与 `RAG_ALLOWED_KNOWLEDGE_BASES` 不一致时，报错只说
「knowledge base is not allowed: cybersec」，不说是哪个变量、允许什么。现在统一走
`RagConfig.require_allowed()` 单一入口，报出变量名与可用值。

**清掉一个死字段**：`AgentConfig.ollama_base_url` 构造后从未被读取（对话用的是 provider 的
base_url，embedding 用的是 `RAG_OLLAMA_BASE_URL`），留着会让人误以为它能配置 embedding。
删除，并在模块文档里写清「对话模型」与「检索 embedding」是两套配置。

测试从 308 增至 311：新增 `StoreLifetimeTests`（租约语义、空闲仍解映射、并发搜索中途 close），
补充 `RAG_MAX_QUERY_LENGTH` 边界与健康检查的用例。每个修复都在回退后确认测试会失败。

九轮累计：280 → 335 个测试（CHANGELOG 每节记录各自的增量，README 与 MCP.md 只写当前值）。

### Fixed — 前三轮质检（克隆可用性、检索状态、agent 交互）

按「克隆下来能不能用 → 检索状态对不对 → 界面会不会卡死」的顺序逐轮复检：

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
