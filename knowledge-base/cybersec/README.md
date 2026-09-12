# 网络安全、渗透测试与 CTF 知识库

这个目录是 `cybersec` 知识库的可复现模板。内容面向授权安全测试、隔离靶场、CTF 和防守研究。

## 使用边界

- 只在明确获得授权的系统、自己搭建的实验环境或 CTF 平台中验证。
- 不要导入真实密码、API Key、Cookie、JWT、SSH 私钥、个人信息或未脱敏的内部地址。
- 对攻击性问题优先回答原理、检测、修复和本地靶场复现，不提供未授权入侵或隐蔽持久化方案。

## 分类

- `00_foundations`：网络、协议、编码和安全基础
- `01_web_security`：Web/API 安全测试与修复
- `02_network_security`：资产、服务、流量和网络防御
- `03_linux_windows`：Linux/Windows 系统安全
- `04_vulnerability`：漏洞分析、验证和修复
- `05_pentest_method`：授权渗透测试流程和报告
- `06_ctf`：CTF 题目分析、复盘和方法论
- `07_defense`：检测、响应、审计和加固
- `08_ctf_des_knowledge`：Des-CTF-Knowledge 外部 Markdown 资料和 WriteUp（运行时导入）
- `09_hacktricks`：HackTricks 的渗透测试、漏洞研究、系统安全和 CTF 资料（运行时导入）
- `10_payloads_all_the_things`：PayloadsAllTheThings 的漏洞分类说明和 Payload 文本清单（运行时导入）

初始化到运行目录：

```powershell
.\scripts\init-cybersec-kb.ps1
```

构建 FAISS 索引（索引构建器不在本仓库，必须显式指出）：

```powershell
.\scripts\rebuild-cybersec.ps1 -ServerRoot C:\path\to\LangGraph-Chatchat\chatchat-server
```

`-ServerRoot` 也可以预先用 `$env:CHATCHAT_SERVER_ROOT` 指定；两者都缺时脚本会直接报错。

## 导入外部资料

一次导入两个上游仓库，并只重建一次 `cybersec`：

```powershell
.\scripts\import-security-sources.ps1 -DataRoot C:\RAG-Agent-Data `
    -ServerRoot C:\path\to\LangGraph-Chatchat\chatchat-server
```

该脚本会把仓库浅克隆到 `C:\RAG-Agent-Data\sources`，固定并记录当前 commit，然后导入到：

- `data\knowledge_base\cybersec\content\09_hacktricks`
- `data\knowledge_base\cybersec\content\10_payloads_all_the_things`

需要更新上游快照时使用 `-Refresh`。只做导入检查、不重建向量时使用 `-SkipRebuild`。

导入规则：

- HackTricks：仅导入 Markdown，去掉 `src/` 前缀，跳过索引、脚本、图片、二进制、许可证和 Agent 配置。
- PayloadsAllTheThings：导入 Markdown 与纯文本 Payload 清单，跳过脚本、图片、压缩包、二进制和许可证。
- 每份文档加入仓库、上游路径、commit、获取日期和授权用途元数据，并在 `imports/<repo>` 保存清单、SHA256、来源说明和许可证副本。
- 对疑似私钥、Bearer Token、API Key、密码和 Cookie 进行 `<REDACTED>` 脱敏；导入前仍应避免使用含真实凭据的资料。
- 不执行任何上游脚本、命令或 Payload；`samples` 知识库不会被修改。

