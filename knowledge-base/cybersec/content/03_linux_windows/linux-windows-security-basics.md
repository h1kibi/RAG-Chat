---
topic: linux_windows
difficulty: intermediate
source: curated-local
authorized_only: true
---
# Linux 与 Windows 系统安全基础

## Linux

审查用户、组、sudo 规则、服务、计划任务、文件权限、SUID/SGID、SSH 配置、环境变量、容器权限和审计日志。重点是判断“谁能以什么身份执行什么操作”，而不是简单罗列命令。

## Windows

审查本地和域账号、组成员关系、服务、计划任务、注册表、PowerShell 日志、Windows Event Log、共享、RDP、补丁和安全策略。域环境要额外关注身份信任、委派、组策略和高权限账户管理。

## 安全原则

- 使用最小权限、分离管理账号和短期凭据。
- 关闭不需要的服务和管理入口。
- 对关键操作启用审计，集中保存并保护日志。
- 在实验室中验证权限边界，生产环境中先评估回滚方案。
