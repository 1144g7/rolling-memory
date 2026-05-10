# Rolling Memory

对话记忆搜索 MCP Server。自动导入你的 Claude Code 和 WorkBuddy 对话历史，FTS5 全文搜索，零配置。

## 功能

**8 个 MCP 工具，开箱即用：**

| 工具 | 功能 |
|------|------|
| `memory_search` | 搜索对话（关键词 / 语义） |
| `memory_recent` | 最近对话活动 |
| `memory_segments` | 对话段列表 |
| `memory_relations` | 段间关系链 |
| `memory_detail` | 段详情 + 关系 |
| `memory_stats` | 数据统计概览 |
| `memory_timeline` | 屏幕时间线 |
| `memory_query` | 自由 SQL 查询（只读） |

**三阶段能力：**

| 阶段 | 能力 | 条件 |
|------|------|------|
| P1 | FTS5 关键词搜索 + 时间过滤 + 最近活动 | 默认，开箱即用 |
| P2 | BGE-M3 语义搜索 | 需要 GPU + `pip install mcp-rolling-memory[semantic]`（正在整理中） |
| P3 | 段落级搜索 + 摘要 + 关系链 | 需要感知引擎跑完索引（正在整理中） |

## 安装

### Claude Code（推荐）

```bash
claude mcp add rolling-memory -- python -m rolling_memory.server
```

或者一行用 uvx（无需 pip install）：

```bash
claude mcp add rolling-memory -- uvx mcp-rolling-memory
```

### 其他 MCP 客户端（Cursor / Windsurf 等）

```json
{
  "mcpServers": {
    "rolling-memory": {
      "command": "uvx",
      "args": ["mcp-rolling-memory"]
    }
  }
}
```

### 从源码

```bash
git clone https://github.com/1144g7/rolling-memory.git
cd rolling-memory
pip install -e .
claude mcp add rolling-memory -- python -m rolling_memory.server
```

## 首次启动

MCP Server 首次启动时会自动：

1. 创建 SQLite 数据库（`~/.rolling-memory/memory.db`）
2. 扫描 `~/.claude/projects/` 下的 Claude Code 对话
3. 扫描 `~/.workbuddy/projects/` 下的 WorkBuddy 对话
4. 建立 FTS5 全文索引

整个过程无需手动操作，通常几秒钟完成。

## 使用示例

在 Claude Code 中直接提问：

- "我之前聊过什么关于 Python 装饰器的内容？"
- "最近一周讨论了哪些话题？"
- "帮我找一下之前讨论的那个数据库设计方案"

## 数据存储

- 数据库位置：`~/.rolling-memory/memory.db`（可通过 `ROLLING_MEMORY_DB` 环境变量修改）
- 只读取本地数据，不上传任何内容
- 纯 SQLite，无外部依赖

## 支持的数据源

| 数据源 | 路径 | 状态 |
|--------|------|------|
| Claude Code | `~/.claude/projects/` | 自动扫描 |
| WorkBuddy | `~/.workbuddy/projects/` | 自动扫描 |

## 依赖

- Python >= 3.10
- `mcp` (MCP Python SDK)

可选：
- `numpy` + `FlagEmbedding`（语义搜索，P2）

## License

MIT
