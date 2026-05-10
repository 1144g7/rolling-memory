"""
Rolling Memory MCP Server — 对话记忆搜索

自动扫描 Claude Code / WorkBuddy 对话目录，
FTS5 全文索引，开箱即用，零配置。

三阶段能力：
  P1 (默认): FTS5 关键词搜索 + 时间过滤 + 最近活动
  P2 (需配置): BGE-M3 语义搜索
  P3 (需引擎): 段落级搜索 + 摘要 + 关系链
"""
import json
import os
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

# ── 配置 ──────────────────────────────────────────────

DB_PATH = os.environ.get(
    "ROLLING_MEMORY_DB",
    str(Path.home() / ".rolling-memory" / "memory.db"),
)

DECAY_FACTOR = 0.05
BASE_WEIGHT = 0.7
RECENCY_WEIGHT = 0.3

mcp = FastMCP(
    name="rolling-memory",
    instructions="对话记忆搜索。自动导入 Claude Code 和 WorkBuddy 对话历史。",
)

READ_ONLY = ToolAnnotations(readOnlyHint=True)

# 当前活跃的 CC 会话（搜索时自动排除）
_active_cc_conv: str | None = None


def _detect_active_session():
    """检测当前正在进行的 Claude Code 会话，搜索时排除"""
    global _active_cc_conv
    cc_dir = Path.home() / ".claude" / "projects"
    if not cc_dir.exists():
        return

    latest_file = None
    latest_mtime = 0
    for project_dir in cc_dir.iterdir():
        if not project_dir.is_dir():
            continue
        for jsonl_file in project_dir.glob("*.jsonl"):
            try:
                mtime = jsonl_file.stat().st_mtime
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_file = jsonl_file
            except OSError:
                continue

    if latest_file:
        _active_cc_conv = f"cc_{latest_file.stem}"
        _log(f"当前活跃会话: {_active_cc_conv}，搜索时自动排除")


def _text(lines: list[str]) -> str:
    return "\n".join(lines)


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=wal")
    return conn


def _log(msg: str):
    print(f"[Rolling Memory] {msg}", file=sys.stderr, flush=True)


# ── 时间衰减 ──────────────────────────────────────────

def _apply_time_decay(scored: list, decay: float) -> list:
    now = datetime.now()
    result = []
    for score, r in scored:
        created = r["created_at"] or ""
        try:
            dt = datetime.fromisoformat(created[:10])
            days_old = max((now - dt).days, 0)
        except (ValueError, IndexError):
            days_old = 365
        recency = 1.0 / (1.0 + days_old * decay)
        final = score * (BASE_WEIGHT + RECENCY_WEIGHT * recency)
        result.append((final, r))
    result.sort(key=lambda x: x[0], reverse=True)
    return result


# ── 过滤器构建 ─────────────────────────────────────────

def _build_filters(source: str, date_from: str, date_to: str,
                   table_prefix: str = "c",
                   exclude_conv: str = "") -> tuple[str, list]:
    filters = []
    params = []
    if source:
        filters.append(f"{table_prefix}.source_type = ?")
        params.append(source)
    if date_from:
        filters.append(f"{table_prefix}.created_at >= ?")
        params.append(date_from)
    if date_to:
        filters.append(f"{table_prefix}.created_at <= ?")
        params.append(date_to + "T99")
    if exclude_conv:
        filters.append(f"{table_prefix}.id != ?")
        params.append(exclude_conv)
    where = ""
    if filters:
        where = "AND " + " AND ".join(filters)
    return where, params


# ── memory_search: 核心搜索 ───────────────────────────

@mcp.tool(annotations=READ_ONLY)
def memory_search(
    query: str,
    mode: str = "keyword",
    source: str = "",
    date_from: str = "",
    date_to: str = "",
    scope: str = "segments",
    time_decay: float = 0.05,
    top_k: int = 10,
    exclude_current: bool = True,
) -> str:
    """搜索对话记忆。
    mode: keyword(FTS5)/semantic(向量,需GPU)。
    source: 过滤平台(claude/gemini/gpt/workbuddy/claude_code)。
    date_from/date_to: 日期范围(YYYY-MM-DD)。
    scope: segments/chunks。
    time_decay: 时间衰减因子(0=关闭, 0.05=默认约20天半衰期)。
    exclude_current: 排除当前活跃的 Claude Code 会话(默认true)。
    """
    conn = _conn()
    try:
        if mode == "semantic":
            return _search_semantic(conn, query, source, date_from, date_to,
                                    scope, time_decay, top_k, exclude_current)
        else:
            return _search_keyword(conn, query, source, date_from, date_to,
                                   scope, time_decay, top_k, exclude_current)
    finally:
        conn.close()


def _search_keyword(conn, query, source, date_from, date_to,
                    scope, time_decay, top_k, exclude_current=True) -> str:
    exclude = _active_cc_conv if exclude_current else ""

    if " " in query or "-" in query or "." in query:
        fts_query = f'"{query}"'
    else:
        fts_query = query

    if scope == "chunks":
        return _search_keyword_chunks(conn, fts_query, query, source,
                                      date_from, date_to, time_decay, top_k,
                                      exclude_conv=exclude)

    # 尝试 segments 级搜索（P3，需要 conv_segments 表）
    where, params = _build_filters(source, date_from, date_to, table_prefix="ci",
                                   exclude_conv=exclude)
    try:
        sql = f"""
            SELECT cs.id, cs.conv_id, cs.seg_idx, cs.name, cs.summary,
                   cs.source,
                   ci.name as conv_name, ci.global_summary, ci.conclusion,
                   ci.created_at
            FROM fts_segments fts
            JOIN conv_segments cs ON cs.id = fts.rowid
            LEFT JOIN conv_index ci ON ci.conv_id = cs.conv_id
            WHERE fts_segments MATCH ? {where}
            ORDER BY rank
            LIMIT ?
        """
        rows = conn.execute(sql, [fts_query] + params + [top_k * 3]).fetchall()
    except sqlite3.OperationalError:
        return _search_keyword_chunks(conn, fts_query, query, source,
                                      date_from, date_to, time_decay, top_k,
                                      exclude_conv=exclude)

    if not rows:
        return f"未找到 '{query}'"

    if time_decay > 0:
        scored = [(1.0, r) for r in rows]
        scored = _apply_time_decay(scored, time_decay)
        rows = [r for _, r in scored[:top_k]]
    else:
        rows = rows[:top_k]

    lines = [f"找到 {min(len(rows), top_k)} 条结果:"]
    for r in rows:
        lines.append(f"- [{r['id']}] {r['name']}  ({r['source']}, {r['created_at']})")
        lines.append(f"  对话: {r['conv_name']}")
        lines.append(f"  摘要: {(r['summary'] or '')[:300]}")
    return _text(lines)


def _search_keyword_chunks(conn, fts_query, query, source, date_from,
                           date_to, time_decay, top_k,
                           exclude_conv: str = "") -> str:
    """搜原始对话消息"""
    where, params = _build_filters(source, date_from, date_to,
                                   exclude_conv=exclude_conv)

    try:
        sql = f"""
            SELECT msg.text,
                   c.source_type as source, c.name as conv_name,
                   c.created_at, msg.sender
            FROM fts_chunks fts
            JOIN conversation_messages msg ON msg.rowid = fts.rowid
            JOIN conversations c ON c.id = msg.conversation_id
            WHERE fts_chunks MATCH ? {where}
            ORDER BY rank
            LIMIT ?
        """
        rows = conn.execute(sql, [fts_query] + params + [top_k]).fetchall()
    except sqlite3.OperationalError:
        like_q = f"%{query}%"
        sql = f"""
            SELECT msg.text,
                   c.source_type as source, c.name as conv_name,
                   c.created_at, msg.sender
            FROM conversation_messages msg
            JOIN conversations c ON c.id = msg.conversation_id
            WHERE msg.text LIKE ? {where}
            LIMIT ?
        """
        rows = conn.execute(sql, [like_q] + params + [top_k]).fetchall()

    if not rows:
        return f"未找到 '{query}'"

    lines = [f"找到 {len(rows)} 条消息:"]
    for r in rows:
        lines.append(f"- [{r['sender']}] {r['conv_name']}  ({r['source']}, {r['created_at']})")
        lines.append(f"  {(r['text'] or '')[:300]}")
    return _text(lines)


def _search_semantic(conn, query, source, date_from, date_to,
                     scope, time_decay, top_k, exclude_current=True) -> str:
    try:
        import numpy as np
        from rolling_memory.embedding import get_embedding
    except ImportError:
        return "语义搜索(P2)需要 BGE-M3 模型，请先用 mode=keyword。P2 正在整理中，敬请期待。"

    bge = get_embedding()
    q_vec = bge.encode([query])[0]
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-8)

    where, params = _build_filters(source, date_from, date_to, table_prefix="cs")

    sql = f"""
        SELECT sce.seg_id, sce.dense_vector,
               cs.name, cs.summary, cs.source,
               ci.name as conv_name, ci.conclusion, ci.created_at
        FROM seg_combined_embeddings sce
        JOIN conv_segments cs ON cs.id = sce.seg_id
        LEFT JOIN conv_index ci ON ci.conv_id = cs.conv_id
        WHERE sce.dense_vector IS NOT NULL {where}
    """
    rows = conn.execute(sql, params).fetchall()

    if not rows:
        return "无向量数据"

    scored = []
    for r in rows:
        vec = np.frombuffer(r["dense_vector"], dtype=np.float32)
        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
        score = float(np.dot(q_norm, vec_norm))
        scored.append((score, r))

    if time_decay > 0:
        scored = _apply_time_decay(scored, time_decay)
    else:
        scored.sort(key=lambda x: x[0], reverse=True)

    lines = []
    for score, r in scored[:top_k]:
        if score < 0.2:
            continue
        lines.append(f"- [{r['seg_id']}] {r['name']}  (score={score:.3f}, {r['source']}, {r['created_at']})")
        lines.append(f"  对话: {r['conv_name']}")
        lines.append(f"  摘要: {(r['summary'] or '')[:300]}")
    if not lines:
        return "未找到相关结果(相似度<0.2)"
    lines.insert(0, f"找到 {len(lines)//3} 条结果:")
    return _text(lines)


# ── memory_segments ───────────────────────────────────

@mcp.tool(annotations=READ_ONLY)
def memory_segments(conv_id: str) -> str:
    """获取对话的所有段摘要"""
    conn = _conn()
    try:
        # Try conv_index first (P3), fallback to conversations (P1)
        conv = conn.execute("""
            SELECT conv_id, name, source, created_at, global_summary, conclusion
            FROM conv_index WHERE conv_id = ?
        """, (conv_id,)).fetchone()

        if not conv:
            # P1 fallback
            conv = conn.execute("""
                SELECT id as conv_id, name, source_type as source, created_at
                FROM conversations WHERE id = ?
            """, (conv_id,)).fetchone()
            if not conv:
                return f"对话 {conv_id} 不存在"
            msgs = conn.execute("""
                SELECT sender, text FROM conversation_messages
                WHERE conversation_id = ? ORDER BY sequence
            """, (conv_id,)).fetchall()
            lines = [
                f"对话: {conv['name']}  ({conv['source']}, {conv['created_at']})",
                f"共 {len(msgs)} 条消息:",
            ]
            for m in msgs:
                lines.append(f"  [{m['sender']}] {(m['text'] or '')[:200]}")
            return _text(lines)

        segs = conn.execute("""
            SELECT id, seg_idx, name, summary
            FROM conv_segments WHERE conv_id = ? ORDER BY seg_idx
        """, (conv_id,)).fetchall()

        lines = [
            f"对话: {conv['name']}  ({conv['source']}, {conv['created_at']})",
            f"全局摘要: {(conv['global_summary'] or '')[:500]}",
            f"共 {len(segs)} 个段:",
        ]
        for s in segs:
            lines.append(f"  [{s['id']}] seg#{s['seg_idx']} {s['name']}")
            lines.append(f"    {s['summary']}")
        return _text(lines)
    finally:
        conn.close()


# ── memory_relations ──────────────────────────────────

@mcp.tool(annotations=READ_ONLY)
def memory_relations(segment_id: int = None, conv_id: str = None,
                     relation_type: str = None) -> str:
    """查询段间关系链。segment_id 或 conv_id 至少提供一个。"""
    conn = _conn()
    try:
        try:
            if segment_id:
                seg = conn.execute(
                    "SELECT conv_id, seg_idx FROM conv_segments WHERE id = ?",
                    (segment_id,)
                ).fetchone()
                if not seg:
                    return f"段 {segment_id} 不存在"
                rels = conn.execute("""
                    SELECT cr.from_idx, cr.to_idx, cr.rel_type,
                           fs.name as from_name, ts.name as to_name
                    FROM conv_relations cr
                    LEFT JOIN conv_segments fs ON fs.conv_id=cr.conv_id AND fs.seg_idx=cr.from_idx
                    LEFT JOIN conv_segments ts ON ts.conv_id=cr.conv_id AND ts.seg_idx=cr.to_idx
                    WHERE cr.conv_id=? AND (cr.from_idx=? OR cr.to_idx=?)
                """, (seg["conv_id"], seg["seg_idx"], seg["seg_idx"])).fetchall()
            elif conv_id:
                type_f = "AND cr.rel_type = ?" if relation_type else ""
                params = [conv_id] + ([relation_type] if relation_type else [])
                rels = conn.execute(f"""
                    SELECT cr.from_idx, cr.to_idx, cr.rel_type,
                           fs.name as from_name, ts.name as to_name
                    FROM conv_relations cr
                    LEFT JOIN conv_segments fs ON fs.conv_id=cr.conv_id AND fs.seg_idx=cr.from_idx
                    LEFT JOIN conv_segments ts ON ts.conv_id=cr.conv_id AND ts.seg_idx=cr.to_idx
                    WHERE cr.conv_id=? {type_f}
                """, params).fetchall()
            else:
                return "需提供 segment_id 或 conv_id"

            lines = [f"共 {len(rels)} 条关系:"]
            for r in rels:
                lines.append(f"- {r['from_name']}(seg#{r['from_idx']}) --[{r['rel_type']}]--> {r['to_name']}(seg#{r['to_idx']})")
            return _text(lines)
        except sqlite3.OperationalError:
            return "关系数据不可用。P3（段落索引+关系链）正在整理中，敬请期待。"
    finally:
        conn.close()


# ── memory_detail ─────────────────────────────────────

@mcp.tool(annotations=READ_ONLY)
def memory_detail(segment_id: int) -> str:
    """获取段完整信息+关联关系"""
    conn = _conn()
    try:
        try:
            seg = conn.execute("""
                SELECT cs.id, cs.conv_id, cs.seg_idx, cs.name, cs.summary,
                       cs.source, cs.msg_span_start, cs.msg_span_end,
                       ci.name as conv_name, ci.global_summary, ci.conclusion, ci.created_at
                FROM conv_segments cs
                LEFT JOIN conv_index ci ON ci.conv_id = cs.conv_id
                WHERE cs.id = ?
            """, (segment_id,)).fetchone()

            if not seg:
                return f"段 {segment_id} 不存在"

            rels = conn.execute("""
                SELECT cr.from_idx, cr.to_idx, cr.rel_type,
                       fs.name as from_name, ts.name as to_name
                FROM conv_relations cr
                LEFT JOIN conv_segments fs ON fs.conv_id=cr.conv_id AND fs.seg_idx=cr.from_idx
                LEFT JOIN conv_segments ts ON ts.conv_id=cr.conv_id AND ts.seg_idx=cr.to_idx
                WHERE cr.conv_id=? AND (cr.from_idx=? OR cr.to_idx=?)
            """, (seg["conv_id"], seg["seg_idx"], seg["seg_idx"])).fetchall()

            span = f"{seg['msg_span_start']}-{seg['msg_span_end']}" if seg["msg_span_start"] is not None else "N/A"
            lines = [
                f"段 #{seg['id']} (seg_idx={seg['seg_idx']}) {seg['name']}",
                f"对话: {seg['conv_name']}  ({seg['source']}, {seg['created_at']})",
                f"msg_span: {span}",
                f"摘要: {seg['summary']}",
            ]
            if rels:
                lines.append(f"关系 ({len(rels)} 条):")
                for r in rels:
                    tag = " <--" if r["to_idx"] == seg["seg_idx"] else " -->"
                    lines.append(f"  {r['from_name']} --[{r['rel_type']}]--{tag}> {r['to_name']}")
            else:
                lines.append("无关系")
            return _text(lines)
        except sqlite3.OperationalError:
            return "段详情不可用。P3（段落索引+摘要+关系链）正在整理中，敬请期待。"
    finally:
        conn.close()


# ── memory_stats ──────────────────────────────────────

@mcp.tool(annotations=READ_ONLY)
def memory_stats() -> str:
    """数据统计概览"""
    conn = _conn()
    try:
        lines = ["=== 记忆统计 ==="]

        # P1 基础统计
        conv_count = conn.execute("SELECT count(*) FROM conversations").fetchone()[0]
        msg_count = conn.execute("SELECT count(*) FROM conversation_messages").fetchone()[0]
        lines.append(f"对话: {conv_count}")
        lines.append(f"消息: {msg_count}")

        # P3 统计（如果可用）
        for table, label in [("conv_segments", "段"), ("conv_relations", "关系"),
                             ("screen_timeline", "屏幕事件")]:
            try:
                cnt = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                lines.append(f"{label}: {cnt}")
            except sqlite3.OperationalError:
                pass

        lines.append("\n按来源:")
        for src, cnt in conn.execute(
            "SELECT source_type, count(*) cnt FROM conversations GROUP BY source_type ORDER BY cnt DESC"
        ).fetchall():
            lines.append(f"  {src}: {cnt}")

        # P3 关系类型（如果可用）
        try:
            lines.append("\n关系类型:")
            for rt, cnt in conn.execute(
                "SELECT rel_type, count(*) cnt FROM conv_relations GROUP BY rel_type ORDER BY cnt DESC"
            ).fetchall():
                lines.append(f"  {rt or '(空)'}: {cnt}")
        except sqlite3.OperationalError:
            pass

        tr = conn.execute("SELECT min(created_at) a, max(created_at) b FROM conversations").fetchone()
        lines.append(f"\n时间范围: {tr['a']} ~ {tr['b']}")

        lines.append("\n最近5条:")
        for r in conn.execute(
            "SELECT name, source_type, created_at FROM conversations ORDER BY created_at DESC LIMIT 5"
        ).fetchall():
            lines.append(f"  {r['name']}  ({r['source_type']}, {r['created_at']})")

        return _text(lines)
    finally:
        conn.close()


# ── memory_recent ─────────────────────────────────────

@mcp.tool(annotations=READ_ONLY)
def memory_recent(days: int = 7, source: str = "", limit: int = 20) -> str:
    """最近对话活动。days: 回看天数, source: 过滤平台。"""
    conn = _conn()
    try:
        where, params = _build_filters(source, "", "")
        if days > 0:
            where += f" AND c.created_at >= date('now', '-{days} days')"

        # Try conv_index (P3) first, fallback to conversations (P1)
        try:
            rows = conn.execute(f"""
                SELECT ci.conv_id, ci.name, ci.source, ci.created_at,
                       ci.global_summary, ci.msg_count, ci.n_segments
                FROM conv_index ci
                WHERE 1=1 {where}
                ORDER BY ci.created_at DESC
                LIMIT ?
            """, params + [limit]).fetchall()
        except sqlite3.OperationalError:
            rows = conn.execute(f"""
                SELECT c.id as conv_id, c.name, c.source_type as source, c.created_at,
                       c.message_count as msg_count
                FROM conversations c
                WHERE 1=1 {where}
                ORDER BY c.created_at DESC
                LIMIT ?
            """, params + [limit]).fetchall()

        lines = [f"最近 {len(rows)} 条对话:"]
        for r in rows:
            extra = f", {r['msg_count']}msg"
            try:
                extra += f", {r['n_segments']}seg"
            except (KeyError, TypeError):
                pass
            lines.append(f"- {r['name']}  ({r['source']}, {r['created_at']}{extra})")
            try:
                if r['global_summary']:
                    lines.append(f"  {r['global_summary'][:200]}")
            except (KeyError, TypeError):
                pass
        return _text(lines)
    finally:
        conn.close()


# ── memory_timeline ───────────────────────────────────

@mcp.tool(annotations=READ_ONLY)
def memory_timeline(date: str = "", app_name: str = "",
                    query: str = "", limit: int = 20) -> str:
    """查屏幕时间线。date: YYYY-MM-DD, app_name: 应用名, query: 搜内容。"""
    conn = _conn()
    try:
        filters = []
        params: list = []
        if date:
            filters.append("date(timestamp) = ?")
            params.append(date)
        if app_name:
            filters.append("app_name LIKE ?")
            params.append(f"%{app_name}%")
        if query:
            filters.append("(narrative LIKE ? OR summary LIKE ? OR visible_text LIKE ?)")
            params += [f"%{query}%"] * 3

        where = ""
        if filters:
            where = "WHERE " + " AND ".join(filters)

        try:
            rows = conn.execute(f"""
                SELECT id, timestamp, app_name, window_title,
                       narrative, summary, model
                FROM screen_timeline
                {where}
                ORDER BY timestamp DESC
                LIMIT ?
            """, params + [limit]).fetchall()
        except sqlite3.OperationalError:
            return "屏幕时间线不可用（需要感知引擎采集数据）。P3 正在整理中，敬请期待。"

        lines = [f"最近 {len(rows)} 条屏幕事件:"]
        for r in rows:
            lines.append(f"- [{r['id']}] {r['timestamp']}  {r['app_name']} - {r['window_title']}")
            if r["narrative"]:
                lines.append(f"  {r['narrative'][:200]}")
        return _text(lines)
    finally:
        conn.close()


# ── memory_query ──────────────────────────────────────

_SAFE_SQL_RE = re.compile(r"^\s*SELECT\s", re.IGNORECASE)
_FORBIDDEN = re.compile(r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|ATTACH|DETACH)\b", re.IGNORECASE)


@mcp.tool(annotations=READ_ONLY)
def memory_query(sql: str, limit: int = 50) -> str:
    """自由SQL查询(只读)。表: conversations, conversation_messages, conv_index, conv_segments, conv_relations。
    只允许SELECT。"""
    if not _SAFE_SQL_RE.match(sql):
        return "只允许 SELECT 查询"
    if _FORBIDDEN.search(sql):
        return "禁止修改操作"

    if "LIMIT" not in sql.upper():
        sql = sql.rstrip(";") + f" LIMIT {limit}"

    conn = _conn()
    try:
        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(limit)

        lines = [f"{len(rows)} 行 | 列: {', '.join(columns)}", ""]
        for r in rows:
            parts = []
            for i, col in enumerate(columns):
                val = r[i]
                if isinstance(val, bytes):
                    val = f"<blob {len(val)}B>"
                elif isinstance(val, str) and len(val) > 100:
                    val = val[:100] + "..."
                parts.append(f"{col}={val}")
            lines.append(" | ".join(parts))
        return _text(lines)
    except Exception as e:
        return f"SQL错误: {e}"
    finally:
        conn.close()


# ── 首次启动：自动建库 + 扫描数据 ──────────────────────

def _init_db(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            source_type TEXT NOT NULL,
            name TEXT,
            summary TEXT,
            message_count INTEGER DEFAULT 0,
            total_chars INTEGER DEFAULT 0,
            created_at TEXT,
            updated_at TEXT,
            imported_at TEXT NOT NULL,
            process_status TEXT DEFAULT 'pending',
            metadata TEXT DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_conv_source ON conversations(source_type);

        CREATE TABLE IF NOT EXISTS conversation_messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            sender TEXT NOT NULL,
            text TEXT,
            sequence INTEGER NOT NULL,
            created_at TEXT,
            metadata TEXT DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_msg_conv ON conversation_messages(conversation_id);

        CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
            text,
            content='conversation_messages',
            content_rowid='rowid',
            tokenize='unicode61'
        );
    """)
    conn.commit()


def _rebuild_fts(conn: sqlite3.Connection):
    _log("重建 FTS5 索引...")
    conn.execute("INSERT INTO fts_chunks(fts_chunks) VALUES('rebuild')")
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM fts_chunks").fetchone()[0]
    _log(f"FTS5 索引完成: {count} 条")


def _scan_data_dirs(conn: sqlite3.Connection) -> dict:
    stats = {"claude_code": 0, "workbuddy": 0, "messages": 0}

    sources = {
        "claude_code": {
            "dir": Path.home() / ".claude" / "projects",
            "prefix": "cc_",
        },
        "workbuddy": {
            "dir": Path.home() / ".workbuddy" / "projects",
            "prefix": "wb_",
        },
    }

    for source_name, cfg in sources.items():
        base_dir = cfg["dir"]
        if not base_dir.exists():
            _log(f"{source_name}: 目录不存在，跳过")
            continue

        for project_dir in sorted(base_dir.iterdir()):
            if not project_dir.is_dir():
                continue
            for jsonl_file in sorted(project_dir.glob("*.jsonl")):
                try:
                    messages = _parse_jsonl(jsonl_file, source_name)
                except Exception as e:
                    _log(f"解析失败 {jsonl_file.name}: {e}")
                    continue

                if len(messages) < 2:
                    continue

                conv_id = f"{cfg['prefix']}{jsonl_file.stem}"
                if conn.execute("SELECT 1 FROM conversations WHERE id=?", (conv_id,)).fetchone():
                    continue

                conv_name = messages[0]["text"][:80]
                created = messages[0].get("created_at", "")
                total_chars = sum(len(m["text"]) for m in messages)

                model = ""
                if source_name == "workbuddy":
                    meta_path = jsonl_file.with_suffix(".meta.json")
                    if meta_path.exists():
                        try:
                            meta = json.loads(meta_path.read_text(encoding="utf-8"))
                            model = meta.get("model", "")
                            mc = meta.get("createdAt", "")
                            if mc and not created:
                                created = datetime.fromtimestamp(mc / 1000).isoformat() if isinstance(mc, (int, float)) else str(mc)
                        except Exception:
                            pass

                meta_dict = {"project": project_dir.name}
                if model:
                    meta_dict["model"] = model

                conn.execute(
                    """INSERT OR IGNORE INTO conversations
                       (id, source_type, name, summary, message_count, total_chars,
                        created_at, updated_at, imported_at, process_status, metadata)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (conv_id, source_name, conv_name, "", len(messages), total_chars,
                     created, messages[-1].get("created_at", created),
                     datetime.now().isoformat(), "pending",
                     json.dumps(meta_dict, ensure_ascii=False)),
                )

                for seq, msg in enumerate(messages):
                    conn.execute(
                        """INSERT OR IGNORE INTO conversation_messages
                           (id, conversation_id, sender, text, sequence, created_at)
                           VALUES (?,?,?,?,?,?)""",
                        (f"{conv_id}_{seq}", conv_id, msg["sender"], msg["text"], seq, msg.get("created_at", "")),
                    )

                conn.commit()
                stats[source_name] += 1
                stats["messages"] += len(messages)

    return stats


# ── JSONL 解析器 ──────────────────────────────────────

def _parse_jsonl(path: Path, source: str) -> list[dict]:
    messages = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            if source == "claude_code":
                msg = _parse_cc_line(obj)
            else:
                msg = _parse_wb_line(obj)

            if msg and len(msg["text"].strip()) >= 5:
                messages.append(msg)
    return messages


def _parse_cc_line(obj: dict) -> dict | None:
    if obj.get("type") not in ("user", "assistant"):
        return None
    msg = obj.get("message", {})
    if not msg:
        return None

    content = msg.get("content", "")
    text = _extract_cc_text(content, obj["type"])
    if not text:
        return None

    if obj["type"] == "user" and re.match(r"^<command-name>/\w+</command-name>", text.strip()):
        return None

    ts = msg.get("timestamp", obj.get("timestamp", ""))
    if isinstance(ts, (int, float)):
        ts = datetime.fromtimestamp(ts / 1000).isoformat()

    return {"sender": "human" if obj["type"] == "user" else "assistant", "text": text.strip(), "created_at": ts}


def _extract_cc_text(content, msg_type: str) -> str:
    if isinstance(content, str):
        if msg_type == "user":
            content = re.sub(r"<local-command-caveat>.*?</local-command-caveat>", "", content, flags=re.DOTALL)
            content = re.sub(r"<system-reminder>.*?</system-reminder>", "", content, flags=re.DOTALL)
            m = re.search(r"<user_query>\s*(.*?)\s*</user_query>", content, re.DOTALL)
            if m:
                return m.group(1).strip()
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            t = item.get("type", "")
            if t == "thinking":
                continue
            text = item.get("text", "")
            if text:
                parts.append(text.strip() if msg_type != "user" else _extract_cc_text(text, "user"))
            elif t == "tool_use":
                name = item.get("name", "")
                inp = item.get("input", {})
                if name:
                    parts.append(f"[Tool: {name}] {json.dumps(inp, ensure_ascii=False)[:500]}")
            elif t == "tool_result":
                inner = item.get("content", "")
                if isinstance(inner, str) and inner:
                    parts.append(f"[Tool Result] {inner[:500]}")
                elif isinstance(inner, list):
                    for sub in inner:
                        if isinstance(sub, dict) and sub.get("text"):
                            parts.append(f"[Tool Result] {sub['text'][:500]}")
        return "\n".join(p for p in parts if p)
    return ""


def _parse_wb_line(obj: dict) -> dict | None:
    if obj.get("type") != "message":
        return None
    role = obj.get("role", "")
    if role not in ("user", "assistant"):
        return None

    content = obj.get("content", [])
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        parts = [item.get("text", "").strip() for item in content if isinstance(item, dict) and item.get("text")]
        text = "\n".join(parts)
    else:
        return None

    if not text:
        return None

    ts = obj.get("timestamp", "")
    if isinstance(ts, (int, float)):
        ts = datetime.fromtimestamp(ts / 1000).isoformat()

    return {"sender": "human" if role == "user" else "assistant", "text": text, "created_at": ts}


# ── 入口 ──────────────────────────────────────────────

def _ensure_db():
    """确保数据库存在且有数据"""
    db_path = Path(DB_PATH)

    if not db_path.exists():
        _log("首次启动：创建数据库 + 扫描数据目录...")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=wal")
        _init_db(conn)
        stats = _scan_data_dirs(conn)
        _rebuild_fts(conn)
        conn.close()
        _log(f"完成: CC={stats['claude_code']} WB={stats['workbuddy']} 消息={stats['messages']}")
    else:
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
        has_fts = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fts_chunks'").fetchone()
        conn.close()
        if count == 0:
            _log("数据库为空，扫描数据目录...")
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA journal_mode=wal")
            _init_db(conn)
            stats = _scan_data_dirs(conn)
            _rebuild_fts(conn)
            conn.close()
            _log(f"完成: CC={stats['claude_code']} WB={stats['workbuddy']} 消息={stats['messages']}")
        elif not has_fts:
            _log("FTS 索引缺失，重建...")
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA journal_mode=wal")
            _init_db(conn)
            _rebuild_fts(conn)
            conn.close()


def main():
    _ensure_db()
    _detect_active_session()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
