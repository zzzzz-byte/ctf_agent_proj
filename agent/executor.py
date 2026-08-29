"""执行 Agent:接口桩 + MockExecutor + RealExecutor(② 真实执行)。

- Executor:契约接口(③ 只调用,不实现)。
- MockExecutor:可配置返回内容的执行 mock(测试/冒烟)。
- RealExecutor:LLM 工具循环 → CommandRunner 路由跑命令 → ExecResult。内部 ReAct
  (chat_with_tools 工具循环)由执行层自管,引擎不感知;tool_exec 复合引擎注入的
  (apply_tool/get_doc/get_record)与执行工具(run_command/run_python/submit_flag/answer)。
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from opslog import ErrorLevel, emit, record_error

from agent.tools import APPLY_TOOL_SPEC, GET_DOC_SPEC, REMOVE_TOOL_SPEC
from agent.llm_api import ToolLoopError


@dataclass
class ExecResult:
    observation: str            # 执行观察(喂给步骤校验)
    result: dict | None = None  # 该步产物(可选,写 dag.step.result 供 ee)
    tool_calls: list[dict] | None = None  # 工具调用轨迹 [{tool, args, result}](喂 trace 通道)
    total_usage: dict | None = None  # {prompt_tokens, completion_tokens, total_tokens}


@dataclass
class ExecState:
    """执行状态上下文:engine 组装"本轮为何重跑/首次"的结构化事实,executor 渲染进系统提示词。

    status: first(首次) / retry_incomplete / retry_drift / retry_other(上轮 ee 判定对应)。
    verdict/diagnosis 仅 retry 有,取自 ee 的 EvalResult。
    """
    status: str
    attempts: int
    max_attempts: int
    verdict: str | None = None
    diagnosis: str | None = None


# 状态注入提示词:镜像 planner.TRIGGER_NOTES,但给的是硬指令(要执行遵守),非只陈述。
EXEC_STATE_NOTES = {
    "first": (
        "本步骤首次执行。务必在工具预算内产出结论性交付物(flag/答案/可检验清单),"
        "优先做针对性验证,不要宽泛枚举式侦察(grep/strings 全家桶烧预算)。"
        "逆向默认用 ghidra;objdump 已禁用。"
    ),
    "retry_incomplete": (
        "上一步执行被评估判定为进度不足(incomplete),未达成验收标准。本次必须:"
        "① 必须先产出结论(flag/答案/清单),② 只做增量验证,禁止重复已执行过的全量命令,"
        "③ 必须遵守下方【本轮评估意见】列出的未达成理由,逐条回应。"
    ),
    "retry_drift": (
        "上一步执行方向偏了(drift),解题路径偏离目标。本次必须回到 criterion 重新解读"
        "该步骤到底要什么,不要延续错误路径;下方【本轮评估意见】会指出偏差点,逐条纠正。"
    ),
    "retry_other": (
        "上一步执行未通过评估。先说明上轮为何失败、本次将如何不同,再行动;"
        "遵守下方【本轮评估意见】。"
    ),
}


def render_exec_state(sc) -> str:
    """把执行状态渲染成系统提示词段落(状态解释 + 尝试进度 + ee 判定)。"""
    parts = ["# 执行状态"]
    note = EXEC_STATE_NOTES.get(sc.status)
    if note:
        parts.append(note)
    parts.append(f"# 当前尝试: {sc.attempts}/{sc.max_attempts}")
    if sc.verdict and sc.diagnosis:
        parts.append(f"# 上一步 ee 判定: {sc.verdict} / {sc.diagnosis}")
    if sc.attempts >= sc.max_attempts:
        parts.append("# 最后一次尝试: 本次失败该步骤将升级,不再重试,必须一次达成。")
    return "\n\n".join(parts)


# objdump 已禁用:全量反汇编反复烧预算,且容器内已有 ghidra 可做逆向。命令边界硬拦——
# 即便 LLM 直接输 objdump 也被拦;readelf/nm/objcopy 等其它 binutils 不受影响。
# 前/后视界避免误伤含 objdump 的文件名(如 cat objdump.txt)。
OBJDUMP_BLOCK_RE = re.compile(r"(?<![\w.])objdump(?=\s|[/;|&]|$)", re.I)
OBJDUMP_DISABLED_MSG = (
    "objdump 已禁用(命令被拦截):逆向改用 ghidra(容器已装)/radare2;"
    "符号/节/段信息用 nm / readelf / objcopy。"
)


def block_objdump(cmd: str) -> str | None:
    """objdump 工具禁用检查:命中返回拦截原因,否则 None。"""
    if OBJDUMP_BLOCK_RE.search(cmd):
        return OBJDUMP_DISABLED_MSG
    return None


class Executor:
    # 系统提示词(经 engine 传入 SystemPromptComponent 渲染;mock 为空)
    system: str = ""

    async def run(self, step, ctx: str, tool_exec=None, runner=None) -> ExecResult:
        raise NotImplementedError

    def system_for(self, state_context=None) -> str:
        """状态化系统提示词:base(self.system)+ 引擎注入的执行状态上下文(镜像 planner)。

        base 为空(如 Mock)或没有状态上下文 → 原样返回,行为零变化。
        """
        base = self.system
        if not base or not state_context:
            return base
        return base + "\n\n" + render_exec_state(state_context)

    def match_experience(self) -> list[dict]:
        """当前挑战精确匹配到的已验证解题经验(engine _init_run 装填 ws.experience);
        mock/无适配器返回空。"""
        return []


class MockExecutor(Executor):
    """可配置返回内容的执行 mock。
    传 fn 则用 callable(step, ctx, tool_exec=None) -> ExecResult;否则固定返回 observation/result/tool_calls。"""

    def __init__(self, observation: str = "", result: dict | None = None,
                 tool_calls: list[dict] | None = None, fn=None):
        self._observation = observation
        self._result = result
        self._tool_calls = tool_calls
        self._fn = fn

    async def run(self, step, ctx: str, tool_exec=None, runner=None) -> ExecResult:
        if self._fn is not None:
            try:
                r = self._fn(step, ctx, tool_exec)
                return await r if asyncio.iscoroutine(r) else r
            except TypeError:
                r = self._fn(step, ctx)  # 兼容旧 2 参 fn(step, ctx)
                return await r if asyncio.iscoroutine(r) else r
        return ExecResult(observation=self._observation, result=self._result,
                          tool_calls=self._tool_calls)


# ===== RealExecutor(② 真实执行) =====

EXEC_SYSTEM = (
    "你是 CTF 解题执行 Agent。根据当前步骤的指令与验收标准，实际执行命令/代码推进解题。\n"
    "\n"
    "【可用的上下文】\n"
    "- 任务：题面原文与目标列表（Task）\n"
    "- 当前步骤：instruction / criterion / status / attempts / skill_id（Dag）\n"
    "- 工具目录：可申请的完整清单（apply_tool 申请后进活动工具集）\n"
    "- 活动工具：已申请可用的工具\n"
    "- 技能文档：检索命中的参考文档索引（经 get_doc 按 id 取全文）\n"
    "- 本轮评估意见：前置评估 Agent 对之前步骤/计划的非 pass 意见（避免重犯）\n"
    "- 工具轨迹：本角色本轮已执行的工具调用记录\n"
    "\n"
    "【执行准则】\n"
    "- 用 run_command 跑 shell,run_python 跑 Python（命令只在沙箱 SSH 容器内执行）\n"
    "- 逆向用 ghidra（容器已装）/radare2；objdump 命令已被禁用，会被拦截\n"
    "- 需要技能文档或工具时，先 get_doc / apply_tool\n"
    "- 逐步对齐 criterion；拿 flag 用 submit_flag 提交，无 flag 的结论用 answer 给出\n"
    "- 命令失败时根据错误信息调整，不盲目重试同一命令"
)

RUN_COMMAND_SPEC = {
    "type": "function",
    "function": {
        "name": "run_command",
        "description": (
            "在沙箱(SSH 容器)内执行一条 shell 命令(含参数)。自动安装缺失依赖;"
            "工作目录缺省为题目附件目录,只能在该目录内操作。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "命令,如 'gdb -q ./pwn1'"},
                "tool_id": {"type": "string",
                            "description": "涉及的工具清单 id(如 'gdb'),用于沙箱自动安装"},
                "cwd": {"type": "string", "description": "工作目录(缺省用题目附件目录)"},
                "timeout": {"type": "number", "description": "超时秒数(缺省 120)"},
            },
            "required": ["command"],
        },
    },
}

RUN_PYTHON_SPEC = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": (
            "在沙箱(SSH 容器)内执行一段 Python 代码(写临时脚本后运行)。"
            "涉及 pip 工具(如 pwntools/angr)时填 tool_id:沙箱自动安装缺失依赖。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python 源码"},
                "tool_id": {"type": "string", "description": "依赖的 pip 工具清单 id"},
                "cwd": {"type": "string", "description": "工作目录(缺省用题目附件目录)"},
                "timeout": {"type": "number", "description": "超时秒数(缺省 120)"},
            },
            "required": ["code"],
        },
    },
}

SUBMIT_FLAG_SPEC = {
    "type": "function",
    "function": {
        "name": "submit_flag",
        "description": (
            "最终解出的 flag 提交(flag{} 格式)。动态 flag 题(靶机容器题,flag 随实例变化):"
            "若 flag 是经脚本/盲注推导出来的,附 provenance={verifier: 提取脚本相对路径, "
            "trace: 提取过程摘要(如逐字符断言), flag_format: 匹配的格式},平台确认后该过程"
            "会登记为已验证,后续实例可重跑推导本地判定。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "flag": {"type": "string", "description": "flag 内容"},
                "provenance": {
                    "type": "object",
                    "description": "动态 flag 题的提取来源(EE 把关:来源可确认而非胡编)",
                    "properties": {
                        "verifier": {"type": "string", "description": "提取脚本相对 challenge 目录的路径"},
                        "trace": {"type": "string", "description": "提取过程摘要(断言序列/脚本输出)"},
                        "flag_format": {"type": "string", "description": "匹配的 flag 格式,如 CTF2{...}"},
                    },
                    "required": [],
                },
            },
            "required": ["flag"],
        },
    },
}

ANSWER_SPEC = {
    "type": "function",
    "function": {
        "name": "answer",
        "description": "给出最终结论(非 flag 类答案,如取证结论/推导结果)。",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "结论文本"}},
            "required": ["text"],
        },
    },
}

GET_RECORD_SPEC = {
    "type": "function",
    "function": {
        "name": "get_record",
        "description": "按 uuid 取历史事件全文(展开索引投影)。",
        "parameters": {
            "type": "object",
            "properties": {"uuid": {"type": "string", "description": "历史事件的 uuid"}},
            "required": ["uuid"],
        },
    },
}

EXEC_TOOL_SPECS: list[dict] = [
    RUN_COMMAND_SPEC, RUN_PYTHON_SPEC, SUBMIT_FLAG_SPEC, ANSWER_SPEC,
    GET_DOC_SPEC, APPLY_TOOL_SPEC, REMOVE_TOOL_SPEC, GET_RECORD_SPEC,
]


def _parse_args(arguments) -> dict:
    if isinstance(arguments, dict):
        return arguments
    try:
        return json.loads(arguments or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}


def _normalize_trace(src) -> tuple[list[dict], dict]:
    """归一工具轨迹 → (tool_calls, result)。

    src 可为 ToolResult-like(带 .trace)或原始 trace 列表(异常抛出时)。
    result 从 submit_flag/answer 提取,喂 engine 的 flag 提取与步骤产物。
    """
    raw = getattr(src, "trace", None)
    if raw is None:
        raw = src if isinstance(src, list) else []
    trace = []
    for t in raw:
        if not (isinstance(t, dict) and t.get("name")):
            continue
        entry = {"tool": t.get("name"), "args": _parse_args(t.get("arguments")),
                 "result": t.get("result")}
        if t.get("round") is not None:
            entry["round"] = t["round"]  # 工具调用轮次(事件编码字段,可缺省)
        trace.append(entry)
    result: dict = {}
    for t in trace:
        if t["tool"] == "submit_flag" and t["args"].get("flag"):
            result["flag"] = t["args"]["flag"]
        elif t["tool"] == "answer" and t["args"].get("text"):
            result["answer"] = t["args"]["text"]
    return trace, result


class RealExecutor(Executor):
    """真实执行 Agent:LLM 工具循环 + 命令路由。

    - `llm_fn`: (system=, prompt=, tools=, tool_exec=) -> ToolResult-like(含 trace/
      rounds/content/total_usage);缺省走 llm_api.chat_with_tools(executor 角色模型)。
    - `runner`: CommandRunner(沙箱唯一执行);缺省真实构造。
    - 工具调用轨迹归一为契约 shape {tool, args, result},喂 engine trace 通道。
    """

    system = EXEC_SYSTEM

    def __init__(self, llm_fn=None, runner=None, workspace=None,
                 max_tool_rounds=None, model=None, workdir=None, adapter=None):
        from agent import llm_api
        from agent.runner import CommandRunner
        from model_config import get_engine_config

        self.runner = runner or CommandRunner()
        self.workspace = workspace
        if max_tool_rounds is None:
            max_tool_rounds = get_engine_config().get("max_tool_rounds", 8)
        self.max_tool_rounds = max_tool_rounds
        self.model = model or llm_api.role_model("executor")
        self.workdir = workdir
        self.adapter = adapter  # 平台 ChallengeAdapter(None=提交仅记录)
        self._llm = llm_fn or self._default_llm()
        self._target_resolved = False  # 目标地址惰性解析,成功即缓存(避免重复开靶)
        self._target_cache: str | None = None
        self._target_info: dict | None = None  # 完整连接信息(host/port/access_type/nc_ssl/...),供提示词渲染
        self._target_failed_at: float | None = None  # 上次解析失败时间戳(退避重试依据)
        self._target_retry_delay: float = 60.0  # 失败后至少隔多久重试一次 start_target
        if self.adapter is not None:
            # 动态 flag 本地判定钩子:adapter._local_verify 重跑已验证脚本推导当前实例 flag
            # (set_procedure_runner 可选的注入缝;测试 fake adapter 无此方法则跳过)
            setter = getattr(self.adapter, "set_procedure_runner", None)
            if setter is not None:
                setter(self._run_verifier)

    def _default_llm(self):
        from agent import llm_api

        async def call(*, system, prompt, tools, tool_exec, **kw):
            return await llm_api.chat_with_tools(
                system=system, prompt=prompt, tools=tools, tool_exec=tool_exec,
                model=self.model, max_tool_rounds=self.max_tool_rounds, **kw)

        return call

    # ===== 上下文 =====

    def _category(self, step) -> str:
        if step is not None and step.skill_id:
            cat = step.skill_id.split(".")[0]
            if cat:
                return cat
        if self.workspace is not None:
            task = getattr(self.workspace, "meta", {}).get("task") or {}
            return task.get("challenge_type") or ""
        return ""

    def _allowed_cwd(self) -> Path | None:
        """允许的工作目录根:workdir 参数优先,其次 workspace meta task 的 challenge_dir/workdir/cwd。"""
        if self.workdir:
            return Path(self.workdir).resolve()
        task = (getattr(self.workspace, "meta", {}).get("task") or {}) if self.workspace else {}
        for k in ("challenge_dir", "workdir", "cwd"):
            v = task.get(k)
            if v:
                return Path(str(v)).resolve()
        return None

    @property
    def allowed_cwd(self) -> str | None:
        """执行工作目录根(题目附件目录);None=无法确定。scheduler 开会话时用。"""
        p = self._allowed_cwd()
        return str(p) if p else None

    def set_sandbox(self, handle) -> None:
        """用外部受限句柄(SandboxHandle)接管沙箱执行面,替换自建 CommandRunner。

        引擎接线用:scheduler.acquire 拿到的会话容器 handle 注入这里,后续 run/run_python
        全经该会话(工具/文件状态横跨步骤持久);执行器只依赖 handle,不直接构造沙箱。
        """
        from agent.runner import CommandRunner

        self.runner = CommandRunner(sandbox=handle, timeout=self.runner.timeout,
                                    max_out=self.runner.max_out, max_err=self.runner.max_err)

    def _cwd(self, args: dict) -> str:
        """解析执行工作目录:默认题目附件目录;args.cwd 必须在允许根内,否则拒绝。

        命令只经沙箱执行,沙箱会同步 cwd 目录到远程——因此 cwd 必须锁定在题目
        附件目录,防 LLM 把敏感宿主目录 sync 进沙箱。
        """
        allowed = self._allowed_cwd()
        if allowed is None:
            raise ValueError("无法确定题目附件目录(workdir/meta 均无 challenge_dir)")
        cwd = args.get("cwd")
        if not cwd:
            return str(allowed)
        cand = Path(str(cwd)).resolve()
        if cand != allowed and not cand.is_relative_to(allowed):
            raise ValueError(f"cwd 必须在题目附件目录内: {allowed}")
        return str(cand)

    def _challenge_id(self) -> str | None:
        """从工作区任务元数据解析平台 challenge_id;本地/无 id 返回 None。"""
        if self.workspace is not None:
            task = getattr(self.workspace, "meta", {}).get("task") or {}
            for k in ("challenge_id", "id", "friendly_id"):
                v = task.get(k)
                if v:
                    return str(v)
        return None

    def _submit_flag(self, args: dict) -> dict:
        """提交回环:有平台适配器且解析出 challenge_id → 真调 adapter.submit 返回判定;
        否则仅记录(历史行为),失败降级为可观察的 ok=False 供 LLM 决策重试。
        动态 flag 题的 provenance(verifier/trace)透传给适配器,平台确认后登记为已验证。"""
        flag = args.get("flag", "")
        if self.adapter is None:
            return {"submitted": True, "flag": flag}  # 无适配器:仅记录
        cid = self._challenge_id()
        if not cid:
            return {"submitted": True, "flag": flag, "ok": None, "correct": None,
                    "message": "缺少 challenge_id,仅记录提交"}
        provenance = args.get("provenance") or None
        try:
            if provenance:
                res = self.adapter.submit(cid, flag, provenance=provenance)
            else:
                res = self.adapter.submit(cid, flag)
        except Exception as exc:
            return {"submitted": True, "flag": flag, "ok": False, "correct": None,
                    "message": f"提交异常: {type(exc).__name__}: {exc}"}
        return {"submitted": True, "flag": flag, "ok": res.ok, "correct": res.correct,
                "message": res.message}

    async def _run_verifier(self, verifier_path: str, target: str | None) -> str | None:
        """重跑已验证提取脚本(沙箱),推导当前实例 flag(stdout 末行);失败返回 None。

        脚本约定:相对 challenge 目录,读 argv[1](或 metadata.yml)取靶机地址,
        推导结果打印在 stdout 末行。用于 adapter._local_verify 的动态题本地判定。
        """
        try:
            cwd = Path(self._cwd({})).resolve()
        except ValueError:
            return None
        try:
            script = (cwd / verifier_path).resolve()
            rel = script.relative_to(cwd)
        except (ValueError, OSError):
            return None
        if not script.is_file():
            return None
        target_arg = target or ""
        code = (
            "import runpy, sys\n"
            f"script = {str(rel)!r}\n"
            f"sys.argv = [script, {target_arg!r}] if {target_arg!r} else [script]\n"
            "try:\n"
            # run_name='__main__' 让脚本的 `if __name__=='__main__'` 守卫生效
            "    runpy.run_path(script, run_name='__main__')\n"
            "except SystemExit:\n"
            "    pass\n"
        )
        out = await self.runner.run_python(code, cwd=cwd, timeout=90)
        lines = [ln.strip() for ln in (out.stdout or "").splitlines() if ln.strip()]
        return lines[-1] if lines else None

    def match_experience(self) -> list[dict]:
        """当前挑战精确匹配到的已验证解题经验(经适配器);无适配器/无 id → 空。"""
        if self.adapter is None:
            return []
        cid = self._challenge_id()
        if not cid:
            return []
        try:
            return self.adapter.match_procedures(cid) or []
        except Exception as exc:
            record_error("executor", "experience_match", exc=exc,
                         level=ErrorLevel.RECOVERABLE, challenge_id=cid)
            return []

    # ===== 目标地址(靶机) =====

    def _target(self) -> str | None:
        """解析靶机地址(惰性,成功才缓存)。

        顺序:task.target_info(理解层已结构化)→ task.target(host:port 串)→
        含容器题 + 有适配器 → 惰性 start_target 开靶(缓存 host:port)。无则 None。
        失败(如平台 429 限流)不置 resolved,按 _target_retry_delay 退避,后续步骤
        自动重试——瞬态错误恢复后不再永久锁死整个 run 的目标解析。
        """
        if self._target_resolved:
            return self._target_cache
        now = time.time()
        if (self._target_failed_at is not None
                and now - self._target_failed_at < self._target_retry_delay):
            return None
        task = (getattr(self.workspace, "meta", {}).get("task") or {}) if self.workspace else {}
        info = task.get("target_info")
        if isinstance(info, dict):
            self._target_info = info
            self._target_cache = self._fmt_target(info) or None
            self._target_resolved = True
            return self._target_cache
        raw = task.get("target")
        if isinstance(raw, str) and raw.strip():
            self._target_info = None
            self._target_cache = raw.strip()
            self._target_resolved = True
            return self._target_cache
        cid = self._challenge_id()
        if cid and task.get("has_container") and self.adapter is not None:
            self._target_failed_at = now
            try:
                r = self.adapter.start_target(cid)
            except Exception as exc:
                emit("executor", "target_resolve_failed", challenge_id=cid,
                     error=f"{type(exc).__name__}: {exc}")
                return None
            host, port = r.get("host") or "", r.get("port")
            if host and port:
                self._target_info = self._access_info(r, host=host, port=port)
                self._target_cache = self._fmt_target(self._target_info) or f"{host}:{port}"
                self._target_resolved = True
                self._target_failed_at = None
                return self._target_cache
        if task.get("has_container"):
            # 容器题但无目标可注入(无适配器 / start_target 未返回 host:port)→ 记失败待重试
            self._target_failed_at = now
        return self._target_cache

    def target_blocked(self) -> bool:
        """容器题且靶机不可达(曾尝试解析失败、当前仍无目标)→ True。引擎用于环境阻塞收口。"""
        task = (getattr(self.workspace, "meta", {}).get("task") or {}) if self.workspace else {}
        if not task.get("has_container"):
            return False
        if self._target_resolved:
            return not bool(self._target_cache)
        return self._target_failed_at is not None

    def retry_target(self) -> str | None:
        """供引擎在派发步骤前复查靶机:按退避再尝试一次解析,返回当前目标(可 None)。"""
        return self._target()

    @staticmethod
    def _fmt_target(info: dict) -> str:
        if info.get("kind") == "url":
            return info.get("url") or f"{info.get('scheme')}://{info.get('host')}"
        host = info.get("host")
        if not host:
            return ""
        if info.get("port") is not None:
            return f"{host}:{info['port']}"
        return str(host)

    @staticmethod
    def _access_info(r: dict, *, host: str = "", port=None) -> dict:
        """从 start_target 响应提取 LLM 需要的连接信息(镜像理解层 target_info 语义)。

        只挑语义字段:地址 + 协议(access_type)+ TLS 包裹(nc_ssl)+ 完整 URL;生命周期
        噪音(environment_id/status/expires_at/raw)不进,避免提示词被无关信息污染。
        """
        info: dict = {"kind": "host_port", "source": "start_target"}
        if host:
            info["host"] = host
        if port is not None:
            try:
                info["port"] = int(port)
            except (TypeError, ValueError):
                info["port"] = port
        for k in ("access_url", "access_urls", "access_type", "nc_ssl"):
            if r.get(k) is not None:
                info[k] = r[k]
        return info

    @staticmethod
    def _render_target_hints(info: dict | None) -> str:
        """按连接信息渲染给 LLM 的访问提示(nc_ssl 必须 TLS、http 用 curl)。"""
        if not isinstance(info, dict):
            return ""
        lines = []
        if info.get("nc_ssl"):
            lines.append(
                "端口被平台 TLS 转发器包裹(nc_ssl=true):必须用 TLS/SSL 连接"
                "(SNI + 关闭证书校验),裸 TCP 只会收到 0 字节。"
            )
        if info.get("access_type") == "http":
            url = info.get("access_url") or ""
            lines.append(
                "协议为 http:用 curl/requests 发 HTTP 请求解题,不是 nc 裸连接。"
                + (f"完整 URL: {url}" if url else "")
            )
        return "\n".join(lines)

    def _build_prompt(self, step, ctx: str) -> str:
        parts = []
        if step is not None:
            head = f"# 当前步骤\nid={step.id}\n指令: {step.instruction}"
            if step.criterion:
                head += f"\n验收标准: {step.criterion}"
            if step.depends_on:
                head += f"\n依赖步骤: {list(step.depends_on)}"
            parts.append(head)
        target = self._target()
        if target:
            parts.append(
                "# 目标地址(靶机)\n"
                f"{target}\n"
                "用 nc/pwntools 连接该地址解题(容器题目标为动态分配,只此一次有效)。"
            )
            hints = self._render_target_hints(self._target_info)
            if hints:
                parts.append("# 连接注意\n" + hints)
        if ctx:
            parts.append("# 上下文\n" + ctx)
        return "\n\n".join(parts)

    # ===== 执行 =====

    async def run(self, step, ctx: str, tool_exec=None, runner=None,
                  system=None) -> ExecResult:
        """执行一个步骤。runner 可选:并行 wave 每步注入独立 CommandRunner(各持各的
        容器租约);缺省用 self.runner(串行会话 runner,引擎已注入 handle)。
        system 可选:引擎按执行状态注入组合后的系统提示词;缺省用 EXEC_SYSTEM。"""
        category = self._category(step)

        async def exec_tool(name: str, args: dict):
            if name == "run_command":
                return await self._run_command(args, category, runner)
            if name == "run_python":
                return await self._run_python(args, category, runner)
            if name == "submit_flag":
                return self._submit_flag(args)
            if name == "answer":
                return {"answer": args.get("text", "")}
            if tool_exec is not None:
                r = tool_exec(name, args)
                return await r if asyncio.iscoroutine(r) else r
            return {"error": f"unknown tool: {name}"}

        prompt = self._build_prompt(step, ctx)
        try:
            tr = await self._llm(system=system or EXEC_SYSTEM, prompt=prompt,
                                 tools=EXEC_TOOL_SPECS, tool_exec=exec_tool)
        except ToolLoopError as exc:
            # 工具循环超上限:保留已达成的部分轨迹(喂 run.log/events.jsonl/flag 提取)
            trace, result = _normalize_trace(exc.trace)
            return ExecResult(
                observation=f"执行 Agent 工具循环超上限({self.max_tool_rounds} 轮),"
                            f"已执行 {len(trace)} 次工具调用",
                result=result or None,
                tool_calls=trace or None,
            )
        except Exception as exc:
            return ExecResult(
                observation=f"执行 Agent LLM 循环异常: {type(exc).__name__}: {exc}")

        trace, result = _normalize_trace(tr)
        return ExecResult(
            observation=self._summarize(tr, trace),
            result=result or None,
            tool_calls=trace or None,
            total_usage=getattr(tr, "total_usage", None),
        )

    async def _run_command(self, args: dict, category: str, runner=None) -> dict:
        cmd = args.get("command")
        if not cmd or not str(cmd).strip():
            return {"error": "run_command 需要 command"}
        err = block_objdump(str(cmd))
        if err:
            return {"error": err}
        try:
            cwd = self._cwd(args)
        except ValueError as exc:
            return {"error": str(exc)}
        out = await (runner or self.runner).run(
            str(cmd),
            cwd=cwd,
            category=category,
            tool_id=args.get("tool_id"),
            timeout=args.get("timeout"),
        )
        return out.as_dict()

    async def _run_python(self, args: dict, category: str, runner=None) -> dict:
        code = args.get("code")
        if not code:
            return {"error": "run_python 需要 code"}
        try:
            cwd = self._cwd(args)
        except ValueError as exc:
            return {"error": str(exc)}
        out = await (runner or self.runner).run_python(
            code,
            cwd=cwd,
            category=category,
            tool_id=args.get("tool_id"),
            timeout=args.get("timeout"),
        )
        return out.as_dict()

    @staticmethod
    def _summarize(tr, trace: list[dict]) -> str:
        rounds = getattr(tr, "rounds", 0)
        line = f"执行 {rounds} 轮工具调用"
        if trace:
            line += f": " + ", ".join(t["tool"] for t in trace)
        tail = getattr(tr, "content", None)
        if tail and str(tail).strip():
            line += f"; 结论: {str(tail).strip()[:200]}"
        return line
