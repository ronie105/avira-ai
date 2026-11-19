"""
╔═════════════════════════════════════════════════════════╗
║         AVIRA — Autonomous Terminal AI Agent        ║
║   ReAct Loop · Tools · Memory · Learn · Safe · Plan    ║
╚═════════════════════════════════════════════════════════╝

Architecture: ReAct (Reason → Act → Observe → Repeat)
Avira thinks, picks tools, runs them, observes results,
and loops until she has a final answer — all autonomously.
"""

import os, re, json, time
from datetime import datetime
from pathlib import Path
from sympy import sympify, SympifyError
from duckduckgo_search import DDGS
from groq import Groq
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich import box
from rich.prompt import Prompt
from rich.text import Text
from rich.rule import Rule

# ════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL        = "llama-3.3-70b-versatile"
MAX_STEPS    = 6          # max ReAct iterations per query
MAX_HISTORY  = 30         # rolling conversation window
MAX_SEARCH   = 5          # DuckDuckGo results per search
MEMORY_FILE  = Path("avira_memory.json")

console = Console()

# ════════════════════════════════════════════════════════
#  SAFETY
# ════════════════════════════════════════════════════════
BLOCKED = [
    r"\b(bomb|explosiv|make\s+weapon|build\s+gun|terrorist|terrorism|jihad|isis|al.?qaeda)\b",
    r"\b(how\s+to\s+(kill|murder|attack)\s+people|shoot\s+up|mass\s+shooting)\b",
    r"\b(child\s+porn|csam|cp\s+link|rape\s+video)\b",
    r"\b(make\s+meth|drug\s+synthesis|synthesise\s+(heroin|cocaine|fentanyl))\b",
    r"\b(hack\s+into|ransomware\s+code|malware\s+script)\b",
    r"\b(suicide\s+method|how\s+to\s+hang|how\s+to\s+overdose)\b",
]
def is_blocked(t: str) -> bool:
    return any(re.search(p, t, re.IGNORECASE) for p in BLOCKED)

# ════════════════════════════════════════════════════════
#  PERSISTENT MEMORY
# ════════════════════════════════════════════════════════
class Memory:
    def __init__(self, path: Path):
        self.path = path
        self.facts: list[dict] = []
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self.facts = json.loads(self.path.read_text())
            except Exception:
                self.facts = []

    def _save(self):
        self.path.write_text(json.dumps(self.facts, indent=2))

    def add(self, fact: str, source: str = "user") -> bool:
        fact = fact.strip().rstrip(".")
        if any(f["fact"].lower() == fact.lower() for f in self.facts):
            return False
        self.facts.append({
            "fact": fact,
            "source": source,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
        })
        self._save()
        return True

    def remove(self, idx: int) -> bool:
        if 0 <= idx < len(self.facts):
            self.facts.pop(idx)
            self._save()
            return True
        return False

    def clear(self):
        self.facts = []
        self._save()

    def as_context(self) -> str:
        if not self.facts:
            return ""
        lines = ["[LONG-TERM MEMORY — facts learned from user, treat as trusted:]"]
        for i, f in enumerate(self.facts):
            lines.append(f"  M{i+1}. {f['fact']}  (source:{f['source']}, when:{f['timestamp']})")
        return "\n".join(lines)

# ════════════════════════════════════════════════════════
#  TOOLS  — each returns a plain string observation
# ════════════════════════════════════════════════════════

def tool_search(query: str) -> str:
    """Search DuckDuckGo and return formatted results."""
    try:
        with DDGS() as d:
            results = list(d.text(query.strip(), max_results=MAX_SEARCH))
        if not results:
            return "No results found."
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r.get('title','')}\n    {r.get('body','')}\n    URL: {r.get('href','')}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"Search failed: {e}"

def tool_calculate(expr: str) -> str:
    """Evaluate a mathematical expression using sympy."""
    clean = re.sub(r"[^0-9\+\-\*\/\^\(\)\.\s\%a-zA-Z_,]", "", expr)
    clean = clean.replace("^", "**").replace("%", "/100")
    try:
        result  = sympify(clean)
        numeric = float(result.evalf())
        if abs(numeric - round(numeric)) < 1e-9:
            return f"{clean} = {int(round(numeric)):,}"
        return f"{clean} = {numeric:,.8g}"
    except Exception as e:
        return f"Calculation error: {e}"

def tool_remember(fact: str, memory: Memory) -> str:
    """Store a fact into long-term memory."""
    added = memory.add(fact, source="agent-learned")
    if added:
        return f"Stored in memory: \"{fact}\" (total facts: {len(memory.facts)})"
    return f"Already in memory: \"{fact}\""

def tool_recall(query: str, memory: Memory) -> str:
    """Search long-term memory for relevant facts."""
    q = query.lower()
    hits = [f for f in memory.facts if q in f["fact"].lower()]
    if not hits:
        return f"No memory entries matching '{query}'."
    return "\n".join(f"- {h['fact']} (from {h['source']})" for h in hits)

def tool_analyse(data: str) -> str:
    """Pass raw data/text back as observation for LLM to synthesise."""
    return f"[Data to analyse]\n{data}"

TOOL_REGISTRY = {
    "search":    ("Search the web for current info, news, or facts",      tool_search),
    "calculate": ("Evaluate a math expression or conversion",              tool_calculate),
    "remember":  ("Save an important fact to long-term memory",            None),   # needs memory
    "recall":    ("Search long-term memory for facts about the user/topic",None),   # needs memory
    "analyse":   ("Analyse or summarise a block of text/data",             tool_analyse),
    "finish":    ("Deliver the final answer to the user",                  None),   # special
}

def run_tool(name: str, arg: str, memory: Memory) -> str:
    if name == "search":
        return tool_search(arg)
    if name == "calculate":
        return tool_calculate(arg)
    if name == "remember":
        return tool_remember(arg, memory)
    if name == "recall":
        return tool_recall(arg, memory)
    if name == "analyse":
        return tool_analyse(arg)
    return f"Unknown tool: {name}"

# ════════════════════════════════════════════════════════
#  REACT SYSTEM PROMPT
# ════════════════════════════════════════════════════════
def react_system(memory: Memory) -> str:
    mem_ctx = memory.as_context()
    today   = datetime.now().strftime("%A, %d %B %Y")
    tools_desc = "\n".join(
        f"  - {name}: {desc}" for name, (desc, _) in TOOL_REGISTRY.items()
    )
    return f"""You are Avira, an autonomous AI agent running in a terminal.
Today: {today}

You operate in a strict ReAct loop:
  Thought → Action → Observation → Thought → ... → Finish

TOOLS AVAILABLE:
{tools_desc}

OUTPUT FORMAT (strictly follow this — no deviation):
For every step output EXACTLY one of:

Thought: <your reasoning about what to do next>
Action: <tool_name>
ActionInput: <input to the tool>

OR when you have enough information:

Thought: <final reasoning>
Action: finish
ActionInput: <your complete, well-formatted final answer to the user>

RULES:
- ALWAYS start with a Thought.
- NEVER output a final answer outside of Action: finish / ActionInput.
- You may use multiple tools across multiple steps.
- For multi-part questions, plan and break them into steps.
- Use 'search' for anything about current events, people, places, news.
- Use 'calculate' for any math, percentages, conversions.
- Use 'remember' to save any important fact the user tells you.
- Use 'recall' to check memory before answering personal questions.
- Use 'analyse' to summarise or compare data you already have.
- Format your final answer in clean markdown.
- Be concise but complete. Never make up facts — search if unsure.

SAFETY — NEVER violate:
- No help with terrorism, violence, weapons, illegal activities.
- No hate speech, adult content, or harmful instructions.
- Refuse clearly with a polite explanation if a request is harmful.

{mem_ctx}
"""

# ════════════════════════════════════════════════════════
#  REACT PARSER
# ════════════════════════════════════════════════════════
def parse_react_step(text: str) -> tuple[str, str, str]:
    """
    Parse one ReAct step from LLM output.
    Returns (thought, action, action_input).
    """
    thought = action = action_input = ""

    t_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", text, re.S | re.I)
    a_match = re.search(r"Action:\s*(\w+)", text, re.I)
    i_match = re.search(r"ActionInput:\s*(.+?)$", text, re.S | re.I)

    if t_match: thought      = t_match.group(1).strip()
    if a_match: action       = a_match.group(1).strip().lower()
    if i_match: action_input = i_match.group(1).strip()

    return thought, action, action_input

# ════════════════════════════════════════════════════════
#  GROQ CALL
# ════════════════════════════════════════════════════════
def call_groq(messages: list[dict]) -> str:
    if not GROQ_API_KEY:
        return "Thought: No API key\nAction: finish\nActionInput: ⚠️ GROQ_API_KEY not set. Run: export GROQ_API_KEY=your_key"
    try:
        client = Groq(api_key=GROQ_API_KEY)
        resp   = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=1024,
            stop=["Observation:"]   # stop before writing its own observation
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Thought: Error\nAction: finish\nActionInput: Groq error: {e}"

# ════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ════════════════════════════════════════════════════════
def print_banner():
    t = Text()
    t.append("  ╔═══════════════════════════════════════════════════╗\n", style="bold purple")
    t.append("  ║  ", style="bold purple")
    t.append("✦  AVIRA  ✦", style="bold magenta")
    t.append("  — Autonomous Terminal Agent  ║\n", style="bold purple")
    t.append("  ║   ReAct · Tools · Memory · Learn · Safe          ║\n", style="bold purple")
    t.append("  ╚═══════════════════════════════════════════════════╝\n", style="bold purple")
    console.print(t)
    console.print("  Type [bold]help[/bold] for commands · [bold]quit[/bold] to exit\n", style="dim")

def print_help():
    tbl = Table(box=box.ROUNDED, border_style="purple",
                title="Avira Commands", title_style="bold magenta")
    tbl.add_column("Command",    style="bold cyan", no_wrap=True)
    tbl.add_column("What it does", style="white")
    rows = [
        ("help",             "Show this help"),
        ("memory",           "Show all learned facts"),
        ("history",          "Show conversation history"),
        ("forget <n>",       "Delete memory fact #n"),
        ("forget all",       "Wipe all memory"),
        ("clear history",    "Clear conversation history"),
        ("quit / exit",      "Exit Avira"),
        ("— just chat —",   "Avira autonomously picks tools & reasons"),
    ]
    for r in rows: tbl.add_row(*r)
    console.print(tbl)

def show_thought(step: int, thought: str):
    console.print(f"\n  [dim]┌─ Step {step} · Thought[/dim]")
    console.print(f"  [dim]│[/dim]  [italic]{thought}[/italic]")

def show_action(action: str, action_input: str):
    colors = {
        "search": "blue", "calculate": "green", "remember": "yellow",
        "recall": "cyan", "analyse": "magenta", "finish": "purple"
    }
    icons = {
        "search": "🔍", "calculate": "🧮", "remember": "📝",
        "recall": "🧠", "analyse": "📊", "finish": "✅"
    }
    c = colors.get(action, "white")
    i = icons.get(action, "⚙")
    console.print(f"  [dim]│[/dim]  [{c}]{i} Action:[/{c}] [bold]{action}[/bold]  [dim]→[/dim]  {action_input[:120]}{'…' if len(action_input)>120 else ''}")

def show_observation(obs: str):
    short = obs[:300] + ("…" if len(obs) > 300 else "")
    console.print(f"  [dim]│  Observation:[/dim] {short}")
    console.print(f"  [dim]└{'─'*60}[/dim]")

def show_final(answer: str):
    console.print(Panel(
        Markdown(answer),
        border_style="purple",
        padding=(0, 2),
        title="[bold magenta]✦ Avira[/bold magenta]",
        title_align="left"
    ))

def show_memory_table(memory: Memory):
    if not memory.facts:
        console.print("[dim]  Nothing in memory yet.[/dim]")
        return
    tbl = Table(box=box.SIMPLE, border_style="dim",
                title="📚 Memory", title_style="bold magenta")
    tbl.add_column("#",       style="dim",       width=4)
    tbl.add_column("Fact",    style="white",     ratio=4)
    tbl.add_column("Source",  style="bold cyan", width=14)
    tbl.add_column("When",    style="dim",       width=17)
    for i, f in enumerate(memory.facts):
        tbl.add_row(str(i+1), f["fact"], f["source"], f["timestamp"])
    console.print(tbl)

def show_history_table(history: list[dict]):
    if not history:
        console.print("[dim]  No history yet.[/dim]")
        return
    tbl = Table(box=box.SIMPLE, border_style="dim")
    tbl.add_column("#",    style="dim",       width=3)
    tbl.add_column("Role", style="bold cyan", width=12)
    tbl.add_column("Content (truncated)", style="white")
    for i, m in enumerate(history, 1):
        content = m["content"][:100] + ("…" if len(m["content"]) > 100 else "")
        tbl.add_row(str(i), m["role"].upper(), content)
    console.print(tbl)

# ════════════════════════════════════════════════════════
#  LEARN-FROM-INPUT  (passive learning from natural speech)
# ════════════════════════════════════════════════════════
LEARN_PATS = [
    r"^(remember|note|learn|know)\s+(that\s+)?(.+)",
    r"^my name is (.+)",
    r"^i am (.+)",
    r"^i work (?:at|for|in) (.+)",
    r"^i live in (.+)",
    r"^i(?:'m| am) from (.+)",
    r"^i like (.+)",
    r"^i prefer (.+)",
    r"^i hate (.+)",
    r"^my (.+) is (.+)",
    r"^by the way[,\s]+(.+)",
    r"^fyi[,:\s]+(.+)",
    r"^just so you know[,\s]+(.+)",
    r"^teach yourself[,:\s]+(.+)",
]
def extract_learning(text: str):
    for pat in LEARN_PATS:
        m = re.match(pat, text.strip(), re.IGNORECASE)
        if m:
            groups = [g for g in m.groups() if g]
            fact = groups[-1].strip().rstrip(".")
            if len(fact) > 3:
                return fact
    return None

# ════════════════════════════════════════════════════════
#  AUTONOMOUS REACT LOOP
# ════════════════════════════════════════════════════════
def react_loop(user_query: str, memory: Memory, conv_history: list[dict]) -> str:
    """
    Run the ReAct loop for a user query.
    Returns the final answer string.
    """
    system = react_system(memory)

    # Build the messages: system + conversation history + new user query
    messages = [{"role": "system", "content": system}]
    # include last few turns for context
    messages += conv_history[-(MAX_HISTORY):]
    messages.append({"role": "user", "content": user_query})

    # The scratchpad tracks all Thought/Action/Observation this turn
    scratchpad = ""

    console.print(f"\n  [dim]Avira is reasoning autonomously… (max {MAX_STEPS} steps)[/dim]")
    console.print(f"  [dim]{'─'*62}[/dim]")

    for step in range(1, MAX_STEPS + 1):
        # Append scratchpad so far so LLM knows what it's already done
        step_messages = messages.copy()
        if scratchpad:
            step_messages.append({"role": "assistant", "content": scratchpad})

        raw = call_groq(step_messages)
        thought, action, action_input = parse_react_step(raw)

        if not action:
            # LLM gave a free-form reply — treat as finish
            action = "finish"
            action_input = raw

        show_thought(step, thought or "(no explicit thought)")
        show_action(action, action_input)

        if action == "finish":
            console.print(f"  [dim]└{'─'*60}[/dim]")
            return action_input   # ← final answer

        # Run the tool
        observation = run_tool(action, action_input, memory)
        show_observation(observation)

        # Extend scratchpad
        scratchpad += (
            f"\nThought: {thought}\nAction: {action}\nActionInput: {action_input}"
            f"\nObservation: {observation}\n"
        )

        time.sleep(0.2)  # slight pause so terminal is readable

    # Hit max steps — ask LLM to wrap up
    step_messages = messages.copy()
    step_messages.append({"role": "assistant", "content": scratchpad})
    step_messages.append({"role": "user", "content":
        "You have reached the maximum number of steps. Please give your best final answer now."})
    raw = call_groq(step_messages)
    _, _, final = parse_react_step(raw)
    return final or raw

# ════════════════════════════════════════════════════════
#  MAIN LOOP
# ════════════════════════════════════════════════════════
def run():
    print_banner()
    memory  = Memory(MEMORY_FILE)
    history: list[dict] = []

    if not GROQ_API_KEY:
        console.print(Panel(
            "[bold yellow]⚠  GROQ_API_KEY not set.[/bold yellow]\n\n"
            "Run this before starting:\n"
            "[bold cyan]  export GROQ_API_KEY=your_key_here[/bold cyan]\n\n"
            "Free key: https://console.groq.com",
            border_style="yellow", title="Setup Required"))

    if memory.facts:
        console.print(f"  [dim]📚 {len(memory.facts)} memory fact(s) loaded.[/dim]\n")

    while True:
        try:
            user_input = Prompt.ask("\n[bold purple]You[/bold purple]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]  Goodbye! — Avira ✦[/dim]")
            break

        if not user_input:
            continue

        lower = user_input.lower().strip()

        # ── Commands ───────────────────────────────────────────────────────
        if lower in ("quit", "exit"):
            console.print("\n[dim]  Goodbye! Stay curious. — Avira ✦[/dim]")
            break
        if lower == "help":
            print_help(); continue
        if lower == "memory":
            show_memory_table(memory); continue
        if lower == "history":
            show_history_table(history); continue
        if lower == "clear history":
            history.clear()
            console.print("[dim]  History cleared.[/dim]"); continue
        if lower == "forget all":
            memory.clear()
            console.print("[dim]  All memory wiped.[/dim]"); continue
        if lower.startswith("forget "):
            try:
                idx = int(lower.split("forget ", 1)[1].strip()) - 1
                console.print(f"[dim]  Fact #{idx+1} {'removed.' if memory.remove(idx) else 'not found.'}[/dim]")
            except ValueError:
                console.print("[dim]  Usage: forget <number>[/dim]")
            continue

        # ── Safety ─────────────────────────────────────────────────────────
        if is_blocked(user_input):
            console.print(Panel(
                "[bold red]🚫 Request blocked.[/bold red]\n\n"
                "This violates my safety guidelines. I don't assist with violent, "
                "harmful, illegal, or extremist content.\n\n"
                "[dim]Ask me about news, analysis, or math instead![/dim]",
                border_style="red", title="[bold red]Blocked[/bold red]", title_align="left"))
            continue

        # ── Passive learning check ─────────────────────────────────────────
        fact = extract_learning(user_input)
        if fact:
            added = memory.add(fact, source="user")
            if added:
                console.print(f"\n  [bold yellow]📝 Noted:[/bold yellow] [dim]\"{fact}\" saved to memory.[/dim]")

        # ── Autonomous ReAct loop ──────────────────────────────────────────
        final_answer = react_loop(user_input, memory, history)

        # Update conversation history
        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": final_answer})
        if len(history) > MAX_HISTORY * 2:
            history = history[-(MAX_HISTORY * 2):]

        show_final(final_answer)

if __name__ == "__main__":
    run()