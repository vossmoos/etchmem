"""etchmem visual demo — before / memorize / after.

Run (etchmem-server must be up on :8000):
    pip install fastapi uvicorn openai requests
    cd demo && python app.py
Then open http://localhost:8080
"""
import json
import os
import time
from pathlib import Path

import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# ── config ──────────────────────────────────────────────────────────────────
ETCHMEM = "http://localhost:8000"
LLM_MODEL = "gpt-5.5"

for line in (Path(__file__).parent.parent / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

from openai import OpenAI  # noqa: E402

client = OpenAI()
app = FastAPI()

# The prompt is IDENTICAL before and after. Simple on purpose.
TASK = (
    "Write outreach to John Doe, CTO of Safe Infrastructure Inc. — a company "
    "that sells cloud-cost / infrastructure-efficiency tooling. We sell an "
    "observability platform. Goal: get him into a technical evaluation. "
    "The example message must be fully written out for John — no placeholders."
)

# Field experience from OTHER accounts and deals — never about Safe
# Infrastructure Inc. or John Doe. The agent learns transferable patterns:
# what works for this persona, and which of OUR OWN products/plays/assets win.
SIGNALS = [
    # ── persona patterns: what kills threads ────────────────────────────────
    ("outreach-agent-07", "CTOs at infrastructure-optimization companies never respond to ROI or cost-savings pitches: 0 replies out of 41 this quarter. They sell efficiency themselves — pitching savings back at them reads as amateur."),
    ("email-tracker",     "Channel stats for the infra-optimization CTO segment: plain-text email 31% reply rate, HTML marketing email 6%, LinkedIn InMail 0% (0/56). LinkedIn is dead for this persona."),
    ("outreach-agent-12", "Asking infra-optimization CTOs for a 'quick 15-min demo call' in the first email got 0/28 replies. First CTA must be zero-commitment — a question answerable in one line."),
    ("outreach-agent-12", "Name-dropping the prospect's direct competitors killed 4 out of 4 threads with infra-optimization CTOs — they treat it as a confidentiality risk."),

    # ── persona patterns: what works ─────────────────────────────────────────
    ("outreach-agent-03", "Infra-optimization CTOs answer peer-style engineering questions: 11 of 17 replies this quarter came from emails asking their technical opinion on a concrete design or benchmark decision, not from pitches."),
    ("email-tracker",     "Best send window for infra-optimization CTOs: Tuesday 07:00–09:00 local, 3x the reply rate of any other slot. Friday sends are never answered."),
    ("web-analytics",     "Infra-optimization accounts that visited /docs/benchmarks or /docs/architecture within 7 days replied to outreach 5x more often. /pricing visits predict nothing for this persona."),

    # ── OUR playbook: which of our assets and plays actually win ────────────
    ("crm-sync",          "Our free '2-week telemetry cost audit' offer converted 5 of 7 infra-optimization prospects into technical evaluations this year — it speaks their own language: cost per span, per metric, per GB."),
    ("crm-sync",          "Proposal history in the infra-optimization segment: the 'audit-first' template (free audit → paid pilot on ONE workload) won 4 deals; the 'annual license with discount' template lost every single time."),
    ("outreach-agent-03", "Our Lumen overhead benchmark whitepaper — '3.1% collector overhead at 1M spans/sec, methodology open-sourced' — earned replies from 6 infra-segment CTOs. The generic Lumen product deck earned zero."),
    ("outreach-agent-07", "Sharing our public benchmark sandbox (synthetic 1M spans/sec dataset, run it yourself) works: 40% of infra-segment CTOs who received the link ran it within a week, and every one of them took a follow-up call."),
    ("crm-sync",          "Nordvik Cloud — an infra-cost-optimization vendor in the Nordics, so a peer but not a competitor of our prospects — runs Lumen in production and is a public reference. Citing Nordvik opened 3 of our last 5 conversations in this segment."),
    ("crm-sync",          "Offering discounts backfires with infra-optimization CTOs: two of them said discounting made them doubt our metering accuracy. Framing must stay engineering-grade, never price-led."),
    ("outreach-agent-12", "Follow-ups only work on this segment when they add a NEW technical artifact (fresh benchmark run, architecture note). 'Just bumping this' follow-ups: zero replies ever recorded."),
    ("crm-sync",          "Every infra-optimization deal that reached evaluation had one of our senior engineers (not sales) join the thread by message three; CTOs disengage from sales-only threads."),
]


SYSTEM = (
    "You are an outreach agent. Answer ONLY as compact JSON: "
    '{"recommendation": "<ONE sentence: what to send, how, when>", '
    '"reasoning": "<max 2 short sentences — the main idea>", '
    '"based_on": ["<up to 4 facts from your field-experience memory that '
    'drove this answer, each under 12 words; EMPTY list if you have no '
    'field-experience memory>"], '
    '"example": {"subject": "<subject line>", "body": "<the message, max 80 words>"}}'
)

MEM_PREAMBLE = (
    "AGENT MEMORY (etchmem recall — consolidated field experience learned "
    "from real outcomes across MANY other accounts and deals; general "
    "patterns plus our own proven plays; OVERRIDES generic best practices):"
)


def ask(memory_block: str | None = None) -> dict:
    # The prompt (system + task) is IDENTICAL before and after.
    # The ONLY difference: the experienced agent gets its memory as extra context.
    messages = [{"role": "system", "content": SYSTEM}]
    if memory_block:
        messages.append({"role": "system", "content": MEM_PREAMBLE + "\n\n" + memory_block})
    messages.append({"role": "user", "content": TASK})
    r = client.chat.completions.create(
        model=LLM_MODEL,
        response_format={"type": "json_object"},
        messages=messages,
    )
    return json.loads(r.choices[0].message.content)


@app.get("/api/prompt")
def prompt():
    return {"system": SYSTEM, "task": TASK, "memory_preamble": MEM_PREAMBLE}


@app.get("/api/signals")
def signals():
    return [{"source": s, "text": t} for s, t in SIGNALS]


@app.post("/api/before")
def before():
    return ask()


@app.post("/api/memorize")
def memorize():
    for source, text in SIGNALS:
        requests.post(f"{ETCHMEM}/remember", json={
            "data": text, "source": source, "scope": "outreach",
            "extract_mode": "immediate",
        }, timeout=10).raise_for_status()
    for _ in range(60):
        tick = requests.post(f"{ETCHMEM}/sleep", timeout=300).json()
        if "detail" in tick:
            return {"error": tick["detail"]}
        stats = requests.get(f"{ETCHMEM}/stats", timeout=10).json()
        if stats["signals_new"] == 0 and stats["signals_batched"] == 0:
            break
        time.sleep(1)
    export = requests.post(f"{ETCHMEM}/export", timeout=60).json()
    top = sorted(export["etches"], key=lambda e: -e["confidence"])[:5]
    return {
        "signals": len(SIGNALS),
        "beliefs": export["count"],
        "top_beliefs": [
            {"text": f"{e['entity_name']} · {e['property']}: {e['current_value']}",
             "confidence": e["confidence"]}
            for e in top
        ],
    }


@app.post("/api/after")
def after():
    recall = requests.post(f"{ETCHMEM}/recall", json={
        "query": "outreach to a CTO of an infrastructure cost-optimization company: "
                 "what works and fails for this persona, our winning plays, assets, "
                 "proposal templates, reference customers, channel and timing",
        "scope": "outreach", "top_k": 18,
    }, timeout=30).json()
    memory_block = "\n".join(f"- {r['content']}" for r in recall["results"])
    result = ask(memory_block)
    result["memory_block"] = memory_block  # so the page can show the real diff
    return result


PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>etchmem — agent memory demo</title>
<style>
  :root { --red:#e5484d; --green:#30a46c; --ink:#1a1a2e; --mut:#6b7280; }
  * { box-sizing:border-box; margin:0; }
  body { font-family:-apple-system,'Segoe UI',Roboto,sans-serif; background:#f6f7f9;
         color:var(--ink); padding:40px 24px; }
  .wrap { max-width:1060px; margin:0 auto; }
  h1 { font-size:26px; margin-bottom:6px; }
  .sub { color:var(--mut); margin-bottom:8px; }
  .task { background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:14px 18px;
          margin:18px 0 26px; font-size:15px; color:#374151; }
  .task b { color:var(--ink); }
  .cols { display:grid; grid-template-columns:1fr 220px 1fr; gap:20px; align-items:start; }
  .card { background:#fff; border:1px solid #e5e7eb; border-radius:16px; padding:22px;
          min-height:340px; }
  .card h2 { font-size:15px; text-transform:uppercase; letter-spacing:.08em; margin-bottom:4px; }
  .before h2 { color:var(--red); } .after h2 { color:var(--green); }
  .hint { font-size:13px; color:var(--mut); margin-bottom:14px; }
  button { width:100%; padding:13px; border:0; border-radius:10px; font-size:15px;
           font-weight:600; color:#fff; cursor:pointer; }
  button:disabled { opacity:.45; cursor:default; }
  .before button { background:var(--red); } .after button { background:var(--green); }
  .mid { text-align:center; padding-top:60px; }
  .mid button { background:var(--ink); }
  .mid .note { font-size:12.5px; color:var(--mut); margin-top:10px; line-height:1.5; }
  .out { margin-top:18px; font-size:14.5px; line-height:1.55; }
  .lbl { font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:var(--mut);
         margin:14px 0 3px; }
  .rec { font-weight:600; }
  .mail { background:#f9fafb; border:1px solid #e5e7eb; border-radius:10px; padding:12px 14px;
          margin-top:4px; white-space:pre-wrap; }
  .mail .subj { font-weight:600; border-bottom:1px solid #e5e7eb; padding-bottom:6px;
                margin-bottom:8px; }
  .chips { display:flex; flex-wrap:wrap; gap:6px; margin-top:4px; }
  .chip { background:#e9f7f0; border:1px solid #b7e3cd; color:#1a7f4e; font-size:12px;
          padding:4px 10px; border-radius:999px; }
  .chip.none { background:#fbeaea; border-color:#f2c4c6; color:#b3383d; }
  .beliefs { text-align:left; font-size:12.5px; margin-top:14px; color:#374151; }
  .beliefs div { padding:4px 0; border-top:1px solid #eee; }
  .spin { color:var(--mut); font-size:14px; margin-top:16px; }
  .ok { color:var(--green); font-weight:600; margin-top:12px; }
  .sigs { margin-top:24px; min-height:0; }
  .sigs h2 { font-size:15px; text-transform:uppercase; letter-spacing:.08em;
             color:var(--ink); margin-bottom:4px; }
  .sig { display:flex; gap:12px; padding:8px 0; border-top:1px solid #f0f1f3;
         font-size:13.5px; line-height:1.5; }
  .sig .src { flex:0 0 130px; font-family:ui-monospace,monospace; font-size:11.5px;
              color:var(--mut); padding-top:2px; }
  .prompts { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
  .plabel { font-size:12px; font-weight:700; text-transform:uppercase;
            letter-spacing:.08em; margin-bottom:8px; }
  .pmsg { background:#f9fafb; border:1px solid #e5e7eb; border-radius:10px;
          padding:10px 12px; margin-bottom:10px; font-family:ui-monospace,monospace;
          font-size:12px; line-height:1.55; white-space:pre-wrap; word-break:break-word; }
  .pmsg .role { display:block; font-size:10px; font-weight:700; text-transform:uppercase;
                letter-spacing:.08em; color:var(--mut); margin-bottom:5px; }
  .pmsg.mem { background:#eef9f3; border-color:#b7e3cd; }
  .pmsg.mem .role { color:#1a7f4e; }
  .membody { margin-top:8px; padding-top:8px; border-top:1px dashed #b7e3cd;
             color:#356451; max-height:260px; overflow-y:auto; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Can an AI agent learn from experience?</h1>
  <div class="sub">Same AI. Same prompt, word for word. The only difference: <b>etchmem</b> — the agent's memory of what actually worked in past deals.</div>
  <div class="task"><b>The task, both times:</b> write outreach to <b>John Doe</b>, CTO of <b>Safe Infrastructure Inc.</b> — a company the agent has <b>never contacted before</b>.</div>

  <div class="cols">
    <div class="card before">
      <h2>1 · Before</h2>
      <div class="hint">A fresh agent. Sounds smart — but it's guessing.</div>
      <button id="bBefore" onclick="run('before')">Ask the agent</button>
      <div id="oBefore" class="out"></div>
    </div>

    <div class="card mid">
      <h2>2 · Learn</h2>
      <button id="bMem" onclick="memorize()">Memorize &amp; sleep</button>
      <div class="note">Feeds 16 field signals from <b>other</b> deals — what worked, what failed, which of our plays won — and consolidates them into beliefs.</div>
      <div id="oMem"></div>
    </div>

    <div class="card after">
      <h2>3 · After</h2>
      <div class="hint">The same agent — now drawing on real wins and losses.</div>
      <button id="bAfter" onclick="run('after')" disabled>Ask again</button>
      <div id="oAfter" class="out"></div>
    </div>
  </div>

  <div class="card sigs">
    <h2>The exact prompts we send</h2>
    <div class="hint">Word for word. The system and task messages are <b>byte-identical</b> in both calls —
      the only difference is one extra context message (green) carrying the agent's memory.</div>
    <div class="prompts">
      <div>
        <div class="plabel" style="color:var(--red)">Before</div>
        <div class="pmsg"><span class="role">system</span><span id="pSystem1"></span></div>
        <div class="pmsg"><span class="role">user</span><span id="pTask1"></span></div>
      </div>
      <div>
        <div class="plabel" style="color:var(--green)">After</div>
        <div class="pmsg"><span class="role">system</span><span id="pSystem2"></span></div>
        <div class="pmsg mem"><span class="role">system</span><span id="pMemPre"></span>
          <div id="pMemBody" class="membody">… the recalled beliefs appear here after you run "Ask again" …</div></div>
        <div class="pmsg"><span class="role">user</span><span id="pTask2"></span></div>
      </div>
    </div>
  </div>

  <div class="card sigs">
    <h2>What the agent learns from</h2>
    <div class="hint">Field signals from past work across <b>other</b> accounts and deals.
      None mention John Doe or Safe Infrastructure Inc. — the agent learns
      <b>transferable patterns</b>: how this persona behaves, and which of our own
      products, offers and proposals actually win.</div>
    <div id="sigList"></div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const cap = s => s.charAt(0).toUpperCase() + s.slice(1);

function renderAnswer(el, d) {
  const chips = (d.based_on && d.based_on.length)
    ? d.based_on.map(f => `<span class="chip">${f}</span>`).join('')
    : '<span class="chip none">none — generic best practices</span>';
  el.innerHTML = `
    <div class="lbl">Recommendation</div><div class="rec">${d.recommendation}</div>
    <div class="lbl">Why</div><div>${d.reasoning}</div>
    <div class="lbl">Experience used</div><div class="chips">${chips}</div>
    <div class="lbl">Example message</div>
    <div class="mail"><div class="subj">${d.example.subject}</div>${d.example.body}</div>`;
}

async function run(mode) {
  const btn = $('b' + cap(mode)), out = $('o' + cap(mode));
  btn.disabled = true;
  out.innerHTML = '<div class="spin">Thinking…</div>';
  try {
    const r = await fetch('/api/' + mode, {method:'POST'});
    const d = await r.json();
    renderAnswer(out, d);
    if (d.memory_block) $('pMemBody').textContent = d.memory_block;
  } catch (e) { out.innerHTML = '<div class="spin">Error: ' + e + '</div>'; }
  btn.disabled = false;
}

async function memorize() {
  const btn = $('bMem'), out = $('oMem');
  btn.disabled = true;
  out.innerHTML = '<div class="spin">Depositing signals &amp; consolidating…</div>';
  try {
    const r = await fetch('/api/memorize', {method:'POST'});
    const d = await r.json();
    if (d.error) { out.innerHTML = '<div class="spin">Error: ' + d.error + '</div>'; btn.disabled = false; return; }
    out.innerHTML = `<div class="ok">✓ ${d.signals} signals → ${d.beliefs} beliefs</div>
      <div class="beliefs">` +
      d.top_beliefs.map(b => `<div>${b.text}</div>`).join('') + '</div>';
    $('bAfter').disabled = false;
  } catch (e) { out.innerHTML = '<div class="spin">Error: ' + e + '</div>'; btn.disabled = false; }
}

(async () => {
  const sigs = await (await fetch('/api/signals')).json();
  $('sigList').innerHTML = sigs.map(s =>
    `<div class="sig"><div class="src">${s.source}</div><div>${s.text}</div></div>`).join('');
  const p = await (await fetch('/api/prompt')).json();
  $('pSystem1').textContent = p.system;  $('pSystem2').textContent = p.system;
  $('pTask1').textContent = p.task;      $('pTask2').textContent = p.task;
  $('pMemPre').textContent = p.memory_preamble;
})();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
