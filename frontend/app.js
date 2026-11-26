// ─────────────────────────────────────────────────────────────
// SNN-LM Dashboard SPA
// ─────────────────────────────────────────────────────────────

const TABS = [
  { id: "overview",  label: "Overview"  },
  { id: "chat",      label: "Chat"      },
  { id: "training",  label: "Training"  },
  { id: "lora",      label: "LoRA"      },
  { id: "rag",       label: "RAG"       },
  { id: "reasoning", label: "Reasoning" },
  { id: "jobs",      label: "Jobs"      },
  { id: "files",     label: "Files"     },
  { id: "admin",     label: "Admin"     },
];

const state = {
  tab: "overview",
  schemas: {},
  jobs: [],
  selectedJob: null,
  jobLogStream: null,
  chatStream: null,
  ckpts: [],
};

// ─────────────────────────────────────────────────────────────
// API
// ─────────────────────────────────────────────────────────────
async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    ...opts,
  });
  const text = await res.text();
  let data;
  try { data = text ? JSON.parse(text) : {}; }
  catch { data = { raw: text }; }
  if (!res.ok) {
    const msg = data.error || data.detail || `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return data;
}

function toast(msg, kind = "info") {
  const el = document.createElement("div");
  const color = kind === "error" ? "bg-red-600" : "bg-slate-700";
  el.className = `${color} px-3 py-2 rounded text-sm shadow`;
  el.textContent = msg;
  document.getElementById("toast").appendChild(el);
  setTimeout(() => el.remove(), 4000);
}

// ─────────────────────────────────────────────────────────────
// Nav + header badges
// ─────────────────────────────────────────────────────────────
function renderNav() {
  const nav = document.getElementById("nav");
  nav.innerHTML = TABS.map(t => `
    <div class="tab ${state.tab === t.id ? "active" : ""}" data-tab="${t.id}">
      ${t.label}
    </div>
  `).join("");
  nav.querySelectorAll(".tab").forEach(el => {
    el.onclick = () => switchTab(el.dataset.tab);
  });
}

function switchTab(tab) {
  state.tab = tab;
  if (state.jobLogStream) { state.jobLogStream.close(); state.jobLogStream = null; }
  if (state.chatStream)   { state.chatStream.close(); state.chatStream = null; }
  renderNav();
  renderTab();
}

async function refreshHeader() {
  try {
    const info = await api("/api/model/info");
    const b = document.getElementById("model-badge");
    if (info.loaded) {
      b.className = "badge bg-emerald-700";
      b.textContent = `model: ${info.core_type}/${info.ffn_mode} · ${(info.params/1e6).toFixed(1)}M · ${info.device}`;
    } else {
      b.className = "badge bg-red-700";
      b.textContent = "model: not loaded";
    }
  } catch {}

  try {
    const s = await api("/api/train/stats");
    const total = Object.values(s).reduce((a, b) => a + b, 0);
    const running = s.running || 0;
    const b = document.getElementById("jobs-badge");
    b.className = "badge " + (running > 0 ? "bg-blue-700" : "bg-slate-700");
    b.textContent = `jobs: ${total} (running ${running})`;
  } catch {}

  try {
    const h = await api("/api/admin/health");
    document.getElementById("health-status").textContent =
      `health: ${h.ok ? "ok" : "fail"} · py ${h.python}`;
  } catch {}
}

// ─────────────────────────────────────────────────────────────
// Router
// ─────────────────────────────────────────────────────────────
function renderTab() {
  const main = document.getElementById("main");
  const fn = {
    overview: renderOverview,
    chat: renderChat,
    training: renderJobTab("train"),
    lora: renderJobTab("lora"),
    rag: renderRag,
    reasoning: renderReasoning,
    jobs: renderJobs,
    files: renderFiles,
    admin: renderAdmin,
  }[state.tab];
  main.innerHTML = "";
  fn(main);
}

// ─────────────────────────────────────────────────────────────
// Overview
// ─────────────────────────────────────────────────────────────
async function renderOverview(root) {
  const wrap = document.createElement("div");
  wrap.className = "space-y-4";
  wrap.innerHTML = `<h2 class="text-lg font-semibold">Overview</h2>
    <div id="ov-info" class="grid grid-cols-2 gap-4"></div>
    <div id="ov-kn" class="card"></div>`;
  root.appendChild(wrap);

  const [info, kn, ckpts] = await Promise.all([
    api("/api/model/info").catch(() => ({})),
    api("/api/model/knowledge").catch(() => ({})),
    api("/api/model/ckpts").catch(() => ({ ckpts: [] })),
  ]);
  state.ckpts = ckpts.ckpts;

  const kv = (k, v) => `<div class="flex justify-between text-sm border-b border-slate-700/50 py-1">
    <span class="text-slate-400">${k}</span><span class="font-mono">${v}</span></div>`;

  wrap.querySelector("#ov-info").innerHTML = `
    <div class="card">
      <div class="label mb-2">Model</div>
      ${info.loaded ? `
        ${kv("ckpt", info.ckpt)}
        ${kv("core", info.core_type)}
        ${kv("ffn", info.ffn_mode)}
        ${kv("params", (info.params/1e6).toFixed(2) + "M")}
        ${kv("device", info.device)}
        ${kv("seq_len", info.seq_len)}
        ${kv("layers", info.n_layer)}
        ${kv("rt", info.rt)}
        ${kv("memory_size", info.memory_size)}
      ` : `<div class="text-slate-500">No model loaded.</div>
        <div class="mt-3"><select id="ckpt-pick" class="input">
          ${ckpts.ckpts.map(c => `<option value="${c.path}">${c.path}</option>`).join("")}
        </select>
        <button id="ckpt-load" class="btn btn-primary mt-2 w-full">Load</button>
        </div>`}
    </div>
    <div class="card">
      <div class="label mb-2">Actions</div>
      <button class="btn w-full mb-2" id="act-consolidate">Consolidate memory</button>
      <button class="btn w-full mb-2" id="act-reset-mem">Reset memory</button>
      <button class="btn btn-danger w-full" id="act-unload">Unload model</button>
    </div>
  `;
  wrap.querySelector("#ov-kn").innerHTML =
    `<div class="label mb-2">Self-knowledge</div>` +
    `<pre class="pre">${JSON.stringify(kn, null, 2)}</pre>`;

  const ckPick = document.getElementById("ckpt-pick");
  if (ckPick) {
    document.getElementById("ckpt-load").onclick = async () => {
      try {
        await api("/api/model/load", { method: "POST",
          body: JSON.stringify({ ckpt: ckPick.value, device: "auto" }) });
        toast("Model loaded"); refreshHeader(); renderTab();
      } catch (e) { toast(e.message, "error"); }
    };
  }

  document.getElementById("act-consolidate").onclick = async () => {
    try {
      const r = await api("/api/model/consolidate", { method: "POST" });
      toast("Consolidated: " + JSON.stringify(r));
    } catch (e) { toast(e.message, "error"); }
  };
  document.getElementById("act-reset-mem").onclick = async () => {
    try { await api("/api/model/reset_memory", { method: "POST" });
      toast("Memory reset"); } catch (e) { toast(e.message, "error"); }
  };
  document.getElementById("act-unload").onclick = async () => {
    try { await api("/api/model/unload", { method: "POST" });
      toast("Unloaded"); refreshHeader(); renderTab(); }
    catch (e) { toast(e.message, "error"); }
  };
}

// ─────────────────────────────────────────────────────────────
// Chat
// ─────────────────────────────────────────────────────────────
function renderChat(root) {
  root.innerHTML = `
    <div class="space-y-3">
      <h2 class="text-lg font-semibold">Chat</h2>
      <div class="card space-y-2">
        <textarea id="chat-prompt" rows="3" class="input"
          placeholder="Prompt..."></textarea>
        <div class="grid grid-cols-4 gap-2 text-sm">
          <label>max_tokens<input id="chat-max" type="number" value="200" class="input"></label>
          <label>temperature<input id="chat-temp" type="number" step="0.1" value="0.8" class="input"></label>
          <label>top_k<input id="chat-topk" type="number" value="40" class="input"></label>
          <label>top_p<input id="chat-topp" type="number" step="0.01" value="0.95" class="input"></label>
        </div>
        <div class="flex gap-4 text-sm">
          <label><input id="chat-rag" type="checkbox"> RAG</label>
          <label><input id="chat-bias" type="checkbox"> Reasoning bias</label>
        </div>
        <div class="flex gap-2">
          <button id="chat-send" class="btn btn-primary">Generate</button>
          <button id="chat-verify" class="btn">Generate + Verify</button>
        </div>
      </div>
      <pre id="chat-out" class="pre min-h-[10rem]"></pre>
      <pre id="chat-verify-out" class="pre hidden"></pre>
    </div>
  `;
  document.getElementById("chat-send").onclick = chatSend;
  document.getElementById("chat-verify").onclick = chatVerify;
}

async function chatSend() {
  const out = document.getElementById("chat-out");
  out.textContent = "";
  const body = {
    prompt: document.getElementById("chat-prompt").value,
    max_tokens: +document.getElementById("chat-max").value,
    temperature: +document.getElementById("chat-temp").value,
    top_k: +document.getElementById("chat-topk").value,
    top_p: +document.getElementById("chat-topp").value,
    use_rag: document.getElementById("chat-rag").checked,
    use_reasoning_bias: document.getElementById("chat-bias").checked,
  };
  try {
    const res = await fetch("/api/chat/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(await res.text());
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split("\n\n");
      buf = parts.pop();
      for (const p of parts) {
        const evMatch = p.match(/^event: (.+)$/m);
        const dataMatch = p.match(/^data: (.+)$/m);
        if (!evMatch || !dataMatch) continue;
        const ev = evMatch[1], data = JSON.parse(dataMatch[1]);
        if (ev === "token") out.textContent += data.text;
        if (ev === "error") { toast(data.error, "error"); return; }
        if (ev === "done") return;
      }
    }
  } catch (e) { toast(e.message, "error"); }
}

async function chatVerify() {
  const out = document.getElementById("chat-verify-out");
  out.classList.remove("hidden");
  out.textContent = "running...";
  const body = {
    prompt: document.getElementById("chat-prompt").value,
    max_tokens: +document.getElementById("chat-max").value,
    temperature: +document.getElementById("chat-temp").value,
    top_k: +document.getElementById("chat-topk").value,
    top_p: +document.getElementById("chat-topp").value,
    verify: true,
  };
  try {
    const r = await api("/api/chat/verify", {
      method: "POST", body: JSON.stringify(body),
    });
    out.textContent =
      `verdict: ${r.verdict}\n\n--- text ---\n${r.text}\n\n` +
      `--- problems ---\n` + JSON.stringify(r.problems, null, 2) +
      `\n\n--- signals ---\n` + JSON.stringify(r.signals, null, 2);
  } catch (e) { out.textContent = "ERROR: " + e.message; }
}

// ─────────────────────────────────────────────────────────────
// Generic job form (train / lora)
// ─────────────────────────────────────────────────────────────
function renderJobTab(jobType) {
  return async function(root) {
    if (!state.schemas[jobType]) {
      const s = await api(`/api/train/schema/${jobType}`);
      state.schemas[jobType] = s;
    }
    const schema = state.schemas[jobType];
    const groups = {};
    for (const f of schema.fields) {
      const g = f.group || "Other";
      (groups[g] = groups[g] || []).push(f);
    }

    root.innerHTML = `
      <div class="space-y-3">
        <h2 class="text-lg font-semibold">${schema.title}</h2>
        <p class="text-sm text-slate-400">${schema.description}</p>
        <form id="job-form" class="space-y-3">
          ${schema.positional.map(p => renderField(p)).join("")}
          ${Object.entries(groups).map(([g, fields]) => `
            <details class="card" open>
              <summary class="cursor-pointer label mb-2">${g}</summary>
              <div class="grid grid-cols-2 gap-2">${fields.map(renderField).join("")}</div>
            </details>
          `).join("")}
          <div class="flex gap-2">
            <button type="submit" class="btn btn-primary">Submit job</button>
          </div>
        </form>
        <div id="job-submit-result"></div>
      </div>
    `;
    root.querySelector("#job-form").onsubmit = async (e) => {
      e.preventDefault();
      const config = collectForm(root, schema);
      try {
        const r = await api("/api/train/jobs", {
          method: "POST",
          body: JSON.stringify({ type: jobType, config }),
        });
        toast("Job submitted: " + r.id);
        root.querySelector("#job-submit-result").innerHTML =
          `<div class="card">Job <a class="text-blue-400 underline cursor-pointer"
            onclick="switchTab('jobs');selectJob('${r.id}')">${r.id}</a> enqueued.</div>`;
        refreshHeader();
      } catch (err) { toast(err.message, "error"); }
    };
  };
}

function renderField(f) {
  const name = f.name;
  const label = `<span class="label">${name}${f.required ? " *" : ""}</span>`;
  const help = f.help ? `<div class="text-xs text-slate-500">${f.help}</div>` : "";
  const defaultVal = f.default !== undefined && f.default !== null ? f.default : "";
  if (f.type === "bool") {
    return `<label class="flex items-center gap-2 text-sm py-1">
      <input type="checkbox" data-field="${name}" ${defaultVal ? "checked" : ""}>
      ${name}
    </label>`;
  }
  if (f.type === "select") {
    return `<label class="block">${label}
      <select class="input" data-field="${name}">
        ${f.options.map(o => `<option ${o === defaultVal ? "selected" : ""}>${o}</option>`).join("")}
      </select>${help}</label>`;
  }
  if (f.type === "int" || f.type === "float") {
    return `<label class="block">${label}
      <input class="input" type="number" step="${f.type === "float" ? "any" : "1"}"
        data-field="${name}" data-type="${f.type}" value="${defaultVal}">${help}</label>`;
  }
  if (f.type === "list" || f.type === "paths") {
    const val = Array.isArray(defaultVal) ? defaultVal.join(",") : "";
    return `<label class="block">${label}
      <input class="input" type="text" data-field="${name}"
        data-type="${f.type}" value="${val}" placeholder="comma-separated">${help}</label>`;
  }
  return `<label class="block">${label}
    <input class="input" type="text" data-field="${name}" value="${defaultVal}">${help}</label>`;
}

function collectForm(root, schema) {
  const cfg = {};
  root.querySelectorAll("[data-field]").forEach(el => {
    const name = el.dataset.field;
    if (el.type === "checkbox") { cfg[name] = el.checked; return; }
    const type = el.dataset.type;
    let v = el.value;
    if (v === "") return;
    if (type === "int") v = parseInt(v);
    else if (type === "float") v = parseFloat(v);
    else if (type === "list" || type === "paths") {
      v = v.split(",").map(x => x.trim()).filter(Boolean);
    }
    cfg[name] = v;
  });
  return cfg;
}

// ─────────────────────────────────────────────────────────────
// RAG
// ─────────────────────────────────────────────────────────────
async function renderRag(root) {
  root.innerHTML = `
    <div class="space-y-3">
      <h2 class="text-lg font-semibold">RAG</h2>
      <div class="card">
        <div class="label mb-2">Build / extend KB (job)</div>
        <input id="rag-corpus" class="input mb-2" placeholder="corpus paths, comma-separated">
        <div class="grid grid-cols-2 gap-2">
          <input id="rag-out" class="input" placeholder="out (new KB)">
          <input id="rag-kb" class="input" placeholder="kb (extend existing)">
          <input id="rag-ckpt" class="input" placeholder="ckpt (encoder=hidden)">
          <input id="rag-chunk" class="input" type="number" value="1000">
        </div>
        <button id="rag-build" class="btn btn-primary mt-2">Submit index job</button>
      </div>
      <div class="card">
        <div class="label mb-2">Load KB</div>
        <div class="flex gap-2">
          <input id="rag-load-path" class="input flex-1" placeholder="path to .pt">
          <button id="rag-load" class="btn">Load</button>
          <button id="rag-unload" class="btn btn-danger">Unload</button>
        </div>
      </div>
      <div class="card">
        <div class="label mb-2">Search</div>
        <div class="flex gap-2">
          <input id="rag-query" class="input flex-1" placeholder="query">
          <input id="rag-k" class="input w-20" type="number" value="4">
          <button id="rag-search" class="btn btn-primary">Search</button>
        </div>
        <pre id="rag-results" class="pre mt-2 hidden"></pre>
      </div>
    </div>
  `;
  document.getElementById("rag-build").onclick = async () => {
    const cfg = {
      corpus: document.getElementById("rag-corpus").value.split(",").map(s => s.trim()).filter(Boolean),
      chunk_size: +document.getElementById("rag-chunk").value,
      chunk_overlap: 200,
      encoder: "hidden",
    };
    const out = document.getElementById("rag-out").value.trim();
    const kb = document.getElementById("rag-kb").value.trim();
    const ckpt = document.getElementById("rag-ckpt").value.trim();
    if (out) cfg.out = out;
    if (kb) cfg.kb = kb;
    if (ckpt) cfg.ckpt = ckpt;
    try {
      const r = await api("/api/train/jobs", { method: "POST",
        body: JSON.stringify({ type: "rag_index", config: cfg }) });
      toast("RAG job: " + r.id); refreshHeader();
    } catch (e) { toast(e.message, "error"); }
  };
  document.getElementById("rag-load").onclick = async () => {
    try {
      const r = await api("/api/rag/load", { method: "POST",
        body: JSON.stringify({ path: document.getElementById("rag-load-path").value }) });
      toast(`KB loaded (${r.docs} docs)`); refreshHeader();
    } catch (e) { toast(e.message, "error"); }
  };
  document.getElementById("rag-unload").onclick = async () => {
    try { await api("/api/rag/unload", { method: "POST" }); toast("RAG unloaded"); }
    catch (e) { toast(e.message, "error"); }
  };
  document.getElementById("rag-search").onclick = async () => {
    const out = document.getElementById("rag-results");
    out.classList.remove("hidden");
    try {
      const r = await api("/api/rag/search", { method: "POST",
        body: JSON.stringify({
          path: "",
          query: document.getElementById("rag-query").value,
          k: +document.getElementById("rag-k").value,
        }) });
      out.textContent = JSON.stringify(r.results, null, 2);
    } catch (e) { out.textContent = "ERROR: " + e.message; }
  };
}

// ─────────────────────────────────────────────────────────────
// Reasoning
// ─────────────────────────────────────────────────────────────
async function renderReasoning(root) {
  root.innerHTML = `
    <div class="space-y-3">
      <h2 class="text-lg font-semibold">Reasoning</h2>
      <div class="card">
        <div class="label mb-2">Mine graph (job)</div>
        <input id="rs-corpus" class="input mb-2" placeholder="corpus paths, comma-separated">
        <input id="rs-out" class="input mb-2" placeholder="graph.pt save path">
        <button id="rs-mine" class="btn btn-primary">Submit mining job</button>
      </div>
      <div class="card">
        <div class="label mb-2">Load / query graph</div>
        <div class="flex gap-2 mb-2">
          <input id="rs-path" class="input flex-1" placeholder="graph.pt path">
          <button id="rs-load" class="btn">Load</button>
          <button id="rs-stats" class="btn">Stats</button>
        </div>
        <div class="flex gap-2">
          <input id="rs-facts" class="input flex-1" placeholder="facts, comma-separated">
          <button id="rs-apply" class="btn btn-primary">Apply</button>
        </div>
        <pre id="rs-results" class="pre mt-2 hidden"></pre>
      </div>
      <div class="card">
        <div class="label mb-2">Reasoning bias on generation</div>
        <div class="grid grid-cols-2 gap-2">
          <input id="rs-graph" class="input" placeholder="graph.pt">
          <input id="rs-mapper" class="input" placeholder="symbol_mapper.pt">
          <input id="rs-strength" class="input" type="number" step="0.01" value="0.1">
          <input id="rs-topk" class="input" type="number" value="16">
        </div>
        <div class="flex gap-2 mt-2">
          <button id="rs-load-bias" class="btn btn-primary">Load bias</button>
          <button id="rs-unload-bias" class="btn btn-danger">Unload bias</button>
        </div>
      </div>
    </div>
  `;
  document.getElementById("rs-mine").onclick = async () => {
    const cfg = {
      corpus: document.getElementById("rs-corpus").value.split(",").map(s => s.trim()).filter(Boolean),
      out: document.getElementById("rs-out").value,
      chunk_size: 500,
    };
    try {
      const r = await api("/api/train/jobs", { method: "POST",
        body: JSON.stringify({ type: "reasoning", config: cfg }) });
      toast("Reasoning job: " + r.id); refreshHeader();
    } catch (e) { toast(e.message, "error"); }
  };
  document.getElementById("rs-load").onclick = async () => {
    try {
      await api("/api/reason/load", { method: "POST",
        body: JSON.stringify({ path: document.getElementById("rs-path").value }) });
      toast("Graph loaded"); refreshHeader();
    } catch (e) { toast(e.message, "error"); }
  };
  document.getElementById("rs-stats").onclick = async () => {
    const out = document.getElementById("rs-results");
    out.classList.remove("hidden");
    try {
      const r = await api("/api/reason/stats", { method: "POST",
        body: JSON.stringify({ path: document.getElementById("rs-path").value || null, top: 10 }) });
      out.textContent = JSON.stringify(r, null, 2);
    } catch (e) { out.textContent = "ERROR: " + e.message; }
  };
  document.getElementById("rs-apply").onclick = async () => {
    const out = document.getElementById("rs-results");
    out.classList.remove("hidden");
    try {
      const r = await api("/api/reason/apply", { method: "POST",
        body: JSON.stringify({
          path: document.getElementById("rs-path").value || null,
          facts: document.getElementById("rs-facts").value.split(",").map(s => s.trim()).filter(Boolean),
        }) });
      out.textContent = JSON.stringify(r, null, 2);
    } catch (e) { out.textContent = "ERROR: " + e.message; }
  };
  document.getElementById("rs-load-bias").onclick = async () => {
    try {
      await api("/api/reason/bias/load", { method: "POST",
        body: JSON.stringify({
          graph_path: document.getElementById("rs-graph").value,
          mapper_path: document.getElementById("rs-mapper").value,
          strength: +document.getElementById("rs-strength").value,
          top_k: +document.getElementById("rs-topk").value,
        }) });
      toast("Bias loaded"); refreshHeader();
    } catch (e) { toast(e.message, "error"); }
  };
  document.getElementById("rs-unload-bias").onclick = async () => {
    try { await api("/api/reason/bias/unload", { method: "POST" });
      toast("Bias unloaded"); refreshHeader(); }
    catch (e) { toast(e.message, "error"); }
  };
}

// ─────────────────────────────────────────────────────────────
// Jobs
// ─────────────────────────────────────────────────────────────
async function renderJobs(root) {
  root.innerHTML = `
    <div class="space-y-3">
      <div class="flex items-center justify-between">
        <h2 class="text-lg font-semibold">Jobs</h2>
        <button id="jobs-refresh" class="btn">Refresh</button>
      </div>
      <div class="card">
        <table class="w-full text-sm" id="jobs-table">
          <thead><tr class="text-left text-slate-400 border-b border-slate-700">
            <th class="py-1">id</th><th>type</th><th>status</th>
            <th>created</th><th>finished</th><th>actions</th>
          </tr></thead>
          <tbody></tbody>
        </table>
      </div>
      <div class="card">
        <div class="label mb-2">Log <span id="log-job-id"></span></div>
        <pre id="job-log" class="pre min-h-[10rem]"></pre>
      </div>
    </div>
  `;
  document.getElementById("jobs-refresh").onclick = loadJobs;
  await loadJobs();
}

async function loadJobs() {
  const tbody = document.querySelector("#jobs-table tbody");
  if (!tbody) return;
  const r = await api("/api/train/jobs?limit=100");
  state.jobs = r.jobs;
  tbody.innerHTML = r.jobs.map(j => `
    <tr class="border-b border-slate-700/40">
      <td class="py-1 font-mono text-xs">${j.id}</td>
      <td>${j.type}</td>
      <td><span class="badge ${statusColor(j.status)}">${j.status}</span></td>
      <td class="text-xs">${(j.created_at || "").slice(0, 19)}</td>
      <td class="text-xs">${(j.finished_at || "").slice(0, 19) || "—"}</td>
      <td class="space-x-1">
        <button class="btn text-xs" onclick="selectJob('${j.id}')">log</button>
        ${j.status === "running" || j.status === "pending"
          ? `<button class="btn btn-danger text-xs" onclick="cancelJob('${j.id}')">cancel</button>`
          : ""}
        ${j.status === "failed" || j.status === "cancelled"
          ? `<button class="btn text-xs" onclick="retryJob('${j.id}')">retry</button>`
          : ""}
        ${["completed","failed","cancelled","timed_out"].includes(j.status)
          ? `<button class="btn btn-danger text-xs" onclick="deleteJob('${j.id}')">delete</button>`
          : ""}
      </td>
    </tr>
  `).join("");
}

function statusColor(s) {
  return {
    running: "bg-blue-700", pending: "bg-yellow-700",
    completed: "bg-emerald-700", failed: "bg-red-700",
    cancelled: "bg-slate-600", timed_out: "bg-orange-700",
  }[s] || "bg-slate-700";
}

async function selectJob(id) {
  state.selectedJob = id;
  switchTab("jobs");
  document.getElementById("log-job-id").textContent = id;
  const logEl = document.getElementById("job-log");
  logEl.textContent = "";
  if (state.jobLogStream) state.jobLogStream.close();
  const es = new EventSource(`/api/train/jobs/${id}/log/stream`);
  es.onmessage = e => {
    if (e.data === "[DONE]") { es.close(); state.jobLogStream = null; return; }
    try {
      const data = JSON.parse(e.data);
      if (data.chunk) {
        logEl.textContent += data.chunk;
        logEl.scrollTop = logEl.scrollHeight;
      }
    } catch {}
  };
  state.jobLogStream = es;
}

async function cancelJob(id) {
  await api(`/api/train/jobs/${id}/cancel`, { method: "POST" }).catch(e => toast(e.message, "error"));
  loadJobs(); refreshHeader();
}
async function retryJob(id) {
  await api(`/api/train/jobs/${id}/retry`, { method: "POST" }).catch(e => toast(e.message, "error"));
  loadJobs(); refreshHeader();
}
async function deleteJob(id) {
  if (!confirm("Delete job " + id + "?")) return;
  await api(`/api/train/jobs/${id}`, { method: "DELETE" }).catch(e => toast(e.message, "error"));
  loadJobs(); refreshHeader();
}

// ─────────────────────────────────────────────────────────────
// Files
// ─────────────────────────────────────────────────────────────
async function renderFiles(root) {
  root.innerHTML = `
    <div class="space-y-3">
      <div class="flex items-center justify-between">
        <h2 class="text-lg font-semibold">Files</h2>
        <div class="flex gap-2">
          <button id="file-up" class="btn">Upload</button>
          <button id="file-mkdir" class="btn">New dir</button>
        </div>
      </div>
      <div class="card">
        <div id="file-path" class="label mb-2">/</div>
        <table class="w-full text-sm" id="file-table">
          <tbody></tbody>
        </table>
      </div>
      <div class="card hidden" id="file-preview-card">
        <div class="label mb-2" id="file-preview-path"></div>
        <pre id="file-preview" class="pre min-h-[10rem]"></pre>
      </div>
      <input type="file" id="file-input" class="hidden">
    </div>
  `;
  await listFiles("");
  document.getElementById("file-up").onclick = () => document.getElementById("file-input").click();
  document.getElementById("file-input").onchange = async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const fd = new FormData();
    fd.append("file", f);
    fd.append("path", document.getElementById("file-path").dataset.path || "");
    try {
      await fetch("/api/files/upload", { method: "POST", body: fd });
      toast("Uploaded"); listFiles(document.getElementById("file-path").dataset.path || "");
    } catch (err) { toast(err.message, "error"); }
  };
  document.getElementById("file-mkdir").onclick = async () => {
    const name = prompt("Directory name:");
    if (!name) return;
    const base = document.getElementById("file-path").dataset.path || "";
    const path = base ? `${base}/${name}` : name;
    try { await api("/api/files/mkdir", { method: "POST",
      body: JSON.stringify({ path }) }); listFiles(base); }
    catch (e) { toast(e.message, "error"); }
  };
}

async function listFiles(path) {
  const r = await api(`/api/files/list?path=${encodeURIComponent(path)}`);
  const tbody = document.querySelector("#file-table tbody");
  const pt = document.getElementById("file-path");
  pt.dataset.path = r.path || "";
  pt.textContent = "/" + (r.path || "");
  const entries = r.entries || [];
  const parent = r.path ? r.path.split("/").slice(0, -1).join("/") : "";
  let html = "";
  if (r.path) {
    html += `<tr><td><span class="text-blue-400 cursor-pointer" onclick="listFiles('${parent}')">..</span></td><td></td></tr>`;
  }
  for (const e of entries) {
    const full = r.path ? `${r.path}/${e.name}` : e.name;
    if (e.type === "dir") {
      html += `<tr><td><span class="text-blue-400 cursor-pointer" onclick="listFiles('${full}')">📁 ${e.name}</span></td><td></td></tr>`;
    } else {
      html += `<tr><td><span class="cursor-pointer" onclick="previewFile('${full}')">📄 ${e.name}</span></td>
        <td class="text-right text-xs">${(e.size/1024).toFixed(1)} KB
        <button class="btn text-xs ml-2" onclick="delFile('${full}')">del</button></td></tr>`;
    }
  }
  tbody.innerHTML = html;
}

async function previewFile(path) {
  const card = document.getElementById("file-preview-card");
  card.classList.remove("hidden");
  document.getElementById("file-preview-path").textContent = path;
  try {
    const txt = await (await fetch(`/api/files/read?path=${encodeURIComponent(path)}`)).text();
    document.getElementById("file-preview").textContent = txt;
  } catch (e) { document.getElementById("file-preview").textContent = "ERROR: " + e.message; }
}

async function delFile(path) {
  if (!confirm("Delete " + path + "?")) return;
  await api(`/api/files/delete?path=${encodeURIComponent(path)}`, { method: "DELETE" })
    .catch(e => toast(e.message, "error"));
  listFiles(document.getElementById("file-path").dataset.path || "");
}

// ─────────────────────────────────────────────────────────────
// Admin
// ─────────────────────────────────────────────────────────────
async function renderAdmin(root) {
  const s = await api("/api/admin/state");
  const m = await api("/api/admin/metrics");
  const l = await api("/api/admin/logs");
  root.innerHTML = `
    <div class="space-y-3">
      <h2 class="text-lg font-semibold">Admin</h2>
      <div class="card">
        <div class="label mb-2">State</div>
        <pre class="pre">${JSON.stringify(s, null, 2)}</pre>
      </div>
      <div class="card">
        <div class="label mb-2">Metrics</div>
        <pre class="pre">${JSON.stringify(m, null, 2)}</pre>
      </div>
      <div class="card">
        <div class="label mb-2">Log files</div>
        <pre class="pre">${JSON.stringify(l, null, 2)}</pre>
      </div>
      <div class="card">
        <div class="label mb-2">Danger zone</div>
        <button id="adm-shutdown" class="btn btn-danger">Shutdown server</button>
      </div>
    </div>
  `;
  document.getElementById("adm-shutdown").onclick = async () => {
    if (!confirm("Really shutdown?")) return;
    await api("/api/admin/shutdown", { method: "POST",
      body: JSON.stringify({ confirm: true }) }).catch(() => {});
    toast("Shutdown requested");
  };
}

// ─────────────────────────────────────────────────────────────
// Boot
// ─────────────────────────────────────────────────────────────
(async function boot() {
  renderNav();
  await refreshHeader();
  renderTab();
  setInterval(refreshHeader, 5000);
})();

// Expose to inline onclick
window.switchTab = switchTab;
window.selectJob = selectJob;
window.cancelJob = cancelJob;
window.retryJob = retryJob;
window.deleteJob = deleteJob;
window.listFiles = listFiles;
window.previewFile = previewFile;
window.delFile = delFile;
