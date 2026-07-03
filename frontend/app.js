/* ============================================================
   MergeSE frontend logic
   - API base auto-detects (same origin) or ?api= override
   - Forms submit JSON to the backend
   - Job list refreshes every 3s; selected job streams via SSE
   ============================================================ */

const params = new URLSearchParams(location.search);
const API = (params.get("api") || "").replace(/\/$/, "") || "";  // same origin by default
const DEMO = !!params.get("demo");

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function toast(msg, kind = "") {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast " + kind;
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { t.hidden = true; }, 3800);
}

async function api(path, opts = {}) {
  const res = await fetch(API + path, {
    headers: { "content-type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    let msg = res.statusText;
    try { const j = await res.json(); msg = j.error || msg; } catch (_) {}
    throw new Error(msg);
  }
  return res.json();
}

/* ---------------------- Health ---------------------- */

async function checkHealth() {
  const el = $("#healthText");
  const line = $("#healthLine");
  if (DEMO) {
    el.textContent = "demo mode - no backend (forms are read-only)";
    line.classList.add("bad");
    document.body.classList.add("demo");
    return;
  }
  try {
    const h = await api("/api/health");
    el.textContent = `backend online · ${h.version}`;
    line.classList.remove("bad");
  } catch (e) {
    el.textContent = `backend offline (${e.message}) - append ?api=https://... to point elsewhere`;
    line.classList.add("bad");
  }
}

/* ---------------------- Tasks ---------------------- */

let TASKS = [];
async function loadTasks() {
  if (DEMO) {
    // Fallback registry for the static demo (kept in sync with mergese_tasks.py)
    TASKS = [
      { name: "clone_detection",         display: "Code clone detection",        input_kind: "pair",   metric: "binary_f1", benchmarks: ["bigclonebench","clcdsa","gptclonebench","poj-104"], csv_columns: ["code1","code2","label"] },
      { name: "vulnerability_detection", display: "Vulnerability detection",     input_kind: "single", metric: "binary_f1", benchmarks: ["devign","reveal","big-vul","d2a","draper"],         csv_columns: ["code","label"] },
      { name: "defect_prediction",       display: "Defect / bug prediction",     input_kind: "single", metric: "binary_f1", benchmarks: ["defects4j","promise","bugs.jar","codexglue-defect"], csv_columns: ["code","label"] },
      { name: "code_smell_detection",    display: "Code-smell detection",        input_kind: "single", metric: "binary_f1", benchmarks: ["mlcq","qualitas"],         csv_columns: ["code","label"] },
      { name: "commit_classification",   display: "Commit classification",       input_kind: "single", metric: "macro_f1",  benchmarks: ["commitbench"],             csv_columns: ["code","label"] },
      { name: "code_review",             display: "Code-review acceptability",   input_kind: "pair",   metric: "binary_f1", benchmarks: ["codereview"],              csv_columns: ["code1","code2","label"] },
      { name: "comment_classification",  display: "Code-comment classification", input_kind: "pair",   metric: "binary_f1", benchmarks: ["ccdetector"],              csv_columns: ["code1","code2","label"] },
      { name: "type_inference",          display: "Type inference",              input_kind: "single", metric: "macro_f1",  benchmarks: ["typilus","type4py","manytypes4py"], csv_columns: ["code","label"] },
      { name: "exception_type",          display: "Exception-type prediction",   input_kind: "single", metric: "macro_f1",  benchmarks: ["codexglue-exception"],     csv_columns: ["code","label"] },
      { name: "custom",                  display: "Custom CSV",                  input_kind: "auto",   metric: "auto",      benchmarks: [],                          csv_columns: [] },
    ];
  } else {
    try { TASKS = await api("/api/tasks"); } catch (_) { TASKS = []; }
  }
  populateTaskDropdowns();
}

function populateTaskDropdowns() {
  const eTask = $("#eTask");
  const mTask = $("#mTask");
  if (eTask) {
    eTask.innerHTML = TASKS.map(t =>
      `<option value="${t.name}"${t.name === "clone_detection" ? " selected" : ""}>${escape(t.display)}</option>`
    ).join("");
    eTask.addEventListener("change", updateDatasetHint);
    updateDatasetHint();
  }
  if (mTask) {
    mTask.innerHTML = `<option value="">- none -</option>` + TASKS
      .filter(t => t.name !== "custom")
      .map(t => `<option value="${t.name}">${escape(t.display)}</option>`)
      .join("");
  }
}

function updateDatasetHint() {
  const eTask = $("#eTask");
  const hint = $("#eDatasetHint");
  if (!eTask || !hint) return;
  const t = TASKS.find(x => x.name === eTask.value);
  if (!t) { hint.textContent = ""; return; }
  const cols = t.csv_columns && t.csv_columns.length
    ? `CSV columns expected: ${t.csv_columns.join(",")}`
    : "CSV columns expected: code,label or code1,code2,label";
  hint.textContent = cols;
}

/* ---------------------- Checkpoint library ---------------------- */
/*
 * The user uploads checkpoints once into the Library (top of the page),
 * and every form below picks from there. The library aggregates four
 * sources from /api/library:
 *
 *   - uploads      -> upload://<token>
 *   - server       -> server://<name>     (admin-mounted)
 *   - jobs         -> job://<id>/merged    (finished merge/export outputs)
 *   - suggestions  -> bare HF Hub IDs
 *
 * Plus the user can freely type any HF Hub id in a per-form text field.
 */

let LIBRARY = {
  uploads: [], server: [], jobs: [], suggestions: [],
  datasets: { bundled: [], uploads: [], server: [], suggestions: [] },
};

async function loadLibrary() {
  if (DEMO) {
    LIBRARY = {
      uploads: [], server: [], jobs: [],
      suggestions: [
        { ref: "microsoft/codebert-base",       label: "microsoft/codebert-base" },
        { ref: "microsoft/graphcodebert-base",  label: "microsoft/graphcodebert-base" },
        { ref: "microsoft/unixcoder-base",      label: "microsoft/unixcoder-base" },
      ],
      datasets: {
        bundled: [
          { ref: "bundled://bigclonebench",  label: "bigclonebench (200 rows)" },
          { ref: "bundled://clcdsa",         label: "clcdsa (200 rows)" },
          { ref: "bundled://gptclonebench",  label: "gptclonebench (200 rows)" },
        ],
        uploads: [], server: [],
        suggestions: [
          { ref: "hf-dataset://google/code_x_glue_cc_clone_detection_big_clone_bench#split=test",
            label: "HF: code_x_glue · BigCloneBench (test)" },
        ],
      },
    };
    renderLibrary(); refreshAllPickers(); return;
  }
  try {
    LIBRARY = await api("/api/library");
    // Defensive: older servers don't return `datasets`
    if (!LIBRARY.datasets) LIBRARY.datasets = { bundled: [], uploads: [], server: [], suggestions: [] };
    renderLibrary();
    refreshAllPickers();
  } catch (e) { /* offline */ }
}

function renderLibrary() {
  const u = $("#libUploadsItems"), s = $("#libServerItems"), j = $("#libJobsItems");
  $("#libUploadsCount").textContent = `(${LIBRARY.uploads.length})`;
  $("#libServerCount").textContent  = `(${LIBRARY.server.length})`;
  $("#libJobsCount").textContent    = `(${LIBRARY.jobs.length})`;

  const renderItem = (it, canDelete) => `
    <div class="lib-item" data-ref="${escape(it.ref)}">
      <div>
        <div class="label">${escape(it.label || it.ref)}</div>
        <div class="meta">${escape(it.ref)} · ${humanBytes(it.size || 0)}</div>
      </div>
      <div class="actions">
        <button type="button" class="iconbtn" data-copy="${escape(it.ref)}" title="Copy reference">⧉</button>
        ${canDelete ? `<button type="button" class="iconbtn danger" data-delete-upload="${escape(it.ref.slice(9))}" title="Delete">✕</button>` : ""}
      </div>
    </div>`;

  u.innerHTML = LIBRARY.uploads.length
    ? LIBRARY.uploads.map(it => renderItem(it, true)).join("")
    : `<div class="lib-empty">No uploads yet - drop a .zip above to add one.</div>`;

  s.innerHTML = LIBRARY.server.length
    ? LIBRARY.server.map(it => renderItem(it, false)).join("")
    : `<div class="lib-empty">No server-mounted checkpoints. Operator can set <code>MERGESE_CHECKPOINTS=/path</code>.</div>`;
  // Hide server group when empty AND no mount configured
  const serverGroup = document.querySelector('.lib-group[data-group="server"]');
  if (serverGroup) serverGroup.hidden = LIBRARY.server.length === 0;

  j.innerHTML = LIBRARY.jobs.length
    ? LIBRARY.jobs.map(it => renderItem(it, false)).join("")
    : `<div class="lib-empty">No finished merges/exports yet.</div>`;

  // -------- datasets --------
  const ds = LIBRARY.datasets;
  const dsb = $("#libDsBundledItems"), dsu = $("#libDsUploadsItems"), dss = $("#libDsServerItems");
  if (dsb) {
    $("#libDsBundledCount").textContent = `(${ds.bundled.length})`;
    $("#libDsUploadsCount").textContent = `(${ds.uploads.length})`;
    $("#libDsServerCount").textContent  = `(${ds.server.length})`;
    dsb.innerHTML = ds.bundled.length
      ? ds.bundled.map(it => renderItem(it, false)).join("")
      : `<div class="lib-empty">No bundled benchmarks on this server.</div>`;
    dsu.innerHTML = ds.uploads.length
      ? ds.uploads.map(it => `
          <div class="lib-item" data-ref="${escape(it.ref)}">
            <div>
              <div class="label">${escape(it.label || it.ref)}</div>
              <div class="meta">${escape(it.ref)} · ${humanBytes(it.size || 0)}</div>
            </div>
            <div class="actions">
              <button type="button" class="iconbtn" data-copy="${escape(it.ref)}" title="Copy reference">⧉</button>
              <button type="button" class="iconbtn danger" data-delete-dataset="${escape(it.ref.slice(10))}" title="Delete">✕</button>
            </div>
          </div>`).join("")
      : `<div class="lib-empty">No uploaded CSVs yet - click "+ Upload .csv" above.</div>`;
    dss.innerHTML = ds.server.length
      ? ds.server.map(it => renderItem(it, false)).join("")
      : `<div class="lib-empty">No server-mounted CSVs (operator can set <code>MERGESE_DATASETS=/path</code>).</div>`;
    const serverGroup = document.querySelector('.lib-group[data-group="ds-server"]');
    if (serverGroup) serverGroup.hidden = ds.server.length === 0;
  }

  // Wire copy + delete buttons
  document.querySelectorAll("[data-copy]").forEach(b => {
    b.addEventListener("click", () => {
      navigator.clipboard.writeText(b.dataset.copy).catch(() => {});
      toast(`Copied ${b.dataset.copy}`, "ok");
    });
  });
  document.querySelectorAll("[data-delete-upload]").forEach(b => {
    b.addEventListener("click", async () => {
      if (!confirm("Delete this upload? This cannot be undone.")) return;
      try {
        await api(`/api/uploads/${b.dataset.deleteUpload}`, { method: "DELETE" });
        toast("Deleted.", "ok");
        loadLibrary();
      } catch (e) { toast("Delete failed: " + e.message, "err"); }
    });
  });
  document.querySelectorAll("[data-delete-dataset]").forEach(b => {
    b.addEventListener("click", async () => {
      if (!confirm("Delete this dataset upload?")) return;
      try {
        await api(`/api/datasets/${b.dataset.deleteDataset}`, { method: "DELETE" });
        toast("Deleted.", "ok");
        loadLibrary();
      } catch (e) { toast("Delete failed: " + e.message, "err"); }
    });
  });

  // Re-apply any active search filter after the lists are rebuilt.
  applyLibraryFilters();
}

/* ---------------------- Library search / filter ---------------------- */
/*
 * Server-mounted checkpoints can number in the dozens. A live filter keeps the
 * library usable: it hides non-matching items across every group as the user
 * types, matching on both the display name and the underlying reference.
 */
const LIB_QUERY = { models: "", datasets: "" };

function filterLibraryScope(scopeSel, query) {
  const scope = document.querySelector(scopeSel);
  if (!scope) return;
  const q = (query || "").trim().toLowerCase();
  scope.querySelectorAll(".lib-group").forEach((group) => {
    if (group.hidden) return;
    const items = Array.from(group.querySelectorAll(".lib-item"));
    let visible = 0;
    items.forEach((it) => {
      const hay = ((it.dataset.ref || "") + " " + it.textContent).toLowerCase();
      const match = !q || hay.includes(q);
      it.hidden = !match;
      if (match) visible++;
    });
    let note = group.querySelector(".lib-nomatch");
    if (q && items.length && visible === 0) {
      if (!note) {
        note = document.createElement("div");
        note.className = "lib-nomatch";
        note.textContent = "No matches in this group.";
        (group.querySelector(".lib-items") || group).appendChild(note);
      }
    } else if (note) {
      note.remove();
    }
  });
}

function applyLibraryFilters() {
  filterLibraryScope("#libGroups", LIB_QUERY.models);
  filterLibraryScope("#libDatasetGroups", LIB_QUERY.datasets);
}

function bindLibrarySearch() {
  const m = document.getElementById("libSearch");
  const d = document.getElementById("libDsSearch");
  if (m) m.addEventListener("input", () => {
    LIB_QUERY.models = m.value;
    filterLibraryScope("#libGroups", m.value);
  });
  if (d) d.addEventListener("input", () => {
    LIB_QUERY.datasets = d.value;
    filterLibraryScope("#libDatasetGroups", d.value);
  });
}

/* ---------------------- Library upload (drag-and-drop) ---------------------- */

function bindLibraryUpload() {
  // Dataset upload via the small "+ Upload .csv" button in the divider row
  const dsInput = $("#libDatasetFile");
  if (dsInput) {
    dsInput.addEventListener("change", () => {
      const f = dsInput.files[0];
      if (!f) return;
      dsInput.value = "";
      uploadDataset(f);
    });
  }

  const dz = $("#libDropzone");
  const input = $("#libFile");
  const progressWrap = $("#libProgress");
  const bar = $("#libBar");
  const text = $("#libProgressText");
  if (!dz || !input) return;

  ["dragenter","dragover"].forEach(ev => dz.addEventListener(ev, (e) => {
    e.preventDefault(); dz.classList.add("drag");
  }));
  ["dragleave","drop"].forEach(ev => dz.addEventListener(ev, () => dz.classList.remove("drag")));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    const f = e.dataTransfer?.files?.[0];
    if (f) uploadFile(f);
  });
  input.addEventListener("change", () => {
    const f = input.files[0];
    if (f) uploadFile(f);
    input.value = "";
  });

  function uploadFile(f) {
    if (DEMO) { toast("Uploads are disabled in the static demo. Run locally to upload.", "err"); return; }
    if (!f.name.toLowerCase().endsWith(".zip")) {
      toast("Drop a .zip checkpoint here (use the small button below for CSV datasets).", "err");
      return;
    }
    progressWrap.hidden = false;
    bar.style.width = "0%";
    text.textContent = `uploading ${f.name} (${humanBytes(f.size)})...`;

    const xhr = new XMLHttpRequest();
    xhr.open("POST", (API || "") + "/api/uploads");
    xhr.upload.onprogress = (ev) => {
      if (ev.lengthComputable) bar.style.width = `${Math.round(100 * ev.loaded / ev.total)}%`;
    };
    xhr.onload = () => {
      let body = {}; try { body = JSON.parse(xhr.responseText); } catch (_) {}
      if (xhr.status >= 200 && xhr.status < 300 && body.token) {
        text.textContent = `✓ uploaded as ${body.ref}`;
        toast(`Added ${f.name} to the library`, "ok");
        setTimeout(() => { progressWrap.hidden = true; }, 1500);
        loadLibrary();
      } else {
        progressWrap.hidden = true;
        toast(`upload failed: ${body.error || xhr.statusText}`, "err");
      }
    };
    xhr.onerror = () => {
      progressWrap.hidden = true;
      toast("upload network error", "err");
    };
    const form = new FormData();
    form.append("file", f);
    form.append("label", f.name.replace(/\.zip$/i, ""));
    xhr.send(form);
  }
}

function uploadDataset(f) {
  if (DEMO) { toast("Uploads disabled in demo mode.", "err"); return; }
  const ok = /\.(csv|zip)$/i.test(f.name);
  if (!ok) { toast("Dataset must be a .csv (or a .zip wrapping one).", "err"); return; }
  toast(`Uploading dataset ${f.name}...`, "ok");

  const xhr = new XMLHttpRequest();
  xhr.open("POST", (API || "") + "/api/datasets");
  xhr.onload = () => {
    let body = {}; try { body = JSON.parse(xhr.responseText); } catch (_) {}
    if (xhr.status >= 200 && xhr.status < 300 && body.token) {
      toast(`Dataset ready: ${body.ref}`, "ok");
      loadLibrary();
    } else {
      toast(`dataset upload failed: ${body.error || xhr.statusText}`, "err");
    }
  };
  xhr.onerror = () => toast("dataset upload network error", "err");
  const form = new FormData();
  form.append("file", f);
  form.append("label", f.name.replace(/\.(csv|zip)$/i, ""));
  xhr.send(form);
}

/* ---------------------- Model picker (library-aware) ---------------------- */
/*
 * One model entry = one row inside a `.model-list` container. It shows:
 *   [ ▼ pick from library (or "HF id...") ] [text input when "HF id..."] [✕]
 */

function makeModelEntry(initial = {}) {
  const wrap = document.createElement("div");
  wrap.className = "model-entry";

  const sel = document.createElement("select");
  sel.className = "value";
  wrap.appendChild(sel);

  // Free-text HF input (shown only when "Type HF Hub id..." is selected)
  const hfInput = document.createElement("input");
  hfInput.type = "text"; hfInput.className = "value hf-input";
  hfInput.placeholder = "e.g. microsoft/codebert-base";
  hfInput.hidden = true;
  wrap.appendChild(hfInput);

  const remove = document.createElement("button");
  remove.type = "button"; remove.className = "remove"; remove.textContent = "Remove";
  remove.addEventListener("click", () => {
    const list = wrap.parentElement;
    const minN = parseInt(list?.dataset.min || "0", 10);
    if (list && list.querySelectorAll(".model-entry").length <= minN) {
      toast(`Need at least ${minN} model${minN === 1 ? "" : "s"}`, "err");
      return;
    }
    wrap.remove();
  });
  wrap.appendChild(remove);

  populateSelect(sel, initial);

  sel.addEventListener("change", () => {
    hfInput.hidden = sel.value !== "__hf__";
    if (sel.value === "__hf__") hfInput.focus();
  });

  return wrap;
}

function populateSelect(sel, initial = {}) {
  sel.innerHTML = "";
  // Detect the parent list kind - models (default) vs datasets
  const listEl = sel.closest(".model-list");
  const kind = listEl?.dataset.kind === "dataset" ? "dataset" : "model";

  let groups, placeholder, hfLabel, hfOptValue;
  if (kind === "dataset") {
    const ds = LIBRARY.datasets || { bundled: [], uploads: [], server: [], suggestions: [] };
    groups = [
      ["Bundled benchmarks",  ds.bundled],
      ["Your uploaded CSVs",  ds.uploads],
      ["Server-mounted CSVs", ds.server],
      ["HuggingFace datasets (test split)", ds.suggestions],
    ];
    placeholder = "- pick a dataset -";
    hfLabel = "↗ Type a HuggingFace dataset id...";
    hfOptValue = "__hfds__";
  } else {
    groups = [
      ["Your uploads",       LIBRARY.uploads],
      ["Server-mounted",     LIBRARY.server],
      ["From finished jobs", LIBRARY.jobs],
      ["HuggingFace suggestions", LIBRARY.suggestions],
    ];
    placeholder = "- pick a model -";
    hfLabel = "↗ Type a HuggingFace Hub id...";
    hfOptValue = "__hf__";
  }

  const ph = document.createElement("option");
  ph.value = ""; ph.textContent = placeholder;
  sel.appendChild(ph);
  for (const [name, items] of groups) {
    if (!items.length) continue;
    const og = document.createElement("optgroup");
    og.label = name;
    for (const it of items) {
      const o = document.createElement("option");
      o.value = it.ref;
      o.textContent = it.size ? `${it.label} (${humanBytes(it.size)})` : it.label;
      og.appendChild(o);
    }
    sel.appendChild(og);
  }
  const hf = document.createElement("option");
  hf.value = hfOptValue; hf.textContent = hfLabel;
  sel.appendChild(hf);

  // Apply initial value
  if (initial.ref) {
    const opt = Array.from(sel.options).find(o => o.value === initial.ref);
    if (opt) {
      sel.value = initial.ref;
    } else {
      sel.value = hfOptValue;
      const hfInput = sel.parentElement.querySelector(".hf-input");
      if (hfInput) {
        hfInput.value = initial.ref;
        hfInput.hidden = false;
        hfInput.placeholder = kind === "dataset"
          ? "e.g. google/code_x_glue_cc_clone_detection_big_clone_bench"
          : "e.g. microsoft/codebert-base";
      }
    }
  }
}

function refreshAllPickers() {
  document.querySelectorAll(".model-list .model-entry").forEach((entry) => {
    const sel = entry.querySelector("select.value");
    const hfInput = entry.querySelector(".hf-input");
    const current = sel.value === "__hf__" ? (hfInput?.value || "") : sel.value;
    populateSelect(sel, { ref: current });
    if (sel.value !== "__hf__") hfInput.hidden = true;
  });
}

function readModelEntry(entryEl) {
  const sel = entryEl.querySelector("select.value");
  const hfInput = entryEl.querySelector(".hf-input");
  if (!sel) return null;
  if (sel.value === "__hf__") {
    const v = (hfInput?.value || "").trim();
    return v || null;
  }
  if (sel.value === "__hfds__") {
    const v = (hfInput?.value || "").trim();
    if (!v) return null;
    return v.startsWith("hf-dataset://") ? v : `hf-dataset://${v}`;
  }
  return sel.value || null;
}

function readModelList(containerId) {
  const list = $("#" + containerId);
  if (!list) return [];
  return Array.from(list.querySelectorAll(".model-entry"))
    .map(readModelEntry)
    .filter(Boolean);
}

function setModelList(containerId, refs) {
  const list = $("#" + containerId);
  if (!list) return;
  const min = parseInt(list.dataset.min || "0", 10);
  list.innerHTML = "";
  const items = refs.length ? refs : Array(min).fill(null);
  for (const ref of items) {
    list.appendChild(makeModelEntry(ref ? { ref } : {}));
  }
}

function initModelLists() {
  document.querySelectorAll(".model-list").forEach((list) => {
    const min = parseInt(list.dataset.min || "0", 10);
    for (let i = 0; i < Math.max(min, list.querySelectorAll(".model-entry").length); i++) {
      list.appendChild(makeModelEntry({}));
    }
  });
  document.querySelectorAll("[data-add-model]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const list = $("#" + btn.dataset.addModel);
      if (!list) return;
      const max = parseInt(list.dataset.max || "0", 10);
      if (max && list.querySelectorAll(".model-entry").length >= max) {
        toast(`At most ${max} entry${max === 1 ? "" : "ies"}`, "err");
        return;
      }
      list.appendChild(makeModelEntry({}));
    });
  });
}

function humanBytes(n) {
  if (!n) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(n >= 10 ? 0 : 1)} ${units[i]}`;
}

/* Legacy compatibility: some older code paths called loadServerCheckpoints. */
async function loadServerCheckpoints() { /* now folded into loadLibrary */ }

/* ---------------------- Presets ---------------------- */

let PRESETS = null;
async function loadPresets() {
  try { PRESETS = (await api("/api/presets")).presets || []; }
  catch (_) { PRESETS = []; }
}

function applyPreset(kind) {
  if (!PRESETS) return;
  const p = PRESETS.find(x => x.kind === kind);
  if (!p) { toast("No preset for " + kind, "err"); return; }
  const b = p.body || {};
  if (kind === "inspect") {
    setModelList("iModelsList", b.models || []);
    setModelList("iBaseList",   b.base ? [b.base] : []);
  } else if (kind === "merge") {
    setModelList("mModelsList", b.models || []);
    setModelList("mBaseList",   b.base ? [b.base] : []);
    $("#mMethod").value = b.method || "ties";
    // Apply method-aware enable/disable + hints, but don't overwrite the
    // preset's explicit trim/drop values - set them after.
    applyMethodDefaults($("#mMethod").value, { overwrite: false });
    $("#mTrim").value   = b.trim_percentile ?? 20;
    $("#mDrop").value   = b.drop_rate ?? 0.3;
    $("#mWeights").value= b.weights || "";
    $("#mSeed").value   = b.seed ?? 42;
    if ($("#mTask"))  $("#mTask").value  = b.task || "";
    if ($("#mHeads")) {
      $("#mHeads").value = b.encoder_only === true ? "encoder_only"
                         : b.encoder_only === false ? "include" : "auto";
    }
  } else if (kind === "evaluate") {
    setModelList("eModelList", b.model ? [b.model] : []);
    // Pick a sensible default dataset for the task
    const defaultDs = b.dataset_ref
                     || (b.dataset && `bundled://${b.dataset}`)
                     || (b.task === "clone_detection" ? "bundled://bigclonebench" : "");
    setModelList("eDatasetList", defaultDs ? [defaultDs] : []);
    $("#eTask").value     = b.task || "clone_detection";
    $("#eBatch").value    = b.batch_size ?? 32;
    $("#eMaxLen").value   = b.max_length ?? 512;
    $("#eLimit").value    = b.limit ?? 0;
    if ($("#eMetric")) $("#eMetric").value = b.metric || "auto";
    updateDatasetHint();
  }
  toast(`Loaded preset: ${p.name}`, "ok");
}

/* ---------------------- Submit handlers ---------------------- */

async function submit(path, body, opts) {
  if (DEMO) {
    toast("Demo mode - backend disabled. Try locally with `python server/app.py`.", "err");
    return null;
  }
  try {
    const r = await api(path, { method: "POST", body: JSON.stringify(body) });
    toast(`Started ${opts.kind} job ${r.job_id}`, "ok");
    refreshJobs(true).then(() => selectJob(r.job_id));
    // Smooth-scroll the user down to the Jobs section so they can see the
    // live log instead of staring at the empty form they just submitted.
    const jobs = $("#jobs");
    if (jobs) jobs.scrollIntoView({ behavior: "smooth", block: "start" });
    // When the job ends, also refresh the library so its merged output shows up
    // under "From finished jobs". A 1.5s delay lets the artifacts settle.
    if (opts.kind === "merge" || opts.kind === "export") {
      setTimeout(() => loadLibrary(), 1500);
    }
    return r;
  } catch (e) {
    toast(`${opts.kind} failed: ${e.message}`, "err");
    return null;
  }
}

/* ---------------------- Method-aware defaults ---------------------- */
/*
 * Switching merge method resets trim / drop / WUDI fields to that method's
 * recommended values. The reset fires only on an actual method change, not on
 * every keystroke, so a user's manual edits are preserved.
 *
 * Recommended values per method:
 *   ties:      trim 20%
 *   dare-ties: trim 20%, drop 0.3
 *   wudi:      300 Adam steps per linear layer, lr 1e-5
 *   average:   no trim/drop
 */
const METHOD_DEFAULTS = {
  "ties":      { trim: 20, drop: 0.0, trimEnabled: true,  dropEnabled: false,
                 wudiEnabled: false,
                 trimHint: "TIES default: 20 - drops the smallest 20% of |Δ|",
                 dropHint: "not used by TIES (only DARE-TIES)",
                 wudiHint: "not used by TIES (only WUDI)" },
  "dare-ties": { trim: 20, drop: 0.3, trimEnabled: true,  dropEnabled: true,
                 wudiEnabled: false,
                 trimHint: "DARE-TIES default: 20",
                 dropHint: "DARE-TIES default: 0.3 - drop 30% of task-vector entries before TIES",
                 wudiHint: "not used by DARE-TIES (only WUDI)" },
  "wudi":      { trim: 0,  drop: 0.0, trimEnabled: false, dropEnabled: false,
                 wudiEnabled: true,
                 trimHint: "ignored - WUDI does not trim",
                 dropHint: "ignored - WUDI does not use DARE",
                 wudiHint: "WUDI defaults: 300 steps, lr 1e-5 per linear layer" },
  "average":   { trim: 0,  drop: 0.0, trimEnabled: false, dropEnabled: false,
                 wudiEnabled: false,
                 trimHint: "ignored - averaging keeps all entries",
                 dropHint: "ignored - averaging doesn't use DARE",
                 wudiHint: "not used by averaging (only WUDI)" },
};

function applyMethodDefaults(method, opts = {}) {
  const d = METHOD_DEFAULTS[method] || METHOD_DEFAULTS.ties;
  const trim = $("#mTrim"), drop = $("#mDrop");
  const wudiSteps = $("#mWudiSteps"), wudiLr = $("#mWudiLr");
  // overwrite=true on method-change events; overwrite=false on initial render
  // (we still want the field disabled state to reflect the method, but we
  // don't want to clobber a value the user just typed).
  if (opts.overwrite !== false) {
    trim.value = d.trim;
    drop.value = d.drop;
  }
  trim.disabled = !d.trimEnabled;
  drop.disabled = !d.dropEnabled;
  $("#mTrimHint").textContent = d.trimHint;
  $("#mDropHint").textContent = d.dropHint;
  if (wudiSteps && wudiLr) {
    wudiSteps.disabled = !d.wudiEnabled;
    wudiLr.disabled = !d.wudiEnabled;
    $("#mWudiStepsHint").textContent = d.wudiHint;
    $("#mWudiLrHint").textContent = d.wudiHint;
  }
}

function bindForms() {
  $("#inspectForm").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const models = readModelList("iModelsList");
    if (models.length < 2) { toast("Add at least 2 models to inspect.", "err"); return; }
    const base = readModelList("iBaseList")[0] || null;
    submit("/api/inspect", { models, base }, { kind: "inspect" });
  });

  $("#mergeForm").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const models = readModelList("mModelsList");
    const base = readModelList("mBaseList")[0];
    if (models.length < 2) { toast("Add at least 2 fine-tuned models.", "err"); return; }
    if (!base) { toast("Set a base checkpoint.", "err"); return; }
    const heads = $("#mHeads") ? $("#mHeads").value : "auto";
    const encoder_only = heads === "encoder_only" ? true
                       : heads === "include"      ? false
                       : null;
    submit("/api/merge", {
      models, base,
      method: $("#mMethod").value,
      trim_percentile: parseFloat($("#mTrim").value),
      drop_rate: parseFloat($("#mDrop").value),
      wudi_steps: parseInt($("#mWudiSteps").value, 10),
      wudi_lr: parseFloat($("#mWudiLr").value),
      weights: $("#mWeights").value.trim() || null,
      seed: parseInt($("#mSeed").value, 10),
      task: $("#mTask") ? ($("#mTask").value || null) : null,
      encoder_only,
    }, { kind: "merge" });
  });

  $("#evalForm").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const model = readModelList("eModelList")[0];
    if (!model) { toast("Choose a model to evaluate.", "err"); return; }
    const dataset_ref = readModelList("eDatasetList")[0];
    if (!dataset_ref) { toast("Choose a dataset to evaluate against.", "err"); return; }
    submit("/api/evaluate", {
      model,
      task: $("#eTask").value,
      dataset_ref,
      batch_size: parseInt($("#eBatch").value, 10),
      max_length: parseInt($("#eMaxLen").value, 10),
      limit: parseInt($("#eLimit").value, 10),
      metric: $("#eMetric") ? $("#eMetric").value : "auto",
    }, { kind: "evaluate" });
  });

  $("#exportForm").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const model = readModelList("xModelList")[0];
    if (!model) { toast("Choose a model to export.", "err"); return; }
    submit("/api/export", {
      model,
      format: $("#xFormat").value,
      max_length: parseInt($("#xMaxLen").value, 10),
    }, { kind: "export" });
  });

  $$("button[data-preset]").forEach(btn => {
    btn.addEventListener("click", () => applyPreset(btn.dataset.preset));
  });

  $("#refreshJobs").addEventListener("click", () => refreshJobs(true));

  // Reactive method defaults - switching method overwrites trim/drop with
  // method-appropriate values. Users can still tweak afterwards.
  const methodSel = $("#mMethod");
  if (methodSel) {
    applyMethodDefaults(methodSel.value, { overwrite: false });
    methodSel.addEventListener("change", () => applyMethodDefaults(methodSel.value));
  }
}

/* ---------------------- Jobs panel ---------------------- */

const JOBS = new Map();
let CURRENT_JOB = null;
let CURRENT_SSE = null;

function formatTime(ts) {
  if (!ts) return "-";
  const d = new Date(ts * 1000);
  return d.toLocaleString();
}

function badgeFor(status) {
  return `<span class="badge b-${status}">${status}</span>`;
}

async function refreshJobs(force = false) {
  if (DEMO) { renderJobs([]); return; }
  try {
    const r = await api("/api/jobs");
    renderJobs(r.jobs || []);
  } catch (e) {
    if (force) toast(`couldn't refresh jobs: ${e.message}`, "err");
  }
}

function renderJobs(jobs) {
  JOBS.clear();
  jobs.forEach(j => JOBS.set(j.id, j));
  const list = $("#jobList");
  list.innerHTML = "";
  if (jobs.length === 0) {
    list.innerHTML = `<li><div class="empty" style="color:var(--muted); padding:14px;">No jobs yet - submit a task above.</div></li>`;
  } else {
    for (const j of jobs) {
      const li = document.createElement("li");
      li.dataset.id = j.id;
      if (CURRENT_JOB === j.id) li.classList.add("active");
      li.innerHTML = `
        <div class="row">
          <span class="jkind">${j.kind}</span>
          ${badgeFor(j.status)}
        </div>
        <div class="row">
          <span class="jid">${j.id}</span>
          <span class="jtime">${formatTime(j.started_at)}</span>
        </div>`;
      li.addEventListener("click", () => selectJob(j.id));
      list.appendChild(li);
    }
  }
  // Keep the open detail panel's badge / finished-at / download button in sync
  refreshJobDetailMeta();
  // Update the active-jobs pill in the topbar
  updateRunningPill(jobs);
}

function renderJobMeta(job) {
  /* Builds the .meta line for the detail panel. Called on selectJob AND on
   * every job-list tick / SSE end, so the badge / finished-at / download
   * button update live. */
  const showDownload = job.status === "done" && (job.kind === "merge" || job.kind === "export");
  return `
    <span><strong>${job.kind}</strong></span>
    <span class="muted">id <code>${job.id}</code></span>
    <span>${badgeFor(job.status)}</span>
    <span class="muted">started ${formatTime(job.started_at)}</span>
    ${job.finished_at ? `<span class="muted">finished ${formatTime(job.finished_at)}</span>` : ``}
    ${job.exit_code != null ? `<span class="muted">exit ${job.exit_code}</span>` : ``}
    ${job.status === "running" ? `<button class="btn btn-ghost" id="cancelBtn" style="margin-left:auto;">Cancel</button>` : ``}
    ${showDownload ? `<a class="download-btn" href="${API}/api/jobs/${job.id}/download" download>⬇ Download ${job.kind === "merge" ? "merged model" : "export"} (.zip)</a>` : ``}
  `;
}

function refreshJobDetailMeta() {
  if (!CURRENT_JOB) return;
  const job = JOBS.get(CURRENT_JOB);
  if (!job) return;
  const meta = document.querySelector("#jobDetail .meta");
  if (!meta) return;
  // Snapshot what's there now so we only re-render if status changed.
  const newMeta = renderJobMeta(job);
  if (meta.dataset.snapshot !== newMeta) {
    meta.innerHTML = newMeta;
    meta.dataset.snapshot = newMeta;
    wireCancelBtn(job);
  }
}

function updateRunningPill(jobs) {
  const pill = $("#runningPill");
  const text = $("#runningPillText");
  if (!pill || !text) return;
  const running = jobs.filter(j => j.status === "running" || j.status === "pending");
  if (running.length === 0) {
    pill.hidden = true;
    document.title = "MergeSE";
  } else {
    pill.hidden = false;
    text.textContent = `${running.length} running`;
    // Tab title gets a hint too - useful when the user is in another tab
    document.title = `(${running.length}) MergeSE`;
  }
}

function wireCancelBtn(job) {
  const c = $("#cancelBtn");
  if (!c) return;
  c.addEventListener("click", async () => {
    try { await api(`/api/jobs/${job.id}/cancel`, { method: "POST" }); }
    catch (e) { toast("cancel failed: " + e.message, "err"); }
    refreshJobs(true);
  });
}

async function selectJob(id) {
  CURRENT_JOB = id;
  $$("#jobList li").forEach(li => li.classList.toggle("active", li.dataset.id === id));

  const detail = $("#jobDetail");
  const job = JOBS.get(id);
  if (!job) { detail.innerHTML = `<div class="empty">Job not found.</div>`; return; }

  const metaHtml = renderJobMeta(job);
  detail.innerHTML = `
    <div class="meta" data-snapshot="${escape(metaHtml)}">${metaHtml}</div>
    <pre class="log" id="logView">loading...</pre>
    <div class="result" id="resultView" hidden></div>
  `;
  // The data-snapshot attribute is just a cheap dirty-check string; the meta is
  // re-rendered by refreshJobDetailMeta() whenever the snapshot differs.
  document.querySelector("#jobDetail .meta").dataset.snapshot = metaHtml;
  wireCancelBtn(job);

  if (CURRENT_SSE) { try { CURRENT_SSE.close(); } catch(_) {} CURRENT_SSE = null; }
  const log = $("#logView");
  log.textContent = "";

  // For active jobs, stream via SSE; for finished jobs, fetch full log once.
  if (job.status === "running" || job.status === "pending") {
    streamJob(job, log);
  } else {
    try {
      const txt = await fetch(API + `/api/jobs/${job.id}/log`).then(r => r.text());
      log.textContent = txt || "(no output)";
      log.scrollTop = log.scrollHeight;
    } catch (e) { log.textContent = `failed to load log: ${e.message}`; }
  }

  // Fetch a structured result if available
  try {
    const r = await api(`/api/jobs/${job.id}/result`);
    if (r && r.result) showResult(r.result);
  } catch (_) {}
}

function streamJob(job, logEl) {
  const url = API + `/api/jobs/${job.id}/stream`;
  const es = new EventSource(url);
  CURRENT_SSE = es;
  es.onmessage = (ev) => {
    try {
      const d = JSON.parse(ev.data);
      if (typeof d.line === "string") {
        logEl.textContent += d.line + "\n";
        logEl.scrollTop = logEl.scrollHeight;
      }
    } catch (_) {}
  };
  es.addEventListener("end", async (ev) => {
    es.close();
    CURRENT_SSE = null;
    // refreshJobs() repopulates JOBS and calls refreshJobDetailMeta() so the
    // detail panel's badge flips immediately (without waiting for the next tick).
    await refreshJobs();
    loadLibrary();   // pull in the newly-finished merge/export output
    try {
      const r = await api(`/api/jobs/${job.id}/result`);
      if (r && r.result) showResult(r.result);
    } catch (_) {}
  });
  es.onerror = () => {
    // Browsers spam onerror when streams close - only show a toast on disconnect.
    es.close();
  };
}

function showResult(result) {
  const el = $("#resultView");
  if (!el) return;
  el.hidden = false;

  // Render flat tables of metrics or compact JSON for nested
  if (result.verdict) {
    el.innerHTML = renderInspectResult(result);
  } else if ("f1" in result || "accuracy" in result) {
    el.innerHTML = renderMetrics(result);
  } else {
    el.innerHTML = `<pre style="margin:0;font-size:12px;">${escape(JSON.stringify(result, null, 2))}</pre>`;
  }
}

function renderInspectResult(r) {
  const rows = (r.pairs || []).map(p => `
    <tr><td class="k">${escape(p.a)} <-> ${escape(p.b)}</td>
        <td>cos ${p.cosine.toFixed(4)} · sign ${(p.sign_agreement*100).toFixed(1)}%</td></tr>`).join("");
  return `
    <table>
      <tr><td class="k">verdict</td><td><span class="badge b-${r.verdict === 'COMPATIBLE' ? 'done' : r.verdict === 'RISKY' ? 'pending' : 'error'}">${r.verdict}</span></td></tr>
      <tr><td class="k">base</td><td>${escape(r.base || '-')}</td></tr>
      ${rows}
    </table>`;
}
function renderMetrics(m) {
  const keys = ["accuracy", "precision", "recall", "f1", "n_examples", "elapsed_sec", "device", "dataset"];
  const rows = keys.filter(k => k in m).map(k =>
    `<tr><td class="k">${k}</td><td>${typeof m[k] === "number" ? m[k].toFixed(4) : escape(String(m[k]))}</td></tr>`
  ).join("");
  return `<table>${rows}</table>`;
}
function escape(s) { return String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]); }

/* ---------------------- Init ---------------------- */

document.addEventListener("DOMContentLoaded", async () => {
  await Promise.all([loadTasks(), loadLibrary()]);
  initModelLists();
  bindForms();
  bindLibraryUpload();
  bindLibrarySearch();
  await Promise.all([checkHealth(), loadPresets()]);
  refreshJobs();
  // refresh jobs every 3s; refresh library every 15s (cheaper, picks up finished
  // jobs so the "From finished jobs" group stays current)
  setInterval(() => refreshJobs(false), 3000);
  setInterval(() => loadLibrary(),    15000);
});
