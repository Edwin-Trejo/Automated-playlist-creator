const { moods, full } = window.APP;
const FEATURES = Object.keys(full);
const state = { mood: null, tracks: [] };
const $ = (id) => document.getElementById(id);

async function api(url, body) {
  const opts = body === undefined
    ? { method: "GET" }
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const res = await fetch(url, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

// ---- sliders ----
function buildSliders() {
  const box = $("sliders");
  for (const f of FEATURES) {
    const [lo, hi] = full[f];
    const step = f === "tempo" ? 1 : 0.01;
    box.insertAdjacentHTML("beforeend", `
      <div class="slider" data-feature="${f}">
        <span class="label">${f}</span>
        <input type="range" class="min" min="${lo}" max="${hi}" step="${step}" value="${lo}" aria-label="${f} minimum">
        <input type="range" class="max" min="${lo}" max="${hi}" step="${step}" value="${hi}" aria-label="${f} maximum">
        <span class="value"></span>
      </div>`);
  }
  box.addEventListener("input", (e) => {
    const row = e.target.closest(".slider");
    if (row) updateLabel(row);
  });
  document.querySelectorAll(".slider").forEach(updateLabel);
}

function updateLabel(row) {
  const min = row.querySelector(".min");
  const max = row.querySelector(".max");
  if (Number(min.value) > Number(max.value)) [min.value, max.value] = [max.value, min.value];
  row.querySelector(".value").textContent = `${min.value} – ${max.value}`;
}

function setSliders(ranges) {
  document.querySelectorAll(".slider").forEach((row) => {
    const [lo, hi] = ranges[row.dataset.feature] || full[row.dataset.feature];
    row.querySelector(".min").value = lo;
    row.querySelector(".max").value = hi;
    updateLabel(row);
  });
}

function readSliders() {
  const out = {};
  document.querySelectorAll(".slider").forEach((row) => {
    out[row.dataset.feature] = [Number(row.querySelector(".min").value), Number(row.querySelector(".max").value)];
  });
  return out;
}

// ---- mood chips ----
function selectMood(mood) {
  state.mood = state.mood === mood ? null : mood;
  document.querySelectorAll("#moods .chip").forEach((b) => b.classList.toggle("active", b.dataset.mood === state.mood));
  setSliders(state.mood ? moods[state.mood] : full);
}

// ---- sync ----
function showSyncError(msg) {
  $("sync-error").textContent = msg;
  $("sync-error").hidden = false;
  $("sync-btn").disabled = false;
}

async function startSync() {
  $("sync-error").hidden = true;
  $("sync-btn").disabled = true;
  $("progress").hidden = false;
  $("progress-text").textContent = "Fetching your liked songs…";
  try {
    await api("/sync", {});
    pollSync();
  } catch (e) {
    showSyncError(e.message);
  }
}

async function pollSync() {
  let p;
  try {
    p = await api("/sync/status");
  } catch (e) {
    return showSyncError(e.message);
  }
  $("progress-bar").max = p.total || 1;
  $("progress-bar").value = p.done;
  $("progress-text").textContent =
    `${p.done}/${p.total} · ${p.analyzed} analyzed · ${p.no_preview} no preview · ${p.failed} failed`;
  $("counts").textContent = `${p.counts.analyzed} analyzed · ${p.counts.no_preview} without preview`;
  if (p.running) {
    $("sync-btn").disabled = true;
    $("progress").hidden = false;
    setTimeout(pollSync, 1500);
  } else {
    $("sync-btn").disabled = false;
    if (p.error) showSyncError(p.error);
    else if (p.total) $("progress-text").textContent += " · done (reload to refresh genres)";
  }
}

// ---- generate + results ----
const fmt = (v, digits = 2) => (v === null || v === undefined ? "–" : Number(v).toFixed(digits));

function renderResults(data) {
  $("results").hidden = false;
  $("save-result").textContent = "";
  const body = $("results-body");
  body.replaceChildren();
  state.tracks.forEach((t, i) => {
    const tr = document.createElement("tr");
    const cells = [t.name, t.artists, t.genre, fmt(t.danceability), fmt(t.energy), fmt(t.valence), fmt(t.tempo, 0)];
    for (const c of cells) {
      const td = document.createElement("td");
      td.textContent = c;
      tr.appendChild(td);
    }
    const td = document.createElement("td");
    const rm = document.createElement("button");
    rm.className = "icon";
    rm.textContent = "×";
    rm.title = "Remove from playlist";
    rm.addEventListener("click", () => {
      state.tracks.splice(i, 1);
      renderResults({ matched: state.tracks.length, requested: data.requested });
    });
    td.appendChild(rm);
    tr.appendChild(td);
    body.appendChild(tr);
  });
  $("results-note").textContent = data.matched < data.requested
    ? `Only ${state.tracks.length} songs match. Try widening the sliders or removing genre filters.`
    : `${state.tracks.length} songs`;
  $("save-btn").disabled = state.tracks.length === 0;
}

async function generate() {
  const genres = [...document.querySelectorAll("#genres input:checked")].map((i) => i.value);
  $("generate-btn").disabled = true;
  try {
    const data = await api("/generate", { mood: state.mood, overrides: readSliders(), genres, size: Number($("size").value) });
    state.tracks = data.tracks;
    renderResults(data);
  } catch (e) {
    $("results").hidden = false;
    $("results-note").textContent = e.message;
  } finally {
    $("generate-btn").disabled = false;
  }
}

async function save() {
  $("save-btn").disabled = true;
  const out = $("save-result");
  out.textContent = "Naming your playlist and saving…";
  try {
    const d = await api("/playlist/save", { mood: state.mood, track_ids: state.tracks.map((t) => t.spotify_id) });
    out.replaceChildren();
    const a = document.createElement("a");
    a.href = d.url;
    a.target = "_blank";
    a.rel = "noopener";
    a.textContent = `Open “${d.name}” in Spotify`;
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = d.description + (d.source === "template" ? " (Ollama unavailable: used a default name)" : "");
    out.append(a, p);
  } catch (e) {
    out.textContent = e.message;
  } finally {
    $("save-btn").disabled = false;
  }
}

// ---- wire up ----
buildSliders();
document.querySelectorAll("#moods .chip").forEach((b) => b.addEventListener("click", () => selectMood(b.dataset.mood)));
$("sync-btn").addEventListener("click", startSync);
$("generate-btn").addEventListener("click", generate);
$("save-btn").addEventListener("click", save);
pollSync(); // resume progress display if a sync is already running
