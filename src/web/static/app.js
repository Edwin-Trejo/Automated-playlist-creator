const FEATURES = window.APP.features;
const state = { mood: null, tracks: [], genres: [], savedMood: null, target: null };
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

// ---- sliders: one per feature, from -1 (less) to +1 (more); the middle keeps the mood as it is ----
const SLIDER_TEXT = {
  danceability: ["Danceability", "less", "more"],
  energy: ["Energy", "less", "more"],
  valence: ["Happiness", "sadder", "happier"],
  tempo: ["Tempo", "slower", "faster"],
};

function buildSliders() {
  const box = $("sliders");
  for (const f of FEATURES) {
    const [label, less, more] = SLIDER_TEXT[f] || [f, "less", "more"];
    box.insertAdjacentHTML("beforeend", `
      <div class="slider" data-feature="${f}" data-less="${less}" data-more="${more}">
        <span class="label">${label}</span>
        <span class="end less">${less}</span>
        <input type="range" min="-1" max="1" step="0.05" value="0" aria-label="${label}: ${less} to ${more}">
        <span class="end more">${more}</span>
        <span class="value"></span>
        <button type="button" class="icon reset" title="Reset ${label}" aria-label="Reset ${label}">×</button>
      </div>`);
  }
  box.addEventListener("input", (e) => {
    const row = e.target.closest(".slider");
    if (row) updateLabel(row);
  });
  box.addEventListener("click", (e) => {
    if (!e.target.classList.contains("reset")) return;
    const row = e.target.closest(".slider");
    row.querySelector("input").value = 0;
    updateLabel(row);
  });
  resetSliders();
}

function updateLabel(row) {
  const v = Number(row.querySelector("input").value);
  const size = Math.abs(v);
  const word = v < 0 ? row.dataset.less : row.dataset.more;
  const text = size < 0.05 ? (state.mood ? "as the mood" : "any")
    : size < 0.4 ? `a bit ${word}` : size < 0.8 ? word : `much ${word}`;
  row.querySelector(".value").textContent = text;
  row.classList.toggle("changed", size >= 0.05);
}

function resetSliders() {
  document.querySelectorAll(".slider").forEach((row) => {
    row.querySelector("input").value = 0;
    updateLabel(row);
  });
}

function readSliders() {
  const out = {};
  document.querySelectorAll(".slider").forEach((row) => {
    const v = Number(row.querySelector("input").value);
    if (v) out[row.dataset.feature] = v;
  });
  return out;
}

// ---- mood chips ----
function selectMood(mood) {
  state.mood = state.mood === mood ? null : mood;
  document.querySelectorAll("#moods .chip").forEach((b) => b.classList.toggle("active", b.dataset.mood === state.mood));
  resetSliders();
}

// ---- sync ----
function showSyncError(msg) {
  $("sync-error").textContent = msg;
  $("sync-error").hidden = false;
  $("sync-btn").disabled = false;
}

async function startSync() {
  const sources = selectedSources();
  if (!sources.length) return showSyncError("Select at least one source to sync.");
  $("sync-error").hidden = true;
  $("sync-btn").disabled = true;
  $("progress").hidden = false;
  $("progress-text").textContent = "Fetching your songs…";
  try {
    await api("/sync", { sources });
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
  $("counts").textContent = `${p.counts.analyzed} analyzed · ${p.counts.no_preview} without preview (all sources)`;
  if (p.running) {
    $("sync-btn").disabled = true;
    $("progress").hidden = false;
    setTimeout(pollSync, 1500);
  } else {
    $("sync-btn").disabled = false;
    if (p.error) showSyncError(p.error);
    else if (p.total) $("progress-text").textContent += " · done (reload to refresh genres)";
    if (state.syncing) loadSources(); // refresh the per-source song counts
  }
  state.syncing = p.running;
}

// ---- sources: Liked Songs + the user's own playlists; the selection is remembered in this browser ----
const SOURCES_KEY = "playlist-creator.sources";

function savedSources() {
  try {
    const saved = JSON.parse(localStorage.getItem(SOURCES_KEY));
    if (Array.isArray(saved)) return saved;
  } catch (e) { /* storage unavailable: use the default */ }
  return ["liked"];
}

function selectedSources() {
  const boxes = [...document.querySelectorAll("#sources input")];
  return boxes.length ? boxes.filter((b) => b.checked).map((b) => b.value) : savedSources();
}

function sourceLabel(s) {
  if (!s.total) return "not synced yet";
  return `${s.analyzed} of ${s.total} analyzed`;
}

function renderSources(list) {
  const chosen = new Set(savedSources());
  const box = $("sources");
  box.replaceChildren();
  for (const s of list) {
    const label = document.createElement("label");
    label.className = "chip";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = s.id;
    input.checked = chosen.has(s.id);
    const count = document.createElement("span");
    count.className = "muted";
    count.textContent = ` · ${sourceLabel(s)}`;
    label.append(input, ` ${s.name}`, count);
    box.appendChild(label);
  }
}

async function loadSources() {
  try {
    const data = await api("/sources");
    $("sources-error").hidden = true;
    renderSources(data.sources);
  } catch (e) {
    renderSources([{ id: "liked", name: "Liked Songs", total: 0, analyzed: 0 }]);
    $("sources-error").textContent = `Couldn't load your playlists (${e.message}). Log out and log in again to allow it.`;
    $("sources-error").hidden = false;
  }
}

$("sources").addEventListener("change", () => {
  try { localStorage.setItem(SOURCES_KEY, JSON.stringify(selectedSources())); } catch (e) { /* not remembered */ }
});

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
  const t = state.target;
  const adding = t && t.exists && !$("new-playlist").checked;
  if (adding && state.tracks.length === 0) {
    $("results-note").textContent = `Every matching song is already in “${t.name}”. Like more songs and sync, or adjust the filters.`;
  } else {
    const kind = adding ? "new songs" : "songs";
    $("results-note").textContent = data.matched < data.requested
      ? `Only ${state.tracks.length} ${kind} match. Try moving the sliders or removing genre filters.`
      : `${state.tracks.length} ${kind}`;
  }
  $("save-btn").disabled = state.tracks.length === 0;
}

// Explains where Save goes: the existing playlist for this mood (+ genres), or a new one.
function renderTarget() {
  const t = state.target;
  if (!t) return;
  const hint = $("target-hint");
  if ($("new-playlist").checked) {
    $("save-btn").textContent = `Create new “${t.name}”`;
    hint.textContent = "A separate new playlist is created. Future songs are added to it.";
  } else if (t.exists) {
    $("save-btn").textContent = `Add to “${t.name}”`;
    hint.replaceChildren("If not selected, songs are added to your existing ");
    const a = document.createElement("a");
    a.href = t.url;
    a.target = "_blank";
    a.rel = "noopener";
    a.textContent = `“${t.name}”`;
    hint.append(a, ` playlist (${t.already_in} songs). Songs already in it are left out.`);
  } else {
    $("save-btn").textContent = `Create “${t.name}”`;
    hint.textContent = `You don't have a “${t.name}” playlist yet, so Save creates it. Next time, new songs are added to it.`;
  }
}

async function generate() {
  const genres = [...document.querySelectorAll("#genres input:checked")].map((i) => i.value);
  const sources = selectedSources();
  if (!sources.length) {
    $("results").hidden = false;
    $("results-note").textContent = "Select at least one source under Your library.";
    return;
  }
  $("generate-btn").disabled = true;
  try {
    const data = await api("/generate", {
      mood: state.mood, adjust: readSliders(), genres, sources, size: Number($("size").value),
      new_playlist: $("new-playlist").checked,
    });
    state.tracks = data.tracks;
    state.genres = genres;
    state.savedMood = state.mood;
    state.target = data.target;
    renderTarget();
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
  out.textContent = "Saving…";
  try {
    const d = await api("/playlist/save", {
      mood: state.savedMood, genres: state.genres, track_ids: state.tracks.map((t) => t.spotify_id),
      new_playlist: $("new-playlist").checked,
    });
    out.replaceChildren();
    const p = document.createElement("p");
    p.textContent = d.created
      ? `Created “${d.name}” with ${d.added} songs. `
      : `Added ${d.added} new songs to “${d.name}” (${d.total} total). `;
    const a = document.createElement("a");
    a.href = d.url;
    a.target = "_blank";
    a.rel = "noopener";
    a.textContent = "Open in Spotify";
    p.append(a);
    out.append(p);
    // That playlist now exists and is where the next save goes.
    state.target = { name: d.name, url: d.url, exists: true, already_in: d.total };
    $("new-playlist").checked = false;
    renderTarget();
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
// Add mode leaves out songs already in the playlist, so switching modes needs a fresh preview.
$("new-playlist").addEventListener("change", () => { if (!$("results").hidden) generate(); });
loadSources();
pollSync(); // resume progress display if a sync is already running
