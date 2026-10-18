const state = { mode: null, mood: null, tracks: [], saved: null, target: null, sources: [], showAllSources: false };
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

// ---- "Make a playlist by": Mood or Genre; the next options appear once one is picked ----
const MODE_KEY = "playlist-creator.mode";

function selectMode(mode, remember = true) {
  state.mode = mode;
  document.querySelectorAll("#modes .chip").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  $("mood-step").hidden = mode !== "mood";
  $("genre-step").hidden = mode !== "genre";
  $("generate-row").hidden = !mode;
  $("results").hidden = true; // a preview from the other mode would save to the wrong playlist
  if (remember) {
    try { localStorage.setItem(MODE_KEY, mode); } catch (e) { /* not remembered */ }
  }
}

function savedMode() {
  try {
    const mode = localStorage.getItem(MODE_KEY);
    if (mode === "mood" || mode === "genre") return mode;
  } catch (e) { /* storage unavailable */ }
  return null;
}

function selectMood(mood) {
  state.mood = state.mood === mood ? null : mood;
  document.querySelectorAll("#moods .chip").forEach((b) => b.classList.toggle("active", b.dataset.mood === state.mood));
}

// What the current picks ask for: a mood playlist or a genre playlist, never both.
function playlistChoice() {
  if (state.mode === "mood") return { mode: "mood", mood: state.mood, genres: [] };
  const genres = [...document.querySelectorAll("#genres input:checked")].map((i) => i.value);
  return { mode: "genre", mood: null, genres };
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
    `${p.done}/${p.total} · ${p.analyzed} analyzed · ${p.from_dataset} from Spotify data · ` +
    `${p.estimated} estimated from artist · ${p.no_preview} no preview · ${p.failed} failed`;
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

const RECENT_PLAYLISTS = 5;

function sourceCover(s) {
  if (s.image) {
    const img = document.createElement("img");
    img.className = "cover";
    img.src = s.image;
    img.alt = "";
    img.loading = "lazy";
    return img;
  }
  const tile = document.createElement("span");
  tile.className = s.id === "liked" ? "cover liked" : "cover blank";
  tile.textContent = s.id === "liked" ? "♥" : "♪";
  tile.setAttribute("aria-hidden", "true");
  return tile;
}

// Liked Songs + the 5 most recent playlists; playlists further down show only when selected or after "Show all".
function renderSources(list = state.sources) {
  state.sources = list;
  const chosen = new Set(savedSources());
  const playlists = list.filter((s) => s.id !== "liked");
  const visible = list.filter((s) => s.id === "liked" || state.showAllSources ||
    playlists.indexOf(s) < RECENT_PLAYLISTS || chosen.has(s.id));
  const box = $("sources");
  box.replaceChildren();
  for (const s of visible) {
    const label = document.createElement("label");
    label.className = "source";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = s.id;
    input.checked = chosen.has(s.id);
    const text = document.createElement("span");
    text.className = "source-text";
    const name = document.createElement("span");
    name.className = "source-name";
    name.textContent = s.name;
    const count = document.createElement("span");
    count.className = "muted";
    count.textContent = sourceLabel(s);
    text.append(name, count);
    label.append(input, sourceCover(s), text);
    box.appendChild(label);
  }
  const hidden = list.length - visible.length;
  const more = $("sources-more");
  more.hidden = playlists.length <= RECENT_PLAYLISTS;
  more.textContent = state.showAllSources ? "Show fewer" : `Show all playlists (${hidden} more)`;
}

async function loadSources() {
  try {
    const data = await api("/sources");
    $("sources-error").hidden = true;
    renderSources(data.sources);
  } catch (e) {
    renderSources([{ id: "liked", name: "Liked Songs", total: 0, analyzed: 0, image: null }]);
    $("sources-error").textContent = `Couldn't load your playlists (${e.message}). Log out and log in again to allow it.`;
    $("sources-error").hidden = false;
  }
}

$("sources").addEventListener("change", () => {
  // Selected playlists are always shown, so the visible checkboxes hold the whole selection.
  try { localStorage.setItem(SOURCES_KEY, JSON.stringify(selectedSources())); } catch (e) { /* not remembered */ }
});
$("sources-more").addEventListener("click", () => {
  state.showAllSources = !state.showAllSources;
  renderSources();
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
    // Classes and labels let the phone layout turn each row into a card (see style.css).
    const cells = [
      ["c-name", t.name], ["c-artist", t.artists], ["c-genre", t.genre],
      ["c-stat", fmt(t.danceability), "Dance"], ["c-stat", fmt(t.energy), "Energy"],
      ["c-stat", fmt(t.valence), "Happy"], ["c-stat", fmt(t.tempo, 0), "BPM"],
    ];
    for (const [cls, text, label] of cells) {
      const td = document.createElement("td");
      td.className = cls;
      td.textContent = text;
      if (label) td.dataset.label = label;
      tr.appendChild(td);
    }
    if (t.analysis_source === "artist") {
      const mark = document.createElement("span");
      mark.className = "est";
      mark.textContent = " est.";
      mark.title = "No preview found: estimated from this artist's other songs";
      tr.children[0].appendChild(mark);
    }
    const td = document.createElement("td");
    td.className = "c-rm";
    const rm = document.createElement("button");
    rm.className = "icon remove";
    rm.textContent = "×";
    rm.title = "Remove from playlist";
    rm.setAttribute("aria-label", `Remove ${t.name}`);
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
    $("results-note").textContent = `Every matching song is already in “${t.name}”. Like more songs and sync, or add more sources.`;
  } else {
    const kind = adding ? "new songs" : "songs";
    const tip = state.saved && state.saved.mode === "genre"
      ? "Pick more genres or sync more sources." : "Sync more sources to find more.";
    $("results-note").textContent = data.matched < data.requested
      ? `Only ${state.tracks.length} ${kind} match. ${tip}`
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
  const choice = playlistChoice();
  const sources = selectedSources();
  if (!sources.length) {
    $("results").hidden = false;
    $("results-note").textContent = "Select at least one source under Your library.";
    return;
  }
  $("generate-btn").disabled = true;
  try {
    const data = await api("/generate", {
      ...choice, sources, size: Number($("size").value), new_playlist: $("new-playlist").checked,
    });
    state.tracks = data.tracks;
    state.saved = choice; // Save uses the picks this preview was made from
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
      ...state.saved, track_ids: state.tracks.map((t) => t.spotify_id),
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
document.querySelectorAll("#modes .chip").forEach((b) => b.addEventListener("click", () => selectMode(b.dataset.mode)));
if (savedMode()) selectMode(savedMode(), false);
document.querySelectorAll("#moods .chip").forEach((b) => b.addEventListener("click", () => selectMood(b.dataset.mood)));
$("sync-btn").addEventListener("click", startSync);
$("generate-btn").addEventListener("click", generate);
$("save-btn").addEventListener("click", save);
// Add mode leaves out songs already in the playlist, so switching modes needs a fresh preview.
$("new-playlist").addEventListener("change", () => { if (!$("results").hidden) generate(); });
loadSources();
pollSync(); // resume progress display if a sync is already running
