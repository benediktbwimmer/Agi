const runsList = document.getElementById("runs-list");
const refreshRunsBtn = document.getElementById("refresh-runs");
const runDetails = document.getElementById("run-details");
const telemetryPane = document.getElementById("telemetry");
const approvalsPane = document.getElementById("approvals");
const memorySelect = document.getElementById("memory-select");
const memoryPane = document.getElementById("memory-records");

let selectedRunId = null;
let telemetryPoll = null;
let approvalsPoll = null;
let memoryPoll = null;

function formatJSON(value) {
  try {
    return JSON.stringify(value, null, 2);
  } catch (err) {
    return String(value);
  }
}

async function loadRuns() {
  try {
    const response = await fetch("/runs");
    if (!response.ok) throw new Error("Failed to load runs");
    const runs = await response.json();
    renderRunList(runs);
  } catch (err) {
    console.error(err);
  }
}

function renderRunList(runs) {
  runsList.innerHTML = "";
  if (!runs.length) {
    const empty = document.createElement("li");
    empty.textContent = "No runs ingested";
    empty.classList.add("empty");
    runsList.appendChild(empty);
    return;
  }
  runs.forEach((run) => {
    const item = document.createElement("li");
    item.dataset.runId = run.run_id;
    item.className = run.run_id === selectedRunId ? "active" : "";
    item.innerHTML = `
      <strong>${run.goal?.goal ?? run.goal ?? run.run_id}</strong><br />
      <span>${run.run_id}</span>
    `;
    item.addEventListener("click", () => {
      selectedRunId = run.run_id;
      Array.from(runsList.children).forEach((el) => el.classList.remove("active"));
      item.classList.add("active");
      loadRunDetails(run.run_id);
    });
    runsList.appendChild(item);
  });
}

async function loadRunDetails(runId) {
  try {
    const response = await fetch(`/runs/${encodeURIComponent(runId)}`);
    if (!response.ok) throw new Error("Run not found");
    const payload = await response.json();
    renderRunDetails(runId, payload);
  } catch (err) {
    runDetails.textContent = err.message;
  }
}

function renderRunDetails(runId, payload) {
  const lines = [];
  lines.push(`<h3>${runId}</h3>`);
  if (payload.manifest) {
    lines.push(`<details open><summary>Manifest</summary><pre>${formatJSON(payload.manifest)}</pre></details>`);
  }
  if (payload.working_memory) {
    lines.push(`<details><summary>Working Memory</summary><pre>${formatJSON(payload.working_memory)}</pre></details>`);
  }
  if (payload.reflection_summary) {
    lines.push(`<details><summary>Reflection Summary</summary><pre>${formatJSON(payload.reflection_summary)}</pre></details>`);
  }
  runDetails.innerHTML = lines.join("\n");
}

async function loadTelemetry() {
  try {
    const response = await fetch("/telemetry?limit=100");
    if (!response.ok) throw new Error("Telemetry fetch failed");
    const events = await response.json();
    telemetryPane.textContent = events.map((event) => formatJSON(event)).join("\n\n");
  } catch (err) {
    console.error(err);
  }
}

async function loadApprovals() {
  try {
    const response = await fetch("/approvals/pending");
    if (!response.ok) throw new Error("Failed to load approvals");
    const approvals = await response.json();
    renderApprovals(approvals);
  } catch (err) {
    console.error(err);
  }
}

function renderApprovals(items) {
  approvalsPane.innerHTML = "";
  if (!items.length) {
    const empty = document.createElement("p");
    empty.textContent = "No approvals pending.";
    approvalsPane.appendChild(empty);
    return;
  }
  items.forEach((item) => {
    const card = document.createElement("div");
    card.className = "approval-card";
    card.innerHTML = `
      <h3>${item.tool ?? "Unknown tool"} <span class="chip">${item.tier}</span></h3>
      <p><strong>Request ID:</strong> ${item.id}</p>
      <p><strong>Requested by:</strong> ${item.requested_by}</p>
      <p><strong>Created at:</strong> ${item.created_at}</p>
      <pre class="pre">${formatJSON(item.context)}</pre>
      <div class="approval-actions">
        <button class="approve">Approve</button>
        <button class="deny">Deny</button>
      </div>
    `;
    const [approveBtn, denyBtn] = card.querySelectorAll("button");
    approveBtn.addEventListener("click", () => submitDecision(item.id, true));
    denyBtn.addEventListener("click", () => submitDecision(item.id, false));
    approvalsPane.appendChild(card);
  });
}

async function submitDecision(approvalId, approved) {
  const reviewer = prompt("Reviewer name", "oversight");
  if (reviewer === null) return;
  const message = approved ? "Approved via console" : "Denied via console";
  try {
    const response = await fetch(`/approvals/${encodeURIComponent(approvalId)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ approved, reviewer, message }),
    });
    if (!response.ok) throw new Error("Failed to record decision");
    await loadApprovals();
  } catch (err) {
    alert(err.message);
  }
}

async function loadMemoryLogs() {
  try {
    const response = await fetch("/memory");
    if (!response.ok) throw new Error("Failed to load memory logs");
    const logs = await response.json();
    renderMemoryLogs(logs);
  } catch (err) {
    console.error(err);
  }
}

function renderMemoryLogs(logs) {
  const names = Object.keys(logs);
  memorySelect.innerHTML = "";
  if (!names.length) {
    memoryPane.textContent = "No memory logs ingested.";
    return;
  }
  names.forEach((name, idx) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = `${name} (${logs[name].length})`;
    if (idx === 0 && !memorySelect.value) {
      option.selected = true;
    }
    memorySelect.appendChild(option);
  });
  const active = memorySelect.value || names[0];
  memoryPane.textContent = logs[active]
    .map((record) => formatJSON(record))
    .join("\n\n");
}

memorySelect?.addEventListener("change", loadMemoryLogs);
refreshRunsBtn?.addEventListener("click", loadRuns);

document.addEventListener("visibilitychange", () => {
  const active = document.visibilityState === "visible";
  if (active) {
    loadTelemetry();
    loadApprovals();
    loadMemoryLogs();
  }
});

document.addEventListener("DOMContentLoaded", () => {
  loadRuns();
  loadTelemetry();
  loadApprovals();
  loadMemoryLogs();

  telemetryPoll = setInterval(loadTelemetry, 5000);
  approvalsPoll = setInterval(loadApprovals, 4000);
  memoryPoll = setInterval(loadMemoryLogs, 15000);
});

window.addEventListener("beforeunload", () => {
  clearInterval(telemetryPoll);
  clearInterval(approvalsPoll);
  clearInterval(memoryPoll);
});
