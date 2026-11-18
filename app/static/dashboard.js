"use strict";
async function loadHealth() {
  const node = document.getElementById("health");
  try {
    const response = await fetch("/health");
    const data = await response.json();
    if (!response.ok) throw new Error("磁盘空间检查未通过");
    node.textContent = `后端在线 · Local Mode · 可用磁盘 ${data.free_disk_gib} GiB`;
  } catch (error) { node.textContent = `后端状态异常：${error.message}`; node.classList.add("error"); }
}
loadHealth();

let selectedId = null;
let loadingDetail = 0;
const byId = (id) => document.getElementById(id);

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "请求校验失败，请检查输入。");
  return data;
}

function message(id, text, error = false) {
  byId(id).textContent = text;
  byId(id).classList.toggle("error", error);
}

function renderTable(id, headers, rows) {
  const table = byId(id);
  table.replaceChildren();
  const head = table.createTHead().insertRow();
  for (const label of headers) { const cell = document.createElement("th"); cell.textContent = label; head.append(cell); }
  const body = table.createTBody();
  for (const row of rows) {
    const tr = body.insertRow();
    for (const value of row) { const td = tr.insertCell(); td.textContent = value === null ? "∅" : String(value); }
  }
}

function renderDetail(data) {
  selectedId = data.dataset_id;
  byId("empty-state").hidden = true;
  byId("detail-content").hidden = false;
  byId("dataset-name").textContent = data.original_filename;
  byId("dataset-id").textContent = data.dataset_id;
  byId("dataset-stats").textContent = `${data.row_count} 行 · ${data.column_count} 列 · ${data.duplicate_rows} 条重复记录 · ${(data.size_bytes / 1024).toFixed(1)} KiB`;
  byId("task-hint").textContent = data.task_hint;
  const select = byId("target-column");
  select.replaceChildren(new Option("不选择目标列", ""));
  for (const column of data.columns) select.add(new Option(column.name, column.name));
  select.value = data.target_column || "";
  renderTable("columns-table", ["列名", "推断类型", "缺失数量", "缺失比例", "非缺失唯一值"], data.columns.map(c => [c.name, c.dtype, c.missing_count, `${(c.missing_ratio * 100).toFixed(2)}%`, `${c.unique_count_is_lower_bound ? "≥" : ""}${c.unique_count}`]));
  renderTable("preview-table", data.columns.map(c => c.name), data.preview);
  byId("dataset-summary").textContent = JSON.stringify({encoding: data.encoding, sha256: data.sha256, missing_tokens: data.missing_tokens, target_summary: data.target_summary, numeric_columns: data.columns.filter(c => c.numeric_statistics).map(c => ({name: c.name, ...c.numeric_statistics}))}, null, 2);
  for (const node of byId("dataset-list").children) node.classList.toggle("active", node.dataset.id === selectedId);
}

async function showDataset(id) {
  const sequence = ++loadingDetail;
  message("target-message", "");
  try {
    const data = await api(`/api/datasets/${encodeURIComponent(id)}`);
    if (sequence === loadingDetail) renderDetail(data);
  } catch (error) { message("upload-message", error.message, true); }
}

async function loadDatasets() {
  try {
    const data = await api("/api/datasets");
    const list = byId("dataset-list");
    list.replaceChildren();
    if (!data.datasets.length) { list.textContent = "还没有数据集。"; return; }
    for (const dataset of data.datasets) {
      const button = document.createElement("button");
      button.type = "button"; button.className = "dataset"; button.dataset.id = dataset.dataset_id;
      button.textContent = `${dataset.original_filename} · ${dataset.row_count} 行 / ${dataset.column_count} 列`;
      button.classList.toggle("active", selectedId === dataset.dataset_id);
      button.addEventListener("click", () => showDataset(dataset.dataset_id)); list.append(button);
    }
  } catch (error) { byId("dataset-list").textContent = error.message; }
}

byId("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = byId("csv-file").files[0];
  if (!file) return;
  byId("upload-button").disabled = true;
  message("upload-message", "正在上传并检查，请稍候…");
  try {
    const data = await api(`/api/datasets?filename=${encodeURIComponent(file.name)}`, {method: "POST", headers: {"Content-Type": "text/csv"}, body: file});
    ++loadingDetail;
    renderDetail(data); await loadDatasets();
    message("upload-message", "上传成功。原始 CSV 和检查结果已保存在本机。");
    loadHealth();
  } catch (error) { message("upload-message", error.message, true); }
  finally { byId("upload-button").disabled = false; }
});

byId("target-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!selectedId) return;
  const id = selectedId;
  const target = byId("target-column").value || null;
  byId("target-button").disabled = true;
  try {
    const data = await api(`/api/datasets/${id}/target`, {method: "PATCH", headers: {"Content-Type": "application/json"}, body: JSON.stringify({target_column: target})});
    if (selectedId === id) { renderDetail(data); message("target-message", "目标列已保存。"); }
  } catch (error) { message("target-message", error.message, true); }
  finally { byId("target-button").disabled = false; }
});
loadDatasets();

async function loadModels() {
  const list = byId("model-list");
  try {
    const data = await api("/api/models");
    list.replaceChildren();
    for (const model of data.models) {
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      summary.textContent = `${model.model_name} · ${model.framework} · ${model.task_type}`;
      const description = document.createElement("p");
      description.textContent = model.description;
      const spec = document.createElement("pre");
      spec.textContent = JSON.stringify({parameter_space: model.parameter_space, supported_metrics: model.supported_metrics,
        capabilities: model.capabilities, execution_requirements: model.execution_requirements}, null, 2);
      details.append(summary, description, spec); list.append(details);
    }
  } catch (error) { list.textContent = error.message; }
}
loadModels();
