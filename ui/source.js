const params = new URLSearchParams(location.search);
const table = params.get("table");
const targetRow = params.get("row_id");
const snapshot = params.get("snapshot");
let currentPage = 0;
let requestNumber = 0;
const description = document.querySelector("#source-description");
const previous = document.querySelector("#previous");
const next = document.querySelector("#next");
const title = document.querySelector("#source-title");
title.textContent = table || "Source tuples";

async function loadPage(page = 0, focus = false) {
  const request = ++requestNumber;
  try {
    let data;
    if (snapshot) {
      const stored = localStorage.getItem(snapshot);
      if (!stored) throw new Error("This run’s generated source snapshot is no longer available.");
      const rows = JSON.parse(stored)[table];
      if (!Array.isArray(rows)) throw new Error("Generated source table is unavailable for this run.");
      const position = rows.findIndex((row) => String(row.__rid__) === targetRow);
      if (position < 0) throw new Error("Source tuple is unavailable in this run’s generated table.");
      const selectedPage = focus ? Math.floor(position / 100) : page;
      data = {rows: rows.slice(selectedPage * 100, (selectedPage + 1) * 100),
        total: rows.length, page: selectedPage, page_size: 100};
    } else {
      const query = new URLSearchParams({dataset: params.get("dataset") || "", table: table || "", page});
      if (focus) query.set("row_id", targetRow || "");
      const response = await fetch(`/ui/data/source-table?${query}`);
      data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Could not load source table.");
    }
    if (request !== requestNumber) return;
    currentPage = data.page;
    description.textContent = `${snapshot ? "Generated table from this query run" : `Original dataset: ${params.get("dataset")}`} · ${data.total} rows. The contributing tuple is highlighted in blue.`;
    previous.disabled = currentPage === 0;
    next.disabled = (currentPage + 1) * data.page_size >= data.total;
    document.querySelector("#page-label").textContent = `Page ${currentPage + 1} of ${Math.max(1, Math.ceil(data.total / data.page_size))}`;
    const columns = [...new Set(data.rows.flatMap((row) => Object.keys(row)))].filter((column) => column !== "__rid__");
    const grid = document.createElement("table");
    const header = grid.createTHead().insertRow();
    for (const column of columns) {
      const cell = document.createElement("th");
      cell.scope = "col";
      cell.textContent = column;
      header.append(cell);
    }
    const body = grid.createTBody();
    for (const row of data.rows) {
      const tr = body.insertRow();
      if (String(row.__rid__) === targetRow) {
        tr.className = "source-highlight";
        tr.setAttribute("aria-label", "Contributing source tuple");
      }
      for (const column of columns) {
        const value = row[column];
        tr.insertCell().textContent = typeof value === "object" && value !== null ? JSON.stringify(value) : String(value ?? "");
      }

    }
    document.querySelector("#source-table").replaceChildren(grid);
    if (focus) document.querySelector(".source-highlight")?.scrollIntoView({block: "center"});
  } catch (error) {
    if (request !== requestNumber) return;
    description.textContent = error.message;
    document.querySelector("#source-table").replaceChildren();
    previous.disabled = next.disabled = true;
  }
}
previous.addEventListener("click", () => loadPage(currentPage - 1));
next.addEventListener("click", () => loadPage(currentPage + 1));
document.querySelector("#focus-source").addEventListener("click", () => loadPage(0, true));
loadPage(0, true);
