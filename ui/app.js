const BACKEND_ENDPOINTS = {
  plan: window.location.protocol === "file:" ? "" : "/ui/plan",
  run: window.location.protocol === "file:" ? "" : "/ui/run",
};

const PIPELINES = [
  { id: "planner-first", label: "Planner-first" },
  { id: "planner-first-explanation", label: "Planner-first explanation" },
  { id: "rag", label: "RAG" },
  { id: "internal-knowledge", label: "Internal knowledge" },
  { id: "manual", label: "Manual review" },
];

const state = {
  plan: null,
  selectedPipelines: {},
  output: null,
};

const els = {
  queryInput: document.querySelector("#query-input"),
  planButton: document.querySelector("#plan-button"),
  runButton: document.querySelector("#run-button"),
  backendStatus: document.querySelector("#backend-status"),
  leafCount: document.querySelector("#leaf-count"),
  planSummary: document.querySelector("#plan-summary"),
  planTree: document.querySelector("#plan-tree"),
  leafList: document.querySelector("#leaf-list"),
  runStatus: document.querySelector("#run-status"),
  answerView: document.querySelector("#answer-view"),
  provenanceView: document.querySelector("#provenance-view"),
  rowsView: document.querySelector("#rows-view"),
  tabs: document.querySelectorAll(".tab"),
};

function hasBackend(endpoint) {
  return typeof endpoint === "string" && endpoint.trim().length > 0;
}

function setStatus(message) {
  els.backendStatus.textContent = message;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function compactList(values) {
  if (!Array.isArray(values) || values.length === 0) {
    return "none";
  }
  return values.join(", ");
}

function firstTableFromSql(sql) {
  const match = sql.match(/\bfrom\s+([a-zA-Z_][\w]*)/i);
  return match ? match[1] : "input_table";
}

function joinedTablesFromSql(sql) {
  const tables = [];
  const fromTable = firstTableFromSql(sql);
  if (fromTable) {
    tables.push(fromTable);
  }

  for (const match of sql.matchAll(/\bjoin\s+([a-zA-Z_][\w]*)/gi)) {
    if (!tables.includes(match[1])) {
      tables.push(match[1]);
    }
  }

  return tables.length ? tables : ["input_table"];
}

function makeMockPlan(sql) {
  const tables = joinedTablesFromSql(sql);
  const hasWhere = /\bwhere\b/i.test(sql);
  const hasLimit = /\blimit\b/i.test(sql);

  return {
    source: "mock",
    sql,
    plan: {
      query_type: "SELECT",
      sql,
      leaf_tasks: tables.map((table, index) => ({
        table_name: table,
        alias: index === 0 ? table[0] : null,
        scan_op: index === 0 && hasWhere ? "FilterLeafScan" : "LeafScan",
        columns: index === 0 ? ["selected_columns", "join_keys"] : ["join_keys"],
        local_predicates: index === 0 && hasWhere ? ["predicate extracted by planner"] : [],
        join_keys: tables.length > 1 ? [`${table}_key`] : [],
        select_columns: index === 0 ? ["display_value"] : [],
        group_by_columns: [],
        aggregate_columns: [],
      })),
      joins: tables.slice(1).map((table) => ({
        join_type: "JOIN",
        table,
        alias: null,
        on_sql: "join condition from planner",
      })),
      post_ops: [
        { op: "Project", payload: { select: ["selected expressions"] } },
        ...(hasLimit ? [{ op: "Limit", payload: { value: "from query" } }] : []),
      ],
    },
  };
}

function makeMockOutput() {
  const leaves = state.plan?.plan?.leaf_tasks ?? [];
  const chosen = leaves.map((leaf) => ({
    table: leaf.table_name,
    pipeline: state.selectedPipelines[leaf.table_name] ?? "planner-first",
  }));

  const first = chosen[0]?.table ?? "result";
  const second = chosen[1]?.table ?? "support";

  return {
    source: "mock",
    answer: [
      {
        result: {
          value: `${first} result 1`,
          related_value: `${second} evidence`,
        },
        provenance: [[`${first}_1`, `${second}_3`]],
      },
      {
        result: {
          value: `${first} result 2`,
          related_value: `${second} evidence`,
        },
        provenance: [[`${first}_2`, `${second}_4`]],
      },
    ],
    rows_by_id: {
      [`${first}_1`]: {
        table: first,
        row: { id: `${first}_1`, produced_by: chosen[0]?.pipeline ?? "planner-first" },
      },
      [`${second}_3`]: {
        table: second,
        row: { id: `${second}_3`, produced_by: chosen[1]?.pipeline ?? "planner-first" },
      },
      [`${first}_2`]: {
        table: first,
        row: { id: `${first}_2`, produced_by: chosen[0]?.pipeline ?? "planner-first" },
      },
      [`${second}_4`]: {
        table: second,
        row: { id: `${second}_4`, produced_by: chosen[1]?.pipeline ?? "planner-first" },
      },
    },
    pipeline_choices: chosen,
  };
}

function planToTree(plan) {
  const leaves = plan?.leaf_tasks ?? [];
  if (!leaves.length) {
    return null;
  }

  const leafNodes = leaves.map((leaf) => {
    const tableNode = {
      label: leaf.table_name,
      type: "table",
      children: [],
    };

    if (Array.isArray(leaf.local_predicates) && leaf.local_predicates.length) {
      return {
        label: `sigma ${leaf.local_predicates.join(" and ")}`,
        type: "filter",
        children: [tableNode],
      };
    }

    return {
      label: leaf.scan_op || "LeafScan",
      type: "operation",
      children: [tableNode],
    };
  });

  let tree = leafNodes[0];
  const joins = plan?.joins ?? [];
  for (let index = 1; index < leafNodes.length; index += 1) {
    const join = joins[index - 1] ?? {};
    tree = {
      label: join.on_sql ? `join ${join.on_sql}` : "join",
      type: "join",
      children: [tree, leafNodes[index]],
    };
  }

  for (const postOp of plan?.post_ops ?? []) {
    tree = {
      label: postOp.op,
      type: "operation",
      children: [tree],
    };
  }

  return tree;
}

function treeLeafCount(node) {
  if (!node?.children?.length) {
    return 1;
  }
  return node.children.reduce((total, child) => total + treeLeafCount(child), 0);
}

function treeDepth(node) {
  if (!node?.children?.length) {
    return 1;
  }
  return 1 + Math.max(...node.children.map(treeDepth));
}

function layoutTree(root) {
  const leafGap = 168;
  const levelGap = 86;
  const marginX = 86;
  const marginY = 34;
  let cursor = 0;
  const nodes = [];
  const edges = [];

  function visit(node, depth) {
    const children = node.children ?? [];
    const laidOutChildren = children.map((child) => visit(child, depth + 1));
    let x;

    if (laidOutChildren.length) {
      x =
        laidOutChildren.reduce((total, child) => total + child.x, 0) /
        laidOutChildren.length;
    } else {
      x = marginX + cursor * leafGap;
      cursor += 1;
    }

    const laidOut = {
      ...node,
      x,
      y: marginY + depth * levelGap,
    };

    nodes.push(laidOut);
    for (const child of laidOutChildren) {
      edges.push({ from: laidOut, to: child });
    }

    return laidOut;
  }

  visit(root, 0);

  return {
    nodes,
    edges,
    width: Math.max(360, treeLeafCount(root) * leafGap + marginX),
    height: Math.max(260, treeDepth(root) * levelGap + marginY),
  };
}

function truncateLabel(label) {
  const clean = String(label ?? "").replace(/\s+/g, " ").trim();
  return clean.length > 34 ? `${clean.slice(0, 31)}...` : clean;
}

function renderPlanTree(plan) {
  const tree = planToTree(plan);
  if (!tree) {
    els.planTree.className = "plan-tree empty-state";
    els.planTree.textContent = "No tree available for this plan.";
    return;
  }

  const layout = layoutTree(tree);
  const nodeWidth = 132;
  const nodeHeight = 42;
  const edges = layout.edges
    .map(
      (edge) => `
        <line
          class="tree-edge"
          x1="${edge.from.x}"
          y1="${edge.from.y + nodeHeight / 2}"
          x2="${edge.to.x}"
          y2="${edge.to.y - nodeHeight / 2}"
        />
      `,
    )
    .join("");
  const nodes = layout.nodes
    .map(
      (node) => `
        <g class="tree-node ${escapeHtml(node.type)}" transform="translate(${node.x - nodeWidth / 2}, ${node.y - nodeHeight / 2})">
          <rect width="${nodeWidth}" height="${nodeHeight}" rx="6"></rect>
          <text x="${nodeWidth / 2}" y="${nodeHeight / 2}">${escapeHtml(truncateLabel(node.label))}</text>
          <title>${escapeHtml(node.label)}</title>
        </g>
      `,
    )
    .join("");

  els.planTree.className = "plan-tree";
  els.planTree.innerHTML = `
    <svg viewBox="0 0 ${layout.width} ${layout.height}" width="${layout.width}" height="${layout.height}" role="img" aria-label="Query plan tree">
      ${edges}
      ${nodes}
    </svg>
  `;
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`Request failed with ${response.status}`);
  }

  return response.json();
}

async function generatePlan() {
  const query = els.queryInput.value.trim();
  if (!query) {
    setStatus("Write a question or SQL query first.");
    return;
  }

  els.planButton.disabled = true;
  setStatus("Building query plan...");

  try {
    if (hasBackend(BACKEND_ENDPOINTS.plan)) {
      state.plan = await postJson(BACKEND_ENDPOINTS.plan, { question: query, sql: query });
      setStatus("Plan loaded from backend.");
    } else {
      state.plan = makeMockPlan(query);
      setStatus("Backend endpoints are empty. Showing local mock plan data.");
    }

    state.selectedPipelines = {};
    for (const leaf of state.plan.plan.leaf_tasks) {
      state.selectedPipelines[leaf.table_name] = "planner-first";
    }

    renderPlan();
    clearOutput();
  } catch (error) {
    setStatus(error.message);
  } finally {
    els.planButton.disabled = false;
  }
}

async function runSelectedPipelines() {
  if (!state.plan) {
    setStatus("Generate a plan before running pipelines.");
    return;
  }

  els.runButton.disabled = true;
  els.runStatus.textContent = "Running";
  els.runStatus.classList.remove("muted-pill");
  setStatus("Running selected leaf pipelines...");

  const payload = {
    question: els.queryInput.value.trim(),
    sql: state.plan.sql,
    plan: state.plan.plan,
    leaf_pipeline_choices: state.selectedPipelines,
  };

  try {
    if (hasBackend(BACKEND_ENDPOINTS.run)) {
      state.output = await postJson(BACKEND_ENDPOINTS.run, payload);
      setStatus("Output loaded from backend.");
    } else {
      state.output = makeMockOutput();
      setStatus("Backend endpoints are empty. Showing local mock output.");
    }

    els.runStatus.textContent = "Complete";
    renderOutput();
  } catch (error) {
    els.runStatus.textContent = "Error";
    setStatus(error.message);
  } finally {
    els.runButton.disabled = false;
  }
}

function renderPlan() {
  const plan = state.plan?.plan;
  const leaves = plan?.leaf_tasks ?? [];

  els.leafCount.textContent = `${leaves.length} ${leaves.length === 1 ? "leaf" : "leaves"}`;
  els.planSummary.innerHTML = [
    ["Type", plan?.query_type ?? "-"],
    ["Joins", plan?.joins?.length ?? 0],
    ["Post-ops", plan?.post_ops?.length ?? 0],
    ["Source", state.plan?.source ?? "backend"],
  ]
    .map(
      ([label, value]) => `
        <div class="summary-item">
          <span>${escapeHtml(label)}</span>
          <strong>${escapeHtml(value)}</strong>
        </div>
      `,
    )
    .join("");

  renderPlanTree(plan);

  if (!leaves.length) {
    els.leafList.className = "leaf-list empty-state";
    els.leafList.textContent = "No leaf tasks returned.";
    return;
  }

  els.leafList.className = "leaf-list";
  els.leafList.innerHTML = leaves.map(renderLeafCard).join("");

  els.leafList.querySelectorAll("select[data-table]").forEach((select) => {
    select.addEventListener("change", (event) => {
      const table = event.target.dataset.table;
      state.selectedPipelines[table] = event.target.value;
    });
  });
}

function renderLeafCard(leaf, index) {
  const selected = state.selectedPipelines[leaf.table_name] ?? "planner-first";
  const options = PIPELINES.map(
    (pipeline) => `
      <option value="${escapeHtml(pipeline.id)}" ${pipeline.id === selected ? "selected" : ""}>
        ${escapeHtml(pipeline.label)}
      </option>
    `,
  ).join("");

  return `
    <article class="leaf-card">
      <div class="leaf-card-header">
        <div>
          <h3>Leaf ${index + 1}: ${escapeHtml(leaf.table_name)}</h3>
          <div class="leaf-meta">
            <span class="meta-chip">${escapeHtml(leaf.scan_op)}</span>
            ${leaf.alias ? `<span class="meta-chip">alias ${escapeHtml(leaf.alias)}</span>` : ""}
          </div>
        </div>
        <label>
          <span class="field-label">Pipeline</span>
          <select data-table="${escapeHtml(leaf.table_name)}">${options}</select>
        </label>
      </div>
      <div class="leaf-details">
        ${renderDetail("Columns", compactList(leaf.columns))}
        ${renderDetail("Predicates", compactList(leaf.local_predicates))}
        ${renderDetail("Join keys", compactList(leaf.join_keys))}
        ${renderDetail("Select columns", compactList(leaf.select_columns))}
        ${renderDetail("Leaf SQL", leaf.question_sql ?? "not available")}
      </div>
    </article>
  `;
}

function renderDetail(label, value) {
  return `
    <div class="detail-box">
      <span>${escapeHtml(label)}</span>
      <code>${escapeHtml(value)}</code>
    </div>
  `;
}

function clearOutput() {
  state.output = null;
  els.runStatus.textContent = "Waiting";
  els.runStatus.classList.add("muted-pill");
  els.answerView.innerHTML = `<div class="empty-state">Run the selected pipelines to see final output.</div>`;
  els.provenanceView.innerHTML = `<div class="empty-state">Provenance will appear here after a run.</div>`;
  els.rowsView.innerHTML = `<div class="empty-state">Supporting rows will appear here after a run.</div>`;
}

function renderOutput() {
  const answer = state.output?.answer ?? [];
  els.answerView.innerHTML = `
    ${state.output?.source === "mock" ? `<p class="notice">Mock output. Wire BACKEND_ENDPOINTS.run in app.js to replace this.</p>` : ""}
    ${state.output?.note ? `<p class="notice">${escapeHtml(state.output.note)}</p>` : ""}
    ${renderErrors(state.output?.errors ?? [])}
    ${renderExplanationOutput(state.output)}
    ${renderAnswerTable(answer)}
  `;
  els.provenanceView.innerHTML = renderProvenance(answer);
  els.rowsView.innerHTML = renderSupportingRows(state.output?.rows_by_id ?? {});
}

function renderExplanationOutput(output) {
  const explanations = Array.isArray(output?.explanations)
    ? output.explanations
    : output?.explanation
      ? [output.explanation]
      : [];

  if (!explanations.length) {
    return "";
  }

  return `
    <div class="provenance-formula">
      ${explanations
        .map(
          (explanation, index) => `
            <div class="formula-card">
              <strong>Planner-first explanation ${index + 1}: ${escapeHtml(explanation.table ?? "leaf")}</strong>
              ${explanation.question_sql ? `<code>${escapeHtml(explanation.question_sql)}</code>` : ""}
              <pre>${escapeHtml(explanation.response_text ?? "")}</pre>
            </div>
          `,
        )
        .join("")}
    </div>
  `;
}

function renderErrors(errors) {
  if (!Array.isArray(errors) || !errors.length) {
    return "";
  }

  return `
    <div class="notice">
      ${errors.map((error) => `<div>${escapeHtml(error)}</div>`).join("")}
    </div>
  `;
}

function renderAnswerTable(answer) {
  const rows = answer.map((item) => item.result ?? {});
  const columns = [...new Set(rows.flatMap((row) => Object.keys(row)))];
  if (!rows.length || !columns.length) {
    return `<div class="empty-state">No answer rows.</div>`;
  }

  return `
    <div class="table-wrap">
      <table>
        <thead>
          <tr>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr>
        </thead>
        <tbody>
          ${rows
            .map(
              (row) => `
                <tr>${columns.map((column) => `<td>${escapeHtml(row[column])}</td>`).join("")}</tr>
              `,
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderProvenance(answer) {
  if (!answer.length) {
    return `<div class="empty-state">No provenance rows.</div>`;
  }

  return `
    <div class="provenance-formula">
      ${answer
        .map((item, index) => {
          const formula = (item.provenance ?? [])
            .map((witnessSet) => witnessSet.join(" AND "))
            .join(" OR ");
          return `
            <div class="formula-card">
              <strong>Result ${index + 1}</strong>
              <code>${escapeHtml(formula || "no provenance")}</code>
            </div>
          `;
        })
        .join("")}
    </div>
  `;
}

function renderSupportingRows(rowsById) {
  const rows = Object.entries(rowsById).map(([rowId, info]) => ({
    row_id: rowId,
    table: info.table,
    values: JSON.stringify(info.row ?? {}, null, 0),
  }));

  if (!rows.length) {
    return `<div class="empty-state">No supporting rows.</div>`;
  }

  return `
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>Row id</th><th>Table</th><th>Values</th></tr>
        </thead>
        <tbody>
          ${rows
            .map(
              (row) => `
                <tr>
                  <td>${escapeHtml(row.row_id)}</td>
                  <td>${escapeHtml(row.table)}</td>
                  <td>${escapeHtml(row.values)}</td>
                </tr>
              `,
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

function switchTab(tabName) {
  for (const tab of els.tabs) {
    tab.classList.toggle("active", tab.dataset.tab === tabName);
  }

  els.answerView.classList.toggle("hidden", tabName !== "answer");
  els.provenanceView.classList.toggle("hidden", tabName !== "provenance");
  els.rowsView.classList.toggle("hidden", tabName !== "rows");
}

els.planButton.addEventListener("click", generatePlan);
els.runButton.addEventListener("click", runSelectedPipelines);
els.tabs.forEach((tab) => {
  tab.addEventListener("click", () => switchTab(tab.dataset.tab));
});

clearOutput();
if (hasBackend(BACKEND_ENDPOINTS.plan) && hasBackend(BACKEND_ENDPOINTS.run)) {
  setStatus("Connected mode. Planning, leaf runs, and explanation runs will call the gateway.");
} else {
  setStatus("Backend endpoints are empty in file mode; the UI is running with mock data.");
}
