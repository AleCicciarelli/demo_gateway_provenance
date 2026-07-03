const BACKEND_ENDPOINTS = {
  plan: window.location.protocol === "file:" ? "" : "/ui/plan",
  run: window.location.protocol === "file:" ? "" : "/ui/run",
  runStream: window.location.protocol === "file:" ? "" : "/ui/run/stream",
};

const PIPELINES = [
  { id: "planner-only", label: "Planner only" },
  { id: "planner-only-explanation", label: "Planner only + explanation" },
  { id: "rag", label: "RAG" },
  { id: "internal-knowledge", label: "Internal knowledge" },
  { id: "manual", label: "Manual review" },
];

const state = {
  plan: null,
  selectedPipelines: {},
  output: null,
  progress: [],
  explanationOpen: false,
};

const DEFAULT_PIPELINE = "planner-only";

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

function pipelineLabel(id) {
  return PIPELINES.find((pipeline) => pipeline.id === id)?.label ?? id ?? "unknown";
}

function formatPipelineSummary(output) {
  const choices = output?.pipeline_choices ?? [];
  if (!Array.isArray(choices) || !choices.length) {
    return "Pipeline executed.";
  }

  const parts = choices.map(
    (choice) => `${choice.table} with ${pipelineLabel(choice.pipeline)}`,
  );
  return `Pipeline executed: ${parts.join(", ")}.`;
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
    pipeline: state.selectedPipelines[leaf.table_name] ?? DEFAULT_PIPELINE,
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
        row: { id: `${first}_1`, produced_by: chosen[0]?.pipeline ?? DEFAULT_PIPELINE },
      },
      [`${second}_3`]: {
        table: second,
        row: { id: `${second}_3`, produced_by: chosen[1]?.pipeline ?? DEFAULT_PIPELINE },
      },
      [`${first}_2`]: {
        table: first,
        row: { id: `${first}_2`, produced_by: chosen[0]?.pipeline ?? DEFAULT_PIPELINE },
      },
      [`${second}_4`]: {
        table: second,
        row: { id: `${second}_4`, produced_by: chosen[1]?.pipeline ?? DEFAULT_PIPELINE },
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
  const leaves = plan?.leaf_tasks ?? [];
  if (!leaves.length) {
    els.planTree.className = "plan-tree empty-state";
    els.planTree.textContent = "No tree available for this plan.";
    return;
  }

  const joins = plan?.joins ?? [];
  const postOps = plan?.post_ops ?? [];
  const joinStep = 2;
  const postOpStep = joins.length ? 3 : 2;

  els.planTree.className = "plan-tree";
  els.planTree.innerHTML = `
    <div class="plan-flow" aria-label="Query execution flow">
      <section class="flow-stage">
        <div class="flow-stage-label">
          <span>Step 1</span>
          <strong>Read source tables</strong>
        </div>
        <div class="flow-leaf-grid">
          ${leaves.map(renderFlowLeaf).join("")}
        </div>
      </section>

      ${joins.length ? renderFlowJoins(joins, joinStep) : ""}
      ${postOps.length ? renderFlowPostOps(postOps, postOpStep) : ""}

      <section class="flow-stage flow-output">
        <div class="flow-stage-label">
          <span>Output</span>
          <strong>Return final tuples with provenance</strong>
        </div>
      </section>
    </div>
  `;
}

function renderFlowLeaf(leaf) {
  const predicates = leaf.local_predicates ?? [];
  const joinKeys = leaf.join_keys ?? [];
  const selectColumns = leaf.select_columns ?? [];

  return `
    <article class="flow-leaf">
      <div class="flow-leaf-title">
        <span class="flow-node-type">${escapeHtml(leaf.scan_op || "LeafScan")}</span>
        <strong>${escapeHtml(leaf.table_name)}</strong>
        ${leaf.alias ? `<small>alias ${escapeHtml(leaf.alias)}</small>` : ""}
      </div>
      <div class="flow-facts">
        ${renderFlowFact("Filter", predicates.length ? predicates.join(" AND ") : "none")}
        ${renderFlowFact("Join keys", compactList(joinKeys))}
        ${renderFlowFact("Output cols", compactList(selectColumns))}
      </div>
    </article>
  `;
}

function renderFlowJoins(joins, stepNumber) {
  return `
    <section class="flow-stage">
      <div class="flow-stage-label">
        <span>Step ${stepNumber}</span>
        <strong>Combine matching rows</strong>
      </div>
      <div class="flow-op-list">
        ${joins
          .map(
            (join, index) => `
              <div class="flow-op join-op">
                <span>${escapeHtml(join.join_type || "JOIN")} ${escapeHtml(join.table || `table ${index + 2}`)}</span>
                <code>${escapeHtml(join.on_sql || "no join condition")}</code>
              </div>
            `,
          )
          .join("")}
      </div>
    </section>
  `;
}

function renderFlowPostOps(postOps, stepNumber) {
  return `
    <section class="flow-stage">
      <div class="flow-stage-label">
        <span>Step ${stepNumber}</span>
        <strong>Shape the result</strong>
      </div>
      <div class="flow-op-list">
        ${postOps
          .map(
            (postOp) => `
              <div class="flow-op">
                <span>${escapeHtml(postOp.op)}</span>
                <code>${escapeHtml(describePostOp(postOp))}</code>
              </div>
            `,
          )
          .join("")}
      </div>
    </section>
  `;
}

function renderFlowFact(label, value) {
  return `
    <div class="flow-fact">
      <span>${escapeHtml(label)}</span>
      <code>${escapeHtml(value)}</code>
    </div>
  `;
}

function describePostOp(postOp) {
  const payload = postOp?.payload ?? {};
  if (postOp?.op === "Project") {
    const select = payload.select ?? payload.columns ?? [];
    if (Array.isArray(select)) {
      return select
        .map((item) => (typeof item === "string" ? item : item.sql ?? JSON.stringify(item)))
        .join(", ");
    }
  }
  if (postOp?.op === "Limit") {
    return `keep first ${payload.value ?? payload.limit ?? "n"} rows`;
  }
  if (postOp?.op === "OrderBy") {
    return JSON.stringify(payload.order_by ?? payload);
  }
  return Object.keys(payload).length ? JSON.stringify(payload) : "no extra parameters";
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

async function postJsonStream(url, payload, onEvent) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`Request failed with ${response.status}`);
  }

  if (!response.body) {
    onEvent({ type: "complete", result: await response.json() });
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const clean = line.trim();
      if (clean) {
        onEvent(JSON.parse(clean));
      }
    }
  }

  buffer += decoder.decode();
  const clean = buffer.trim();
  if (clean) {
    onEvent(JSON.parse(clean));
  }
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
      state.selectedPipelines[leaf.table_name] = DEFAULT_PIPELINE;
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
  state.progress = [];
  state.explanationOpen = false;

  const payload = {
    question: els.queryInput.value.trim(),
    sql: state.plan.sql,
    plan: state.plan.plan,
    leaf_pipeline_choices: state.selectedPipelines,
  };

  try {
    if (hasBackend(BACKEND_ENDPOINTS.runStream)) {
      state.output = makeEmptyRunOutput();
      renderOutput();
      await postJsonStream(BACKEND_ENDPOINTS.runStream, payload, applyRunEvent);
    } else {
      state.output = makeMockOutput();
      setStatus("Backend endpoints are empty. Showing local mock output.");
      els.runStatus.textContent = "Complete";
      renderOutput();
    }
  } catch (error) {
    els.runStatus.textContent = "Error";
    setStatus(error.message);
  } finally {
    els.runButton.disabled = false;
  }
}

function makeEmptyRunOutput() {
  const choices = (state.plan?.plan?.leaf_tasks ?? []).map((leaf) => ({
    table: leaf.table_name,
    pipeline: state.selectedPipelines[leaf.table_name] ?? DEFAULT_PIPELINE,
  }));

  return {
    source: "gateway",
    answer: [],
    rows_by_id: {},
    pipeline_choices: choices,
    errors: [],
    explanations: [],
    csv_page_url: "/ui/csv",
    generated_csv_files: [],
    csv_ready: false,
    progress: state.progress,
  };
}

function pushProgress(event, tone = "info") {
  state.progress.push({
    type: event.type,
    tone,
    message: event.message ?? event.type,
    table: event.table,
    pipeline: event.pipeline,
    rows: event.rows,
    files: event.files,
    contextPreview: event.context_preview,
  });
}

function applyRunEvent(event) {
  if (!state.output) {
    state.output = makeEmptyRunOutput();
  }

  if (event.message) {
    setStatus(event.message);
  }

  if (event.type === "leaf_start") {
    pushProgress(event);
  } else if (event.type === "leaf_context") {
    pushProgress(event, "success");
  } else if (event.type === "leaf_done") {
    pushProgress(event, "success");
  } else if (event.type === "csv_done") {
    state.output.generated_csv_files = event.files ?? [];
    state.output.csv_page_url = event.csv_page_url ?? "/ui/csv";
    state.output.csv_ready = true;
    pushProgress(event, "success");
  } else if (event.type === "answer_done") {
    state.output.answer = event.answer ?? [];
    pushProgress(event, "success");
  } else if (event.type === "ap_explanation_done") {
    state.output.explanations = [event.explanation];
    pushProgress(event, "success");
  } else if (event.type === "error") {
    state.output.errors = event.errors ?? [event.message];
    pushProgress(event, "warning");
  } else if (event.type === "fatal_error") {
    state.output.errors = [event.message];
    pushProgress(event, "warning");
    els.runStatus.textContent = "Error";
  } else if (event.type === "complete") {
    state.output = event.result ?? state.output;
    state.output.csv_ready = Array.isArray(state.output.generated_csv_files)
      && state.output.generated_csv_files.length > 0;
    state.output.progress = state.progress;
    els.runStatus.textContent = "Complete";
  } else if (event.type === "start") {
    pushProgress(event);
  }

  state.output.progress = state.progress;
  renderOutput();
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
  const selected = state.selectedPipelines[leaf.table_name] ?? DEFAULT_PIPELINE;
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
  state.progress = [];
  state.explanationOpen = false;
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
    ${renderErrors(state.output?.errors ?? [])}
    ${renderRunSummary(state.output)}
    ${renderProgress(state.output?.progress ?? [])}
    ${renderAnswerTable(answer)}
    ${renderInlineExplanation(state.output)}
  `;
  els.provenanceView.innerHTML = renderProvenance(answer);
  els.rowsView.innerHTML = renderSupportingRows(state.output?.rows_by_id ?? {});
  bindOutputActions();
}

function renderRunSummary(output) {
  const csvUrl = output?.csv_page_url ?? "/ui/csv";
  const csvReady = Boolean(output?.csv_ready || output?.generated_csv_files?.length);
  const hasExplanation = Array.isArray(output?.explanations) && output.explanations.length > 0;
  return `
    <div class="run-summary">
      <span>${escapeHtml(formatPipelineSummary(output))}</span>
      <div class="run-summary-actions">
        ${
          csvReady
            ? `<a class="button-link" href="${escapeHtml(csvUrl)}" target="_blank" rel="noreferrer">
                View generated CSVs
              </a>`
            : `<button class="button-link disabled-link" type="button" disabled>
                CSVs not ready
              </button>`
        }
        ${
          hasExplanation
            ? `<button class="secondary small-button" type="button" data-action="toggle-explanation">
                ${state.explanationOpen ? "Hide AP explanation" : "Show AP explanation"}
              </button>`
            : ""
        }
      </div>
    </div>
  `;
}

function renderProgress(progress) {
  if (!Array.isArray(progress) || !progress.length) {
    return "";
  }

  return `
    <div class="progress-list" aria-label="Pipeline progress">
      ${progress
        .map((item) => {
          const detail = [
            item.table ? `table ${item.table}` : "",
            item.pipeline ? pipelineLabel(item.pipeline) : "",
            Number.isFinite(item.rows) ? `${item.rows} rows` : "",
            Array.isArray(item.files) && item.files.length ? item.files.join(", ") : "",
          ]
            .filter(Boolean)
            .join(" / ");
          return `
            <div class="progress-item ${escapeHtml(item.tone ?? "info")}">
              <div class="progress-item-line">
                <span>${escapeHtml(item.message)}</span>
                ${detail ? `<small>${escapeHtml(detail)}</small>` : ""}
              </div>
              ${renderContextPreview(item.contextPreview)}
            </div>
          `;
        })
        .join("")}
    </div>
  `;
}

function renderContextPreview(contextPreview) {
  if (!contextPreview || typeof contextPreview !== "object" || !Object.keys(contextPreview).length) {
    return "";
  }

  return `
    <details class="context-preview">
      <summary>View retrieved context</summary>
      <pre>${escapeHtml(JSON.stringify(contextPreview, null, 2))}</pre>
    </details>
  `;
}

function renderInlineExplanation(output) {
  const explanations = Array.isArray(output?.explanations) ? output.explanations : [];
  if (!explanations.length) {
    return "";
  }

  return `
    <div class="inline-explanation ${state.explanationOpen ? "" : "hidden"}">
      ${renderExplanationOutput(output)}
    </div>
  `;
}

function bindOutputActions() {
  els.answerView.querySelector("[data-action='toggle-explanation']")?.addEventListener("click", () => {
    state.explanationOpen = !state.explanationOpen;
    renderOutput();
  });
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
              <strong>AP explanation ${index + 1}: ${escapeHtml(explanation.scope ?? "query")}</strong>
              ${explanation.query_sql ? `<code>${escapeHtml(explanation.query_sql)}</code>` : ""}
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
          const formula = formatProvenance(item.provenance);
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

function formatProvenance(provenance) {
  if (Array.isArray(provenance)) {
    return provenance
      .map((witnessSet) => (Array.isArray(witnessSet) ? witnessSet.join(" AND ") : String(witnessSet)))
      .join(" OR ");
  }

  if (provenance && typeof provenance === "object") {
    const preferred = provenance.formula ?? provenance.why ?? provenance.how ?? provenance.which;
    if (preferred && typeof preferred === "object" && "expression" in preferred) {
      return preferred.expression ?? "";
    }
    return JSON.stringify(provenance);
  }

  return "";
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
