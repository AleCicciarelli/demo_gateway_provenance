const IS_DEMO = document.body.dataset.view === "demo";
const BACKEND_ENDPOINTS = {
  plan: window.location.protocol === "file:" ? "" : "/ui/plan",
  run: window.location.protocol === "file:" ? "" : "/ui/run",
  runStream: window.location.protocol === "file:" ? "" : "/ui/run/stream",
};

const PIPELINES = [
  { id: "rag", label: "RAG" },
  { id: "sql-table", label: "PostgreSQL" },
  { id: "llm-internal", label: "LLM" },
];

const state = {
  dataset: "relf1",
  plan: null,
  planInput: null,
  planRevision: 0,
  selectedPipelines: {},
  leafRagOptions: {},
  unavailableTables: {},
  output: null,
  progress: [],
  explanationOpen: false,
  ragAnnotationsOpen: false,
};

const DEFAULT_PIPELINE = "rag";

const els = {
  queryInput: document.querySelector("#query-input"),
  inputType: document.querySelector("#input-type"),
  sqlPreview: document.querySelector("#resolved-sql"),
  sqlPreviewPanel: document.querySelector("#resolved-sql-panel"),
  datasetSelect: document.querySelector("#dataset-select"),
  planButton: document.querySelector("#plan-button"),
  runButton: document.querySelector("#run-button"),
  executionMode: document.querySelector("#execution-mode"),
  semanticMatchingView: document.querySelector("#semantic-matching-view"),
  backendStatus: document.querySelector("#backend-status"),
  leafCount: document.querySelector("#leaf-count"),
  planSummary: document.querySelector("#plan-summary"),
  planTree: document.querySelector("#plan-tree"),
  leafList: document.querySelector("#leaf-list"),
  runStatus: document.querySelector("#run-status"),
  answerView: document.querySelector("#answer-view"),
  comparisonView: document.querySelector("#comparison-view"),
  generationPromptView: document.querySelector("#generation-prompt-view"),
  provenanceView: document.querySelector("#provenance-view"),
  executionView: document.querySelector("#execution-view"),
  evidenceView: document.querySelector("#evidence-view"),
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

function renderMarkdown(markdown) {
  const renderInline = (value) => escapeHtml(value)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/__([^_]+)__/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    .replace(/_([^_]+)_/g, "<em>$1</em>");

  const lines = String(markdown ?? "").trim().split(/\r?\n/);
  const html = [];
  let paragraph = [];
  let listOpen = false;

  const flushParagraph = () => {
    if (paragraph.length) {
      html.push(`<p>${renderInline(paragraph.join(" "))}</p>`);
      paragraph = [];
    }
  };
  const closeList = () => {
    if (listOpen) {
      html.push("</ul>");
      listOpen = false;
    }
  };

  lines.forEach((line) => {
    const heading = line.match(/^(#{1,6})\s+(.+)$/);
    const listItem = line.match(/^\s*[-*]\s+(.+)$/);

    if (!line.trim()) {
      flushParagraph();
      closeList();
    } else if (/^\s*(---+|___+|\*\*\*+)\s*$/.test(line)) {
      flushParagraph();
      closeList();
      html.push("<hr>");
    } else if (heading) {
      flushParagraph();
      closeList();
      const level = Math.min(heading[1].length + 1, 6);
      html.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
    } else if (listItem) {
      flushParagraph();
      if (!listOpen) {
        html.push("<ul>");
        listOpen = true;
      }
      html.push(`<li>${renderInline(listItem[1])}</li>`);
    } else {
      closeList();
      paragraph.push(line.trim());
    }
  });

  flushParagraph();
  closeList();
  return html.join("");
}

function explanationMarkdown(explanation) {
  const output = explanation?.explanation_output;
  const candidates = [
    output?.result?.explanation,
    output?.result?.markdown,
    output?.explanation,
    explanation?.explanation,
  ];
  return candidates.find((value) => typeof value === "string" && value.trim()) ?? "";
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
    dataset: state.dataset,
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
    ${(plan?.warnings ?? []).map((warning) => `<p role="status">${escapeHtml(warning)}</p>`).join("")}
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

async function responseError(response) {
  const fallback = `Request failed with ${response.status}`;
  try {
    const body = await response.json();
    const detail = body.detail;
    if (typeof detail === "string" && detail.trim()) {
      return new Error(`${fallback}: ${detail}`);
    }
    if (Array.isArray(detail)) {
      const messages = detail.map((item) => {
        const location = Array.isArray(item.loc) ? item.loc.join(".") : "";
        return [location, item.msg].filter(Boolean).join(": ");
      }).filter(Boolean);
      if (messages.length) return new Error(`${fallback}: ${messages.join("; ")}`);
    }
  } catch {
    // Proxies may return an empty or non-JSON error response.
  }
  return new Error(fallback);
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw await responseError(response);
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
    throw await responseError(response);
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

function currentPlanInput() {
  return {
    question: els.queryInput.value.trim(),
    dataset: els.datasetSelect?.value ?? "relf1",
    input_type: els.inputType?.value ?? "auto",
  };
}

function invalidatePlan(message) {
  state.planRevision += 1;
  state.plan = null;
  state.planInput = null;
  state.selectedPipelines = {};
  state.leafRagOptions = {};
  state.unavailableTables = {};
  els.sqlPreview.textContent = "";
  els.sqlPreviewPanel.hidden = true;
  els.runButton.disabled = true;
  els.planButton.disabled = false;
  renderPlan();
  clearOutput();
  if (message) setStatus(message);
}

async function generatePlan() {
  const query = els.queryInput.value.trim();
  if (!query) {
    invalidatePlan("Write a question or SQL query first.");
    return;
  }

  const input = currentPlanInput();
  invalidatePlan();
  const revision = state.planRevision;
  els.planButton.disabled = true;
  setStatus("Preparing SQL and building query plan...");

  try {
    state.dataset = input.dataset;
    let preparedPlan;
    if (hasBackend(BACKEND_ENDPOINTS.plan)) {
      preparedPlan = await postJson(BACKEND_ENDPOINTS.plan, input);
      if (revision !== state.planRevision) return;
      setStatus(preparedPlan.query_resolution?.translated
        ? "Question translated to SQL. Review the SQL before running."
        : "Plan loaded from backend.");
    } else {
      preparedPlan = makeMockPlan(query);
      setStatus("Backend endpoints are empty. Showing local mock plan data.");
    }
    state.plan = preparedPlan;
    state.planInput = input;

    state.selectedPipelines = {};
    state.leafRagOptions = {};
    state.unavailableTables = {};
    for (const leaf of state.plan.plan.leaf_tasks) {
      state.selectedPipelines[leaf.table_name] = DEFAULT_PIPELINE;
      state.leafRagOptions[leaf.table_name] = { pushdown: false, iterative: false };
    }

    els.sqlPreview.textContent = state.plan.sql;
    els.sqlPreviewPanel.hidden = false;
    els.runButton.disabled = false;
    renderPlan();
    clearOutput();
  } catch (error) {
    if (revision === state.planRevision) setStatus(error.message);
  } finally {
    if (revision === state.planRevision) els.planButton.disabled = false;
  }
}

async function runSelectedPipelines() {
  if (!state.plan || JSON.stringify(state.planInput) !== JSON.stringify(currentPlanInput())) {
    invalidatePlan("Prepare the current question, dataset and input format before running pipelines.");
    return;
  }

  els.runButton.disabled = true;
  const queryControls = [els.queryInput, els.datasetSelect, els.inputType, els.planButton].filter(Boolean);
  queryControls.forEach((control) => { control.disabled = true; });
  els.runStatus.textContent = "Running";
  els.runStatus.classList.remove("muted-pill");
  setStatus("Running selected leaf pipelines...");
  state.progress = [];
  state.explanationOpen = false;

  const payload = {
    question: state.planInput.question,
    sql: state.plan.sql,
    plan: state.plan.plan,
    leaf_pipeline_choices: state.selectedPipelines,
    leaf_pipeline_options: state.leafRagOptions,
    unavailable_tables: Object.keys(state.unavailableTables).filter((table) => state.unavailableTables[table]),
    semantic_matching: !IS_DEMO && els.executionMode?.value === "semantic",
    dataset: state.plan.dataset ?? state.dataset,
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
    queryControls.forEach((control) => { control.disabled = false; });
  }
}

function makeEmptyRunOutput() {
  const choices = (state.plan?.plan?.leaf_tasks ?? []).map((leaf) => ({
    table: leaf.table_name,
    pipeline: state.selectedPipelines[leaf.table_name] ?? DEFAULT_PIPELINE,
  }));

  return {
    source: "gateway",
    dataset: state.plan?.dataset ?? state.dataset,
    answer: [],
    rows_by_id: {},
    annotations: [],
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
    retrievalQuery: event.retrieval_query,
    iterativeStep: event.iterative_step,
    inheritedBindings: event.inherited_bindings,
    sourceRowIds: event.source_row_ids,
    sourceRowSummaries: event.source_row_summaries,
    contextPreview: event.context_preview,
    stage: event.stage,
    elapsedSeconds: event.elapsed_seconds,
    model: event.model,
    modelProvider: event.model_provider,
    requestStatus: event.request_status,
    durationSeconds: event.duration_seconds,
  });
}

function matchingRequestSize(call = {}) {
  if (Number.isFinite(call.pair_count)) {
    return `${call.pair_count} ${call.pair_count === 1 ? "pair" : "pairs"}`;
  }
  if (Number.isFinite(call.left_row_count) && Number.isFinite(call.right_row_count)) {
    return `${call.left_row_count} left + ${call.right_row_count} right rows`;
  }
  return "";
}

function matchingEdgeLabel(trace) {
  if (!Number.isFinite(trace.edge_count)) return "";
  const completed = trace.completed_edges ?? 0;
  const current = trace.current_edge;
  return `${completed}/${trace.edge_count} join edges complete${current
    ? ` · Edge ${current.index} (${current.left_table} → ${current.right_table})` : ""} — `;
}

function updateMatchingProgress(event) {
  const trace = event.semantic_matching ?? {};
  const calls = trace.calls ?? [];
  const received = calls.filter((call) => typeof call.response === "string").length;
  const edge = trace.current_edge;
  const total = edge ? edge.total_requests : trace.total_requests;
  const completed = edge ? (edge.completed_requests ?? 0) : (trace.completed_requests ?? received);
  const edgeLabel = matchingEdgeLabel(trace);
  const percentage = Number.isFinite(total) && total > 0
    ? Math.floor(100 * Math.min(completed, total) / total) : (trace.status === "complete" ? 100 : null);
  const countText = Number.isFinite(total) ? `${completed}/${total} requests complete` : `${received} responses received`;
  let message = "Preparing semantic matching…";
  let tone = "info";
  if (trace.status === "skipped") {
    message = `Matching skipped — ${trace.reason}`;
    tone = "success";
  } else if (event.type === "matching_prompt") {
    const size = matchingRequestSize(calls[calls.length - 1]);
    const pairsText = size ? `${size} — ` : "";
    const requestNumber = edge ? completed + 1 : calls.length;
    message = `${edgeLabel}Comparing records — ${pairsText}request ${requestNumber}${Number.isFinite(total) ? ` of ${total}` : ""} in progress (${percentage ?? 0}%).`;
  } else if (["matching_response", "matching_progress"].includes(event.type)) {
    message = `${edgeLabel}Comparing records — ${countText}${percentage !== null ? ` (${percentage}%)` : ""}…`;
  } else if (event.type === "matching_done" || event.type === "matching_sql") {
    const count = trace.accepted_count ?? 0;
    message = `${edgeLabel}Matching complete — ${count} accepted ${count === 1 ? "pair" : "pairs"}.`;
    tone = "success";
  } else if (event.type === "matching_failed") {
    message = `Matching failed: ${trace.error ?? "Unable to complete matching."}`;
    tone = "warning";
  }
  let item = state.progress.find((entry) => entry.type === "semantic_matching");
  if (!item) {
    item = { type: "semantic_matching" };
    state.progress.push(item);
  }
  Object.assign(item, { message, tone, matchingProgress: { total, completed, percentage,
    edgeIndex: edge?.index,
    retries: Math.max(0, (total ?? 0) - (edge?.planned_requests ?? trace.planned_requests ?? total ?? 0)) } });
  setStatus(message);
}

function updateWaitingProgress(event) {
  const existing = [...state.progress]
    .reverse()
    .find((item) => (
      item.type === "heartbeat"
      || (["leaf_model_start", "model_request_start", "model_fallback_start", "model_response_received"].includes(item.type)
        && item.table === event.table)
    ));
  const update = {
    type: event.type,
    tone: "info",
    message: event.message,
    table: event.table,
    pipeline: event.pipeline,
    stage: event.stage,
    elapsedSeconds: event.elapsed_seconds,
    model: event.model,
    modelProvider: event.model_provider,
    requestStatus: event.request_status,
    durationSeconds: event.duration_seconds,
  };
  if (existing) {
    Object.assign(existing, update);
  } else {
    state.progress.push(update);
  }
}

let streamRenderPending = false;

function updateModelStream(event) {
  let item = state.progress.find((entry) => entry.streamId === event.stream_id);
  if (!item) {
    item = {
      type: "model_stream", streamId: event.stream_id, tone: "info",
      table: event.table, pipeline: event.pipeline, stage: event.stage,
      model: event.model, modelProvider: event.model_provider,
      reasoningText: "", answerText: "",
    };
    state.progress.push(item);
  }
  item.requestStatus = event.request_status;
  item.message = event.message ?? `Receiving live output from ${event.model}.`;
  if (event.type === "model_stream_delta") {
    const field = event.channel === "reasoning" ? "reasoningText" : "answerText";
    // Bound the preview while the backend retains the complete answer.
    item[field] = (item[field] + event.text).slice(-50000);
  }
  if (event.request_status === "failed") item.tone = "warning";
  if (event.request_status === "response_received") item.tone = "success";
}

function renderModelStream(item) {
  if (item.type !== "model_stream") return "";
  return `
    <div class="model-stream">
      ${item.stage === "confidence_assessment" ? "<strong>Answer assessment</strong>" : ""}
      <details open><summary>Reasoning exposed by the provider</summary>
        <pre>${escapeHtml(item.reasoningText || "No reasoning text received.")}</pre>
      </details>
      <details open><summary>Answer preview</summary>
        <pre>${escapeHtml(item.answerText || "Waiting for answer text…")}</pre>
      </details>
    </div>`;
}

function applyRunEvent(event) {
  if (!state.output) {
    state.output = makeEmptyRunOutput();
  }

  if (event.type.startsWith("model_stream_")) {
    updateModelStream(event);
    state.output.progress = state.progress;
    if (!streamRenderPending) {
      streamRenderPending = true;
      requestAnimationFrame(() => {
        streamRenderPending = false;
        renderOutput();
      });
    }
    return;
  }

  if (event.message) {
    setStatus(event.message);
  }

  if (event.type.startsWith("matching_")) {
    state.output.semantic_matching = event.semantic_matching;
    if (event.type === "matching_done") {
      state.output.generated_csv_files = [...new Set([
        ...(state.output.generated_csv_files ?? []),
        ...(event.semantic_matching.mapping_files ?? []),
        ...(event.semantic_matching.mapping_file ? [event.semantic_matching.mapping_file] : []),
      ])];
    }
    updateMatchingProgress(event);
  } else if (event.type === "leaf_start") {
    pushProgress(event);
  } else if (event.type === "iterative_join_leaf_start") {
    pushProgress(event);
  } else if (event.type === "leaf_retrieval_start") {
    pushProgress(event);
  } else if (event.type === "leaf_context") {
    if (Array.isArray(event.annotations) && event.annotations.length) {
      state.output.annotations.push(...event.annotations);
    }
    pushProgress(event, "success");
  } else if (event.type === "leaf_model_start") {
    pushProgress(event);
  } else if (event.type === "generation_prompt") {
    state.output.leaf_outputs ??= [];
    let leaf = state.output.leaf_outputs.find((item) => (
      item.table_name === event.table && item.pipeline === event.pipeline
    ));
    if (!leaf) {
      leaf = { table_name: event.table, pipeline: event.pipeline, generation_prompts: [] };
      state.output.leaf_outputs.push(leaf);
    }
    leaf.prompt ??= event.prompt;
    leaf.generation_prompts ??= [];
    leaf.generation_prompts.push(event.prompt);
  } else if (event.type === "model_request_start") {
    updateWaitingProgress(event);
  } else if (event.type === "model_fallback_start") {
    updateWaitingProgress(event);
  } else if (event.type === "model_response_received") {
    updateWaitingProgress(event);
  } else if (event.type === "leaf_model_done") {
    state.progress = state.progress.filter((item) => item.type !== "heartbeat");
    pushProgress(event, "success");
  } else if (event.type === "iterative_join_leaf_done") {
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
    if (state.output.comparison) state.output.comparison.status = "unavailable";
    pushProgress(event, "warning");
    els.runStatus.textContent = "Error";
  } else if (event.type === "complete") {
    state.output = event.result ?? state.output;
    state.output.csv_ready = Array.isArray(state.output.generated_csv_files)
      && state.output.generated_csv_files.length > 0;
    state.output.progress = state.progress;
    els.runStatus.textContent = state.output.errors?.length ? "Error" : "Complete";
  } else if (event.type === "reference_start" || event.type === "reference_done") {
    if (event.type === "reference_done") state.output.comparison = { status: "pending", reference: event.reference };
    setStatus(event.type === "reference_start" ? "Computing the SQL reference answer and provenance…" : "SQL reference finished. Running your selected pipelines…");
    pushProgress(event);
  } else if (event.type === "start") {
    pushProgress(event);
  } else if (event.type === "heartbeat") {
    updateWaitingProgress(event);
  }

  state.output.progress = state.progress;
  renderOutput();
}

function renderPlan() {
  const plan = state.plan?.plan;
  const leaves = plan?.leaf_tasks ?? [];

  els.leafCount.textContent = `${leaves.length} ${leaves.length === 1 ? "leaf" : "leaves"}`;
  els.planSummary.innerHTML = IS_DEMO ? "" : [
    ["Type", plan?.query_type ?? "-"],
    ["Joins", plan?.joins?.length ?? 0],
    ["Post-ops", plan?.post_ops?.length ?? 0],
    ["Dataset", state.plan?.dataset ?? state.dataset],
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

  if (IS_DEMO) renderDemoPlanTree(plan);
  else renderPlanTree(plan);

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
      renderPlan();
    });
  });
  els.leafList.querySelectorAll("input[data-source-unavailable]").forEach((input) => {
    input.addEventListener("change", (event) => {
      const table = event.target.dataset.table;
      state.unavailableTables[table] = event.target.checked;
      if (event.target.checked) {
        state.selectedPipelines[table] = "llm-internal";
        for (const options of Object.values(state.leafRagOptions)) options.iterative = false;
      }
      clearOutput();
      renderPlan();
    });
  });
  els.leafList.querySelectorAll("input[data-rag-option]").forEach((input) => {
    input.addEventListener("change", (event) => {
      const table = event.target.dataset.table;
      const option = event.target.dataset.ragOption;
      state.leafRagOptions[table] ??= { pushdown: false, iterative: false };
      state.leafRagOptions[table][option] = event.target.checked;
      if (event.target.checked) {
        const otherOption = option === "pushdown" ? "iterative" : "pushdown";
        state.leafRagOptions[table][otherOption] = false;
      }
      renderPlan();
    });
  });
}

function renderDemoPlanTree(plan) {
  const tree = planToTree(plan);
  els.planTree.replaceChildren();
  els.planTree.className = "plan-tree demo-query-tree";
  if (!tree) {
    els.planTree.textContent = "Prepare your query to see its plan.";
    return;
  }
  const labels = { Project: "Choose result columns", GroupBy: "Group related rows", Aggregate: "Calculate summaries", OrderBy: "Sort results", Limit: "Limit results", Distinct: "Remove duplicates", LeafScan: "Read records" };
  function branch(node) {
    const item = document.createElement("li");
    const label = document.createElement("div");
    label.className = `demo-plan-node demo-node-${node.type}`;
    const title = document.createElement("strong");
    const detail = document.createElement("span");
    if (node.type === "table") {
      title.textContent = "Source table";
      detail.textContent = node.label;
    } else if (node.type === "filter") {
      title.textContent = "Filter rows";
      detail.textContent = node.label.replace(/^sigma /, "");
    } else if (node.type === "join") {
      title.textContent = "Combine matching rows";
      detail.textContent = node.label.replace(/^join\s*/, "");
    } else {
      title.textContent = labels[node.label] ?? node.label;
    }
    label.append(title);
    if (detail.textContent) label.append(detail);
    item.append(label);
    if (node.children.length) {
      const children = document.createElement("ul");
      children.append(...node.children.map(branch));
      item.append(children);
    }
    return item;
  }
  const root = document.createElement("ul");
  root.className = "demo-plan-tree";
  root.setAttribute("aria-label", "Query plan; read from source tables up to the results");
  root.append(branch(tree));
  const hint = document.createElement("p");
  hint.className = "demo-tree-hint";
  hint.textContent = "Read from the source tables at the bottom ↑ to the final result at the top.";
  els.planTree.append(hint, root);
}

function renderLeafCard(leaf, index) {
  const selected = state.selectedPipelines[leaf.table_name] ?? DEFAULT_PIPELINE;
  const ragOptions = state.leafRagOptions[leaf.table_name] ?? {
    pushdown: false,
    iterative: false,
  };
  const sourceUnavailable = Boolean(state.unavailableTables[leaf.table_name]);
  const hasUnavailableSources = Object.values(state.unavailableTables).some(Boolean);
  const options = PIPELINES.map(
    (pipeline) => `
      <option value="${escapeHtml(pipeline.id)}" ${pipeline.id === selected ? "selected" : ""} ${sourceUnavailable && pipeline.id !== "llm-internal" ? "disabled" : ""}>
        ${escapeHtml(IS_DEMO ? ({rag: "RAG · retrieve records from documents", "sql-table": "Database · use source table records", "llm-internal": "LLM · generate records from model knowledge"}[pipeline.id]) : pipeline.label)}
      </option>
    `,
  ).join("");
  return `
    <article class="leaf-card">
      <div class="leaf-card-header">
        <div>
          <h3>${IS_DEMO ? "Table" : `Leaf ${index + 1}:`} ${escapeHtml(leaf.table_name)}</h3>
          <div class="leaf-meta">
            <span class="meta-chip">${escapeHtml(leaf.scan_op)}</span>
            ${leaf.alias ? `<span class="meta-chip">alias ${escapeHtml(leaf.alias)}</span>` : ""}
          </div>
        </div>
        <div class="leaf-pipeline-control">
          <label>
            <span class="field-label">Pipeline</span>
            <select data-table="${escapeHtml(leaf.table_name)}">${options}</select>
          </label>
          <label>
            <input type="checkbox" data-source-unavailable data-table="${escapeHtml(leaf.table_name)}"
              ${sourceUnavailable ? "checked" : ""} />
            Source data unavailable
          </label>
          <small>${sourceUnavailable ? "Stored rows blocked for this run. Uses LLM knowledge; SQL reference may be unavailable." : "Source data available. Check to block stored rows for this run."}</small>
          ${selected === "rag" ? `
            <div class="leaf-rag-options">
              <label>
                <input type="checkbox" data-table="${escapeHtml(leaf.table_name)}"
                  data-rag-option="pushdown" ${ragOptions.pushdown ? "checked" : ""}
                  ${ragOptions.iterative ? "disabled" : ""} />
                Pushdown
              </label>
              <label>
                <input type="checkbox" data-table="${escapeHtml(leaf.table_name)}"
                  data-rag-option="iterative" ${ragOptions.iterative ? "checked" : ""}
                  ${ragOptions.pushdown || hasUnavailableSources ? "disabled" : ""} />
                Iterative RAG (whole query)
              </label>
              <small>Iterative RAG runs every leaf with RAG and includes local predicates.</small>
            </div>
          ` : ""}
        </div>
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
  if (els.executionView) els.executionView.innerHTML = "";
  if (els.evidenceView) els.evidenceView.innerHTML = "";
  state.progress = [];
  state.explanationOpen = false;
  els.runStatus.textContent = "Waiting";
  els.runStatus.classList.add("muted-pill");
  els.answerView.innerHTML = `<div class="empty-state">Run the selected pipelines to see final output.</div>`;
  els.comparisonView.innerHTML = "";
  els.semanticMatchingView.innerHTML = "";
  delete els.semanticMatchingView.dataset.rendered;
  els.generationPromptView.innerHTML = "";
  delete els.generationPromptView.dataset.rendered;
  els.provenanceView.innerHTML = `<div class="empty-state">Provenance will appear here after a run.</div>`;
}

function renderMatchedRecord(record, table) {
  return `<div class="matched-record">
    <h5>${escapeHtml(table || "Record")}</h5>
    <dl>${Object.entries(record.attributes ?? {}).map(([key, value]) => `
      <div><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value ?? "Unknown")}</dd></div>
    `).join("")}</dl>
    <small>Row: <code>${escapeHtml(record.id)}</code></small>
  </div>`;
}

function renderSemanticMatching(trace) {
  if (!trace) return "";
  if (Array.isArray(trace.joins)) {
    return `<section class="matching-summary"><h3>Join rewriting</h3>
      ${trace.error ? `<p>${escapeHtml(trace.error)}</p>` : ""}
      ${trace.rewritten_sql ? `<details><summary>SQL sent for execution</summary><pre>${escapeHtml(trace.service_sql ?? trace.rewritten_sql)}</pre></details>` : ""}
      ${trace.joins.map((join) => `<section><h4>${escapeHtml(join.left_table)} → ${escapeHtml(join.right_table)}</h4>${renderSemanticMatching(join)}</section>`).join("")}
    </section>`;
  }
  if (trace.status === "skipped") {
    return `<section class="matching-summary"><h3>Matching skipped</h3><p>${escapeHtml(trace.reason)}</p></section>`;
  }
  const accepted = trace.audit?.accepted_pairs ?? [];
  const coverage = trace.audit?.assessment_summary;
  const coverageHtml = coverage && Object.keys(coverage).length ? `
    <p>Assessments across completed blocks: ${coverage.assessed_left_rows ?? 0} left-row assessments ·
      ${coverage.explicit_matches ?? 0} explicit matches · ${coverage.explicit_rejections ?? 0} explicit rejections ·
      ${coverage.uncertain_pairs ?? 0} uncertain pairs · ${coverage.unresolved_left_rows ?? 0} unresolved row assessments ·
      ${coverage.no_partner_left_rows ?? 0} no-partner assessments ·
      ${coverage.omitted_pairs ?? 0} pairs without an explicit pair decision ·
      ${coverage.validation_downgrades ?? 0} validation downgrades.</p>
    <p>Omitted pairs are not explicit rejections. Row assessments apply to one candidate block;
      a row may be assessed in several blocks. Completed requests do not establish factual correctness.</p>` : "";
  const summary = trace.status === "complete" ? `<section class="matching-summary">
    <h3>Retained pairs (${escapeHtml(trace.accepted_count ?? accepted.length)})</h3>
    <p>The join retains only explicit matches, with a fixed weight of 0.9. This weight is not calibrated factual confidence. Unmatch and uncertain decisions remain in the audit.</p>
    ${coverageHtml}
    ${accepted.length ? `<div class="matched-pairs">${accepted.map((pair, index) => `
      <article class="matched-pair">
        <h4>Pair ${index + 1}</h4>
        <p>Decision: ${escapeHtml(pair.decision ?? "match")} · Probability: ${escapeHtml(pair.probability ?? "Unknown")}</p>
        <div class="matched-records">
          ${renderMatchedRecord(pair.left, trace.left_table)}
          <span class="match-link" aria-label="matched with">↔</span>
          ${renderMatchedRecord(pair.right, trace.right_table)}
        </div>
        <p class="match-reason"><strong>Reason:</strong> ${escapeHtml(pair.reason)}</p>
        ${trace.mapping_rows?.[index]?.[trace.rownum_column] ? `<small>Mapping row: <code>${escapeHtml(trace.mapping_rows[index][trace.rownum_column])}</code></small>` : ""}
      </article>`).join("")}</div>` : `<p class="empty-state">${trace.accepted_count ? "Pair details are unavailable for this run." : "No explicit matches were retained. Consult row assessments and pair decisions below."}</p>`}
  </section>` : "";
  const calls = (trace.calls ?? []).map((call, index) => `
    <details class="matching-call" data-matching-detail="request-${index}">
      <summary>Request ${index + 1}${matchingRequestSize(call) ? ` · ${matchingRequestSize(call)}` : ""}${call.is_retry ? " · Retry" : ""} · ${call.response !== undefined ? "Response received" : trace.status === "failed" ? "Failed" : "In progress"}</summary>
      <h4>Prompt</h4><pre>${escapeHtml(call.prompt)}</pre>
      <h4>Model output</h4><pre>${escapeHtml(call.response ?? "Waiting for response…")}</pre>
    </details>`).join("");
  return `${summary}<details class="semantic-trace" data-matching-detail="overview">
    <summary>Matching requests and details · ${escapeHtml(trace.status)}</summary>
    ${Number.isFinite(trace.total_requests) ? `<p>${escapeHtml(matchingEdgeLabel(trace))}${trace.completed_requests ?? 0} of ${trace.total_requests} ${trace.current_edge ? "current-edge " : ""}requests complete. Missing assessments get at most one retry per block. Request completion does not validate matching accuracy.</p>` : ""}
    ${trace.error ? `<p>${escapeHtml(trace.error)}</p>` : ""}
    ${calls}
    ${trace.rewritten_sql ? `<h4>Rewritten SQL</h4><pre>${escapeHtml(trace.service_sql ?? trace.rewritten_sql)}</pre>` : ""}
    ${trace.mapping_csv !== undefined ? `<h4>${escapeHtml(trace.mapping_file)}</h4><pre>${escapeHtml(trace.mapping_csv)}</pre>` : ""}
  </details>`;
}

function renderSqlComparison(output) {
  const comparison = output?.comparison;
  if (!comparison) return "";
  const reference = comparison.reference;
  const complete = comparison.status === "complete";
  const summary = comparison.status === "pending" ? "SQL reference ready. Waiting for the selected pipelines to finish…" : complete
    ? `${comparison.answers_equal ? "Answers match" : "Answers differ"} · ${comparison.matched_count} shared · ${comparison.missing_count} missing from selected pipelines · ${comparison.extra_count} extra`
    : `Comparison unavailable: ${comparison.reason ?? reference.error ?? "one of the executions did not complete."}`;
  function table(answer, statuses) {
    if (!answer.length) return '<p class="empty-state">No result rows.</p>';
    const columns = [...new Set(answer.flatMap((item) => Object.keys(item.result ?? {})))];
    return `<div class="table-wrap"><table><thead><tr><th>Comparison</th>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr></thead><tbody>${answer.map((item, index) => {
      const status = complete ? statuses[index] : "unavailable";
      const label = {shared: "Shared", missing: "Missing from selected", extra: "Extra in selected", unavailable: "Not compared"}[status];
      return `<tr class="comparison-${status}"><td><strong>${label}</strong></td>${columns.map((column) => `<td>${escapeHtml(item.result?.[column] == null ? "—" : typeof item.result[column] === "object" ? JSON.stringify(item.result[column]) : item.result[column])}</td>`).join("")}</tr>`;
    }).join("")}</tbody></table></div>`;
  }
  return `<section class="sql-comparison" aria-labelledby="comparison-title">
    <h3 id="comparison-title">Output tuples: SQL vs selected pipelines</h3>
    <p role="status">${escapeHtml(summary)}</p>
    <p class="status-note">The SQL reference uses source tables permitted for this run. Unavailable source tables are excluded; queries requiring them cannot execute. With full access, the original query runs through the explanation service. Rows are compared by all column names and values, including duplicate counts; row order is ignored. Queries with LIMIT without a unique ordering may return different valid subsets.</p>
    <div class="comparison-grid">
      <section><h4>SQL reference · ${reference.answer.length} rows</h4>${reference.status !== "complete" ? `<p>${escapeHtml(reference.error ?? "SQL reference unavailable.")}</p>` : table(reference.answer, comparison.reference_status ?? [])}</section>
      <section><h4>Selected pipelines · ${output.answer?.length ?? 0} rows</h4>${table(output.answer ?? [], comparison.selected_status ?? [])}</section>
    </div>
  </section>`;
}

function renderRunProvenance(output) {
  const selected = renderProvenance(output?.answer ?? []);
  const reference = output?.comparison?.reference;
  if (!reference || reference.status !== "complete") return selected;
  return `<h3>Selected pipelines</h3>${selected}<h3>SQL reference</h3>${renderProvenance(reference.answer, { reference: true })}`;
}

function renderOutput() {
  renderComparisonViews();
  preparedSourceSnapshots = new WeakSet();
  els.comparisonView.innerHTML = renderSqlComparison(state.output);
  if (IS_DEMO) {
    const answer = state.output?.answer ?? [];
    els.answerView.innerHTML = `
      ${state.output?.errors?.length ? '<p role="status">The query could not finish completely. Any available results are shown below. See the technical workspace for diagnostics.</p>' : ""}
      ${renderAnswerTable(answer)}
      ${state.output?.explanations?.length ? `<button class="secondary" type="button" data-action="toggle-explanation">${state.explanationOpen ? "Hide" : "Show"} explanation</button>` : ""}
      ${renderInlineExplanation(state.output)}
    `;
    els.provenanceView.innerHTML = renderRunProvenance(state.output);
    bindOutputActions();
    return;
  }
  const semanticHtml = renderSemanticMatching(state.output?.semantic_matching);
  if (els.semanticMatchingView.dataset.rendered !== semanticHtml) {
    const openDetails = new Set([...els.semanticMatchingView.querySelectorAll("details[open][data-matching-detail]")]
      .map((detail) => detail.dataset.matchingDetail));
    els.semanticMatchingView.innerHTML = semanticHtml;
    els.semanticMatchingView.querySelectorAll("details[data-matching-detail]").forEach((detail) => {
      detail.open = openDetails.has(detail.dataset.matchingDetail);
    });
    els.semanticMatchingView.dataset.rendered = semanticHtml;
  }
  const answer = state.output?.answer ?? [];
  const promptHtml = renderGenerationPrompts(state.output?.leaf_outputs ?? []);
  // Avoid replacing this panel on every progress event, preserving its scroll
  // position and open/closed state while the model is generating.
  if (els.generationPromptView.dataset.rendered !== promptHtml) {
    els.generationPromptView.innerHTML = promptHtml;
    els.generationPromptView.dataset.rendered = promptHtml;
  }
  els.answerView.innerHTML = `
    ${state.output?.source === "mock" ? `<p class="notice">Mock output. Wire BACKEND_ENDPOINTS.run in app.js to replace this.</p>` : ""}
    ${renderErrors(state.output?.errors ?? [])}
    ${renderRunSummary(state.output)}
    ${renderProgress(state.output?.progress ?? [])}
    ${renderRagAnnotations(state.output?.annotations ?? [])}
    ${renderInternalKnowledgeOutputs(
      state.output?.leaf_outputs ?? [],
      state.output?.annotations ?? [],
    )}
    ${renderAnswerTable(answer)}
    ${renderInlineExplanation(state.output)}
  `;
  els.provenanceView.innerHTML = renderRunProvenance(state.output);
  bindOutputActions();
}

function renderGenerationPrompts(leafOutputs) {
  const leaves = Array.isArray(leafOutputs) ? leafOutputs.filter((leaf) => (
    typeof leaf?.prompt === "string"
    && leaf.prompt.trim()
    && leaf.prompt !== "SQL TABLE MODE"
    && leaf.prompt !== "MANUAL REVIEW MODE"
  )) : [];
  if (!leaves.length) return "";
  return `
    <section class="annotation-section" aria-label="LLM generation prompts">
      <h3>LLM generation prompts</h3>
      ${leaves.map((leaf) => {
        const batches = Array.isArray(leaf.batches) ? leaf.batches : [];
        const prompts = batches.length
          ? batches.flatMap((batch, batchIndex) => {
              const attempts = Array.isArray(batch.generation_prompts) && batch.generation_prompts.length
                ? batch.generation_prompts : [batch.prompt];
              return attempts.map((prompt, attemptIndex) => ({
                prompt,
                label: `batch ${batchIndex + 1}/${batches.length} · attempt ${attemptIndex + 1}${attemptIndex ? " (retry)" : ""}${batch.status === "split" ? " · split after output limit" : ""}`,
              }));
            })
          : (Array.isArray(leaf.generation_prompts) && leaf.generation_prompts.length
              ? leaf.generation_prompts : [leaf.prompt]).map((prompt, index) => ({
                prompt, label: `request ${index + 1}`,
              }));
        return `
          <div class="annotation-card">
            <strong>${escapeHtml(leaf.table_name ?? "query")} · ${escapeHtml(pipelineLabel(leaf.pipeline))}</strong>
            ${leaf.retrieval_query ? `<details class="context-preview">
              <summary>View retrieval query</summary>
              <pre>${escapeHtml(leaf.retrieval_query)}</pre>
            </details>` : ""}
            ${prompts.map(({ prompt, label }) => `
              <details class="context-preview internal-prompt" open>
                <summary>View generation prompt · ${escapeHtml(label)}</summary>
                <pre>${escapeHtml(prompt)}</pre>
              </details>
            `).join("")}
          </div>
        `;
      }).join("")}
    </section>
  `;
}

function renderInternalKnowledgeOutputs(leafOutputs, annotations) {
  if (!Array.isArray(leafOutputs)) {
    return "";
  }

  const internalOutputs = leafOutputs.filter((leaf) => (
    leaf?.pipeline === "llm-internal"
    && typeof leaf.output_text === "string"
    && leaf.output_text.trim()
  ));
  if (!internalOutputs.length) {
    return "";
  }

  return `
    <section class="annotation-section" aria-label="Internal-knowledge model output">
      <h3>Internal-knowledge model output</h3>
      ${internalOutputs
        .map((leaf) => {
          const confidence = [
            ...(Array.isArray(leaf.annotations) ? leaf.annotations : []),
            ...(Array.isArray(annotations) ? annotations : []),
          ].find((annotation) => (
            annotation?.type === "llm_confidence"
            && annotation.scope?.table === leaf.table_name
          ));
          const level = String(confidence?.level ?? "unknown").toLowerCase();
          const levelClass = {
            high: "similarity-green",
            medium: "similarity-yellow",
            low: "similarity-red",
          }[level] ?? "similarity-unknown";
          const pipelineAnswer = Array.isArray(leaf.parsed_output)
            ? leaf.parsed_output.map((item) => ({
                result: item?.values ?? {},
                confidence_annotations: item?.confidence_annotations ?? [],
              }))
            : [];
          return `
            <div class="annotation-card confidence-card internal-output-card">
              <div class="annotation-heading">
                <strong>${escapeHtml(leaf.table_name ?? "query")} · internal-knowledge answer</strong>
                <span class="annotation-badge">LLM output</span>
              </div>
              <details class="context-preview internal-prompt">
                <summary>View prompt sent to the model</summary>
                <pre>${escapeHtml(leaf.prompt ?? "Prompt unavailable.")}</pre>
              </details>
              ${renderAnswerTable(pipelineAnswer, "Row assessment")}
              <div class="internal-confidence-summary">
                <strong>Block assessment</strong>
                <span class="confidence-level ${levelClass}">${escapeHtml(level)}</span>
                <span>${escapeHtml(confidence?.reason ?? "No confidence assessment was returned.")}</span>
              </div>
              <p>
                ${escapeHtml(confidence?.model ?? "unknown model")} ·
                ${escapeHtml(confidence?.assessment_method ?? "self-assessment")}
              </p>
            </div>
          `;
        })
        .join("")}
    </section>
  `;
}

function renderRagAnnotations(annotations) {
  const ragAnnotations = Array.isArray(annotations)
    ? annotations.filter((annotation) => annotation?.type === "rag_similarity")
    : [];
  if (!ragAnnotations.length) {
    return "";
  }

  return `
    <details class="annotation-section rag-annotations" data-rag-annotations
      ${state.ragAnnotationsOpen ? "open" : ""}>
      <summary>RAG annotations · ${ragAnnotations.length} retrieval ${ragAnnotations.length === 1 ? "step" : "steps"}</summary>
      <div class="rag-annotations-content">
      ${ragAnnotations
        .map((annotation) => {
          const evidence = Array.isArray(annotation.evidence)
            ? annotation.evidence.filter((item) => (
                item?.source_type !== "correlated"
                && item?.retrieval_method !== "relational_correlation"
              ))
            : [];
          const table = annotation.scope?.table ?? "unknown leaf";
          const embedding = annotation.embedding_model ?? annotation.embedding_strategy ?? "unknown";
          return `
            <div class="annotation-card">
              <div class="annotation-heading">
                <strong>RAG retrieval · ${escapeHtml(table)}</strong>
                <span class="annotation-badge">RAG annotation</span>
              </div>
              <p>
                ${escapeHtml(pipelineLabel(annotation.pipeline))} ·
                ${escapeHtml(embedding)}
                ${annotation.embedding_strategy ? ` (${escapeHtml(annotation.embedding_strategy)})` : ""}
              </p>
              <p>
                FAISS relevance score; higher values indicate greater similarity to the retrieval query.
              </p>
              <div class="similarity-legend" aria-label="Similarity score legend">
                <span class="similarity-chip similarity-green">Close ≥ 0.75</span>
                <span class="similarity-chip similarity-yellow">Relevant ≥ 0.50</span>
                <span class="similarity-chip similarity-orange">Weak ≥ 0.25</span>
                <span class="similarity-chip similarity-red">Far &lt; 0.25</span>
              </div>
              <code>${escapeHtml(annotation.retrieval_query ?? "")}</code>
              ${
                evidence.length
                  ? `<div class="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>Rank</th>
                            <th>Retrieval method</th>
                            <th>Source identifier</th>
                            <th>Table</th>
                            <th>Row id</th>
                            <th>Chunk id</th>
                            <th>Similarity score</th>
                          </tr>
                        </thead>
                        <tbody>
                          ${evidence
                            .map(
                              (item) => `
                                <tr>
                                  <td>${escapeHtml(item.rank ?? "—")}</td>
                                  <td>${escapeHtml(
                                    item.retrieval_method === "relational_correlation"
                                      ? `Correlated from ${item.source_table ?? "row"} ${item.source_row_id ?? ""}`
                                      : "FAISS similarity",
                                  )}</td>
                                  <td><code>${escapeHtml(item.source_id ?? "")}</code></td>
                                  <td>${escapeHtml(item.table)}</td>
                                  <td>${escapeHtml(item.row_id)}</td>
                                  <td>${escapeHtml(item.chunk_id ?? item.row_id)}</td>
                                  <td>${renderSimilarityScore(item.score)}</td>
                                </tr>
                              `,
                            )
                            .join("")}
                        </tbody>
                      </table>
                    </div>`
                  : `<div class="empty-state">No scored evidence rows were retrieved.</div>`
              }
            </div>
          `;
        })
        .join("")}
      </div>
    </details>
  `;
}

function similarityBand(score) {
  if (!Number.isFinite(score)) {
    return { className: "similarity-unknown", label: "Unknown" };
  }
  if (score >= 0.75) {
    return { className: "similarity-green", label: "Close" };
  }
  if (score >= 0.5) {
    return { className: "similarity-yellow", label: "Relevant" };
  }
  if (score >= 0.25) {
    return { className: "similarity-orange", label: "Weak" };
  }
  return { className: "similarity-red", label: "Far" };
}

function renderSimilarityScore(score) {
  const numericScore = typeof score === "number" ? score : Number(score);
  const band = similarityBand(numericScore);
  const value = Number.isFinite(numericScore) ? numericScore.toFixed(4) : "N/A";
  return `
    <span class="similarity-score ${band.className}">
      <strong>${escapeHtml(value)}</strong>
      <small>${escapeHtml(band.label)}</small>
    </span>
  `;
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
              ${renderMatchingProgress(item)}
              ${renderProgressDetails(item)}
              ${renderModelStream(item)}
              ${renderContextPreview(item.contextPreview)}
            </div>
          `;
        })
        .join("")}
    </div>
  `;
}

function renderMatchingProgress(item) {
  const progress = item.matchingProgress;
  if (!progress || progress.percentage === null) return "";
  return `<div class="matching-progress-meter">
    <progress max="100" value="${progress.percentage}" aria-label="${progress.edgeIndex ? "Current join matching" : "Matching"} requests completed"></progress>
    <small>${progress.edgeIndex ? `Edge ${escapeHtml(progress.edgeIndex)} · ` : ""}${escapeHtml(progress.completed)}/${escapeHtml(progress.total)} requests complete · ${progress.percentage}%${progress.retries ? ` · includes ${progress.retries} extra retries` : ""}</small>
  </div>`;
}

function renderProgressDetails(item) {
  const hasRetrieval = typeof item.retrievalQuery === "string" && item.retrievalQuery.trim();
  const hasBindings = item.inheritedBindings
    && typeof item.inheritedBindings === "object"
    && Object.keys(item.inheritedBindings).length;
  const hasSources = Array.isArray(item.sourceRowIds) && item.sourceRowIds.length;
  const hasSourceSummaries = Array.isArray(item.sourceRowSummaries) && item.sourceRowSummaries.length;
  const hasStep = Number.isFinite(item.iterativeStep);
  const hasModel = typeof item.model === "string" && item.model.trim();
  const hasProvider = typeof item.modelProvider === "string" && item.modelProvider.trim();
  const hasDuration = Number.isFinite(item.durationSeconds);
  const modelLabel = item.requestStatus === "response_received"
    ? "Active model"
    : item.requestStatus === "fallback_started" ? "Fallback model" : "Requested model";
  const providerLabel = item.requestStatus === "response_received"
    ? "Active provider"
    : item.requestStatus === "fallback_started" ? "Fallback provider" : "Requested provider";

  if (!hasRetrieval && !hasBindings && !hasSources && !hasSourceSummaries && !hasStep
      && !hasModel && !hasProvider && !hasDuration) {
    return "";
  }

  return `
    <div class="progress-details">
      ${hasStep ? renderProgressDetail("Step", item.iterativeStep) : ""}
      ${hasRetrieval ? renderProgressDetail("Retrieval query", item.retrievalQuery) : ""}
      ${hasBindings ? renderProgressDetail("Inherited bindings", JSON.stringify(item.inheritedBindings)) : ""}
      ${hasSources ? renderProgressDetail("Source rows", item.sourceRowIds.join(", ")) : ""}
      ${hasSourceSummaries ? renderProgressDetail("Source values", item.sourceRowSummaries.join(" | ")) : ""}
      ${hasModel ? renderProgressDetail(modelLabel, item.model) : ""}
      ${hasProvider ? renderProgressDetail(providerLabel, item.modelProvider) : ""}
      ${hasDuration ? renderProgressDetail("Duration", `${item.durationSeconds.toFixed(2)}s`) : ""}
    </div>
  `;
}

function renderProgressDetail(label, value) {
  return `
    <div class="progress-detail">
      <span>${escapeHtml(label)}</span>
      <code>${escapeHtml(value)}</code>
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
  els.answerView.querySelector("[data-rag-annotations]")?.addEventListener("toggle", (event) => {
    // Keep the user's choice when streaming progress re-renders the answer panel.
    if (event.currentTarget.isConnected) {
      state.ragAnnotationsOpen = event.currentTarget.open;
    }
  });
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
              ${!IS_DEMO && explanation.query_sql ? `<code>${escapeHtml(explanation.query_sql)}</code>` : ""}
              ${!IS_DEMO && explanation.question_sql ? `<code>${escapeHtml(explanation.question_sql)}</code>` : ""}
              <div class="explanation-markdown">
                ${
                  explanationMarkdown(explanation)
                    ? renderMarkdown(explanationMarkdown(explanation))
                    : "<p>No written explanation was returned by the AP explanation service.</p>"
                }
              </div>
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

const sourceSnapshots = new WeakMap();
let preparedSourceSnapshots = new WeakSet();

function resultSources(item) {
  const annotated = (item.provenance_annotations ?? []).flatMap((entry) => entry.source_rows ?? []);
  if (annotated.length) return annotated;
  const ids = new Set((item.provenance ?? []).flat?.(Infinity) ?? []);
  return Object.entries(state.output?.rows_by_id ?? {})
    .filter(([id]) => ids.has(id))
    .map(([row_id, source]) => ({ row_id, ...source }));
}

function sourceLink(source) {
  const params = new URLSearchParams({
    dataset: state.output?.dataset ?? state.dataset,
    table: source.table,
    row_id: source.row_id,
  });
  const generated = ["llm-internal", "semantic-matching"].includes(source.pipeline);
  if (generated) {
    let key = sourceSnapshots.get(state.output);
    if (!key) {
      const runId = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`;
      key = `provenance-sources-${runId}`;
      sourceSnapshots.set(state.output, key);
    }
    if (!preparedSourceSnapshots.has(state.output)) {
      const tables = {};
      const seen = new Map();
      for (const leaf of state.output?.leaf_outputs ?? []) {
        if (leaf.pipeline !== "llm-internal") continue;
        tables[leaf.table_name] ??= [];
        if (!seen.has(leaf.table_name)) seen.set(leaf.table_name, new Set());
        for (const item of leaf.parsed_output ?? []) {
          if (!seen.get(leaf.table_name).has(item.row_id)) {
            tables[leaf.table_name].push({ ...item.values, __rid__: item.row_id });
            seen.get(leaf.table_name).add(item.row_id);
          }
        }
      }
      const matching = state.output?.semantic_matching;
      for (const trace of matching?.joins ?? (matching ? [matching] : [])) {
        if (trace.mapping_table) {
          tables[trace.mapping_table] = (trace.mapping_rows ?? []).map((row) => ({
            ...row, __rid__: row[`${trace.mapping_table}_rownum`],
          }));
        }
      }
      try {
        localStorage.setItem(key, JSON.stringify(tables));
        preparedSourceSnapshots.add(state.output);
      } catch {
        return `<span title="Browser storage is unavailable">${escapeHtml(source.table)} · ${escapeHtml(source.row_id)} (preview unavailable)</span>`;
      }
    }
    params.set("snapshot", key);
  }
  return `<a class="source-link" href="/ui/source.html?${escapeHtml(params.toString())}" target="_blank" rel="noopener"><span class="source-link-heading"><strong>${escapeHtml(source.table)}</strong>${generated ? `<span class="source-kind">Generated</span>` : ""}<span class="source-open" aria-hidden="true">↗</span></span><span class="source-row-id">${escapeHtml(source.row_id)}</span></a>`;
}

function renderAnswerTable(answer, confidenceLabel = "Probabilities") {
  const rows = answer.map((item) => item.result ?? {});
  const columns = [...new Set(rows.flatMap((row) => Object.keys(row)))];
  const hasSources = answer.some((item) => resultSources(item).length);
  const hasProbabilities = answer.some((item) => item.probability_annotations?.some((entry) => entry.type === "final_answer_probability"));
  const hasConfidence = answer.some(
    (item) =>
      (Array.isArray(item.probability_annotations) && item.probability_annotations.length) ||
      (Array.isArray(item.confidence_annotations) && item.confidence_annotations.length),
  );
  if (!rows.length || !columns.length) {
    return `<div class="empty-state">No answer rows.</div>`;
  }

  return `
    ${hasProbabilities ? `<div class="probability-legend" aria-label="Probability color scale">
      <span>Probability:</span>
      <span><i class="legend-green" aria-hidden="true"></i>75–100%</span>
      <span><i class="legend-yellow" aria-hidden="true"></i>50–&lt;75%</span>
      <span><i class="legend-red" aria-hidden="true"></i>&lt;50%</span>
    </div>` : ""}
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            ${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}
            ${hasConfidence ? `<th>${escapeHtml(confidenceLabel)}</th>` : ""}
            ${hasSources ? "<th>Sources</th>" : ""}
          </tr>
        </thead>
        <tbody>
          ${answer
            .map(
              (item) => `
                <tr>
                  ${columns
                    .map((column) => `<td>${escapeHtml((item.result ?? {})[column])}</td>`)
                    .join("")}
                  ${
                    hasConfidence
                      ? `<td>${renderFinalAnswerConfidence([
                          ...(item.probability_annotations ?? []),
                          ...(item.confidence_annotations ?? []),
                        ])}</td>`
                      : ""
                  }
                  ${hasSources ? `<td><div class="source-links">${resultSources(item).map(sourceLink).join("") || "No source tuples available"}</div></td>` : ""}
                </tr>
              `,
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

function probabilityColorClass(probability) {
  if (!Number.isFinite(probability) || probability < 0 || probability > 1) {
    return "similarity-unknown";
  }
  if (probability >= 0.75) return "similarity-green";
  if (probability >= 0.5) return "similarity-yellow";
  return "similarity-red";
}

function renderFinalAnswerConfidence(annotations) {
  if (!Array.isArray(annotations) || !annotations.length) {
    return `<span class="muted">Assessment unavailable</span>`;
  }
  const probability = annotations.find((entry) => entry.type === "final_answer_probability");
  const sources = probability?.source_probabilities ?? [];
  const valid = (value) => typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1;
  const percent = (value) => valid(value) ? `${(value * 100).toFixed(2)}%` : "Unavailable";
  const tone = (value) => !valid(value) ? "probability-tone-unknown"
    : value >= 0.75 ? "probability-tone-green" : value >= 0.5 ? "probability-tone-amber" : "probability-tone-red";
  const pipelineName = (source) => ({
    deterministic_execution: "SQL execution",
    llm_self_assessment: "LLM self-assessment",
    faiss_relevance: "RAG relevance",
    semantic_match_decision: "Semantic match",
    mixed_pipeline: "Mixed pipelines",
  }[source.metric] ?? source.pipeline ?? "Source assessment");
  const sourceRows = sources.map((source) => `
    <div class="probability-source" title="${escapeHtml(source.row_id ?? "")}">
      <div class="probability-source-label">
        <strong>${escapeHtml(source.table || "Source")}</strong>
        <span>${escapeHtml(pipelineName(source))}</span>
      </div>
      <div class="probability-indicator ${tone(source.probability)}">
        <span class="probability-value">${percent(source.probability)}</span>
        ${valid(source.probability) ? `<span class="probability-track" aria-hidden="true"><span style="width:${source.probability * 100}%"></span></span>` : ""}
      </div>
    </div>`).join("");
  const supplemental = annotations.filter((entry) => {
    if (entry.type === "final_answer_probability") return false;
    if (entry.source_type === "rag_retrieval") return !probability;
    if (entry.source_type === "llm_internal") return !sources.some((source) => source.metric === "llm_self_assessment");
    if (entry.source_type === "deterministic_sql") return !sources.some((source) => source.metric === "deterministic_execution");
    return true;
  }).map((entry) => {
    const label = entry.type === "llm_row_confidence" || entry.source_type === "llm_internal"
      ? "LLM self-assessed confidence"
      : entry.source_type === "deterministic_sql" ? "SQL execution" : "Retrieval assessment";
    const level = ({high: "High", medium: "Medium", low: "Low", unknown: "Unavailable",
      verified_execution: "Verified", close: "Close match", relevant: "Relevant", weak: "Weak match", far: "Distant match"})[entry.level] || "Unavailable";
    return `<div class="assessment-note" title="${escapeHtml(entry.reason ?? "")}"><span>${label}</span><strong>${escapeHtml(level)}</strong></div>`;
  }).join("");
  const finalAvailable = probability?.complete && valid(probability.probability);
  return `<div class="probability-panel">
    ${probability ? `<div class="result-probability ${tone(finalAvailable ? probability.probability : null)}">
      <span>Final result probability</span>
      <strong title="${escapeHtml(probability.reason ?? "Computed by the explanation service")}">${finalAvailable ? percent(probability.probability) : "Unavailable"}</strong>
    </div>` : ""}
    ${sources.length ? `<div class="probability-source-heading">Contributing sources</div>${sourceRows}` : ""}
    ${supplemental}
  </div>`;
}

function provenanceTupleSources(item, reference = false) {
  const sources = new Map();
  function add(source) {
    if (!source.table || !source.row_id || source.row_id.includes("@")) return;
    sources.set(`${source.table}:${source.row_id}`, source);
  }
  // AP resolves each UUID reference to its actual exported row data.
  function visit(value) {
    if (!value || typeof value !== "object") return;
    if (typeof value.reference === "string" && value.data && typeof value.data === "object") {
      const table = value.reference.split("@")[0];
      const rowId = value.data[`${table}_rownum`] ?? value.data.__rid__;
      if (typeof rowId === "string" && rowId) {
        const pipeline = reference ? "sql-table"
          : state.output?.leaf_outputs?.find((leaf) => leaf.table_name === table)?.pipeline
            ?? (semanticMappingTraces().some((trace) => trace.mapping_table === table) ? "semantic-matching" : "sql-table");
        add({ table, row_id: rowId, pipeline, reference: value.reference });
      }
    }
    for (const child of Object.values(value)) visit(child);
  }
  visit(item.provenance);
  for (const source of (item.provenance_annotations ?? []).flatMap((annotation) => annotation.source_rows ?? [])) add(source);
  return [...sources.values()];
}

function semanticMappingTraces() {
  const matching = state.output?.semantic_matching;
  return matching?.joins ?? (matching ? [matching] : []);
}

function renderProvenance(answer, options = {}) {
  if (!answer.length) {
    return `<div class="empty-state">No provenance rows.</div>`;
  }

  return `
    <div class="provenance-formula">
      ${answer
        .map((item, index) => {
          const formula = formatProvenance(item.provenance);
          const sourceRows = provenanceTupleSources(item, options.reference);
          return `
            <div class="formula-card">
              <strong>Result ${index + 1}</strong>
              <code>${escapeHtml(formula || "no provenance")}</code>
              ${
                sourceRows.length
                  ? `<p class="provenance-row-label">Contributing tuples</p><div class="provenance-sources">
                      ${sourceRows
                        .map(
                          (source) =>
                            `<div class="provenance-tuple"><code class="provenance-row-id">${escapeHtml(source.row_id)}</code>${sourceLink(source)}</div>`,
                        )
                        .join("")}
                    </div>`
                  : '<p class="status-note">The service did not return a row identifier mapping for this result.</p>'
              }
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

function switchTab(tabName) {
  for (const tab of els.tabs) {
    tab.classList.toggle("active", tab.dataset.tab === tabName);
  }

  els.answerView.classList.toggle("hidden", tabName !== "answer");
  els.provenanceView.classList.toggle("hidden", tabName !== "provenance");
  els.executionView?.classList.toggle("hidden", tabName !== "execution");
  els.evidenceView?.classList.toggle("hidden", tabName !== "evidence");
}

els.planButton.addEventListener("click", generatePlan);
els.runButton.addEventListener("click", runSelectedPipelines);
els.datasetSelect?.addEventListener("change", () => {
  state.dataset = els.datasetSelect.value;
  invalidatePlan("Dataset changed. Prepare a new plan before running pipelines.");
});
els.inputType?.addEventListener("change", () => {
  invalidatePlan("Input format changed. Prepare the query again before running pipelines.");
});
els.queryInput.addEventListener("input", () => {
  invalidatePlan("Query changed. Prepare it again before running pipelines.");
});
els.tabs.forEach((tab) => {
  tab.addEventListener("click", () => switchTab(tab.dataset.tab));
});

clearOutput();
if (hasBackend(BACKEND_ENDPOINTS.plan) && hasBackend(BACKEND_ENDPOINTS.run)) {
  setStatus(IS_DEMO ? "Prepare your query, choose a pipeline for each table, then find answers." : "Connected mode. Planning, leaf runs, and explanation runs will call the gateway.");
} else {
  setStatus("Backend endpoints are empty in file mode; the UI is running with mock data.");
}

// Runs stay in this browser tab; source snapshots are carried with each result.
const comparisonRuns = [];
let inspectedRun = null;
let inspectedAnswer = 0;
function metricSeconds(value) {
  return typeof value === 'number' && Number.isFinite(value) ? `${value.toFixed(2)} s` : 'Unavailable';
}
function probabilityText(item) {
  return typeof item?.probability === 'number' && Number.isFinite(item.probability)
    ? `${(item.probability * 100).toFixed(2)}%` : 'Unavailable';
}
function renderComparisonViews() {
  if (!els.executionView || !els.evidenceView) return;
  const output = state.output;
  if (output?.execution_metrics && !comparisonRuns.includes(output)) {
    comparisonRuns.push(output);
    inspectedRun = comparisonRuns.length - 1;
    inspectedAnswer = 0;
  }
  const compatible = comparisonRuns.map((run, index) => ({run, index}))
    .filter(({run}) => run.sql === output?.sql && run.dataset === output?.dataset);
  const rows = compatible.flatMap(({run, index}) => [
    {label: `Run ${index + 1}: SQL reference`, metrics: run.comparison?.reference?.execution_metrics, reference: true},
    {label: `Run ${index + 1}: ${(run.leaf_outputs ?? []).map(l => `${l.table_name}: ${l.pipeline}`).join(', ')}`, metrics: run.execution_metrics},
  ]);
  const maximum = Math.max(1, ...rows.map(r => r.metrics?.total_seconds ?? 0));
  els.executionView.innerHTML = `<h3>Execution through AP</h3><p>Same query and dataset. Runs are retained in this browser tab. AP time includes submission, processing and polling; total excludes the other comparison execution.</p>
    <p class="status-note">Blue: evidence preparation · Amber: identity matching · Purple: AP submission to result · Gray: other finalization and waiting</p>
    ${rows.map(({label, metrics: m, reference}) => {
      const total = m?.total_seconds ?? 0;
      const ap = m?.ap_seconds ?? 0;
      const matching = m?.matching_seconds ?? 0;
      const preparation = reference ? (typeof m?.ap_seconds === "number" ? Math.max(0, total - ap) : 0) : m?.evidence_seconds ?? 0;
      const rest = Math.max(0, total - preparation - matching - ap);
      return `<div class="execution-row"><strong>${escapeHtml(label)}</strong><div class="execution-bar" role="img" aria-label="${escapeHtml(label)}: ${metricSeconds(m?.total_seconds)}">
        ${[[preparation,'preparation'],[matching,'matching'],[ap,'ap'],[rest,'other']].map(([seconds, cls]) => `<span class="execution-${cls}" style="width:${seconds / maximum * 100}%" title="${cls}: ${metricSeconds(seconds)}"></span>`).join('')}</div><span>${metricSeconds(m?.total_seconds)}</span></div>`;
    }).join('')}
    <div class="table-wrap"><table><thead><tr><th>Execution</th><th>Total</th><th>Identity matching</th><th>AP</th><th>Tables supplied</th><th>Materialized rows</th></tr></thead><tbody>${rows.map(({label,metrics:m}) => `<tr><td>${escapeHtml(label)}</td><td>${metricSeconds(m?.total_seconds)}</td><td>${metricSeconds(m?.matching_seconds)}</td><td>${metricSeconds(m?.ap_seconds)}</td><td>${m?.table_count ?? 'Unavailable'}</td><td>${m?.materialized_rows ?? 'Unavailable'}</td></tr>`).join('')}</tbody></table></div><p class="status-note">Model calls, retries and token usage are unavailable in this summary; no counts are inferred from leaf count.</p>`;
  const run = comparisonRuns[inspectedRun];
  if (!run || !compatible.some(r => r.index === inspectedRun)) {
    els.evidenceView.innerHTML = '<p class="empty-state">Finish a run to compare its evidence with the AP reference.</p>';
    return;
  }
  const reference = run.comparison?.reference;
  // Pair answer occurrences, preserving duplicates rather than merging equal rows.
  const remaining = (reference?.answer ?? []).map((item,index) => ({item,index}));
  const pairs = (run.answer ?? []).map(item => {
    const key = JSON.stringify(Object.entries(item.result ?? {}).sort());
    const match = remaining.findIndex(r => JSON.stringify(Object.entries(r.item.result ?? {}).sort()) === key);
    return {selected:item, reference:match < 0 ? null : remaining.splice(match,1)[0].item};
  });
  pairs.push(...remaining.map(r => ({reference:r.item,selected:null})));
  inspectedAnswer = Math.min(inspectedAnswer, Math.max(0,pairs.length - 1));
  const pair = pairs[inspectedAnswer];
  els.evidenceView.innerHTML = `<h3>Answer and evidence comparison</h3>
    <label>Saved run <select id="evidence-run">${compatible.map(({index}) => `<option value="${index}" ${index === inspectedRun ? 'selected' : ''}>Run ${index + 1}</option>`).join('')}</select></label>
    <p>Reference: ${escapeHtml(reference?.status ?? 'unavailable')}. Matching answer values do not establish matching provenance. Probabilities are computed from assigned inputs, not calibrated factual correctness.</p>
    <div class="table-wrap"><table><thead><tr><th>Answer</th><th>Reference comparison</th><th>Selected probability</th><th>Reference probability</th></tr></thead><tbody>${pairs.map((p,i) => `<tr><td><button type="button" class="secondary" data-evidence-answer="${i}" aria-pressed="${i === inspectedAnswer}">${escapeHtml(JSON.stringify((p.selected ?? p.reference).result))}</button></td><td>${p.selected && p.reference ? 'Matches' : p.selected ? (reference?.status === 'complete' ? 'Extra' : 'Not compared') : 'Missing'}</td><td>${probabilityText(p.selected)}</td><td>${probabilityText(p.reference)}</td></tr>`).join('')}</tbody></table></div>
    ${pair ? `<div class="comparison-grid">${evidenceGraph(pair.reference, true, pair.selected, run)}${evidenceGraph(pair.selected, false, pair.reference, run)}</div>` : '<p>No answer rows on either side.</p>'}`;
  document.querySelector('#evidence-run')?.addEventListener('change', event => { inspectedRun = Number(event.target.value); inspectedAnswer = 0; renderComparisonViews(); });
  els.evidenceView.querySelectorAll('[data-evidence-answer]').forEach(button => button.addEventListener('click', () => {inspectedAnswer = Number(button.dataset.evidenceAnswer); renderComparisonViews();}));
}
function evidenceGraph(item, reference, other, run) {
  const title = reference ? 'Full-data AP reference' : 'Selected evidence → AP';
  if (!item) return `<section><h4>${title}</h4><p>No corresponding answer.</p></section>`;
  // Resolve pipeline identities against the inspected run, including saved runs.
  const original = state.output;
  state.output = run;
  const sources = provenanceTupleSources(item, reference);
  const others = provenanceTupleSources(other ?? {}, !reference);
  state.output = original;
  const key = s => `${s.table}:${s.row_id}`;
  const isGenerated = s => ['llm-internal','semantic-matching'].includes(s.pipeline);
  const shared = new Set(others.filter(s => !isGenerated(s)).map(key));
  const height = Math.max(130, sources.length * 66 + 20);
  const annotations = (item.probability_annotations ?? []).find(a => a.type === 'final_answer_probability');
  const graph = sources.length ? `<svg class="evidence-graph" viewBox="0 0 560 ${height}" role="img" aria-label="Contributing evidence for ${title}">
    ${sources.map((s,i) => {
      const y = 10 + i * 66;
      const status = isGenerated(s) ? 'generated' : shared.has(key(s)) ? 'shared' : 'different';
      const assessment = annotations?.source_probabilities?.find(p => p.table === s.table && p.row_id === s.row_id);
      const score = assessment ? `${assessment.metric}: ${probabilityText(assessment)}` : reference ? 'SQL input: 100%' : `${s.pipeline}; score unavailable`;
      return `<path d="M 288 ${y+24} C 325 ${y+24}, 325 ${height/2}, 360 ${height/2}" fill="none" stroke="#8795a5"/><g class="evidence-${status}"><title>${escapeHtml(`${s.table}: ${s.row_id}; ${status}; ${score}`)}</title><rect x="5" y="${y}" width="283" height="50" rx="7"/><text x="14" y="${y+19}">${escapeHtml(s.row_id)}</text><text x="14" y="${y+38}" class="evidence-score">${escapeHtml(score)}</text></g>`;
    }).join('')}
    <g class="evidence-answer"><rect x="360" y="${height/2-27}" width="190" height="54" rx="7"/><text x="370" y="${height/2-5}">Answer</text><text x="370" y="${height/2+15}">AP: ${probabilityText(item)}</text></g></svg>` : '<p>AP did not return resolvable source row identities.</p>';
  return `<section><h4>${title}</h4><p class="status-note">Green: shared source identity · Amber: different source · Purple: generated or mapping record</p>${graph}<p class="status-note">Edges show contributing records; AND/OR derivations are retained in the service provenance below.</p><details><summary>Probability inputs and service provenance</summary>${renderFinalAnswerConfidence(item.probability_annotations ?? [])}<pre>${escapeHtml(JSON.stringify(item.provenance,null,2))}</pre></details></section>`;
}
