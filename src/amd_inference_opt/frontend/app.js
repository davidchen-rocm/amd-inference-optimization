const app = document.querySelector("#app");
const connection = document.querySelector("#connection");
let meta = null;
let builderMeta = null;
let activeMapDrawing = null;

function node(tag, attributes = {}, children = []) {
  const element = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (key === "className") element.className = value;
    else if (key === "text") element.textContent = value == null ? "" : value;
    else if (key === "href") element.setAttribute("href", value);
    else element.setAttribute(key, value);
  }
  for (const child of Array.isArray(children) ? children : [children]) {
    if (child !== null && child !== undefined) {
      element.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
  }
  return element;
}

async function api(path) {
  const response = await fetch(path, {cache: "no-store", credentials: "same-origin"});
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.message || `HTTP ${response.status}`);
  return payload;
}

async function apiPost(path, payload) {
  if (!builderMeta || !builderMeta.draft_writes_enabled) throw new Error("Draft database is disabled");
  const response = await fetch(path, {
    method: "POST", cache: "no-store", credentials: "same-origin",
    headers: {"Content-Type": "application/json", "X-GPUOPT-CSRF": builderMeta.csrf_token},
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.message || `HTTP ${response.status}`);
  return result;
}

// 1. Connection Status
function ConnectionStatus(ok, message) {
  connection.className = `status ${ok ? "ok" : "error"}`;
  connection.textContent = message;
}

// 2. Summary Cards
function SummaryCards(run) {
  const values = [
    ["Stage", run.stage], ["Status", run.status], ["Experiments", run.experiment_count],
    ["Artifacts", run.artifact_count], ["Accepted", run.accepted], ["Rejected", run.rejected],
  ];
  return node("section", {className: "grid"}, values.map(([label, value]) =>
    node("article", {className: "panel"}, [
      node("div", {className: "muted", text: label}),
      node("div", {className: "metric-value", text: value}),
    ])
  ));
}

function TableHead(labels) {
  return node(
    "thead",
    {},
    node("tr", {}, labels.map(value => node("th", {text: value}))),
  );
}

// 3. Run Table
function RunTable(runs) {
  const table = node("table");
  table.append(TableHead(["Run", "Model", "GPU", "Stage", "Status", "Experiments"]));
  const body = node("tbody");
  for (const run of runs) {
    const href = `/runs/${encodeURIComponent(run.source_id)}/${encodeURIComponent(run.run_id)}`;
    body.append(node("tr", {}, [
      node("td", {}, node("a", {href, text: `${run.source_id}/${run.run_id}`})),
      node("td", {text: run.model_name || "—"}), node("td", {text: run.gfx || run.gpu || "—"}),
      node("td", {text: run.stage}), node("td", {text: run.status}), node("td", {text: run.experiment_count}),
    ]));
  }
  table.append(body);
  return node("section", {className: "panel scroll"}, runs.length ? table : node("p", {className: "empty", text: "No readable runs found."}));
}

// Model catalogue: a database-like projection over immutable experiment stores.
function ModelCatalogTable(catalog) {
  const table = node("table");
  table.append(TableHead(["Model", "Architecture", "Quantizations", "Methods", "Runs", "Action"]));
  const body = node("tbody");
  for (const model of catalog.items) {
    const latest = model.runs[0];
    const optimize = latest ? `/builder?source=${encodeURIComponent(latest.source_id)}&run=${encodeURIComponent(latest.run_id)}` : "/builder";
    body.append(node("tr", {}, [
      node("td", {}, [
        node("a", {href: `/models/${encodeURIComponent(model.id)}`, text: model.name}),
        node("span", {className: "origin-badge", text: "ORIGIN"}),
        node("div", {className: "muted compact", text: model.summary}),
      ]),
      node("td", {text: model.architecture || "—"}),
      node("td", {}, node("div", {className: "chip-row"}, model.quantizations.map(value => node("span", {className: "map-chip tag", text: value})))),
      node("td", {}, node("div", {className: "chip-row"}, model.methods.map(value => node("span", {className: "map-chip tag", text: value})))),
      node("td", {text: model.runs.length}),
      node("td", {}, node("a", {className: "primary-link small", href: optimize, text: "Optimize"})),
    ]));
  }
  table.append(body);
  return node("section", {className: "panel scroll"}, [
    node("div", {className: "section-title"}, [node("h2", {text: "Model optimization catalog"}), node("a", {href: "/models", text: "View all →"})]),
    catalog.items.length ? table : node("p", {className: "empty", text: "No indexed models. Add an Experiment Store to the UI command."}),
  ]);
}

function CheckOption(group, value, label, checked = true) {
  return node("label", {className: "check-option"}, [
    node("input", {type: "checkbox", name: group, value, ...(checked ? {checked: "checked"} : {})}),
    node("span", {text: label}),
  ]);
}

function SelectField(label, name, values, selected) {
  const select = node("select", {name});
  for (const value of values) select.append(node("option", {value, text: value.replaceAll("_", " "), ...(value === selected ? {selected: "selected"} : {})}));
  return node("label", {className: "field"}, [node("span", {text: label}), select]);
}

function NumberField(label, name, value, min, max, step = 1) {
  return node("label", {className: "field"}, [node("span", {text: label}), node("input", {type: "number", name, value, min, max, step})]);
}

function ModelRunOptions(catalog, selectedSource, selectedRun) {
  const select = node("select", {name: "model_run", required: "required"});
  for (const model of catalog.items) for (const run of model.runs) {
    const value = `${run.source_id}/${run.run_id}`;
    const selected = run.source_id === selectedSource && run.run_id === selectedRun;
    select.append(node("option", {value, text: `${model.name} · ${run.quantization || "unknown"} · ${run.gfx || "GPU?"} · ${run.run_id}`, ...(selected ? {selected: "selected"} : {})}));
  }
  return select;
}

function PrecisionAssignments(options) {
  const sensitive = new Set(["embedding", "output", "attention_q", "attention_k", "attention_v", "attention_o"]);
  return node("div", {className: "assignment-grid"}, options.tensor_groups.map(group => {
    const select = node("select", {name: `precision_${group}`, "data-tensor-group": group});
    const preferred = sensitive.has(group) ? "Q6_K" : "Q5_K";
    for (const precision of options.tensor_precisions) select.append(node("option", {value: precision, text: precision, ...(precision === preferred ? {selected: "selected"} : {})}));
    return node("label", {className: "field compact-field"}, [node("span", {text: group}), select]);
  }));
}

function OptimizationBuilderForm(catalog, options, selectedSource, selectedRun) {
  const form = node("form", {className: "builder-form"});
  const message = node("div", {className: "form-message muted", text: "Creates an immutable draft only; no GPU command will run."});
  const modelSelect = ModelRunOptions(catalog, selectedSource, selectedRun);
  const refs = node("div", {className: "check-grid"}, options.reference_precisions.map(value => CheckOption("reference_precision", value, value, value !== "Q8_0")));
  const assignments = PrecisionAssignments(options);
  const shapes = node("div", {className: "check-grid"}, options.kernel_shapes.map(value => CheckOption("kernel_shape", value, value.replaceAll("_", " "))));
  const knobs = node("div", {className: "check-grid"}, options.kernel_knobs.map(value => CheckOption("kernel_knob", value, value.replaceAll("_", " "), value !== "gate_up_fusion")));
  const techniqueLabels = {
    matrix_shape_wave_mapping: "Matrix shape → wave/tile mapping (llama.cpp source)",
    weight_layout_fused_dequant: "Weight layout + fused dequant/GEMV",
    gate_up_fusion: "Gate + up projection fusion",
    hip_graph_ab: "HIP Graph strict A/B",
    kv_cache_quantization: "KV cache Q8/Q4 for long context",
    buffer_reuse_audit: "Buffer/memory reuse audit",
  };
  const techniques = node("div", {className: "technique-list"}, options.llama_cpp_techniques.map(value =>
    CheckOption("llama_technique", value, techniqueLabels[value] || value.replaceAll("_", " "), !["weight_layout_fused_dequant", "gate_up_fusion"].includes(value))));
  form.append(
    node("section", {className: "panel builder-section"}, [
      node("h2", {text: "1 · Model"}),
      node("label", {className: "field"}, [node("span", {text: "Existing model/run"}), modelSelect]),
      node("label", {className: "field"}, [node("span", {text: "Draft name"}), node("input", {name: "draft_name", maxlength: "100", value: "Mixed-bit optimization"})]),
    ]),
    node("section", {className: "panel builder-section"}, [
      node("h2", {text: "2 · Mixed-bit design"}),
      node("p", {className: "muted", text: "Default: attention, embedding and output stay Q6; FFN gate/up/down use Q5. Stock Q6/Q5/Q4 arms remain as references."}),
      SelectField("Workflow", "mixed_mode", options.mixed_bit_modes, "SENSITIVITY_GUIDED"),
      node("strong", {className: "field-label", text: "Reference candidates"}), refs,
      node("strong", {className: "field-label", text: "Tensor group assignment"}), assignments,
    ]),
    node("section", {className: "panel builder-section"}, [
      node("h2", {text: "3 · llama.cpp matrix execution mapping"}),
      node("p", {className: "muted", text: "This is the source-level optimization you remembered: profile real matrix shapes, then change wave/tile/split-K mapping in llama.cpp only when evidence shows a gap."}),
      SelectField("Kernel action", "kernel_mode", options.kernel_modes, "EVIDENCE_THEN_TUNE"),
      SelectField("Trigger", "kernel_trigger", ["ONLY_IF_PROFILED_GAP", "ALWAYS"], "ONLY_IF_PROFILED_GAP"),
      NumberField("Maximum kernel candidates", "kernel_max", 4, 1, 16),
      node("strong", {className: "field-label", text: "Real shape groups"}), shapes,
      node("strong", {className: "field-label", text: "Allowed tuning knobs"}), knobs,
    ]),
    node("section", {className: "panel builder-section"}, [
      node("h2", {text: "4 · Other important llama.cpp methods"}),
      node("p", {className: "muted", text: "High-value framework methods only. Deeper compiler/ISA development is intentionally excluded."}),
      techniques,
      NumberField("Maximum source patches", "max_source_patches", 2, 0, 4),
    ]),
    node("section", {className: "panel builder-section"}, [
      node("h2", {text: "5 · Benchmark and quality suites"}),
      SelectField("Quality policy", "quality_policy", options.quality_policies || [options.quality_policy], "balanced-200.v1"),
      node("div", {className: "form-grid"}, [
        NumberField("Repetitions", "repetitions", 5, 3, 12),
        NumberField("Maximum CV %", "max_cv", 2, 0.1, 10, 0.1),
        NumberField("Maximum math drop (points)", "math_drop", 2, 0, 10, 0.1),
        NumberField("Maximum general drop (points)", "general_drop", 2, 0, 10, 0.1),
        NumberField("Maximum PPL regression %", "ppl_drop", 0.5, 0, 10, 0.1),
      ]),
      node("p", {className: "muted", text: "balanced-200.v1 runs 100 math + 100 general questions, PPL and greedy canary. The legacy math-only policy remains available."}),
    ]),
    node("section", {className: "panel builder-actions"}, [message, node("button", {type: "submit", className: "action-button", text: "Create optimization draft"})]),
  );
  form.addEventListener("submit", async event => {
    event.preventDefault();
    const data = new FormData(form);
    const [source_id, run_id] = String(data.get("model_run")).split("/", 2);
    const selectedModel = catalog.items.flatMap(item => item.runs.map(run => ({model: item, run}))).find(item => item.run.source_id === source_id && item.run.run_id === run_id);
    const mixedMode = String(data.get("mixed_mode"));
    const kernelMode = String(data.get("kernel_mode"));
    const qualityPolicy = String(data.get("quality_policy"));
    const payload = {
      schema_version: 1, name: String(data.get("draft_name")),
      model: {source_id, run_id, model_sha256: selectedModel ? selectedModel.run.model_sha256 : null},
      mixed_bit: {
        mode: mixedMode, source_precision: "BF16", sensitivity_evidence: "REUSE_OR_COLLECT_IMATRIX",
        reference_candidates: mixedMode === "DISABLED" ? [] : data.getAll("reference_precision"),
        assignments: mixedMode === "SENSITIVITY_GUIDED" ? [...form.querySelectorAll("[data-tensor-group]")].map(input => ({group: input.getAttribute("data-tensor-group"), precision: input.value})) : [],
        require_effective_bpw: true, require_pareto_ranking: true,
      },
      kernel_mapping: {
        mode: kernelMode, trigger: String(data.get("kernel_trigger")),
        shapes: kernelMode === "NO_CHANGE" ? [] : data.getAll("kernel_shape"),
        knobs: kernelMode === "NO_CHANGE" ? [] : data.getAll("kernel_knob"),
        max_candidates: Number(data.get("kernel_max")), independent_clean_base: true,
      },
      llama_cpp: {
        policy: "EVIDENCE_FIRST", techniques: data.getAll("llama_technique"),
        max_source_patches: Number(data.get("max_source_patches")),
      },
      benchmark: {generation_lengths: [128, 512], repetitions: Number(data.get("repetitions")), max_cv_percent: Number(data.get("max_cv"))},
      quality: {policy: qualityPolicy, math_problem_count: 100, general_problem_count: qualityPolicy === "balanced-200.v1" ? 100 : null, perplexity_enabled: true, greedy_canary_enabled: true, max_math_accuracy_drop_points: Number(data.get("math_drop")), max_general_accuracy_drop_points: Number(data.get("general_drop")), max_perplexity_regression_percent: Number(data.get("ppl_drop"))},
      notes: "",
    };
    try {
      message.className = "form-message muted"; message.textContent = "Saving validated draft…";
      const record = await apiPost("/api/v1/drafts", payload);
      history.pushState({}, "", `/drafts/${encodeURIComponent(record.id)}`);
      loadCurrentScreen();
    } catch (error) {
      message.className = "form-message REJECT"; message.textContent = error.message;
    }
  });
  return form;
}

function DraftTable(records) {
  const table = node("table");
  table.append(TableHead(["Draft", "Model run", "Mixed-bit", "Kernel", "Created"]));
  const body = node("tbody");
  for (const record of records.items) body.append(node("tr", {}, [
    node("td", {}, node("a", {href: `/drafts/${encodeURIComponent(record.id)}`, text: record.request.name})),
    node("td", {text: `${record.request.model.source_id}/${record.request.model.run_id}`}),
    node("td", {text: record.request.mixed_bit.mode}), node("td", {text: record.request.kernel_mapping.mode}),
    node("td", {text: new Date(record.created_at).toLocaleString()}),
  ]));
  table.append(body);
  return node("section", {className: "panel scroll"}, records.items.length ? table : node("p", {className: "empty", text: "No drafts yet."}));
}

function formatMetric(value, suffix = "") {
  return value == null ? "—" : `${Number(value).toFixed(2)}${suffix}`;
}

function VariantValidationDetails(variant) {
  const evidence = variant.evidence_paths.map(path => {
    const target = `/runs/${encodeURIComponent(variant.source_id)}/${encodeURIComponent(variant.run_id)}/artifacts?path=${encodeURIComponent(path)}`;
    return node("li", {}, node("a", {href: target, text: path}));
  });
  return node("details", {className: "validation-details"}, [
    node("summary", {text: "Validation & evidence"}),
    node("div", {className: "validation-grid"}, [
      node("div", {}, [node("strong", {text: "Perplexity"}), node("p", {text: formatMetric(variant.quality.perplexity)}), node("p", {className: "muted", text: `Δ ${formatMetric(variant.quality.perplexity_delta_percent, "%")}`})]),
      node("div", {}, [node("strong", {text: "Disposition"}), node("p", {text: variant.disposition})]),
      node("div", {}, [node("strong", {text: "Evidence"}), evidence.length ? node("ul", {}, evidence) : node("p", {className: "muted", text: "None linked"})]),
    ]),
    variant.reasons.length ? node("ul", {className: "reason-list"}, variant.reasons.map(value => node("li", {text: value}))) : null,
  ]);
}

function ModelPerformanceComparison(model) {
  const table = node("table", {className: "comparison-table"});
  table.append(TableHead(["Role", "Variant", "tg128", "tg512", "Math", "General", "PPL", "Result"]));
  const body = node("tbody");
  for (const variant of model.variants) {
    const mathValue = variant.quality.math_accuracy_percent == null ? variant.quality.accuracy_percent : variant.quality.math_accuracy_percent;
    const mathCorrect = variant.quality.math_correct == null ? variant.quality.correct : variant.quality.math_correct;
    const mathTotal = variant.quality.math_total == null ? variant.quality.total : variant.quality.math_total;
    const math = mathValue == null ? "Not run" : `${mathValue.toFixed(1)}%${mathCorrect == null ? "" : ` (${mathCorrect}/${mathTotal})`}`;
    const general = variant.quality.general_accuracy_percent == null ? "Not run" : `${variant.quality.general_accuracy_percent.toFixed(1)}%${variant.quality.general_correct == null ? "" : ` (${variant.quality.general_correct}/${variant.quality.general_total})`}`;
    body.append(node("tr", {}, [
      node("td", {}, node("span", {className: `role-badge role-${variant.role}`, text: variant.role})),
      node("td", {}, [node("strong", {text: variant.label}), node("div", {className: "muted compact", text: [variant.quantization, variant.effective_bpw == null ? null : `${variant.effective_bpw} bpw`].filter(Boolean).join(" · ")})]),
      node("td", {}, [node("strong", {text: formatMetric(variant.performance.tg128)}), node("div", {className: "muted compact", text: `Δ ${formatMetric(variant.performance.tg128_delta_percent, "%")}`})]),
      node("td", {}, [node("strong", {text: formatMetric(variant.performance.tg512)}), node("div", {className: "muted compact", text: `Δ ${formatMetric(variant.performance.tg512_delta_percent, "%")}`})]),
      node("td", {}, [node("strong", {text: math}), node("div", {className: "muted compact", text: `Δ ${formatMetric(variant.quality.math_accuracy_delta_points == null ? variant.quality.accuracy_delta_points : variant.quality.math_accuracy_delta_points, " points")}`})]),
      node("td", {}, [node("strong", {text: general}), node("div", {className: "muted compact", text: `Δ ${formatMetric(variant.quality.general_accuracy_delta_points, " points")}`})]),
      node("td", {}, [node("strong", {text: formatMetric(variant.quality.perplexity)}), node("div", {className: "muted compact", text: `Δ ${formatMetric(variant.quality.perplexity_delta_percent, "%")}`})]),
      node("td", {className: variant.role === "ACCEPTED" ? "ACCEPT" : variant.disposition.includes("REJECT") ? "REJECT" : "muted", text: variant.selected_as_winner ? "WINNER" : variant.disposition}),
    ]));
    body.append(node("tr", {className: "validation-row"}, [node("td", {colspan: "8"}, VariantValidationDetails(variant))]));
  }
  table.append(body);
  return node("section", {className: "panel scroll"}, [
    node("h2", {text: "Performance and accuracy comparison"}),
    node("p", {className: "muted", text: "Primary view: decode tokens/s, math, general accuracy and PPL. Open Validation & evidence for reasons and artifacts."}),
    model.variants.length ? table : node("p", {className: "empty", text: "No model variants projected."}),
  ]);
}

function AdvancedRunDetails(detail, source, run) {
  return node("details", {className: "advanced-disclosure"}, [
    node("summary", {text: "Advanced workflow, validation and evidence"}),
    node("div", {className: "grid"}, [CapabilityMatrix(detail.capabilities), StagePipeline(detail.workflow_stages)]),
    EvidenceProviderMatrix(detail.evidence_providers || []), EventTimeline(detail.recent_events), ArtifactBrowser(detail.artifacts, source, run),
  ]);
}

// 4. Capability Matrix
function CapabilityMatrix(capabilities) {
  return node("section", {className: "panel"}, [
    node("h2", {text: "Capability matrix"}),
    ...capabilities.map(item => node("div", {className: "capability"}, [
      node("div", {}, [node("strong", {text: item.title}), node("div", {className: "muted", text: item.summary})]),
      node("span", {className: item.status, text: item.status}),
    ])),
  ]);
}

function EvidenceProviderMatrix(providers) {
  return node("section", {className: "panel"}, [
    node("h2", {text: "Evidence providers"}),
    ...providers.map(item => node("div", {className: "capability"}, [
      node("div", {}, [
        node("strong", {text: item.name}),
        node("div", {className: "muted", text: `${item.integration} · ${item.capability_ids.length} capabilities`}),
        node("div", {className: "muted", text: item.detail}),
      ]),
      node("span", {className: item.availability === "AVAILABLE" ? "COMPLETE" : "INCONCLUSIVE", text: item.availability}),
    ])),
  ]);
}

// 5. Stage Pipeline
function StagePipeline(stages) {
  return node("section", {className: "panel"}, [node("h2", {text: "Workflow stages"}),
    node("div", {className: "pipeline"}, stages.map(item => node("div", {className: `stage ${item.status}`}, [
      node("strong", {text: item.name}), node("div", {className: "muted", text: item.status}),
    ]))),
  ]);
}

// 6. Candidate / Experiment Table
function ExperimentMetric(item, coordinate) {
  const metrics = Array.isArray(item.candidate_metrics) ? item.candidate_metrics : [];
  const needle = coordinate.toLowerCase();
  return metrics.find(metric => String(metric.name || "").toLowerCase().includes(needle)) || null;
}

function ExperimentMetricCell(metric) {
  if (!metric) return node("span", {text: "—"});
  return node("div", {}, [
    node("strong", {text: `${formatMetric(metric.candidate)} ${metric.unit || ""}`.trim()}),
    node("div", {className: "muted compact", text: `Δ ${formatMetric(metric.delta_percent, "%")}`}),
  ]);
}

function ExperimentQualityCell(item) {
  const quality = item.quality || {};
  const summary = item.quality_summary || null;
  if (summary) return node("div", {}, [
    node("strong", {text: `Math ${formatMetric(summary.math.candidate_percent, "%")}`}),
    node("div", {className: "muted compact", text: `General ${formatMetric(summary.general.candidate_percent, "%")}`}),
    node("div", {className: "muted compact", text: `PPL ${formatMetric(summary.perplexity.candidate)}`}),
  ]);
  let accuracy = quality.candidate_accuracy;
  if (accuracy == null) accuracy = quality.accuracy_percent;
  if (accuracy == null && quality.accuracy != null) accuracy = quality.accuracy;
  if (accuracy != null && Math.abs(Number(accuracy)) <= 1) accuracy = Number(accuracy) * 100;
  const ppl = quality.candidate_perplexity == null ? quality.perplexity : quality.candidate_perplexity;
  const headline = accuracy == null ? (quality.status || "Not run") : `${formatMetric(accuracy, "%")} accuracy`;
  return node("div", {}, [
    node("strong", {text: headline}),
    node("div", {className: "muted compact", text: ppl == null ? "PPL —" : `PPL ${formatMetric(ppl)}`}),
  ]);
}

function ExperimentAdvancedDetails(item, source, run) {
  const evidence = Array.isArray(item.evidence) ? item.evidence : [];
  const reasons = Array.isArray(item.gate_reasons) ? item.gate_reasons : (Array.isArray(item.reasons) ? item.reasons : []);
  const links = evidence.map(value => {
    const path = typeof value === "string" ? value : value.path;
    if (!path) return null;
    const target = `/runs/${encodeURIComponent(source)}/${encodeURIComponent(run)}/artifacts?path=${encodeURIComponent(path)}`;
    return node("li", {}, node("a", {href: target, text: path}));
  }).filter(Boolean);
  return node("details", {className: "validation-details"}, [
    node("summary", {text: `Advanced artifacts (${links.length})`}),
    item.hypothesis ? node("p", {}, [node("strong", {text: "Hypothesis: "}), node("span", {text: item.hypothesis})]) : null,
    reasons.length ? node("ul", {className: "reason-list"}, reasons.map(value => node("li", {text: value}))) : null,
    links.length ? node("ul", {className: "artifact-list"}, links) : node("p", {className: "muted", text: "No linked artifacts."}),
  ]);
}

function CandidateTable(detail, source, run) {
  const items = detail.experiments.length ? detail.experiments : detail.candidates;
  const table = node("table");
  table.append(TableHead(["Experiment", "Change", "tg128", "tg512", "Quality", "Decision"]));
  const body = node("tbody");
  for (const item of items) {
    const id = item.id || item.candidate_id || "unknown";
    const href = `/runs/${encodeURIComponent(source)}/${encodeURIComponent(run)}/experiments/${encodeURIComponent(id)}`;
    const decision = item.decision || item.status || item.disposition || item.winner || "—";
    body.append(node("tr", {}, [
      node("td", {}, [
        node("a", {href, text: id}),
        item.projection ? node("div", {className: "muted compact", text: item.projection}) : null,
      ]),
      node("td", {}, [
        node("strong", {text: item.change_summary || item.strategy || item.kind || "—"}),
        item.change_summary && item.strategy ? node("div", {className: "muted compact", text: item.strategy}) : null,
      ]),
      node("td", {}, ExperimentMetricCell(ExperimentMetric(item, "tg128"))),
      node("td", {}, ExperimentMetricCell(ExperimentMetric(item, "tg512"))),
      node("td", {}, ExperimentQualityCell(item)),
      node("td", {className: ["ACCEPT", "REJECT", "INCONCLUSIVE"].includes(decision) ? decision : "muted", text: decision}),
    ]));
    body.append(node("tr", {className: "validation-row"}, [
      node("td", {colspan: "6"}, ExperimentAdvancedDetails(item, source, run)),
    ]));
  }
  table.append(body);
  return node("section", {className: "panel scroll"}, [
    node("h2", {text: "Experiments"}),
    node("p", {className: "muted", text: "Primary view: change, decode throughput, quality and deterministic decision. Raw evidence stays collapsed."}),
    items.length ? table : node("p", {className: "empty", text: "No projected experiments."}),
  ]);
}

// 7. Metric and Quality Panel
function MetricQualityPanel(detail) {
  const table = node("table");
  table.append(TableHead(["Metric", "Baseline", "Candidate", "Delta"]));
  const body = node("tbody");
  for (const metric of detail.metrics) body.append(node("tr", {}, [
    node("td", {text: `${metric.name} (${metric.unit})`}),
    node("td", {text: metric.baseline == null ? "—" : metric.baseline}),
    node("td", {text: metric.candidate == null ? "—" : metric.candidate}),
    node("td", {text: metric.delta_percent == null ? "—" : `${metric.delta_percent}%`}),
  ]));
  table.append(body);
  const quality = detail.quality_summary;
  const qualityGrid = quality ? node("div", {className: "validation-grid"}, [
    node("div", {}, [node("strong", {text: "Math"}), node("p", {text: formatMetric(quality.math.candidate_percent, "%")}), node("p", {className: "muted", text: `Δ ${formatMetric(quality.math.delta_points, " points")}`})]),
    node("div", {}, [node("strong", {text: "General"}), node("p", {text: formatMetric(quality.general.candidate_percent, "%")}), node("p", {className: "muted", text: `Δ ${formatMetric(quality.general.delta_points, " points")}`})]),
    node("div", {}, [node("strong", {text: "Perplexity"}), node("p", {text: formatMetric(quality.perplexity.candidate)}), node("p", {className: "muted", text: `Δ ${formatMetric(quality.perplexity.delta_percent, "%")}`})]),
  ]) : null;
  return node("section", {className: "panel scroll"}, [node("h2", {text: "Metrics and quality"}),
    detail.metrics.length ? table : node("p", {className: "empty", text: "No normalized metrics."}),
    qualityGrid,
    detail.quality ? node("details", {className: "validation-details"}, [node("summary", {text: "Raw quality record"}), node("pre", {text: JSON.stringify(detail.quality, null, 2)})]) : null,
  ]);
}

// 8. Event Timeline
function EventTimeline(events) {
  return node("section", {className: "panel"}, [node("h2", {text: "Recent events"}),
    events.length ? node("ol", {className: "events"}, events.map(item => node("li", {}, [
      node("time", {text: new Date(item.timestamp).toLocaleString()}), node("strong", {text: item.event}), node("span", {text: item.summary}),
    ]))) : node("p", {className: "empty", text: "No readable events."}),
  ]);
}

// 9. Artifact Browser
function ArtifactBrowser(artifacts, source, run) {
  return node("section", {className: "panel"}, [node("h2", {text: "Evidence artifacts"}),
    artifacts.length ? node("div", {className: "artifact-list"}, artifacts.map(item => {
      const target = `/runs/${encodeURIComponent(source)}/${encodeURIComponent(run)}/artifacts?path=${encodeURIComponent(item.path)}`;
      return node("div", {className: "artifact"}, [
        item.preview_available ? node("a", {href: target, text: item.path}) : node("span", {text: item.path}),
        node("span", {className: "muted", text: `${item.size} bytes`}), node("span", {className: item.integrity === "mismatch" ? "REJECT" : "muted", text: item.integrity}),
      ]);
    })) : node("p", {className: "empty", text: "No registered artifacts."}),
  ]);
}

// 10. Record Inspector (used by the experiment and artifact screens)
function RecordInspector(title, record) {
  return node("section", {className: "panel"}, [node("h2", {text: title}), node("pre", {text: typeof record === "string" ? record : JSON.stringify(record, null, 2)})]);
}

// 11. Optimization Map
function MapLegend() {
  const availability = ["IMPLEMENTED", "PARTIAL", "PLANNED", "NOT_IMPLEMENTED"];
  const states = ["RUNNING", "ACCEPT", "COMPLETE", "REJECT", "INCONCLUSIVE", "NOT_STARTED"];
  return node("section", {className: "map-legend", "aria-label": "Optimization map legend"}, [
    node("div", {}, [node("strong", {text: "Framework"}), ...availability.map(value => node("span", {className: `map-chip availability-${value}`, text: value.replace("_", " ")}))]),
    node("div", {}, [node("strong", {text: "This run"}), ...states.map(value => node("span", {className: `map-chip run-${value}`, text: value.replace("_", " ")}))]),
  ]);
}

function OptimizationNodeInspector(item, source, run) {
  if (!item) return node("aside", {className: "panel map-inspector"}, [
    node("h2", {text: "Component details"}),
    node("p", {className: "empty", text: "Select a component to inspect its inputs, outputs, experiments, and evidence."}),
  ]);
  const runHref = `/runs/${encodeURIComponent(source)}/${encodeURIComponent(run)}`;
  const labeledList = (label, values) => node("div", {className: "inspector-group"}, [
    node("strong", {text: label}),
    values.length ? node("ul", {}, values.map(value => node("li", {text: value}))) : node("span", {className: "muted", text: "None recorded"}),
  ]);
  const experiments = node("div", {className: "inspector-group"}, [
    node("strong", {text: "Experiments"}),
    item.experiment_ids.length ? node("ul", {}, item.experiment_ids.map(identifier => {
      const target = `${runHref}/experiments/${encodeURIComponent(identifier)}`;
      return node("li", {}, node("a", {href: target, text: identifier}));
    })) : node("span", {className: "muted", text: "None recorded"}),
  ]);
  const evidence = node("div", {className: "inspector-group"}, [
    node("strong", {text: "Evidence"}),
    item.evidence.length ? node("ul", {}, item.evidence.map(link => {
      const target = `${runHref}/artifacts?path=${encodeURIComponent(link.path)}`;
      return node("li", {}, node("a", {href: target, text: link.path}));
    })) : node("span", {className: "muted", text: "None recorded"}),
  ]);
  return node("aside", {className: "panel map-inspector"}, [
    node("div", {className: "inspector-title"}, [
      node("h2", {text: item.title}),
      node("span", {className: `map-chip run-${item.run_status}`, text: item.run_status.replace("_", " ")}),
    ]),
    node("p", {text: item.summary}),
    node("p", {className: "status-summary", text: item.status_summary}),
    node("div", {className: "chip-row"}, [
      node("span", {className: `map-chip availability-${item.availability}`, text: item.availability.replace("_", " ")}),
      ...item.tags.map(value => node("span", {className: "map-chip tag", text: value})),
    ]),
    labeledList("Inputs", item.inputs), labeledList("Outputs", item.outputs), experiments, evidence,
  ]);
}

function drawMapEdges(board, graph) {
  const svg = board.querySelector(".map-edges");
  if (!svg) return;
  svg.replaceChildren();
  const bounds = board.getBoundingClientRect();
  const width = board.scrollWidth;
  const height = board.scrollHeight;
  svg.setAttribute("width", width);
  svg.setAttribute("height", height);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  for (const edge of graph.edges) {
    const source = board.querySelector(`[data-node-id="${edge.source}"]`);
    const target = board.querySelector(`[data-node-id="${edge.target}"]`);
    if (!source || !target) continue;
    const from = source.getBoundingClientRect();
    const to = target.getBoundingClientRect();
    const x1 = from.right - bounds.left;
    const y1 = from.top + from.height / 2 - bounds.top;
    const x2 = to.left - bounds.left;
    const y2 = to.top + to.height / 2 - bounds.top;
    const curve = Math.max(28, Math.abs(x2 - x1) * 0.42);
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", `M ${x1} ${y1} C ${x1 + curve} ${y1}, ${x2 - curve} ${y2}, ${x2} ${y2}`);
    path.setAttribute("class", `map-edge edge-${edge.state}`);
    const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
    title.textContent = `${edge.label} · ${edge.state}`;
    path.append(title);
    svg.append(path);
  }
}

function OptimizationMapBoard(graph, source, run) {
  let selected = graph.nodes.find(item => item.run_status === "RUNNING") ||
    graph.nodes.find(item => item.run_status === "ACCEPT") || graph.nodes[0] || null;
  const inspectorHost = node("div", {className: "map-inspector-host"});
  const lanes = node("div", {className: "map-lanes"});
  const board = node("div", {className: "map-board"}, [
    document.createElementNS("http://www.w3.org/2000/svg", "svg"), lanes,
  ]);
  board.firstChild.setAttribute("class", "map-edges");
  board.firstChild.setAttribute("aria-hidden", "true");

  const select = item => {
    selected = item;
    for (const element of board.querySelectorAll(".map-node")) {
      element.classList.toggle("selected", element.getAttribute("data-node-id") === item.id);
    }
    inspectorHost.replaceChildren(OptimizationNodeInspector(item, source, run));
  };
  for (const layer of graph.layers) {
    const items = graph.nodes.filter(item => item.layer === layer);
    const lane = node("section", {className: `map-lane layer-${layer}`}, [
      node("header", {}, [node("h2", {text: layer}), node("span", {className: "pill", text: `${items.length} components`})]),
    ]);
    for (const item of items) {
      const card = node("button", {
        type: "button", className: `map-node run-${item.run_status}`,
        "data-node-id": item.id, "aria-label": `${item.title}: ${item.run_status}`,
      }, [
        node("strong", {text: item.title}),
        node("span", {className: "node-summary", text: item.status_summary}),
        node("span", {className: "node-badges"}, [
          node("span", {className: `map-chip availability-${item.availability}`, text: item.availability.replace("_", " ")}),
          node("span", {className: `map-chip run-${item.run_status}`, text: item.run_status.replace("_", " ")}),
        ]),
      ]);
      card.addEventListener("click", () => select(item));
      lane.append(card);
    }
    lanes.append(lane);
  }
  const scroll = node("section", {className: "map-scroll", "aria-label": "Optimization component map"}, board);
  const wrapper = node("div", {className: "map-layout"}, [scroll, inspectorHost]);
  inspectorHost.replaceChildren(OptimizationNodeInspector(selected, source, run));
  if (selected) requestAnimationFrame(() => select(selected));
  activeMapDrawing = () => drawMapEdges(board, graph);
  requestAnimationFrame(activeMapDrawing);
  return wrapper;
}

function breadcrumbs(parts) {
  const children = [node("a", {href: "/", text: "Runs"})];
  for (const [label, href] of parts) children.push("/", href ? node("a", {href, text: label}) : node("span", {text: label}));
  return node("nav", {className: "breadcrumbs", "aria-label": "Breadcrumb"}, children);
}

async function overviewScreen() {
  const [runs, catalog, drafts] = await Promise.all([
    api("/api/v1/runs"), api("/api/v1/catalog/models"), api("/api/v1/drafts"),
  ]);
  app.replaceChildren(node("h1", {text: "AMD optimization control plane"}),
    node("p", {className: "subtitle", text: `${catalog.items.length} models · ${runs.total} runs · ${drafts.items.length} optimization drafts`}),
    node("div", {className: "action-row"}, [
      node("a", {className: "action-button", href: "/builder", text: "New optimization draft"}),
      node("a", {className: "primary-link", href: "/models", text: "Model catalog"}),
      node("a", {className: "primary-link", href: "/drafts", text: "Drafts"}),
    ]),
    ModelCatalogTable(catalog), node("h2", {className: "section-heading", text: "Recent runs"}), RunTable(runs.items));
}

async function modelCatalogScreen() {
  const catalog = await api("/api/v1/catalog/models");
  app.replaceChildren(breadcrumbs([["Models", null]]), node("h1", {text: "Model optimization catalog"}),
    node("p", {className: "subtitle", text: "One concise view of each model, its quantizations, optimization methods, GPUs, and evidence runs."}),
    ModelCatalogTable(catalog));
}

async function modelDetailScreen(modelId) {
  const model = await api(`/api/v1/catalog/models/${encodeURIComponent(modelId)}`);
  const latest = model.runs[0];
  const optimize = latest ? `/builder?source=${encodeURIComponent(latest.source_id)}&run=${encodeURIComponent(latest.run_id)}` : "/builder";
  app.replaceChildren(
    breadcrumbs([["Models", "/models"], [model.name, null]]),
    node("div", {className: "model-title-row"}, [
      node("div", {}, [node("h1", {text: model.name}), node("p", {className: "subtitle", text: `${model.architecture || "Unknown architecture"} · ${model.summary}`})]),
      node("a", {className: "action-button", href: optimize, text: "Create optimization draft"}),
    ]),
    node("section", {className: "panel origin-card"}, [
      node("span", {className: "origin-badge", text: "ORIGIN MODEL"}),
      node("h2", {text: model.origin.name}),
      node("div", {className: "origin-meta"}, [
        node("span", {text: `Precision: ${model.origin.precision || "unknown"}`}),
        node("span", {text: `Source: ${model.origin.source || "unknown"}`}),
        node("span", {text: `Revision: ${model.origin.revision || "not recorded"}`}),
      ]),
    ]),
    ModelPerformanceComparison(model),
    node("details", {className: "advanced-disclosure"}, [
      node("summary", {text: "Runs and optimization methods"}),
      node("div", {className: "chip-row detail-chips"}, model.methods.map(value => node("span", {className: "map-chip tag", text: value}))),
      RunTable(model.runs.map(run => ({...run, model_name: model.name, stage: "—", experiment_count: 0}))),
    ]),
  );
}

async function builderScreen(query) {
  const parameters = new URLSearchParams(query);
  const [catalog, options] = await Promise.all([api("/api/v1/catalog/models"), api("/api/v1/builder/options")]);
  app.replaceChildren(breadcrumbs([["Experiment builder", null]]), node("h1", {text: "Optimization experiment builder"}),
    node("p", {className: "subtitle", text: "Choose the weight precision plan and whether kernel mapping should be tested. This creates a reviewable draft; execution remains controlled by the Framework."}),
    catalog.items.length ? OptimizationBuilderForm(catalog, options, parameters.get("source"), parameters.get("run")) : node("div", {className: "error-box", text: "No model runs are available. Configure at least one Experiment Store."}));
}

async function draftsScreen() {
  const drafts = await api("/api/v1/drafts");
  app.replaceChildren(breadcrumbs([["Drafts", null]]), node("h1", {text: "Optimization drafts"}),
    node("p", {className: "subtitle", text: "Immutable parameter plans. Drafts do not execute commands or change experiment evidence."}), DraftTable(drafts));
}

async function draftDetailScreen(draftId) {
  const draft = await api(`/api/v1/drafts/${encodeURIComponent(draftId)}`);
  app.replaceChildren(breadcrumbs([["Drafts", "/drafts"], [draft.id, null]]),
    node("h1", {text: draft.request.name}),
    node("p", {className: "subtitle", text: `${draft.status} · SHA256 ${draft.content_sha256}`}),
    node("div", {className: "grid"}, [
      node("section", {className: "panel"}, [node("h2", {text: "Mixed-bit"}), node("p", {text: draft.request.mixed_bit.mode}), node("p", {className: "muted", text: `${draft.request.mixed_bit.assignments.length} tensor group assignments · references ${draft.request.mixed_bit.reference_candidates.join(", ") || "none"}`})]),
      node("section", {className: "panel"}, [node("h2", {text: "Kernel mapping"}), node("p", {text: draft.request.kernel_mapping.mode}), node("p", {className: "muted", text: `${draft.request.kernel_mapping.shapes.length} shapes · up to ${draft.request.kernel_mapping.max_candidates} candidates`})]),
      node("section", {className: "panel"}, [node("h2", {text: "llama.cpp methods"}), node("p", {text: draft.request.llama_cpp.policy}), node("p", {className: "muted", text: draft.request.llama_cpp.techniques.join(", ") || "No additional methods"})]),
      node("section", {className: "panel"}, [node("h2", {text: "Validation"}), node("p", {text: "tg128 + tg512"}), node("p", {className: "muted", text: `${draft.request.quality.math_problem_count} math${draft.request.quality.general_problem_count ? ` + ${draft.request.quality.general_problem_count} general` : ""} questions · PPL · greedy canary`})]),
    ]), RecordInspector("Validated draft", draft.request));
}

async function runScreen(source, run) {
  const detail = await api(`/api/v1/runs/${encodeURIComponent(source)}/${encodeURIComponent(run)}`);
  const mapHref = `/runs/${encodeURIComponent(source)}/${encodeURIComponent(run)}/map`;
  app.replaceChildren(breadcrumbs([[run, null]]), node("h1", {text: run}),
    node("p", {className: "subtitle", text: `${detail.summary.model_name || "Unknown model"} · ${detail.summary.gfx || "Unknown GPU"}`}),
    node("p", {}, node("a", {className: "primary-link", href: mapHref, text: "Open optimization map →"})),
    SummaryCards(detail.summary),
    MetricQualityPanel(detail), CandidateTable(detail, source, run),
    AdvancedRunDetails(detail, source, run));
}

async function optimizationMapScreen(source, run) {
  const endpoint = `/api/v1/runs/${encodeURIComponent(source)}/${encodeURIComponent(run)}/optimization-map`;
  const graph = await api(endpoint);
  const runHref = `/runs/${encodeURIComponent(source)}/${encodeURIComponent(run)}`;
  const implemented = graph.nodes.filter(item => item.availability === "IMPLEMENTED").length;
  const completed = graph.nodes.filter(item => ["ACCEPT", "COMPLETE", "NO_ACTION", "OPPORTUNITY_FOUND"].includes(item.run_status)).length;
  app.replaceChildren(
    breadcrumbs([[run, runHref], ["Optimization map", null]]),
    node("h1", {text: "Optimization workflow map"}),
    node("p", {className: "subtitle", text: `${graph.nodes.length} components · ${implemented} implemented · ${completed} evidenced in this run`}),
    node("p", {className: "map-help", text: "Read left to right. Framework badges show what the platform can do; run badges show what this experiment actually did. Click any component for details."}),
    MapLegend(), OptimizationMapBoard(graph, source, run),
  );
}

async function experimentScreen(source, run, experiment) {
  const endpoint = `/api/v1/runs/${encodeURIComponent(source)}/${encodeURIComponent(run)}/experiments/${encodeURIComponent(experiment)}`;
  const record = await api(endpoint);
  const runHref = `/runs/${encodeURIComponent(source)}/${encodeURIComponent(run)}`;
  app.replaceChildren(
    breadcrumbs([[run, runHref], [experiment, null]]),
    node("h1", {text: experiment}),
    node("p", {className: "subtitle", text: `${record.projection || "legacy"} projection · ${record.status}`}),
    CandidateTable({experiments: [record], candidates: []}, source, run),
    node("details", {className: "advanced-disclosure"}, [
      node("summary", {text: "Full experiment projection"}),
      RecordInspector("Experiment record", record),
    ]),
  );
}

async function artifactScreen(source, run, query) {
  const artifactPath = new URLSearchParams(query).get("path");
  if (!artifactPath) throw new Error("Artifact path is missing");
  const endpoint = `/api/v1/runs/${encodeURIComponent(source)}/${encodeURIComponent(run)}/artifacts/preview?path=${encodeURIComponent(artifactPath)}`;
  const preview = await api(endpoint);
  const runHref = `/runs/${encodeURIComponent(source)}/${encodeURIComponent(run)}`;
  app.replaceChildren(breadcrumbs([[run, runHref], [artifactPath, null]]), node("h1", {text: artifactPath}),
    node("p", {className: "subtitle", text: `${preview.artifact.media_type} · ${preview.artifact.size} bytes · SHA256 ${preview.artifact.sha256}`}),
    RecordInspector("Verified artifact preview", preview.content));
}

async function loadCurrentScreen() {
  try {
    activeMapDrawing = null;
    meta = meta || await api("/api/v1/meta");
    builderMeta = builderMeta || await api("/api/v1/builder/meta");
    const parts = location.pathname.split("/").filter(Boolean).map(decodeURIComponent);
    if (!parts.length) await overviewScreen();
    else if (parts[0] === "models" && parts.length === 1) await modelCatalogScreen();
    else if (parts[0] === "models" && parts.length === 2) await modelDetailScreen(parts[1]);
    else if (parts[0] === "builder" && parts.length === 1) await builderScreen(location.search);
    else if (parts[0] === "drafts" && parts.length === 1) await draftsScreen();
    else if (parts[0] === "drafts" && parts.length === 2) await draftDetailScreen(parts[1]);
    else if (parts[0] === "runs" && parts.length === 3) await runScreen(parts[1], parts[2]);
    else if (parts[0] === "runs" && parts[3] === "map" && parts.length === 4) await optimizationMapScreen(parts[1], parts[2]);
    else if (parts[0] === "runs" && parts[3] === "experiments" && parts.length === 5) await experimentScreen(parts[1], parts[2], parts[4]);
    else if (parts[0] === "runs" && parts[3] === "artifacts" && parts.length === 4) await artifactScreen(parts[1], parts[2], location.search);
    else throw new Error("Unknown UI route");
    ConnectionStatus(true, `Connected · refresh ${meta.refresh_seconds}s`);
  } catch (error) {
    ConnectionStatus(false, "Read failed");
    app.replaceChildren(node("div", {className: "error-box", text: error.message}));
  }
}

loadCurrentScreen();
setInterval(() => {
  const liveScreen = location.pathname === "/" || location.pathname.startsWith("/runs/");
  if (liveScreen && !document.querySelector("details[open]")) loadCurrentScreen();
}, ((meta && meta.refresh_seconds) || 5) * 1000);
window.addEventListener("resize", () => {
  if (activeMapDrawing) requestAnimationFrame(activeMapDrawing);
});
