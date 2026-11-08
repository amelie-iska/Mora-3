(() => {
  "use strict";

  const loopback = ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname);
  if (!loopback || !["http:", "https:"].includes(location.protocol)) return;

  document.querySelector(".play-runtime").hidden = false;
  document.querySelector(".play-grid").hidden = false;
  document.querySelector(".play-guide").hidden = false;
  document.getElementById("play-availability").textContent = "Synthetic examples and result inspection run locally. Model serving requires a configured local endpoint.";

  const protocol = "mora3-playground-v1";
  const unit = 1024 * 1024;
  const responseLimit = unit;
  const modalities = {
    sequence: ["Sequence", [".fa", ".fasta", ".faa", ".fna", ".txt"], "Protein, DNA, or RNA sequence with an identifier and alphabet. FASTA can carry several entries."],
    molecule: ["Small molecule", [".smi", ".smiles", ".sdf", ".mol", ".txt"], "SMILES describes chemical connectivity and stereochemistry. SDF and MOL can also carry atom coordinates."],
    structure: ["Structure or complex", [".pdb", ".cif", ".mmcif", ".xyz"], "Atomic coordinates condition geometry and binding tasks. Preserve chain IDs, atom order, assemblies, and the coordinate frame."],
    dynamics: ["Coordinate trajectory", [".xyz", ".pdb", ".csv", ".json"], "A trajectory supplies ordered frames. State whether its index is simulation time, experimental time, or a generation step."],
    expression: ["Expression or perturbation", [".csv", ".tsv", ".h5ad", ".json"], "A gene vector or cell matrix can condition perturbation response. Specify feature IDs, units, control, organism, dose, and time."],
    image: ["Microscopy or mask", [".png", ".jpg", ".jpeg", ".tif", ".tiff"], "Images and masks carry spatial measurements. Include channels, pixel scale, and whether the image is raw or processed."],
    assay: ["Assay or bioelectric measurements", [".csv", ".tsv", ".json"], "Tables can carry affinity, voltage, current, coupling, or phenotype measurements. Give the method, conditions, units, and censoring."],
    reaction: ["Reaction or enzyme context", [".rxn", ".smi", ".smiles", ".sdf", ".json"], "An atom-mapped reaction or catalytic context defines roles and transformation constraints."],
    text: ["Text and metadata", [".txt", ".md", ".json", ".csv", ".tsv"], "An instruction, annotation, provenance record, or other structured context. Describe the fields and the question."],
  };
  const output = {
    affinity: ["Binding affinity / assay table", "predict", "Report an assay-specific quantity, unit, uncertainty if available, and the declared conditions. Experimental values and computed feedback use separate fields."],
    annotation: ["Functional or phenotype annotation", "predict", "Return labels with evidence and uncertainty. Predicted function, phenotype, and observed outcomes should be distinguished."],
    expression: ["Expression response table", "predict", "Return aligned feature IDs and direction or magnitude relative to a declared baseline."],
    bioelectric: ["Bioelectric measurements", "predict", "Return condition-linked electrical or coupling endpoints with units. Preserve the biological system and measurement context."],
    answer: ["Structured research answer", "predict", "Return concise text with evidence provenance and unresolved fields."],
    structure: ["3D coordinates / XYZ", "generate", "Return element-labeled Cartesian coordinates in Å, with an axis or frame declaration and an exportable XYZ file."],
    dynamics: ["Trajectory or ensemble summary", "generate", "Return a frame-indexed trace. Distinguish physical time from sampler steps and attach units where known."],
    sequence: ["Sequence / FASTA", "generate", "Return typed candidate sequences in FASTA with their identifiers and a provenance record."],
    molecule: ["Candidate molecules / SMILES", "generate", "Return chemical strings with candidate IDs and whatever geometry or validation evidence is available."],
    image: ["Image or segmentation grid", "generate", "Return a raster or categorical mask with dimensions, channel semantics, and a palette or scale."],
    reaction: ["Reaction and catalytic candidates", "generate", "Return reaction strings or structured candidate hypotheses with roles and atom mapping when supplied."],
  };
  const presets = {
    custom: ["Custom request", "Compose your own combination of instruction, attachments, and requested output."],
    binding: ["Complex affinity", "Condition a binding request on a complex and assay context.", "structure", "affinity", "Summarize the expected binding affinity for this complex. Report the assay type, units, and uncertainty."],
    perturbation: ["Cell perturbation", "Request a response against a stated baseline.", "expression", "expression", "Predict a differential expression profile relative to the matched control. Keep gene IDs and units."],
    repair: ["Bioelectric phenotype", "Organize a tissue phenotype question by measured condition and endpoint.", "assay", "bioelectric", "Summarize the bioelectric measurements and distinguish observed endpoints from hypotheses about tissue repair."],
    coordinates: ["Protein structure", "Request mapped coordinates conditioned on a sequence.", "sequence", "structure", "Generate a structure candidate, preserving the provided sequence and reporting the Cartesian frame."],
    ensemble: ["Molecular ensemble", "Request a trace with explicit frame and time semantics.", "dynamics", "dynamics", "Return an ensemble trace and label the frame index and each measurement unit."],
    segmentation: ["Image segmentation", "Request a mask associated with a microscopy image.", "image", "image", "Return a segmentation mask with its dimensions and explain each label."],
    catalytic: ["Catalytic design", "Request a structured candidate for an atom-mapped transformation.", "reaction", "reaction", "Propose a reaction candidate and identify the mapped atoms and evidence gaps."],
  };

  const samples = {
    affinity: {format: "table", summary: "Illustrative assay output for example complex EX-01.", columns: ["Quantity", "Estimate", "Unit", "Method", "Condition"], rows: [["Kd", "120", "nM", "example binding assay", "298 K"], ["Interval", "80–180", "nM", "synthetic interval", "same condition"]]},
    annotation: {format: "table", summary: "Illustrative label and provenance fields.", columns: ["Label", "Scope", "Provenance"], rows: [["Candidate function A", "protein example", "synthetic label"], ["Morphology B", "cell example", "synthetic label"]]},
    expression: {format: "table", summary: "Illustrative log₂ fold changes relative to a synthetic control.", columns: ["Feature", "Log₂ fold change", "Baseline"], rows: [["GENE_A", "+0.85", "control A"], ["GENE_B", "−0.32", "control A"], ["GENE_C", "+0.04", "control A"]]},
    bioelectric: {format: "table", summary: "Illustrative electrical endpoints with example conditions.", columns: ["Endpoint", "Value", "Unit", "Condition"], rows: [["Membrane potential", "−42", "mV", "example control"], ["Membrane potential", "−55", "mV", "example perturbation"], ["Junctional conductance", "1.8", "nS", "example perturbation"]]},
    answer: {format: "text", summary: "Example research answer layout.", text: "Observation\nThis example summarizes one declared assay.\n\nEvidence\nA synthetic source record provides an endpoint and units.\n\nOpen question\nThe same endpoint has not been measured under the alternative condition."},
    structure: {format: "coordinates", summary: "Synthetic water geometry in a declared Cartesian frame. Coordinates are shown in Å.", atoms: [["O", 0, 0, 0], ["H", 0.9572, 0, 0], ["H", -0.239, 0.927, 0]]},
    dynamics: {format: "series", summary: "Example distance along six sampler steps.", xLabel: "Sampler step", yLabel: "Distance (Å)", points: [[0, 2.12], [1, 2.4], [2, 2.28], [3, 2.6], [4, 2.54], [5, 2.8]]},
    sequence: {format: "text", summary: "Example FASTA formatting. The short string is synthetic.", text: ">example_candidate_01 alphabet=protein\nMKTAYIAKQRG"},
    molecule: {format: "text", summary: "Two example SMILES rows with candidate identifiers.", text: "CCO\texample_01\nCC(=O)O\texample_02"},
    image: {format: "raster", summary: "A synthetic segmentation mask. Dark pixels are background. Bright pixels mark an example region.", pixels: [[0,0,0,0,0,0,0,0],[0,0,0,180,180,0,0,0],[0,0,180,230,230,180,0,0],[0,0,180,230,230,180,0,0],[0,0,0,180,180,0,0,0],[0,0,0,0,0,0,0,0]]},
    reaction: {format: "table", summary: "An illustrative reaction candidate record.", columns: ["Role", "Mapped string", "Evidence"], rows: [["Reactant", "[CH3:1][CH2:2][OH:3]", "example"], ["Product", "[CH3:1][CH:2]=[O:3]", "example"]]},
  };
  const formats = Object.fromEntries(Object.entries(samples).map(([id, sample]) => [id, sample.format]));

  const names = ["availability", "form", "preset", "preset-help", "prompt", "add-message", "conversation",
    "modality", "modality-help", "files", "file-list", "mode", "output", "output-help", "example", "run", "export",
    "message", "result-label", "summary", "result", "import", "result-export", "native-export", "guide",
    "runtime-status", "checkpoint-picker", "checkpoint", "load", "capabilities"];
  const view = Object.fromEntries(names.map(name => [name, document.getElementById("play-" + name)]));
  const messages = [];
  const attachments = [];
  let lastResult;
  let active = null;
  let loading = false;
  let busy = false;

  const put = (name, value) => { view[name].textContent = value; };
  const note = text => put("message", text);
  const make = (tag, value) => {
    const item = document.createElement(tag);
    if (value !== undefined) item.textContent = value;
    return item;
  };
  const options = (picker, entries) => picker.replaceChildren(...entries.map(([id, label]) => new Option(label, id)));
  const selectedOutput = () => view.output.value;
  const selectedCapability = () => active?.outputs.find(item => item.id === selectedOutput() && item.tasks.includes(view.mode.value));
  const capabilityNote = item => item.notes || (item.id === "image" && item.tasks.includes("predict")
    ? "Attach one 8-bit grayscale PNG, up to 64 × 64 pixels and 64 KiB, using the Image modality. This route takes no instructions or conversation turns."
    : "Use the checkpoint’s documented input format.");
  const updateRun = () => {
    view.run.hidden = !selectedCapability();
    view.run.disabled = busy || loading;
  };

  function describeModality() {
    const [, extensions, description] = modalities[view.modality.value];
    view.files.accept = extensions.join(",");
    put("modality-help", description + " File formats " + extensions.join(", ") + ".");
  }
  function chooseOutput(id) {
    const available = Object.entries(output).filter(([key, spec]) => spec[1] === view.mode.value ||
      active?.outputs.some(item => item.id === key && item.tasks.includes(view.mode.value)));
    options(view.output, available.map(([key, spec]) => [key, key === "image" && view.mode.value === "predict"
      ? "Foreground probability raster" : spec[0]]));
    if (id && available.some(([key]) => key === id)) view.output.value = id;
    describeOutput();
  }
  function describeOutput() {
    const capability = selectedCapability();
    put("output-help", capability ? capabilityNote(capability) : output[selectedOutput()][2] +
      (active ? " The selected checkpoint does not support this output. A synthetic example is available." : ""));
    updateRun();
  }
  function describePreset() {
    const [, description, modality, target, prompt] = presets[view.preset.value];
    put("preset-help", description);
    if (!target) return;
    view.modality.value = modality;
    view.mode.value = output[target][1];
    if (!view.prompt.value.trim()) view.prompt.value = prompt;
    describeModality();
    chooseOutput(target);
  }
  function removable(label, handle) {
    const row = make("li");
    const remove = make("button", "Remove");
    remove.type = "button";
    remove.className = "play-remove";
    remove.setAttribute("aria-label", "Remove " + label.slice(0, 90));
    remove.addEventListener("click", handle);
    row.append(make("span", label), remove);
    return row;
  }
  function conversation() {
    view.conversation.replaceChildren(...messages.map((entry, index) => removable(
      (entry.role === "user" ? "You · " : "Model · ") + entry.content,
      () => { messages.splice(index, 1); conversation(); }
    )));
  }
  function addTurn() {
    const content = view.prompt.value.trim();
    if (!content) return note("Write an instruction before adding a turn.");
    if (messages.length >= 20) return note("The conversation holds up to 20 turns.");
    messages.push({role: "user", content});
    view.prompt.value = "";
    conversation();
    note("Instruction added to the conversation.");
  }
  function files() {
    view["file-list"].replaceChildren(...attachments.map(({modality, file}, index) => removable(
      modalities[modality][0] + " · " + file.name + " · " + (file.size / 1024).toFixed(1) + " KiB",
      () => { attachments.splice(index, 1); files(); }
    )));
  }
  function addFiles() {
    const modality = view.modality.value;
    const [, extensions] = modalities[modality];
    const selected = Array.from(view.files.files);
    view.files.value = "";
    if (!selected.length) return;
    const currentSize = attachments.reduce((sum, entry) => sum + entry.file.size, 0);
    if (attachments.length + selected.length > 8 || currentSize + selected.reduce((sum, file) => sum + file.size, 0) > 12 * unit
      || selected.some(file => file.size < 1 || file.size > 6 * unit)) return note("The selected files exceed the file count or size limits.");
    if (selected.some(file => !extensions.some(extension => file.name.toLowerCase().endsWith(extension)))) {
      return note("Choose the attachment modality that matches the file extension.");
    }
    attachments.push(...selected.map(file => ({modality, file})));
    files();
    note(selected.length + " attachment(s) ready.");
  }

  const bounded = (value, limit) => typeof value === "string" && value.length > 0 && value.length <= limit;
  const numbers = (list, count) => Array.isArray(list) && list.length === count && list.every(Number.isFinite);
  const rowsOf = (value, limit, test) => Array.isArray(value) && value.length > 0 && value.length <= limit && value.every(test);

  function validateResult(value) {
    if (!value || value.protocol !== protocol || !Object.hasOwn(formats, value.output) ||
      value.format !== formats[value.output] || !bounded(value.summary, 4000) || !bounded(value.provenance, 1000)) {
      throw new Error("This result does not match the Mora-3 output contract.");
    }
    const {format} = value;
    let valid;
    if (format === "table") {
      valid = rowsOf(value.columns, 16, item => bounded(item, 80)) && rowsOf(value.rows, 500, row =>
        Array.isArray(row) && row.length === value.columns.length &&
        row.every(cell => cell === null || typeof cell === "boolean" || Number.isFinite(cell) || bounded(cell, 2000)));
    } else if (format === "text") {
      valid = bounded(value.text, 50000);
    } else if (format === "series") {
      valid = bounded(value.xLabel, 100) && bounded(value.yLabel, 100) &&
        rowsOf(value.points, 1000, point => numbers(point, 2) && point.every(v => Math.abs(v) <= 1e15));
    } else if (format === "coordinates") {
      valid = rowsOf(value.atoms, 5000, atom => Array.isArray(atom) && atom.length === 4 &&
        typeof atom[0] === "string" && /^[A-Z][a-z]?$/.test(atom[0]) && numbers(atom.slice(1), 3) &&
        atom.slice(1).every(v => Math.abs(v) <= 1e6));
    } else if (format === "raster") {
      valid = rowsOf(value.pixels, 64, row => rowsOf(row, 64, pixel =>
        Number.isInteger(pixel) && pixel >= 0 && pixel <= 255) && row.length === value.pixels[0].length);
    }
    if (!valid) throw new Error("The result fields are invalid or exceed the display limits.");
    return value;
  }

  function table(columns, rows) {
    const wrapper = make("div");
    wrapper.className = "play-table-wrap";
    wrapper.tabIndex = 0;
    wrapper.setAttribute("role", "region");
    wrapper.setAttribute("aria-label", "Result table");
    const content = make("table");
    const header = content.createTHead().insertRow();
    for (const name of columns) {
      const heading = make("th", name);
      heading.scope = "col";
      header.append(heading);
    }
    const body = content.createTBody();
    for (const row of rows) {
      const entry = body.insertRow();
      for (const cell of row) entry.insertCell().textContent = cell === null ? "—" : cell;
    }
    wrapper.append(content);
    return wrapper;
  }
  function svg(tag, attrs, text) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, value);
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function axes(name, width = 480, height = 250) {
    return svg("svg", {viewBox: "0 0 " + width + " " + height, role: "img", "aria-label": name});
  }
  function scale(values, low, high) {
    const min = Math.min(...values);
    const max = Math.max(...values);
    return value => min === max ? (high + low) / 2 : low + (high - low) * (value - min) / (max - min);
  }
  function plot(value) {
    const {points, xLabel, yLabel} = value;
    const x = scale(points.map(p => p[0]), 57, 442);
    const y = scale(points.map(p => p[1]), 184, 43);
    const chart = axes(yLabel + " versus " + xLabel);
    chart.append(svg("path", {d: "M 47 34 V 197 H 453", fill: "none", stroke: "#638895"}));
    chart.append(svg("polyline", {points: points.map(p => x(p[0]) + "," + y(p[1])).join(" "), fill: "none", stroke: "#97e6b8", "stroke-width": 3}));
    for (const [px, py] of points) chart.append(svg("circle", {cx: x(px), cy: y(py), r: 3, fill: "#97e6b8"}));
    chart.append(svg("text", {x: 48, y: 25, fill: "#e5f0f3", "font-size": 14}, yLabel));
    chart.append(svg("text", {x: 48, y: 226, fill: "#e5f0f3", "font-size": 14}, xLabel));
    return [chart, table([xLabel, yLabel], points)];
  }
  function coordinates(value) {
    const atoms = value.atoms.slice(0, 256);
    const x = scale(atoms.map(a => a[1] + a[3] * 0.32), 48, 430);
    const y = scale(atoms.map(a => a[2] - a[3] * 0.21), 210, 55);
    const figure = axes("Projected atom positions, up to 256 points. Export XYZ for all coordinates.");
    const colors = {H: "#dbe4e6", C: "#96b9d1", N: "#7fa7f1", O: "#ff987e", S: "#f5ce77"};
    figure.append(svg("text", {x: 22, y: 28, fill: "#d3e4e8", "font-size": 13}, "CARTESIAN PROJECTION · " + atoms.length + " / " + value.atoms.length + " ATOMS"));
    for (const [symbol, px, py, pz] of atoms) {
      const dot = svg("circle", {cx: x(px + pz * 0.32), cy: y(py - pz * 0.21), r: atoms.length < 16 ? 11 : 4, fill: colors[symbol] || "#97e6b8"});
      dot.append(svg("title", {}, symbol + " (" + [px, py, pz].join(", ") + ") Å"));
      figure.append(dot);
    }
    return [figure, table(["Element", "X (Å)", "Y (Å)", "Z (Å)"], value.atoms.slice(0, 30))];
  }
  function raster(value) {
    const {pixels} = value;
    const figure = axes("Raster result, " + pixels[0].length + " by " + pixels.length + " pixels", pixels[0].length, pixels.length);
    figure.setAttribute("shape-rendering", "crispEdges");
    pixels.forEach((row, y) => row.forEach((pixel, x) =>
      figure.append(svg("rect", {x, y, width: 1, height: 1, fill: "rgb(" + pixel + "," + pixel + "," + pixel + ")"}))));
    return figure;
  }
  function render(value, origin) {
    lastResult = validateResult(value);
    put("result-label", origin + " · " + value.provenance);
    put("summary", value.summary);
    const renderers = {
      table: () => [table(value.columns, value.rows)],
      text: () => [make("pre", value.text)],
      series: () => plot(value),
      coordinates: () => coordinates(value),
      raster: () => [raster(value)],
    };
    view.result.replaceChildren(...renderers[value.format]());
    view["result-export"].hidden = false;
    view["native-export"].hidden = !["structure", "sequence", "molecule"].includes(value.output);
  }

  function fileData(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result).split(",", 2)[1]);
      reader.onerror = () => reject(new Error("An attachment could not be read."));
      reader.readAsDataURL(file);
    });
  }
  async function request() {
    const content = view.prompt.value.trim();
    const turns = content ? [...messages, {role: "user", content}] : [...messages];
    if (turns.length > 20) throw new Error("The conversation holds up to 20 turns. Remove a turn to send this instruction.");
    if (!turns.some(turn => turn.role === "user") && !attachments.length) throw new Error("Add an instruction or attach an input file first.");
    return {
      protocol,
      task: view.mode.value,
      output: selectedOutput(),
      messages: turns,
      inputs: await Promise.all(attachments.map(async ({modality, file}) => ({
        modality, name: file.name, bytes: file.size, encoding: "base64", data: await fileData(file)
      })))
    };
  }
  function download(name, data, mime) {
    const url = URL.createObjectURL(new Blob([data], {type: mime}));
    const link = make("a");
    link.href = url;
    link.download = name;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1500);
  }
  const jsonDownload = (name, value) => download(name, JSON.stringify(value, null, 2) + "\n", "application/json");
  async function operate(action) {
    if (busy || loading) return;
    busy = true;
    view.run.disabled = view.export.disabled = view.load.disabled = true;
    try {
      await action();
    } catch (error) {
      note(error.message || "The request could not be completed.");
    } finally {
      busy = false;
      view.export.disabled = view.load.disabled = false;
      updateRun();
    }
  }

  async function localAPI(path, options = {}) {
    const response = await fetch(path, {credentials: "omit", redirect: "error", cache: "no-store",
      signal: AbortSignal.timeout(120000), ...options});
    const body = await response.text();
    if (new TextEncoder().encode(body).length > responseLimit) throw new Error("The response exceeds the display size limit.");
    let value;
    try { value = JSON.parse(body); } catch (_) { throw new Error("The local checkpoint service is unavailable."); }
    if (!response.ok) throw new Error(typeof value.error === "string" ? value.error : "The local service returned HTTP " + response.status + ".");
    return value;
  }

  const postJSON = (path, value, headers = {}) => localAPI(path, {method: "POST",
    headers: {"Content-Type": "application/json", "Accept": "application/json", ...headers}, body: JSON.stringify(value)});

  function updateRuntime(status, selectSupported = false) {
    if (status.protocol !== protocol || !Array.isArray(status.checkpoints)) throw new Error("The local service protocol is incompatible.");
    put("runtime-status", status.detail);
    const previous = view.checkpoint.value;
    options(view.checkpoint, status.checkpoints.map(item => [item.id, item.label]));
    if (status.checkpoints.some(item => item.id === previous)) view.checkpoint.value = previous;
    view["checkpoint-picker"].hidden = !status.can_select;
    active = status.active;
    view.capabilities.hidden = !active;
    view.capabilities.replaceChildren();
    const selected = selectedOutput();
    if (active) {
      const list = make("ul");
      active.outputs.forEach(item => list.append(make("li", item.tasks.join(" / ") + " · " +
        (item.id === "image" && item.tasks.includes("predict") ? "Foreground probability raster" : output[item.id][0]) +
        ". " + capabilityNote(item))));
      view.capabilities.append(make("strong", active.model), make("p", "Checkpoint " + active.checkpoint), list);
      if (selectSupported && !selectedCapability()) {
        view.mode.value = active.outputs[0].tasks[0];
        if (active.outputs[0].id === "image") { view.modality.value = "image"; describeModality(); }
        chooseOutput(active.outputs[0].id);
        return;
      }
    }
    chooseOutput(selected);
  }

  async function enableServing() {
    try {
      updateRuntime(await localAPI("/api/runtime"));
    } catch (_) {
      put("runtime-status", "Static preview mode. Start the local checkpoint launcher to select a weights directory.");
    }
  }

  view.load.addEventListener("click", async () => {
    if (busy || loading) return;
    loading = true;
    view.load.disabled = view.checkpoint.disabled = true;
    updateRun();
    put("runtime-status", "Checking the selected checkpoint…");
    try {
      updateRuntime(await postJSON("/api/checkpoint", {id: view.checkpoint.value}), true);
    } catch (error) {
      put("runtime-status", error.message);
    } finally {
      loading = false;
      view.load.disabled = view.checkpoint.disabled = false;
      updateRun();
    }
  });

  view.modality.addEventListener("change", describeModality);
  view.preset.addEventListener("change", describePreset);
  view.mode.addEventListener("change", () => chooseOutput());
  view.output.addEventListener("change", describeOutput);
  view["add-message"].addEventListener("click", addTurn);
  view.files.addEventListener("change", addFiles);
  view.example.addEventListener("click", () => {
    const target = selectedOutput();
    render({...samples[target], protocol, output: target, provenance: "Toy example"}, "Synthetic result");
    note("Sample shown. Your instruction and attachments were kept local.");
  });
  view.export.addEventListener("click", () => operate(async () => {
    jsonDownload("mora3-request.json", await request());
    note("Request exported with its conversation and attachments.");
  }));
  view.form.addEventListener("submit", event => {
    event.preventDefault();
    if (!selectedCapability()) return;
    operate(async () => {
      const pending = view.prompt.value.trim();
      const payload = await request();
      note("Waiting for the model endpoint…");
      let result;
      try {
        result = await postJSON("/api/infer", payload, {"X-Mora3-Revision": String(active.revision)});
      } catch (error) {
        await enableServing();
        throw error;
      }
      const value = validateResult(result);
      if (value.output !== payload.output) throw new Error("The model returned a different output type.");
      render(value, "Model result");
      if (pending && messages.length < 20) messages.push({role: "user", content: pending});
      if (view.prompt.value.trim() === pending) view.prompt.value = "";
      if (pending && messages.length < 20) messages.push({role: "assistant", content: value.summary});
      conversation();
      note("Model result received.");
    });
  });
  view.import.addEventListener("change", () => {
    const [file] = view.import.files;
    view.import.value = "";
    if (!file) return;
    operate(async () => {
      if (file.size > responseLimit) throw new Error("Open a result JSON file no larger than 1 MiB.");
      render(JSON.parse(await file.text()), "Imported result");
      note("Result opened locally.");
    });
  });
  view["result-export"].addEventListener("click", () => jsonDownload("mora3-result.json", lastResult));
  view["native-export"].addEventListener("click", () => {
    if (lastResult.format === "coordinates") {
      const {atoms} = lastResult;
      download("mora3-result.xyz", atoms.length + "\nMora-3 Cartesian coordinates (Angstrom)\n"
        + atoms.map(atom => atom.join(" ")).join("\n") + "\n", "text/plain");
    } else {
      download("mora3-result." + (lastResult.output === "sequence" ? "fasta" : "smi"), lastResult.text + "\n", "text/plain");
    }
  });

  options(view.preset, Object.entries(presets).map(([id, preset]) => [id, preset[0]]));
  options(view.modality, Object.entries(modalities).map(([id, entry]) => [id, entry[0]]));
  for (const [name, , description] of Object.values(modalities)) {
    const entry = make("div");
    entry.append(make("dt", name), make("dd", description));
    view.guide.append(entry);
  }
  describePreset();
  describeModality();
  chooseOutput();
  enableServing();
})();
