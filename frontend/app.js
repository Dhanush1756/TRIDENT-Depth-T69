const form = document.querySelector('#reconstruct-form');
const message = document.querySelector('#message');
const run = document.querySelector('#run');
const objectUrls = [];
function downloadLink(id, encoded, mime) {
  const link = document.querySelector(id);
  link.hidden = !encoded;
  if (!encoded) return;
  const bytes = Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
  const url = URL.createObjectURL(new Blob([bytes], {type: mime}));
  objectUrls.push(url);
  link.href = url;
}
let capabilities = {};
run.disabled = true;
function updateMethods() {
  const protocol = document.querySelector('#protocol').value;
  const scales = form.elements.scale;
  scales.disabled = protocol === 'sparse_completion';
  if (scales.disabled) scales.value = '4';
  for (const option of document.querySelector('#method').options) {
    const spec = capabilities[option.value];
    const allowedScales = spec?.protocol_scales?.[protocol] || spec?.scales || [];
    option.disabled = !spec || !spec.protocols.includes(protocol) || !allowedScales.includes(Number(scales.value));
  }
  const methods = document.querySelector('#method');
  if (methods.selectedOptions[0]?.disabled) methods.value = 'classical';
  document.querySelector('#shape-note').textContent = protocol === 'dense_sr'
    ? 'RGB and reference must match depth dimensions multiplied by scale. Learned dense SR requires clean, fully valid depth.'
    : 'Completion keeps input resolution. Use a validity mask for missing depth; learned completion uses conditioning scale 4.';
}
fetch('/api/health').then(r => {if (!r.ok) throw new Error('Backend unavailable'); return r.json();}).then(health => {
  capabilities = health.capabilities;
  for (const [value, label] of [['deep', 'Deep learning'], ['hybrid', 'Hybrid refinement']]) {
    const option = document.querySelector(`#method option[value="${value}"]`);
    option.textContent = health.methods.includes(value) ? label : `${label} - unavailable`;
  }
  document.querySelector('#connection').textContent = 'Backend connected';
  document.querySelector('#model-note').textContent = health.trained_checkpoint_loaded
    ? `Loaded ${health.checkpoint?.model || 'trained model'}. Available methods depend on the selected protocol and scale. Model accuracy is not guaranteed for arbitrary uploads.`
    : 'Classical methods are ready. Learned methods require a configured, verified checkpoint.';
  updateMethods(); run.disabled = false;
}).catch(() => {document.querySelector('#connection').textContent = 'Backend unavailable';document.querySelector('#model-note').textContent = 'Cannot connect. Start the server and reload this page.';});
document.querySelector('#protocol').addEventListener('change', updateMethods);
form.elements.scale.addEventListener('change', updateMethods);
form.addEventListener('submit', async event => {
  event.preventDefault();
  document.querySelector('#result').hidden = true;
  document.querySelector('#empty').hidden = false;
  const spec = capabilities[form.elements.method.value];
  const rgb = form.elements.rgb.files[0];
  if (spec?.rgb_required && !rgb) {message.className='error';message.textContent='This model requires registered RGB guidance.';return;}
  for (const input of form.querySelectorAll('input[type=file]')) {
    if (input.files[0]?.size > 8*1024*1024) {message.className='error';message.textContent='Each file must be at most 8 MiB.';return;}
  }
  form.setAttribute('aria-busy','true');
  document.querySelector('#result').hidden = true;
  run.disabled = true; message.className = ''; message.textContent = 'Reconstructing depth…';
  const data = new FormData(form);
  for (const name of ['rgb', 'mask', 'ground_truth']) if (!data.get(name)?.size) data.delete(name);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120000);
  try {
    const response = await fetch('/api/reconstruct', {method: 'POST', body: data, signal: controller.signal});
    const result = await response.json().catch(() => {throw new Error(`Server returned ${response.status}. Check upload limits and server logs.`);});
    if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : JSON.stringify(result.detail));
    objectUrls.splice(0).forEach(url => URL.revokeObjectURL(url));
    document.querySelector('#input-image').src = `data:image/png;base64,${result.input_preview}`;
    document.querySelector('#output-image').src = `data:image/png;base64,${result.output_preview}`;
    const guidance = document.querySelector('#rgb-image');
    guidance.parentElement.hidden = !result.rgb_preview;
    if (result.rgb_preview) guidance.src = `data:image/png;base64,${result.rgb_preview}`;
    const info = document.querySelector('#run-info'); info.replaceChildren();
    for (const text of [result.method, `${result.output_shape[1]} × ${result.output_shape[0]} pixels`, `${result.latency_ms.toFixed(1)} ms`, 'Metric depth / m']) {
      const span = document.createElement('span'); span.textContent = text; info.append(span);
    }
    document.querySelector('#metric-note').textContent = result.metric_note;
    const metrics = document.querySelector('#metrics'); metrics.replaceChildren();
    if (result.metrics) {
      for (const [key, label] of [['rmse_m','RMSE (m)'],['mae_m','MAE (m)'],['psnr_db','PSNR (dB)'],['ssim','SSIM'],['boundary_precision','Boundary precision'],['boundary_recall','Boundary recall'],['boundary_f1','Boundary F1'],['edge_preservation','Edge preservation']]) {
        const dt = document.createElement('dt'); dt.textContent = label;
        const dd = document.createElement('dd'); const value = result.metrics[key];
        dd.textContent = key === 'psnr_db' && result.metrics.psnr_infinite ? '∞ (exact match)' : value == null ? 'N/A' : value.toFixed(5);
        metrics.append(dt, dd);
      }
    }
    const intermediates = document.querySelector('#intermediates'); intermediates.replaceChildren();
    for (const [name, encoded] of Object.entries(result.intermediate_previews)) {
      const figure = document.createElement('figure'); const img = document.createElement('img');
      img.src = `data:image/png;base64,${encoded}`; img.alt = name.replaceAll('_',' ');
      const caption = document.createElement('figcaption'); caption.textContent = img.alt; figure.append(img,caption); intermediates.append(figure);
    }
    downloadLink('#download-npz', result.output_npz, 'application/octet-stream');
    downloadLink('#download-png', result.output_png16, 'image/png');
    document.querySelector('#empty').hidden = true; document.querySelector('#result').hidden = false;
    message.textContent = 'Reconstruction complete. Downloads preserve numeric depth values.';
  } catch (error) {message.className = 'error';message.textContent = error.name === 'AbortError' ? 'The request timed out. Try a smaller image.' : error.message;}
  finally {clearTimeout(timeout);run.disabled = false;form.setAttribute('aria-busy','false');}
});
