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
fetch('/api/health').then(r => {if (!r.ok) throw new Error('Backend unavailable'); return r.json();}).then(health => {
  document.querySelector('#connection').textContent = '● Backend connected';
  document.querySelector('#model-note').textContent = health.trained_checkpoint_loaded
    ? 'Trained checkpoint loaded. Classical, deep learning, and hybrid methods are available.'
    : 'Classical methods are ready. Deep learning and hybrid methods become available when a trained checkpoint is configured.';
  for (const option of document.querySelector('#method').options) option.disabled = !health.methods.includes(option.value);
}).catch(() => {document.querySelector('#connection').textContent = 'Backend unavailable';document.querySelector('#model-note').textContent = 'Start the FastAPI server to reconstruct depth maps.';});
document.querySelector('#protocol').addEventListener('change', event => {
  document.querySelector('#shape-note').textContent = event.target.value === 'dense_sr'
    ? 'RGB and reference dimensions must equal depth dimensions × scale.'
    : 'Completion keeps the input resolution. Scale conditions the learned model; RGB and reference must match the depth dimensions.';
});
form.addEventListener('submit', async event => {
  event.preventDefault();
  run.disabled = true; message.className = ''; message.textContent = 'Reconstructing depth…';
  const data = new FormData(form);
  for (const name of ['rgb', 'mask', 'ground_truth']) if (!data.get(name)?.size) data.delete(name);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120000);
  try {
    const response = await fetch('/api/reconstruct', {method: 'POST', body: data, signal: controller.signal});
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : JSON.stringify(result.detail));
    objectUrls.splice(0).forEach(url => URL.revokeObjectURL(url));
    document.querySelector('#input-image').src = `data:image/png;base64,${result.input_preview}`;
    document.querySelector('#output-image').src = `data:image/png;base64,${result.output_preview}`;
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
  finally {clearTimeout(timeout);run.disabled = false;}
});
