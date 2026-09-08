// MapRot Advanced Geospatial & Vision AI Web Dashboard

let currentSource = 'sample';
let selectedFile = null;
let currentResult = null;
let isDraggingCurtain = false;

document.addEventListener('DOMContentLoaded', () => {
  loadSamples();
  setupClassChips();
  setupCurtainSlider();
});

// Switch between Sample rasters and Custom Upload
function switchSource(source) {
  currentSource = source;
  document.getElementById('tab-sample').classList.toggle('active', source === 'sample');
  document.getElementById('tab-upload').classList.toggle('active', source === 'upload');
  document.getElementById('sample-view').classList.toggle('active', source === 'sample');
  document.getElementById('upload-view').classList.toggle('active', source === 'upload');
}

// Load test sample rasters from backend
async function loadSamples() {
  try {
    const res = await fetch('/api/samples');
    const data = await res.json();
    const select = document.getElementById('sample-select');
    select.innerHTML = '';

    data.samples.forEach((sample, idx) => {
      const opt = document.createElement('option');
      opt.value = sample.id;
      opt.textContent = `${sample.name} (${sample.size_mb} MB)`;
      if (idx === 0) opt.selected = true;
      select.appendChild(opt);
    });

    onSampleChange();
  } catch (err) {
    console.error('Failed to load samples:', err);
  }
}

function onSampleChange() {
  const select = document.getElementById('sample-select');
  const info = document.getElementById('sample-info');
  if (select.value) {
    info.textContent = `Selected: ${select.value}`;
  }
}

// Handle file upload
function handleFileSelect(e) {
  const file = e.target.files[0];
  if (file) {
    selectedFile = file;
    document.getElementById('upload-filename').textContent = `✓ Selected: ${file.name} (${(file.size / (1024*1024)).toFixed(2)} MB)`;
  }
}

// Setup class toggle chips
function setupClassChips() {
  const chips = document.querySelectorAll('.class-chip');
  chips.forEach(chip => {
    chip.addEventListener('click', (e) => {
      if (e.target.tagName !== 'INPUT') {
        const checkbox = chip.querySelector('input');
        checkbox.checked = !checkbox.checked;
      }
      chip.classList.toggle('active', chip.querySelector('input').checked);
    });
  });
}

function getSelectedClasses() {
  const checked = document.querySelectorAll('.class-chip input:checked');
  const classes = Array.from(checked).map(c => c.value);
  if (!classes.includes('background')) {
    classes.unshift('background');
  }
  return classes;
}

// Execute segmentation API
async function executeSegmentation() {
  const runBtn = document.getElementById('run-btn');
  const placeholder = document.getElementById('placeholder');
  const loading = document.getElementById('loading');
  const displayContainer = document.getElementById('display-container');
  const analyticsCard = document.getElementById('analytics-card');
  const exportSection = document.getElementById('export-section');

  const classes = getSelectedClasses();
  const useTTA = document.getElementById('tta-toggle').checked;

  runBtn.disabled = true;
  placeholder.style.display = 'none';
  displayContainer.style.display = 'none';
  loading.style.display = 'block';

  document.getElementById('loading-msg').textContent = useTTA 
    ? 'Executing 4-fold Test-Time Augmentation (TTA) multi-flip ensemble, stitching GeoTIFF, and generating GIS polygons...'
    : 'Slicing into 512x512 tiles, executing deep neural inference, and vectorizing GIS polygons...';

  document.getElementById('stat-crs').textContent = 'Analyzing...';

  try {
    let res;
    if (currentSource === 'upload' && selectedFile) {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('use_tta', useTTA ? 'true' : 'false');
      classes.forEach(c => formData.append('classes', c));

      res = await fetch('/api/segment', {
        method: 'POST',
        body: formData,
      });
    } else {
      const sampleId = document.getElementById('sample-select').value;
      res = await fetch('/api/segment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sample_id: sampleId, classes: classes, use_tta: useTTA }),
      });
    }

    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Segmentation failed');

    currentResult = data;
    renderResults(data);
    exportSection.style.display = 'block';

  } catch (err) {
    alert(`Error: ${err.message}`);
    placeholder.style.display = 'block';
  } finally {
    loading.style.display = 'none';
    runBtn.disabled = false;
  }
}

// Render prediction outputs
function renderResults(data) {
  document.getElementById('display-container').style.display = 'block';
  document.getElementById('analytics-card').style.display = 'flex';

  // Metrics Bar
  document.getElementById('stat-time').textContent = `${data.time_seconds}s`;
  document.getElementById('stat-area').textContent = `${data.total_area_hectares} ha`;
  document.getElementById('stat-polygons').textContent = `${data.num_polygons} vectors`;
  document.getElementById('stat-crs').textContent = data.crs.split(':').pop().slice(0, 10);

  // Set images for Curtain Swipe View
  document.getElementById('curtain-bg').src = data.images.original;
  document.getElementById('curtain-fg').src = data.images.overlay;

  // Set images for other views
  document.getElementById('img-blend-base').src = data.images.original;
  document.getElementById('img-blend-mask').src = data.images.mask;
  document.getElementById('img-split-orig').src = data.images.original;
  document.getElementById('img-split-mask').src = data.images.mask;
  document.getElementById('img-raw-mask').src = data.images.mask;

  // Align curtain overlay image width to container
  const curtainBox = document.getElementById('curtain-box');
  const fgImg = document.getElementById('curtain-fg');
  if (curtainBox && fgImg) {
    fgImg.style.width = `${curtainBox.offsetWidth}px`;
  }

  // Real-world ground surface breakdown table
  const statsTable = document.getElementById('stats-table');
  statsTable.innerHTML = '';

  data.stats.forEach(st => {
    const box = document.createElement('div');
    box.className = 'stat-box';
    box.innerHTML = `
      <div class="stat-box-top">
        <span class="stat-box-title">
          <span class="stat-box-dot" style="background:${st.color}"></span>
          ${st.class.toUpperCase()}
        </span>
        <span class="stat-box-pct" style="color:${st.color}">${st.percentage}%</span>
      </div>
      <div class="stat-box-details">
        <span>${st.hectares} ha</span>
        <span>${st.sq_meters.toLocaleString()} m²</span>
        <span>${st.acres} ac</span>
      </div>
    `;
    statsTable.appendChild(box);
  });
}

// Switch between Swipe Curtain, Alpha Overlay, Split view, and Raw mask
function switchView(mode) {
  document.querySelectorAll('.view-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.view-mode').forEach(v => v.classList.remove('active'));

  event.target.classList.add('active');
  document.getElementById(`view-${mode}`).classList.add('active');

  const opacityContainer = document.getElementById('opacity-container');
  opacityContainer.style.display = mode === 'blend' ? 'flex' : 'none';

  if (mode === 'curtain') {
    const curtainBox = document.getElementById('curtain-box');
    const fgImg = document.getElementById('curtain-fg');
    if (curtainBox && fgImg) {
      fgImg.style.width = `${curtainBox.offsetWidth}px`;
    }
  }
}

// Dynamically update overlay opacity
function updateOpacity(val) {
  document.getElementById('opacity-value').textContent = `${val}%`;
  const mask = document.getElementById('img-blend-mask');
  if (mask) {
    mask.style.opacity = val / 100;
  }
}

// Setup Interactive Curtain Swipe Slider
function setupCurtainSlider() {
  const container = document.getElementById('curtain-box');
  const handle = document.getElementById('curtain-handle');
  const overlay = document.getElementById('curtain-fg-wrap');
  const fgImg = document.getElementById('curtain-fg');

  if (!container || !handle || !overlay) return;

  function setCurtainPosition(clientX) {
    const rect = container.getBoundingClientRect();
    let x = clientX - rect.left;
    x = Math.max(0, Math.min(x, rect.width));

    const pct = (x / rect.width) * 100;
    overlay.style.width = `${pct}%`;
    handle.style.left = `${pct}%`;

    if (fgImg) {
      fgImg.style.width = `${rect.width}px`;
    }
  }

  handle.addEventListener('mousedown', () => { isDraggingCurtain = true; });
  container.addEventListener('mousedown', (e) => {
    isDraggingCurtain = true;
    setCurtainPosition(e.clientX);
  });

  window.addEventListener('mousemove', (e) => {
    if (!isDraggingCurtain) return;
    setCurtainPosition(e.clientX);
  });

  window.addEventListener('mouseup', () => { isDraggingCurtain = false; });

  // Touch Support
  handle.addEventListener('touchstart', () => { isDraggingCurtain = true; }, { passive: true });
  container.addEventListener('touchstart', (e) => {
    isDraggingCurtain = true;
    if (e.touches[0]) setCurtainPosition(e.touches[0].clientX);
  }, { passive: true });

  window.addEventListener('touchmove', (e) => {
    if (!isDraggingCurtain || !e.touches[0]) return;
    setCurtainPosition(e.touches[0].clientX);
  }, { passive: true });

  window.addEventListener('touchend', () => { isDraggingCurtain = false; });
}
