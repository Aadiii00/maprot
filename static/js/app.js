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
      const isChecked = chip.querySelector('input').checked;
      chip.classList.toggle('active', isChecked);

      if (viewer3D) {
        const clsName = chip.getAttribute('data-class');
        if (clsName) viewer3D.setClassVisibility(clsName, isChecked);
      }
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

  // Update 3D Photorealistic scene if 3D mode is active
  if (document.getElementById('view-3d') && document.getElementById('view-3d').classList.contains('active')) {
    load3DSceneWithProgress(data);
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

// Switch between Swipe Curtain, Alpha Overlay, Split view, Raw mask, and 3D Digital Twin
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
  } else if (mode === '3d') {
    activate3DPhotorealisticView();
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

// ==============================================================================
// Photorealistic Three.js 3D Viewer Integration
// ==============================================================================
let viewer3D = null;
let current3DLoadedResult = null;

function init3DViewer() {
  if (viewer3D) return;
  const container = document.getElementById('threejs-container');
  if (!container || !window.MapRot3DViewer) return;

  viewer3D = new MapRot3DViewer('threejs-container', {
    onBuildingSelect: (info) => show3DBuildingInfo(info),
  });
}

function activate3DPhotorealisticView() {
  if (!viewer3D) init3DViewer();
  setTimeout(() => {
    if (viewer3D) viewer3D.onResize();
    if (currentResult && currentResult !== current3DLoadedResult) {
      load3DSceneWithProgress(currentResult);
    }
  }, 40);
}

function load3DSceneWithProgress(data) {
  if (!viewer3D) init3DViewer();
  if (!data || !data.geojson || !viewer3D) return;
  current3DLoadedResult = data;

  const loadingOverlay = document.getElementById('threejs-loading');
  const loadingText = document.getElementById('threejs-loading-text');
  if (loadingOverlay) loadingOverlay.style.display = 'flex';

  setTimeout(() => {
    viewer3D.load(data.geojson, data.images.original, (msg, pct) => {
      if (loadingText) loadingText.textContent = msg;
      if (pct >= 100 && loadingOverlay) {
        setTimeout(() => { loadingOverlay.style.display = 'none'; }, 250);
      }
    });

    // Sync active classes
    ['building', 'woodland', 'water', 'road'].forEach(cls => {
      const chip = document.querySelector(`.class-chip[data-class="${cls}"] input`);
      const isVisible = chip ? chip.checked : true;
      viewer3D.setClassVisibility(cls, isVisible);
    });
  }, 30);
}

// 3D HUD Event Handlers
function on3DTimeChange(val) {
  if (!viewer3D) return;
  viewer3D.setTimeOfDay(parseFloat(val));
  const h = Math.floor(val);
  const m = (val % 1) === 0.5 ? '30' : '00';
  document.getElementById('threejs-time-label').textContent = `${h}:${m}`;
}

function on3DHeightChange(val) {
  if (!viewer3D) return;
  viewer3D.setBuildingHeightScale(val);
  document.getElementById('threejs-height-label').textContent = `${val}x`;
}

function on3DToggleWireframe() {
  if (!viewer3D) return;
  const isW = viewer3D.toggleWireframe();
  const btn = document.getElementById('wireframe-btn');
  if (btn) btn.textContent = `Wireframe: ${isW ? 'On' : 'Off'}`;
}

function on3DResetCamera() {
  if (viewer3D) viewer3D.resetCamera();
}

function show3DBuildingInfo(info) {
  const inspector = document.getElementById('building-inspector');
  const body = document.getElementById('inspector-body');
  if (!inspector || !body || !info) return;

  const feat = info.feature || {};
  const props = feat.properties || {};
  const area = props.area_map_units || (info.height * info.height);

  body.innerHTML = `
    <div><strong>Class:</strong> ${(props.class || 'Building').toUpperCase()}</div>
    <div><strong>Footprint Area:</strong> ${Math.round(area).toLocaleString()} m²</div>
    <div><strong>Estimated Height:</strong> ${info.height.toFixed(1)} m</div>
    <div><strong>Est. Stories:</strong> ${Math.max(1, Math.round(info.height / 3.5))} floors</div>
  `;
  inspector.style.display = 'block';
}

function close3DInspector() {
  const inspector = document.getElementById('building-inspector');
  if (inspector) inspector.style.display = 'none';
}
