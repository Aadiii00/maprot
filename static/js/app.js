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

  // Update 3D Digital Twin scene if 3D mode is active
  if (document.getElementById('view-3d') && document.getElementById('view-3d').classList.contains('active')) {
    build3DScene(data);
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
    activate3DView();
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
// Three.js 3D Digital Twin Engine
// ==============================================================================
let threeScene, threeCamera, threeRenderer, threeControls;
let threeGroundMesh, threeBuildingGroup, threeWoodlandGroup, threeWaterGroup;
let threeInitialized = false;
let threeBuildingHeightScale = 1.5;
let threeTerrainScale = 0.8;
let threeWireframe = false;
let current3DData = null;

function initThreeJS() {
  if (threeInitialized) return;
  const container = document.getElementById('threejs-container');
  if (!container || !window.THREE) return;

  const width = container.clientWidth || 800;
  const height = container.clientHeight || 560;

  // 1. Scene & Atmospheric Fog
  threeScene = new THREE.Scene();
  threeScene.background = new THREE.Color(0x0a0f1d);
  threeScene.fog = new THREE.FogExp2(0x0a0f1d, 0.001);

  // 2. Perspective Camera (Oblique Bird's-Eye View)
  threeCamera = new THREE.PerspectiveCamera(45, width / height, 1, 5000);
  threeCamera.position.set(0, 360, 480);

  // 3. WebGL Renderer with Antialiasing and Soft Shadows
  threeRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
  threeRenderer.setSize(width, height);
  threeRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  threeRenderer.shadowMap.enabled = true;
  threeRenderer.shadowMap.type = THREE.PCFSoftShadowMap;
  threeRenderer.toneMapping = THREE.ACESFilmicToneMapping;
  threeRenderer.toneMappingExposure = 1.15;
  container.appendChild(threeRenderer.domElement);

  // 4. OrbitControls (Smooth 360 Orbit, Tilt, Pan, Zoom)
  if (window.THREE.OrbitControls) {
    threeControls = new THREE.OrbitControls(threeCamera, threeRenderer.domElement);
    threeControls.enableDamping = true;
    threeControls.dampingFactor = 0.05;
    threeControls.maxPolarAngle = Math.PI / 2 - 0.05; // Prevent camera from going under terrain
    threeControls.minDistance = 80;
    threeControls.maxDistance = 2500;
    threeControls.target.set(0, 0, 0);
  }

  // 5. Lighting Setup: Sun Directional + Sky Hemisphere
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.65);
  threeScene.add(ambientLight);

  const hemiLight = new THREE.HemisphereLight(0x93c5fd, 0x1e293b, 0.55);
  hemiLight.position.set(0, 500, 0);
  threeScene.add(hemiLight);

  const sunLight = new THREE.DirectionalLight(0xfff7ed, 1.35);
  sunLight.position.set(280, 520, 260);
  sunLight.castShadow = true;
  sunLight.shadow.mapSize.width = 2048;
  sunLight.shadow.mapSize.height = 2048;
  sunLight.shadow.camera.near = 20;
  sunLight.shadow.camera.far = 1600;
  const sDist = 480;
  sunLight.shadow.camera.left = -sDist;
  sunLight.shadow.camera.right = sDist;
  sunLight.shadow.camera.top = sDist;
  sunLight.shadow.camera.bottom = -sDist;
  sunLight.shadow.bias = -0.0004;
  threeScene.add(sunLight);

  // 6. Object Groups
  threeBuildingGroup = new THREE.Group();
  threeScene.add(threeBuildingGroup);

  threeWoodlandGroup = new THREE.Group();
  threeScene.add(threeWoodlandGroup);

  threeWaterGroup = new THREE.Group();
  threeScene.add(threeWaterGroup);

  window.addEventListener('resize', onThreeResize);
  threeInitialized = true;
  animate3D();
}

function onThreeResize() {
  const container = document.getElementById('threejs-container');
  if (!container || !threeRenderer || !threeCamera) return;
  const width = container.clientWidth;
  const height = container.clientHeight;
  if (width === 0 || height === 0) return;
  threeCamera.aspect = width / height;
  threeCamera.updateProjectionMatrix();
  threeRenderer.setSize(width, height);
}

function animate3D() {
  requestAnimationFrame(animate3D);
  if (threeControls) threeControls.update();
  if (threeRenderer && threeScene && threeCamera) {
    threeRenderer.render(threeScene, threeCamera);
  }
}

function activate3DView() {
  if (!threeInitialized) initThreeJS();
  setTimeout(() => {
    onThreeResize();
    if (currentResult && (!current3DData || current3DData !== currentResult)) {
      build3DScene(currentResult);
    }
  }, 50);
}

// Build 3D meshes from raster texture and AI polygon contours
function build3DScene(data) {
  if (!threeInitialized) initThreeJS();
  if (!data || !data.images) return;
  current3DData = data;

  const origW = data.dimensions.width;
  const origH = data.dimensions.height;
  const planeW = 600;
  const planeH = 600 * (origH / origW);

  // Clear previous scene elements
  if (threeGroundMesh) {
    threeScene.remove(threeGroundMesh);
    if (threeGroundMesh.geometry) threeGroundMesh.geometry.dispose();
  }
  while (threeBuildingGroup.children.length > 0) {
    const obj = threeBuildingGroup.children[0];
    threeBuildingGroup.remove(obj);
    if (obj.geometry) obj.geometry.dispose();
  }
  while (threeWoodlandGroup.children.length > 0) {
    const obj = threeWoodlandGroup.children[0];
    threeWoodlandGroup.remove(obj);
    if (obj.geometry) obj.geometry.dispose();
  }
  while (threeWaterGroup.children.length > 0) {
    const obj = threeWaterGroup.children[0];
    threeWaterGroup.remove(obj);
    if (obj.geometry) obj.geometry.dispose();
  }

  // 1. Texture Satellite Terrain
  const textureLoader = new THREE.TextureLoader();
  textureLoader.load(data.images.original, (texture) => {
    texture.generateMipmaps = true;
    texture.minFilter = THREE.LinearMipmapLinearFilter;

    // Organic Terrain Relief Geometry
    const groundGeo = new THREE.PlaneGeometry(planeW, planeH, 96, 96);
    groundGeo.rotateX(-Math.PI / 2);

    const pos = groundGeo.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const z = pos.getZ(i);
      const elev = (Math.sin(x * 0.015) * Math.cos(z * 0.015) * 8 +
                    Math.sin(x * 0.03 + z * 0.02) * 3) * threeTerrainScale;
      pos.setY(i, elev);
    }
    groundGeo.computeVertexNormals();

    const groundMat = new THREE.MeshStandardMaterial({
      map: texture,
      roughness: 0.85,
      metalness: 0.05,
      wireframe: threeWireframe,
    });

    threeGroundMesh = new THREE.Mesh(groundGeo, groundMat);
    threeGroundMesh.receiveShadow = true;
    threeScene.add(threeGroundMesh);
  });

  // 2. Extrude Buildings & Volumes from AI Contours
  const contours = data.contours_3d || [];

  const wallMat = new THREE.MeshStandardMaterial({
    color: 0x94a3b8, // sleek architectural concrete/slate
    roughness: 0.6,
    metalness: 0.2,
    wireframe: threeWireframe,
  });

  const roofMat = new THREE.MeshStandardMaterial({
    color: 0xef4444, // sleek modern building rooftop red
    roughness: 0.45,
    metalness: 0.15,
    wireframe: threeWireframe,
  });

  const buildingMats = [wallMat, roofMat];

  contours.forEach((cnt) => {
    const cls = cnt.class.toLowerCase();
    const pts = cnt.points;
    if (!pts || pts.length < 3) return;

    if (cls === 'building') {
      const shape = new THREE.Shape();
      pts.forEach((p, idx) => {
        // Map 2D pixel coord into centered 3D terrain space
        const x3d = (p[0] / origW - 0.5) * planeW;
        const z3d = (p[1] / origH - 0.5) * planeH;
        if (idx === 0) shape.moveTo(x3d, -z3d);
        else shape.lineTo(x3d, -z3d);
      });

      // Extrude height based on footprint surface area
      const baseHeight = Math.min(45, Math.max(10, Math.sqrt(cnt.area) * 0.7));
      const extrudeSettings = {
        depth: baseHeight,
        bevelEnabled: true,
        bevelSegments: 1,
        steps: 1,
        bevelSize: 0.4,
        bevelThickness: 0.4,
      };

      try {
        const extrudeGeo = new THREE.ExtrudeGeometry(shape, extrudeSettings);
        extrudeGeo.rotateX(Math.PI / 2);

        const buildingMesh = new THREE.Mesh(extrudeGeo, buildingMats);
        buildingMesh.position.y = 0.5;
        buildingMesh.castShadow = true;
        buildingMesh.receiveShadow = true;
        threeBuildingGroup.add(buildingMesh);
      } catch (err) {
        // Silently skip degenerate polygon self-intersections
      }
    } else if (cls === 'woodland') {
      // Procedural 3D Canopy volume
      let cx = 0, cz = 0;
      pts.forEach(p => {
        cx += (p[0] / origW - 0.5) * planeW;
        cz += (p[1] / origH - 0.5) * planeH;
      });
      cx /= pts.length;
      cz /= pts.length;

      const radius = Math.min(16, Math.max(4, Math.sqrt(cnt.area) * 0.32));
      const canopyGeo = new THREE.DodecahedronGeometry(radius, 1);
      const canopyMat = new THREE.MeshStandardMaterial({
        color: 0x22c55e,
        roughness: 0.85,
        wireframe: threeWireframe,
      });
      const canopyMesh = new THREE.Mesh(canopyGeo, canopyMat);
      canopyMesh.position.set(cx, radius * 0.9, cz);
      canopyMesh.scale.set(1.15, 0.85, 1.15);
      canopyMesh.castShadow = true;
      threeWoodlandGroup.add(canopyMesh);
    } else if (cls === 'water') {
      // Water body plane
      const shape = new THREE.Shape();
      pts.forEach((p, idx) => {
        const x3d = (p[0] / origW - 0.5) * planeW;
        const z3d = (p[1] / origH - 0.5) * planeH;
        if (idx === 0) shape.moveTo(x3d, -z3d);
        else shape.lineTo(x3d, -z3d);
      });
      try {
        const waterGeo = new THREE.ShapeGeometry(shape);
        waterGeo.rotateX(Math.PI / 2);
        const waterMat = new THREE.MeshStandardMaterial({
          color: 0x0284c7,
          roughness: 0.1,
          metalness: 0.85,
          transparent: true,
          opacity: 0.75,
          wireframe: threeWireframe,
        });
        const waterMesh = new THREE.Mesh(waterGeo, waterMat);
        waterMesh.position.y = 0.8;
        threeWaterGroup.add(waterMesh);
      } catch (e) {}
    }
  });

  threeBuildingGroup.scale.y = threeBuildingHeightScale;
}

// 3D HUD Control Handlers
function update3DHeight(val) {
  threeBuildingHeightScale = parseFloat(val);
  document.getElementById('threejs-height-label').textContent = `${val}x`;
  if (threeBuildingGroup) {
    threeBuildingGroup.scale.y = threeBuildingHeightScale;
  }
}

function update3DTerrain(val) {
  threeTerrainScale = parseFloat(val);
  document.getElementById('threejs-terrain-label').textContent = `${val}x`;
  if (currentResult) {
    build3DScene(currentResult);
  }
}

function toggle3DWireframe() {
  threeWireframe = !threeWireframe;
  document.getElementById('wireframe-btn').textContent = `Wireframe: ${threeWireframe ? 'On' : 'Off'}`;
  if (threeScene) {
    threeScene.traverse((obj) => {
      if (obj.isMesh && obj.material) {
        if (Array.isArray(obj.material)) {
          obj.material.forEach(m => m.wireframe = threeWireframe);
        } else {
          obj.material.wireframe = threeWireframe;
        }
      }
    });
  }
}

function reset3DCamera() {
  if (!threeCamera || !threeControls) return;
  threeCamera.position.set(0, 360, 480);
  threeControls.target.set(0, 0, 0);
  threeControls.update();
}
