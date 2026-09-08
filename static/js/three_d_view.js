/**
 * MapRot Photorealistic 3D Terrain & Structure Viewer
 * 
 * Powered directly by the standard MapRot GeoJSON FeatureCollection export
 * and source satellite rasters.
 * 
 * Realistic Render Engine:
 *  - High-res draped satellite terrain
 *  - Architecturally detailed buildings: PBR facade window fenestration, diverse roof materials,
 *    and rooftop utility boxes (HVAC/elevator structures)
 *  - Botanical tree rendering: Instanced multi-layered organic foliage + wooden bark trunks with
 *    natural botanical color variations and organic lean
 *  - Animated reflective water with specular sun glints
 *  - Dynamic Time-of-Day solar lighting with soft shadow mapping
 *  - Orbit controls with smooth damping & Fly-To camera focus
 *  - Real-time synchronization with 2D class toggles
 */

class MapRot3DViewer {
  constructor(containerId, options = {}) {
    this.container = document.getElementById(containerId);
    this.options = Object.assign({
      planeSize: 600,
      buildingBaseScale: 1.5,
      timeOfDay: 14, // 2 PM default
      treeDensity: 1.0,
      maxTrees: 3500,
    }, options);

    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();

    // Lighting
    this.sunLight = null;
    this.hemiLight = null;
    this.ambientLight = null;

    // Groups
    this.terrainGroup = null;
    this.buildingGroup = null;
    this.woodlandGroup = null;
    this.waterGroup = null;
    this.roadGroup = null;

    // State
    this.isInitialized = false;
    this.clock = new THREE.Clock();
    this.waterMeshes = [];
    this.currentData = null;
    this.buildingHeightScale = this.options.buildingBaseScale;
    this.timeOfDay = this.options.timeOfDay;
    this.isWireframe = false;
    this.selectedBuilding = null;

    // Procedural Shared Textures
    this.facadeTexture = null;
    this.roofTextures = [];

    this.init();
  }

  init() {
    if (this.isInitialized || !this.container || !window.THREE) return;

    const width = this.container.clientWidth || 800;
    const height = this.container.clientHeight || 560;

    // 1. Scene & Atmospheric Horizon Fog
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0c1322);
    this.scene.fog = new THREE.FogExp2(0x0c1322, 0.00085);

    // 2. Camera (Oblique Perspective)
    this.camera = new THREE.PerspectiveCamera(45, width / height, 1, 6000);
    this.camera.position.set(0, 380, 520);

    // 3. WebGL2 Renderer with Soft Shadows & ACES Tone Mapping
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
    });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.2;
    this.container.appendChild(this.renderer.domElement);

    // 4. OrbitControls
    if (window.THREE.OrbitControls) {
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.05;
      this.controls.maxPolarAngle = Math.PI / 2 - 0.03; // Prevent dipping underground
      this.controls.minDistance = 60;
      this.controls.maxDistance = 2800;
      this.controls.target.set(0, 0, 0);
    }

    // 5. Lighting Setup
    this.ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(this.ambientLight);

    this.hemiLight = new THREE.HemisphereLight(0xa5f3fc, 0x1e293b, 0.6);
    this.hemiLight.position.set(0, 500, 0);
    this.scene.add(this.hemiLight);

    this.sunLight = new THREE.DirectionalLight(0xfffaed, 1.45);
    this.sunLight.castShadow = true;
    this.sunLight.shadow.mapSize.width = 2048;
    this.sunLight.shadow.mapSize.height = 2048;
    this.sunLight.shadow.camera.near = 20;
    this.sunLight.shadow.camera.far = 1800;
    const sDist = 520;
    this.sunLight.shadow.camera.left = -sDist;
    this.sunLight.shadow.camera.right = sDist;
    this.sunLight.shadow.camera.top = sDist;
    this.sunLight.shadow.camera.bottom = -sDist;
    this.sunLight.shadow.bias = -0.0003;
    this.scene.add(this.sunLight);

    // 6. Layer Groups
    this.terrainGroup = new THREE.Group();
    this.buildingGroup = new THREE.Group();
    this.woodlandGroup = new THREE.Group();
    this.waterGroup = new THREE.Group();
    this.roadGroup = new THREE.Group();

    this.scene.add(this.terrainGroup);
    this.scene.add(this.waterGroup);
    this.scene.add(this.roadGroup);
    this.scene.add(this.woodlandGroup);
    this.scene.add(this.buildingGroup);

    // Prepare procedural PBR architectural textures
    this.generateProceduralMaterials();

    // Apply default time-of-day lighting
    this.setTimeOfDay(this.timeOfDay);

    // Events
    window.addEventListener('resize', () => this.onResize());
    this.renderer.domElement.addEventListener('click', (e) => this.onCanvasClick(e));

    this.isInitialized = true;
    this.animate();
  }

  // Generate realistic procedural building textures (facade windows & roof materials)
  generateProceduralMaterials() {
    // 1. Facade Window Grid Texture
    const fCanvas = document.createElement('canvas');
    fCanvas.width = 512;
    fCanvas.height = 512;
    const fCtx = fCanvas.getContext('2d');

    // Concrete facade base
    fCtx.fillStyle = '#94a3b8';
    fCtx.fillRect(0, 0, 512, 512);

    // Floor dividers (spandrels)
    fCtx.fillStyle = '#64748b';
    for (let y = 0; y < 512; y += 64) {
      fCtx.fillRect(0, y, 512, 12);
    }

    // Vertical structural columns
    fCtx.fillStyle = '#475569';
    for (let x = 0; x < 512; x += 64) {
      fCtx.fillRect(x, 0, 8, 512);
    }

    // Modern glass windows with subtle specular glint
    for (let y = 16; y < 512; y += 64) {
      for (let x = 12; x < 512; x += 64) {
        // Deep tinted glass
        fCtx.fillStyle = '#0f172a';
        fCtx.fillRect(x, y, 48, 42);

        // Window mullion frame
        fCtx.strokeStyle = '#334155';
        fCtx.lineWidth = 1.5;
        fCtx.strokeRect(x, y, 48, 42);

        // Glass reflection sheen
        fCtx.fillStyle = 'rgba(186, 230, 253, 0.38)';
        fCtx.fillRect(x + 3, y + 3, 16, 14);
      }
    }

    this.facadeTexture = new THREE.CanvasTexture(fCanvas);
    this.facadeTexture.wrapS = THREE.RepeatWrapping;
    this.facadeTexture.wrapT = THREE.RepeatWrapping;
    this.facadeTexture.repeat.set(1.5, 3);

    // 2. Varied Roof Textures (Slate, Terracotta, Industrial Gravel, Zinc)
    const roofColors = [
      { base: '#334155', lines: '#1e293b' }, // Dark slate
      { base: '#b91c1c', lines: '#7f1d1d' }, // Warm terracotta
      { base: '#475569', lines: '#334155' }, // Industrial gravel
      { base: '#64748b', lines: '#475569' }, // Zinc / sheet metal
    ];

    this.roofTextures = roofColors.map(cfg => {
      const rCanvas = document.createElement('canvas');
      rCanvas.width = 128;
      rCanvas.height = 128;
      const rCtx = rCanvas.getContext('2d');
      rCtx.fillStyle = cfg.base;
      rCtx.fillRect(0, 0, 128, 128);

      rCtx.strokeStyle = cfg.lines;
      rCtx.lineWidth = 2;
      for (let y = 0; y < 128; y += 16) {
        rCtx.beginPath(); rCtx.moveTo(0, y); rCtx.lineTo(128, y); rCtx.stroke();
      }

      const rTex = new THREE.CanvasTexture(rCanvas);
      rTex.wrapS = THREE.RepeatWrapping;
      rTex.wrapT = THREE.RepeatWrapping;
      rTex.repeat.set(2, 2);
      return rTex;
    });
  }

  onResize() {
    if (!this.container || !this.renderer || !this.camera) return;
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    if (width === 0 || height === 0) return;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    const elapsed = this.clock.getElapsedTime();

    // Animate water ripple displacement
    this.waterMeshes.forEach(mesh => {
      if (mesh.material && mesh.material.opacity) {
        mesh.position.y = -0.35 + Math.sin(elapsed * 2.2 + mesh.id) * 0.18;
      }
    });

    if (this.controls) this.controls.update();
    if (this.renderer && this.scene && this.camera) {
      this.renderer.render(this.scene, this.camera);
    }
  }

  /**
   * Adjusts Sun direction and atmospheric color based on time of day (0 to 24 hours).
   */
  setTimeOfDay(hour) {
    this.timeOfDay = Math.max(0, Math.min(24, hour));

    const sunAngle = ((this.timeOfDay - 6) / 12) * Math.PI;
    const isDay = this.timeOfDay >= 5.5 && this.timeOfDay <= 19.5;

    const sunDist = 650;
    const x = Math.cos(sunAngle) * sunDist;
    const y = Math.max(30, Math.sin(sunAngle) * sunDist);
    const z = Math.sin(sunAngle * 0.8) * 320;

    this.sunLight.position.set(x, y, z);

    if (isDay) {
      const elevation = Math.sin(sunAngle);
      if (elevation < 0.25) {
        // Golden hour (sunrise / sunset)
        this.sunLight.color.setHex(0xffaa55);
        this.sunLight.intensity = 1.6;
        this.hemiLight.color.setHex(0xfbbf24);
        this.hemiLight.groundColor.setHex(0x3b1c06);
        this.ambientLight.intensity = 0.5;
        this.scene.background.setHex(0x1a1218);
        this.scene.fog.color.setHex(0x1a1218);
      } else {
        // Daylight / Crisp noon
        this.sunLight.color.setHex(0xfffaed);
        this.sunLight.intensity = 1.45;
        this.hemiLight.color.setHex(0xa5f3fc);
        this.hemiLight.groundColor.setHex(0x1e293b);
        this.ambientLight.intensity = 0.65;
        this.scene.background.setHex(0x0c1322);
        this.scene.fog.color.setHex(0x0c1322);
      }
    } else {
      // Moonlight night mode
      this.sunLight.color.setHex(0x38bdf8);
      this.sunLight.intensity = 0.35;
      this.hemiLight.color.setHex(0x1e293b);
      this.hemiLight.groundColor.setHex(0x050811);
      this.ambientLight.intensity = 0.25;
      this.scene.background.setHex(0x030712);
      this.scene.fog.color.setHex(0x030712);
    }
  }

  /**
   * Load GeoJSON features and source raster texture into the photorealistic 3D scene.
   */
  load(geojson, rasterUrl, onProgress = null) {
    if (!this.isInitialized) this.init();
    if (!geojson || !geojson.features) return;

    this.currentData = { geojson, rasterUrl };
    if (onProgress) onProgress("Parsing vector geometry...", 20);

    // 1. Clear previous layers
    this.clearGroup(this.terrainGroup);
    this.clearGroup(this.buildingGroup);
    this.clearGroup(this.woodlandGroup);
    this.clearGroup(this.waterGroup);
    this.clearGroup(this.roadGroup);
    this.waterMeshes = [];

    // 2. Compute Coordinate Bounding Box across all GeoJSON polygons
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    const features = geojson.features;

    features.forEach(f => {
      const geom = f.geometry;
      if (!geom) return;
      const rings = geom.type === 'Polygon' ? [geom.coordinates] : (geom.type === 'MultiPolygon' ? geom.coordinates : []);
      rings.forEach(polyRings => {
        polyRings.forEach(ring => {
          ring.forEach(pt => {
            const px = pt[0];
            const py = pt[1];
            if (px < minX) minX = px;
            if (px > maxX) maxX = px;
            if (py < minY) minY = py;
            if (py > maxY) maxY = py;
          });
        });
      });
    });

    const bboxW = Math.max(1, maxX - minX);
    const bboxH = Math.max(1, maxY - minY);
    const aspect = bboxH / bboxW;

    const planeW = this.options.planeSize;
    const planeH = planeW * aspect;

    const mapCoord = (gx, gy) => {
      const u = (gx - minX) / bboxW;
      const v = (gy - minY) / bboxH;
      const x = (u - 0.5) * planeW;
      const z = (v - 0.5) * planeH;
      return { x, z };
    };

    if (onProgress) onProgress("Draping high-resolution satellite terrain...", 40);

    // 3. Build Draped Satellite Terrain Base
    const textureLoader = new THREE.TextureLoader();
    textureLoader.load(rasterUrl, (texture) => {
      texture.generateMipmaps = true;
      texture.minFilter = THREE.LinearMipmapLinearFilter;
      texture.wrapS = THREE.ClampToEdgeWrapping;
      texture.wrapT = THREE.ClampToEdgeWrapping;

      const groundGeo = new THREE.PlaneGeometry(planeW, planeH, 64, 64);
      groundGeo.rotateX(-Math.PI / 2);

      const groundMat = new THREE.MeshStandardMaterial({
        map: texture,
        roughness: 0.85,
        metalness: 0.05,
        wireframe: this.isWireframe,
      });

      const groundMesh = new THREE.Mesh(groundGeo, groundMat);
      groundMesh.receiveShadow = true;
      this.terrainGroup.add(groundMesh);
    });

    if (onProgress) onProgress(`Constructing ${features.length} 3D architectural structures...`, 65);

    // Shared Water & Road Materials
    const waterMat = new THREE.MeshStandardMaterial({
      color: 0x0284c7, // Azure water
      roughness: 0.08,
      metalness: 0.85,
      transparent: true,
      opacity: 0.84,
      wireframe: this.isWireframe,
    });

    const roadMat = new THREE.MeshStandardMaterial({
      color: 0x334155, // Asphalt dark slate
      roughness: 0.88,
      metalness: 0.1,
      wireframe: this.isWireframe,
    });

    const treeData = [];

    // 4. Ingest and Render Each Class Feature
    features.forEach((feature, fIndex) => {
      const cls = (feature.properties.class || '').toLowerCase();
      const geom = feature.geometry;
      if (!geom) return;

      const polyList = geom.type === 'Polygon' ? [geom.coordinates] : (geom.type === 'MultiPolygon' ? geom.coordinates : []);

      polyList.forEach((rings) => {
        if (!rings || rings.length === 0 || rings[0].length < 3) return;

        const exteriorRing = rings[0];
        const shape = new THREE.Shape();

        exteriorRing.forEach((pt, idx) => {
          const { x, z } = mapCoord(pt[0], pt[1]);
          if (idx === 0) shape.moveTo(x, -z);
          else shape.lineTo(x, -z);
        });

        // Add interior courtyards/holes
        for (let h = 1; h < rings.length; h++) {
          const holeRing = rings[h];
          if (holeRing.length < 3) continue;
          const holePath = new THREE.Path();
          holeRing.forEach((pt, idx) => {
            const { x, z } = mapCoord(pt[0], pt[1]);
            if (idx === 0) holePath.moveTo(x, -z);
            else holePath.lineTo(x, -z);
          });
          shape.holes.push(holePath);
        }

        if (cls === 'building') {
          const area = feature.properties.area_map_units || 120;
          const estHeight = Math.min(50, Math.max(10, Math.sqrt(area) * 0.75));

          const extrudeSettings = {
            depth: estHeight,
            bevelEnabled: true,
            bevelSegments: 1,
            steps: 1,
            bevelSize: 0.4,
            bevelThickness: 0.4,
          };

          try {
            const extrudeGeo = new THREE.ExtrudeGeometry(shape, extrudeSettings);
            extrudeGeo.rotateX(Math.PI / 2);

            // Architectural Materials: Textured Facade Walls + Varied Realistic Roof
            const roofTex = this.roofTextures[fIndex % this.roofTextures.length];
            const wallMat = new THREE.MeshStandardMaterial({
              map: this.facadeTexture,
              roughness: 0.65,
              metalness: 0.2,
              wireframe: this.isWireframe,
            });
            const roofMat = new THREE.MeshStandardMaterial({
              map: roofTex,
              roughness: 0.5,
              metalness: 0.15,
              wireframe: this.isWireframe,
            });

            const buildingMaterials = [wallMat, roofMat];
            const bMesh = new THREE.Mesh(extrudeGeo, buildingMaterials);
            bMesh.position.y = 0.4;
            bMesh.castShadow = true;
            bMesh.receiveShadow = true;
            bMesh.userData = {
              feature: feature,
              height: estHeight,
              class: 'building',
            };
            this.buildingGroup.add(bMesh);

            // On larger buildings (area > 350 m2), add realistic rooftop utility structures (HVAC/Elevator boxes)
            if (area > 350) {
              let cx = 0, cz = 0;
              exteriorRing.forEach(p => {
                const { x, z } = mapCoord(p[0], p[1]);
                cx += x; cz += z;
              });
              cx /= exteriorRing.length;
              cz /= exteriorRing.length;

              const boxW = Math.max(4, Math.sqrt(area) * 0.18);
              const boxH = Math.max(2.5, estHeight * 0.12);
              const hvacGeo = new THREE.BoxGeometry(boxW, boxH, boxW * 0.8);
              const hvacMat = new THREE.MeshStandardMaterial({
                color: 0x475569, // Metal equipment gray
                roughness: 0.5,
                metalness: 0.4,
              });
              const hvacMesh = new THREE.Mesh(hvacGeo, hvacMat);
              hvacMesh.position.set(cx, estHeight + boxH / 2 + 0.4, cz);
              hvacMesh.castShadow = true;
              this.buildingGroup.add(hvacMesh);
            }

          } catch (e) {}

        } else if (cls === 'water') {
          try {
            const waterGeo = new THREE.ShapeGeometry(shape);
            waterGeo.rotateX(Math.PI / 2);
            const wMesh = new THREE.Mesh(waterGeo, waterMat);
            wMesh.position.y = -0.3;
            wMesh.receiveShadow = true;
            this.waterGroup.add(wMesh);
            this.waterMeshes.push(wMesh);
          } catch (e) {}

        } else if (cls === 'road') {
          try {
            const roadGeo = new THREE.ShapeGeometry(shape);
            roadGeo.rotateX(Math.PI / 2);
            const rMesh = new THREE.Mesh(roadGeo, roadMat);
            rMesh.position.y = 0.15;
            rMesh.receiveShadow = true;
            this.roadGroup.add(rMesh);
          } catch (e) {}

        } else if (cls === 'woodland') {
          // Natural Tree Distribution across polygon
          const area = feature.properties.area_map_units || 250;
          const numTrees = Math.min(65, Math.max(3, Math.floor(Math.sqrt(area) * 0.85 * this.options.treeDensity)));

          let cx = 0, cz = 0;
          exteriorRing.forEach(p => {
            const { x, z } = mapCoord(p[0], p[1]);
            cx += x; cz += z;
          });
          cx /= exteriorRing.length;
          cz /= exteriorRing.length;

          for (let t = 0; t < numTrees; t++) {
            if (treeData.length >= this.options.maxTrees) break;
            const rOffset = (Math.random() - 0.5) * Math.sqrt(area) * 0.5;
            const thOffset = Math.random() * Math.PI * 2;
            const tx = cx + Math.cos(thOffset) * rOffset;
            const tz = cz + Math.sin(thOffset) * rOffset;
            
            // Randomize size and rotation for natural organic forest look
            const scaleY = 0.75 + Math.random() * 0.65;
            const scaleXZ = 0.8 + Math.random() * 0.45;
            const rotY = Math.random() * Math.PI * 2;
            const tiltX = (Math.random() - 0.5) * 0.08; // Subtle organic lean
            const tiltZ = (Math.random() - 0.5) * 0.08;

            // Botanical color palette: deep pine, forest emerald, olive moss, spring canopy
            const botanicalColors = [
              0x15803d, // Forest emerald
              0x166534, // Deep pine shadow
              0x14532d, // Rich spruce
              0x3f6212, // Olive moss
              0x22c55e, // Fresh spring green
              0x4d7c0f, // Golden canopy highlight
            ];
            const leafColor = botanicalColors[Math.floor(Math.random() * botanicalColors.length)];

            treeData.push({ x: tx, z: tz, scaleXZ, scaleY, rotY, tiltX, tiltZ, leafColor });
          }
        }
      });
    });

    // 5. Realistic Dual-Instanced Tree System (Wooden Bark Trunk + Multi-Tier Canopy)
    if (treeData.length > 0) {
      if (onProgress) onProgress(`Growing ${treeData.length} photorealistic botanical trees...`, 85);

      const count = treeData.length;

      // A. Trunk Instanced Mesh (Wooden Bark)
      const trunkGeo = new THREE.CylinderGeometry(0.3, 0.55, 3.5, 6);
      trunkGeo.translate(0, 1.75, 0);
      const trunkMat = new THREE.MeshStandardMaterial({
        color: 0x3e2723, // Deep woody bark brown
        roughness: 0.95,
        metalness: 0.02,
        wireframe: this.isWireframe,
      });
      const instancedTrunks = new THREE.InstancedMesh(trunkGeo, trunkMat, count);
      instancedTrunks.castShadow = true;
      instancedTrunks.receiveShadow = true;

      // B. Multi-Tier Organic Foliage Canopy (Merged Organic Dodecahedrons)
      // Tier 1: Wide lower canopy
      const tier1 = new THREE.DodecahedronGeometry(2.4, 1);
      tier1.scale(1.2, 0.85, 1.2);
      tier1.translate(0, 4.0, 0);

      // Tier 2: Middle crown
      const tier2 = new THREE.DodecahedronGeometry(1.9, 1);
      tier2.scale(1.0, 0.9, 1.0);
      tier2.translate(0, 5.7, 0);

      // Tier 3: Tapered top
      const tier3 = new THREE.DodecahedronGeometry(1.3, 1);
      tier3.translate(0, 7.1, 0);

      // Merge foliage tiers into one organic canopy geometry
      // Using buffer attributes merge for high performance
      const foliageGeo = this.mergeBufferGeometries([tier1, tier2, tier3]);

      const foliageMat = new THREE.MeshStandardMaterial({
        roughness: 0.88,
        metalness: 0.05,
        flatShading: true,
        wireframe: this.isWireframe,
      });

      const instancedCanopies = new THREE.InstancedMesh(foliageGeo, foliageMat, count);
      instancedCanopies.castShadow = true;
      instancedCanopies.receiveShadow = true;

      const dummy = new THREE.Object3D();
      const col = new THREE.Color();

      treeData.forEach((td, i) => {
        dummy.position.set(td.x, 0, td.z);
        dummy.rotation.set(td.tiltX, td.rotY, td.tiltZ);
        dummy.scale.set(td.scaleXZ, td.scaleY, td.scaleXZ);
        dummy.updateMatrix();

        instancedTrunks.setMatrixAt(i, dummy.matrix);
        instancedCanopies.setMatrixAt(i, dummy.matrix);

        // Apply natural botanical green shade per tree instance
        col.setHex(td.leafColor);
        instancedCanopies.setColorAt(i, col);
      });

      instancedTrunks.instanceMatrix.needsUpdate = true;
      instancedCanopies.instanceMatrix.needsUpdate = true;
      if (instancedCanopies.instanceColor) instancedCanopies.instanceColor.needsUpdate = true;

      this.woodlandGroup.add(instancedTrunks);
      this.woodlandGroup.add(instancedCanopies);
    }

    // Apply global building height scale
    this.buildingGroup.scale.y = this.buildingHeightScale;

    if (onProgress) onProgress("Photorealistic 3D Scene ready!", 100);
  }

  // Simple, fast buffer geometry merger for compound tree canopies
  mergeBufferGeometries(geos) {
    let totalVerts = 0;
    geos.forEach(g => { totalVerts += g.attributes.position.count; });

    const posArray = new Float32Array(totalVerts * 3);
    const normArray = new Float32Array(totalVerts * 3);
    let offset = 0;

    geos.forEach(g => {
      const pos = g.attributes.position.array;
      const norm = g.attributes.normal.array;
      posArray.set(pos, offset * 3);
      normArray.set(norm, offset * 3);
      offset += g.attributes.position.count;
    });

    const merged = new THREE.BufferGeometry();
    merged.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
    merged.setAttribute('normal', new THREE.BufferAttribute(normArray, 3));
    return merged;
  }

  setBuildingHeightScale(val) {
    this.buildingHeightScale = parseFloat(val);
    if (this.buildingGroup) {
      this.buildingGroup.scale.y = this.buildingHeightScale;
    }
  }

  setClassVisibility(className, isVisible) {
    const cls = className.toLowerCase();
    if (cls === 'building' && this.buildingGroup) this.buildingGroup.visible = isVisible;
    if (cls === 'woodland' && this.woodlandGroup) this.woodlandGroup.visible = isVisible;
    if (cls === 'water' && this.waterGroup) this.waterGroup.visible = isVisible;
    if (cls === 'road' && this.roadGroup) this.roadGroup.visible = isVisible;
  }

  toggleWireframe() {
    this.isWireframe = !this.isWireframe;
    if (this.scene) {
      this.scene.traverse(obj => {
        if (obj.isMesh && obj.material) {
          if (Array.isArray(obj.material)) {
            obj.material.forEach(m => m.wireframe = this.isWireframe);
          } else {
            obj.material.wireframe = this.isWireframe;
          }
        }
      });
    }
    return this.isWireframe;
  }

  resetCamera() {
    if (!this.camera || !this.controls) return;
    this.camera.position.set(0, 380, 520);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  flyTo(targetX, targetZ, distance = 140) {
    if (!this.camera || !this.controls) return;
    const startPos = this.camera.position.clone();
    const endPos = new THREE.Vector3(targetX + 60, distance, targetZ + 80);
    const startTarget = this.controls.target.clone();
    const endTarget = new THREE.Vector3(targetX, 0, targetZ);

    let progress = 0;
    const duration = 50;
    const animateFly = () => {
      progress++;
      const t = progress / duration;
      const ease = t * (2 - t);
      this.camera.position.lerpVectors(startPos, endPos, ease);
      this.controls.target.lerpVectors(startTarget, endTarget, ease);
      this.controls.update();

      if (progress < duration) {
        requestAnimationFrame(animateFly);
      }
    };
    animateFly();
  }

  onCanvasClick(event) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    this.raycaster.setFromCamera(this.mouse, this.camera);
    const intersects = this.raycaster.intersectObjects(this.buildingGroup.children);

    if (intersects.length > 0) {
      const hit = intersects[0].object;
      if (hit.userData && hit.userData.feature) {
        this.selectedBuilding = hit;
        const pt = intersects[0].point;
        this.flyTo(pt.x, pt.z, 160);

        if (typeof this.options.onBuildingSelect === 'function') {
          this.options.onBuildingSelect(hit.userData);
        }
      }
    }
  }

  clearGroup(group) {
    if (!group) return;
    while (group.children.length > 0) {
      const obj = group.children[0];
      group.remove(obj);
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {
        if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
        else obj.material.dispose();
      }
    }
  }
}

// Export for browser global scope
window.MapRot3DViewer = MapRot3DViewer;
