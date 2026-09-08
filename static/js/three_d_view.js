/**
 * MapRot Photorealistic 3D Terrain & Structure Viewer
 * 
 * Powered directly by the standard MapRot GeoJSON FeatureCollection export
 * and source satellite rasters.
 * 
 * Features:
 *  - High-res draped terrain mesh
 *  - Extruded PBR buildings with height estimation from area_map_units
 *  - Instanced 3D low-poly tree scattering for woodland polygons (GPU single draw call)
 *  - Animated water surface with specular reflections & wave displacement
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
      maxTrees: 2500,
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
    this.isAnimating = false;
    this.clock = new THREE.Clock();
    this.waterMeshes = [];
    this.currentData = null;
    this.buildingHeightScale = this.options.buildingBaseScale;
    this.timeOfDay = this.options.timeOfDay;
    this.isWireframe = false;
    this.selectedBuilding = null;

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
    this.renderer.toneMappingExposure = 1.15;
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

    this.hemiLight = new THREE.HemisphereLight(0x93c5fd, 0x1e293b, 0.6);
    this.hemiLight.position.set(0, 500, 0);
    this.scene.add(this.hemiLight);

    this.sunLight = new THREE.DirectionalLight(0xfffaed, 1.4);
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

    // Apply default time-of-day lighting
    this.setTimeOfDay(this.timeOfDay);

    // Events
    window.addEventListener('resize', () => this.onResize());
    this.renderer.domElement.addEventListener('click', (e) => this.onCanvasClick(e));

    this.isInitialized = true;
    this.animate();
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
        mesh.position.y = -0.4 + Math.sin(elapsed * 2.0 + mesh.id) * 0.25;
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

    // Calculate solar angle
    // Dawn: 6, Noon: 12, Sunset: 18, Night: 22-5
    const sunAngle = ((this.timeOfDay - 6) / 12) * Math.PI;
    const isDay = this.timeOfDay >= 5.5 && this.timeOfDay <= 19.5;

    const sunDist = 650;
    const x = Math.cos(sunAngle) * sunDist;
    const y = Math.max(30, Math.sin(sunAngle) * sunDist);
    const z = Math.sin(sunAngle * 0.8) * 320;

    this.sunLight.position.set(x, y, z);

    if (isDay) {
      // Dawn / Sunset golden hour vs crisp noon
      const elevation = Math.sin(sunAngle); // 0 at dawn/dusk, 1 at noon
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
        // Broad daylight / Noon
        this.sunLight.color.setHex(0xfffaed);
        this.sunLight.intensity = 1.35;
        this.hemiLight.color.setHex(0xa5f3fc);
        this.hemiLight.groundColor.setHex(0x1e293b);
        this.ambientLight.intensity = 0.65;
        this.scene.background.setHex(0x0c1322);
        this.scene.fog.color.setHex(0x0c1322);
      }
    } else {
      // Night mode (subtle moonlight)
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

    // Helper: Map GeoJSON coordinate [gx, gy] to centered 3D terrain space [x, z]
    const mapCoord = (gx, gy) => {
      const u = (gx - minX) / bboxW; // 0 to 1
      const v = (gy - minY) / bboxH; // 0 to 1
      const x = (u - 0.5) * planeW;
      const z = (v - 0.5) * planeH;
      return { x, z };
    };

    if (onProgress) onProgress("Draping high-resolution satellite terrain...", 45);

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

    if (onProgress) onProgress(`Extruding ${features.length} vector features...`, 70);

    // Materials Palette (PBR)
    const buildingWallMat = new THREE.MeshStandardMaterial({
      color: 0x94a3b8, // Concrete / modern facade
      roughness: 0.65,
      metalness: 0.15,
      wireframe: this.isWireframe,
    });
    const buildingRoofMat = new THREE.MeshStandardMaterial({
      color: 0xdc2626, // Crimson terracotta / urban architectural red
      roughness: 0.45,
      metalness: 0.15,
      wireframe: this.isWireframe,
    });
    const buildingMaterials = [buildingWallMat, buildingRoofMat];

    const waterMat = new THREE.MeshStandardMaterial({
      color: 0x0284c7, // Azure water
      roughness: 0.08,
      metalness: 0.85,
      transparent: true,
      opacity: 0.82,
      wireframe: this.isWireframe,
    });

    const roadMat = new THREE.MeshStandardMaterial({
      color: 0x334155, // Asphalt dark slate
      roughness: 0.88,
      metalness: 0.1,
      wireframe: this.isWireframe,
    });

    const treePositions = [];

    // 4. Ingest and Render Each Class Feature
    features.forEach((feature) => {
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

        // Add interior holes if any
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
          // Estimate height from area_map_units or footprint bounds
          const area = feature.properties.area_map_units || 100;
          const estHeight = Math.min(48, Math.max(9, Math.sqrt(area) * 0.75));

          const extrudeSettings = {
            depth: estHeight,
            bevelEnabled: true,
            bevelSegments: 1,
            steps: 1,
            bevelSize: 0.35,
            bevelThickness: 0.35,
          };

          try {
            const extrudeGeo = new THREE.ExtrudeGeometry(shape, extrudeSettings);
            extrudeGeo.rotateX(Math.PI / 2);

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
          } catch (e) {}

        } else if (cls === 'water') {
          // Water surface plane with slight negative offset
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
          // Thin flat road ribbon
          try {
            const roadGeo = new THREE.ShapeGeometry(shape);
            roadGeo.rotateX(Math.PI / 2);
            const rMesh = new THREE.Mesh(roadGeo, roadMat);
            rMesh.position.y = 0.15;
            rMesh.receiveShadow = true;
            this.roadGroup.add(rMesh);
          } catch (e) {}

        } else if (cls === 'woodland') {
          // Sample interior points for Instanced Tree scattering
          const area = feature.properties.area_map_units || 200;
          const numTrees = Math.min(60, Math.max(3, Math.floor(Math.sqrt(area) * 0.8 * this.options.treeDensity)));
          
          let cx = 0, cz = 0;
          exteriorRing.forEach(p => {
            const { x, z } = mapCoord(p[0], p[1]);
            cx += x; cz += z;
          });
          cx /= exteriorRing.length;
          cz /= exteriorRing.length;

          for (let t = 0; t < numTrees; t++) {
            if (treePositions.length >= this.options.maxTrees) break;
            const rOffset = (Math.random() - 0.5) * Math.sqrt(area) * 0.45;
            const thOffset = Math.random() * Math.PI * 2;
            const tx = cx + Math.cos(thOffset) * rOffset;
            const tz = cz + Math.sin(thOffset) * rOffset;
            const tScale = 0.75 + Math.random() * 0.55;
            treePositions.push({ x: tx, z: tz, scale: tScale });
          }
        }
      });
    });

    // 5. Instanced Tree Meshes for Woodland (Single GPU Draw Call)
    if (treePositions.length > 0) {
      if (onProgress) onProgress(`Instancing ${treePositions.length} woodland trees...`, 90);

      const foliageGeo = new THREE.ConeGeometry(3.5, 7, 6);
      foliageGeo.translate(0, 5.5, 0);

      const treeMat = new THREE.MeshStandardMaterial({
        color: 0x15803d, // Forest emerald green
        roughness: 0.85,
        metalness: 0.05,
        wireframe: this.isWireframe,
      });

      const instancedFoliage = new THREE.InstancedMesh(foliageGeo, treeMat, treePositions.length);
      instancedFoliage.castShadow = true;
      instancedFoliage.receiveShadow = true;

      const dummy = new THREE.Object3D();
      treePositions.forEach((tp, i) => {
        dummy.position.set(tp.x, 0, tp.z);
        dummy.scale.set(tp.scale, tp.scale, tp.scale);
        dummy.rotation.y = Math.random() * Math.PI * 2;
        dummy.updateMatrix();
        instancedFoliage.setMatrixAt(i, dummy.matrix);
      });
      instancedFoliage.instanceMatrix.needsUpdate = true;

      this.woodlandGroup.add(instancedFoliage);
    }

    // Apply global building height scale
    this.buildingGroup.scale.y = this.buildingHeightScale;

    if (onProgress) onProgress("3D Scene ready!", 100);
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
