/* ══════════════════════════════════════
   Gesture Controlled Particles — script.js
   ══════════════════════════════════════ */

import * as THREE from 'three';
import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

// ══════════════════════════════════════
// CONFIGURATION
// ══════════════════════════════════════

const PARTICLE_COUNT = 5000;

let handLandmarker;
let webcamRunning = false;
let lastVideoTime = -1;

// ── Shared State ──
const state = {
    shapeType: 'sphere',
    scale: 1,
    colorHue: 0.5,
    targetPositions: new Float32Array(PARTICLE_COUNT * 3),
    currentPositions: new Float32Array(PARTICLE_COUNT * 3)
};

// ══════════════════════════════════════
// THREE.JS SETUP
// ══════════════════════════════════════

const scene    = new THREE.Scene();
const camera   = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 5;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// ── Particle Geometry ──
const geometry  = new THREE.BufferGeometry();
const positions = new Float32Array(PARTICLE_COUNT * 3);
const colors    = new Float32Array(PARTICLE_COUNT * 3);

// Scatter particles randomly at startup
for (let i = 0; i < PARTICLE_COUNT; i++) {
    positions[i * 3]     = (Math.random() - 0.5) * 10;
    positions[i * 3 + 1] = (Math.random() - 0.5) * 10;
    positions[i * 3 + 2] = (Math.random() - 0.5) * 10;
}

geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
geometry.setAttribute('color',    new THREE.BufferAttribute(colors, 3));

// ── Particle Material ──
const material = new THREE.PointsMaterial({
    size: 0.05,
    vertexColors: true,
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false
});

const particleSystem = new THREE.Points(geometry, material);
scene.add(particleSystem);

// ══════════════════════════════════════
// SHAPE GENERATORS
// ══════════════════════════════════════

/**
 * Populate state.targetPositions with coordinates for the given shape.
 * @param {string} type - 'sphere' | 'heart' | 'saturn' | 'flower'
 */
function updateTargetShape(type) {
    const pos = state.targetPositions;

    for (let i = 0; i < PARTICLE_COUNT; i++) {
        const i3 = i * 3;

        if (type === 'heart') {
            // Parametric heart curve
            const t = Math.random() * Math.PI * 2;
            const x = 16 * Math.pow(Math.sin(t), 3);
            const y = 13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t);
            pos[i3]     = x * 0.1;
            pos[i3 + 1] = y * 0.1;
            pos[i3 + 2] = (Math.random() - 0.5) * 0.5;

        } else if (type === 'saturn') {
            if (i < PARTICLE_COUNT * 0.6) {
                // Planet body — Fibonacci sphere distribution
                const phi   = Math.acos(-1 + (2 * i) / (PARTICLE_COUNT * 0.6));
                const theta = Math.sqrt(PARTICLE_COUNT * 0.6 * Math.PI) * phi;
                pos[i3]     = Math.cos(theta) * Math.sin(phi) * 1.5;
                pos[i3 + 1] = Math.sin(theta) * Math.sin(phi) * 1.5;
                pos[i3 + 2] = Math.cos(phi) * 1.5;
            } else {
                // Rings
                const angle  = Math.random() * Math.PI * 2;
                const radius = 2.2 + Math.random() * 0.8;
                pos[i3]     = Math.cos(angle) * radius;
                pos[i3 + 1] = Math.sin(angle) * radius * 0.4; // Tilted
                pos[i3 + 2] = Math.sin(angle) * radius * 0.2;
            }

        } else if (type === 'flower') {
            // Rose curve (5 petals)
            const t = Math.random() * Math.PI * 2;
            const r = 2 * Math.cos(5 * t);
            pos[i3]     = r * Math.cos(t);
            pos[i3 + 1] = r * Math.sin(t);
            pos[i3 + 2] = (Math.random() - 0.5) * 0.5;

        } else {
            // Default: Sphere — Fibonacci distribution
            const phi   = Math.acos(-1 + (2 * i) / PARTICLE_COUNT);
            const theta = Math.sqrt(PARTICLE_COUNT * Math.PI) * phi;
            pos[i3]     = Math.cos(theta) * Math.sin(phi) * 2;
            pos[i3 + 1] = Math.sin(theta) * Math.sin(phi) * 2;
            pos[i3 + 2] = Math.cos(phi) * 2;
        }
    }
}

// ══════════════════════════════════════
// MEDIAPIPE — HAND TRACKING
// ══════════════════════════════════════

/**
 * Load the MediaPipe HandLandmarker model and start the webcam.
 */
async function initHandTracking() {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );

    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 1
    });

    startWebcam();
}

/**
 * Request webcam access and pipe it into the <video> element.
 */
async function startWebcam() {
    const video  = document.getElementById("webcam");
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.addEventListener("loadeddata", () => {
        webcamRunning = true;
    });
}

// ══════════════════════════════════════
// GESTURE → ACTION MAPPING
// ══════════════════════════════════════

/**
 * Interpret hand landmark data and update the scene state.
 * @param {object} results - MediaPipe detection results.
 */
function processHands(results) {
    if (!results.landmarks || results.landmarks.length === 0) return;

    const hand = results.landmarks[0];

    // 1. Scale via Pinch distance (Thumb tip [4] ↔ Index tip [8])
    const pinchDist = Math.hypot(hand[4].x - hand[8].x, hand[4].y - hand[8].y);
    state.scale = THREE.MathUtils.lerp(state.scale, pinchDist * 10, 0.1);

    // 2. Color via horizontal hand position (wrist X coordinate)
    state.colorHue = hand[0].x;

    // 3. Shape via extended finger count
    //    A finger is "up" when its tip Y < the PIP joint Y (screen coords, Y grows downward)
    let count = 0;
    if (hand[8].y  < hand[6].y)  count++; // Index finger
    if (hand[12].y < hand[10].y) count++; // Middle finger
    if (hand[16].y < hand[14].y) count++; // Ring finger
    if (hand[20].y < hand[18].y) count++; // Pinky finger

    const lastShape = state.shapeType;

    if      (count <= 1) state.shapeType = 'sphere';
    else if (count === 2) state.shapeType = 'heart';
    else if (count === 3) state.shapeType = 'saturn';
    else                  state.shapeType = 'flower';

    // Regenerate target only on shape change
    if (lastShape !== state.shapeType) updateTargetShape(state.shapeType);

    // 4. Rotation: continuous Y spin + tilt based on hand height
    particleSystem.rotation.y += 0.01;
    particleSystem.rotation.x  = (hand[0].y - 0.5) * 2;
}

// ══════════════════════════════════════
// ANIMATION LOOP
// ══════════════════════════════════════

function animate() {
    // ── Run hand detection on each new video frame ──
    if (webcamRunning) {
        const video = document.getElementById("webcam");
        if (video.currentTime !== lastVideoTime) {
            const results = handLandmarker.detectForVideo(video, performance.now());
            processHands(results);
            lastVideoTime = video.currentTime;
        }
    }

    // ── Lerp particles toward target shape ──
    const posArray   = geometry.attributes.position.array;
    const colorArray = geometry.attributes.color.array;
    const colorObj   = new THREE.Color().setHSL(state.colorHue, 0.8, 0.5);

    for (let i = 0; i < PARTICLE_COUNT; i++) {
        const i3 = i * 3;

        // Smooth position interpolation
        posArray[i3]     += (state.targetPositions[i3]     * state.scale - posArray[i3])     * 0.1;
        posArray[i3 + 1] += (state.targetPositions[i3 + 1] * state.scale - posArray[i3 + 1]) * 0.1;
        posArray[i3 + 2] += (state.targetPositions[i3 + 2] * state.scale - posArray[i3 + 2]) * 0.1;

        // Update vertex colors
        colorArray[i3]     = colorObj.r;
        colorArray[i3 + 1] = colorObj.g;
        colorArray[i3 + 2] = colorObj.b;
    }

    geometry.attributes.position.needsUpdate = true;
    geometry.attributes.color.needsUpdate    = true;

    renderer.render(scene, camera);
    requestAnimationFrame(animate);
}

// ══════════════════════════════════════
// RESPONSIVE RESIZE
// ══════════════════════════════════════

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// ══════════════════════════════════════
// BOOT
// ══════════════════════════════════════

updateTargetShape('sphere');
initHandTracking();
animate();
