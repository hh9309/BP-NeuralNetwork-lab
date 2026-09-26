import React, { useRef, useEffect, useState, useMemo } from "react";
import {
  Sparkles,
  ArrowRight,
  ArrowLeft,
  Info,
  Layers,
  Zap,
  Activity,
  Play,
  SkipForward,
  Cpu,
  HelpCircle,
  Maximize2,
  Minimize2,
} from "lucide-react";
import { NeuralNetwork } from "../lib/neural-engine";
import { StepTraceFrame, TrainingSample } from "../types";
import { BlockMath, InlineMath } from "../lib/katex-helper";

interface TopologyPulseCanvasProps {
  nn: NeuralNetwork;
  currentSample?: TrainingSample;
  isPlaying: boolean;
  stepTrace?: StepTraceFrame[];
  currentStepIndex: number;
  onSetStepIndex: (idx: number) => void;
  onNextStep: () => void;
  epoch: number;
  onApplyTopology?: (layers: number[], acts: ("sigmoid" | "tanh" | "relu" | "leaky_relu" | "linear" | "softmax")[]) => void;
}

interface Particle {
  id: number;
  direction: "forward" | "backward";
  fromLayer: number;
  fromNode: number;
  toLayer: number;
  toNode: number;
  progress: number; // 0 to 1
  speed: number;
  color: string;
  size: number;
}

export const TopologyPulseCanvas: React.FC<TopologyPulseCanvasProps> = ({
  nn,
  currentSample,
  isPlaying,
  stepTrace,
  currentStepIndex,
  onSetStepIndex,
  onNextStep,
  epoch,
  onApplyTopology,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [showNeuronAnatomy, setShowNeuronAnatomy] = useState(false);
  const [hoveredNode, setHoveredNode] = useState<{ layer: number; index: number; x: number; y: number } | null>(null);
  const [hoveredEdge, setHoveredEdge] = useState<{
    fromLayer: number;
    fromIndex: number;
    toLayer: number;
    toIndex: number;
    weight: number;
    x: number;
    y: number;
  } | null>(null);

  const particlesRef = useRef<Particle[]>([]);
  const nextParticleId = useRef(1);

  // Compute forward and backward states for the current sample
  const networkState = useMemo(() => {
    if (!currentSample || !currentSample.input || !currentSample.target) return null;
    const fwd = nn.forward(currentSample.input);
    const bwd = nn.backwardSingle(currentSample.input, currentSample.target);
    return { fwd, bwd };
  }, [nn, currentSample, epoch]);

  // Current topology string representation
  const topologyStr = nn.layerSizes.join(" → ");
  const is241 = nn.layerSizes.length === 3 && nn.layerSizes[0] === 2 && nn.layerSizes[1] === 4 && nn.layerSizes[2] === 1;
  const is2841 = nn.layerSizes.length === 4 && nn.layerSizes[0] === 2 && nn.layerSizes[1] === 8 && nn.layerSizes[2] === 4 && nn.layerSizes[3] === 1;
  const is261 = nn.layerSizes.length === 3 && nn.layerSizes[0] === 2 && nn.layerSizes[1] === 6 && nn.layerSizes[2] === 1;

  // Main Canvas Rendering Loop with Retina / High-DPI support
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animId: number;

    const render = () => {
      const dpr = window.devicePixelRatio || 1;
      const displayWidth = canvas.clientWidth || 800;
      const displayHeight = canvas.clientHeight || 380;

      if (canvas.width !== displayWidth * dpr || canvas.height !== displayHeight * dpr) {
        canvas.width = displayWidth * dpr;
        canvas.height = displayHeight * dpr;
      }

      ctx.save();
      ctx.scale(dpr, dpr);

      const width = displayWidth;
      const height = displayHeight;

      // 1. Clear background (Deep sleek dark theme)
      ctx.fillStyle = "#090d16";
      ctx.fillRect(0, 0, width, height);

      // Draw subtle mathematical coordinate grid
      ctx.strokeStyle = "rgba(30, 41, 59, 0.4)";
      ctx.lineWidth = 1;
      const gridSize = 32;
      for (let x = 0; x < width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = 0; y < height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      // 2. Calculate node positions
      const layerSizes = nn.layerSizes;
      const numLayers = layerSizes.length;
      const paddingX = 85;
      const paddingTop = 75;
      const paddingBottom = 45;
      const availableW = width - 2 * paddingX;
      const availableH = height - paddingTop - paddingBottom;

      const layerX: number[] = [];
      for (let l = 0; l < numLayers; l++) {
        layerX.push(paddingX + (l / (numLayers - 1 || 1)) * availableW);
      }

      // Draw vertical column lanes (Standard Hierarchical Layer Lanes)
      for (let l = 0; l < numLayers; l++) {
        const x = layerX[l];
        const colW = Math.min(100, (availableW / (numLayers || 1)) * 0.7);

        // Column background tint
        ctx.fillStyle = l === 0
          ? "rgba(16, 185, 129, 0.03)"
          : l === numLayers - 1
          ? "rgba(245, 158, 11, 0.03)"
          : "rgba(99, 102, 241, 0.03)";
        ctx.fillRect(x - colW / 2, paddingTop - 25, colW, availableH + 45);

        // Vertical lane guide lines
        ctx.strokeStyle = l === 0
          ? "rgba(16, 185, 129, 0.15)"
          : l === numLayers - 1
          ? "rgba(245, 158, 11, 0.15)"
          : "rgba(99, 102, 241, 0.15)";
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(x, paddingTop - 20);
        ctx.lineTo(x, height - paddingBottom + 15);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      const nodePositions: { x: number; y: number }[][] = [];
      for (let l = 0; l < numLayers; l++) {
        const count = layerSizes[l];
        const layerPositions: { x: number; y: number }[] = [];
        const x = layerX[l];

        // For large layers (like MNIST 64 inputs), display compact capped layout
        const maxDisplay = Math.min(count, 8);
        const stepY = availableH / (maxDisplay + 1);

        for (let i = 0; i < count; i++) {
          let y = paddingTop + (i + 1) * (availableH / (count + 1));
          if (count > 8) {
            // Compress visually
            if (i < 4) {
              y = paddingTop + (i + 1) * stepY;
            } else if (i === 4 && count > 8) {
              y = paddingTop + 4.5 * stepY;
            } else {
              y = paddingTop + (maxDisplay - (count - 1 - i)) * stepY;
            }
          }
          layerPositions.push({ x, y });
        }
        nodePositions.push(layerPositions);
      }

      // 3. Draw Synaptic Weights / Connections
      for (let l = 0; l < numLayers - 1; l++) {
        const W = nn.weights[l];
        if (!W) continue;
        const fromCount = layerSizes[l];
        const toCount = layerSizes[l + 1];

        for (let j = 0; j < toCount; j++) {
          for (let i = 0; i < fromCount; i++) {
            // If many nodes, only draw a subset to prevent clutter
            if (fromCount > 10 && i % 4 !== 0 && i !== fromCount - 1) continue;

            const pFrom = nodePositions[l][i];
            const pTo = nodePositions[l + 1][j];
            if (!pFrom || !pTo) continue;

            const weight = W[j]?.[i] ?? 0;
            const absW = Math.min(6, Math.abs(weight));
            const isPositive = weight >= 0;

            ctx.beginPath();
            ctx.moveTo(pFrom.x, pFrom.y);

            // Bezier curve for standard hierarchical neural wiring
            const cpx = (pFrom.x + pTo.x) / 2;
            ctx.bezierCurveTo(cpx, pFrom.y, cpx, pTo.y, pTo.x, pTo.y);

            // Color coding based on weight sign and magnitude
            if (isPositive) {
              ctx.strokeStyle = `rgba(99, 102, 241, ${0.18 + Math.min(0.7, absW * 0.22)})`; // Indigo/Cyan
            } else {
              ctx.strokeStyle = `rgba(244, 63, 94, ${0.18 + Math.min(0.7, absW * 0.22)})`; // Rose
            }
            ctx.lineWidth = Math.max(0.8, Math.min(4.5, 0.9 + absW * 0.8));
            ctx.stroke();
          }
        }
      }

      // 4. Update and Draw Dynamic Pulse Particles
      if (Math.random() < (isPlaying ? 0.5 : 0.18)) {
        // Forward particle (Sky blue photon)
        const l = Math.floor(Math.random() * (numLayers - 1));
        const fCount = layerSizes[l];
        const tCount = layerSizes[l + 1];
        const fromIdx = Math.floor(Math.random() * fCount);
        const toIdx = Math.floor(Math.random() * tCount);

        particlesRef.current.push({
          id: nextParticleId.current++,
          direction: "forward",
          fromLayer: l,
          fromNode: fromIdx,
          toLayer: l + 1,
          toNode: toIdx,
          progress: 0,
          speed: 0.015 + Math.random() * 0.02,
          color: "#38bdf8", // Sky blue forward photon
          size: 3.5 + Math.random() * 1.5,
        });

        // Backward error particle (Amber/Orange delta gradient pulse)
        if (Math.random() < 0.85) {
          particlesRef.current.push({
            id: nextParticleId.current++,
            direction: "backward",
            fromLayer: l + 1,
            fromNode: toIdx,
            toLayer: l,
            toNode: fromIdx,
            progress: 0,
            speed: 0.018 + Math.random() * 0.02,
            color: "#f59e0b", // Amber backward delta pulse
            size: 3.5 + Math.random() * 1.5,
          });
        }
      }

      // Render & prune particles
      const remainingParticles: Particle[] = [];
      for (const p of particlesRef.current) {
        p.progress += p.speed;
        if (p.progress >= 1) continue;

        const pFrom = nodePositions[p.fromLayer]?.[p.fromNode];
        const pTo = nodePositions[p.toLayer]?.[p.toNode];

        if (pFrom && pTo) {
          const t = p.progress;
          // Interpolate along bezier curve
          const cpx = (pFrom.x + pTo.x) / 2;
          const u = 1 - t;
          const curX = u * u * u * pFrom.x + 3 * u * u * t * cpx + 3 * u * t * t * cpx + t * t * t * pTo.x;
          const curY = u * u * u * pFrom.y + 3 * u * u * t * pFrom.y + 3 * u * t * t * pTo.y + t * t * t * pTo.y;

          // Glowing particle dot
          ctx.beginPath();
          ctx.arc(curX, curY, p.size, 0, Math.PI * 2);
          ctx.fillStyle = p.color;
          ctx.shadowColor = p.color;
          ctx.shadowBlur = 8;
          ctx.fill();
          ctx.shadowBlur = 0; // reset
        }

        remainingParticles.push(p);
      }
      particlesRef.current = remainingParticles;

      // 5. Draw Layer Headers & Clear Prominent Badges (输入层 / 隐藏层 / 输出层)
      for (let l = 0; l < numLayers; l++) {
        const x = layerX[l];
        let layerMainTitle = `隐藏层 L${l}`;
        let layerType = "hidden";
        if (l === 0) {
          layerMainTitle = "输入层 L₀";
          layerType = "input";
        } else if (l === numLayers - 1) {
          layerMainTitle = `输出层 L${l}`;
          layerType = "output";
        }

        const badgeW = 100;
        const badgeH = 26;
        const badgeY = 16;

        // Draw clear high-contrast layer badge background
        ctx.save();
        ctx.beginPath();
        const r = 6;
        const bx = x - badgeW / 2;
        const by = badgeY;
        ctx.moveTo(bx + r, by);
        ctx.lineTo(bx + badgeW - r, by);
        ctx.quadraticCurveTo(bx + badgeW, by, bx + badgeW, by + r);
        ctx.lineTo(bx + badgeW, by + badgeH - r);
        ctx.quadraticCurveTo(bx + badgeW, by + badgeH, bx + badgeW - r, by + badgeH);
        ctx.lineTo(bx + r, by + badgeH);
        ctx.quadraticCurveTo(bx, by + badgeH, bx, by + badgeH - r);
        ctx.lineTo(bx, by + r);
        ctx.quadraticCurveTo(bx, by, bx + r, by);
        ctx.closePath();

        if (layerType === "input") {
          ctx.fillStyle = "rgba(16, 185, 129, 0.9)"; // Emerald solid
          ctx.strokeStyle = "#34d399";
        } else if (layerType === "output") {
          ctx.fillStyle = "rgba(245, 158, 11, 0.9)"; // Amber solid
          ctx.strokeStyle = "#fbbf24";
        } else {
          ctx.fillStyle = "rgba(99, 102, 241, 0.9)"; // Indigo solid
          ctx.strokeStyle = "#818cf8";
        }
        ctx.lineWidth = 1.5;
        ctx.fill();
        ctx.stroke();

        // Layer Main Title (Pure white, bold, high contrast)
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 12px 'Noto Sans SC', sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(layerMainTitle, x, badgeY + badgeH / 2);

        // Subtitle (Neuron count & activation function info)
        const actName = l === 0 ? "特征输入 x" : `f(z)=${nn.activations[l - 1] || "σ"}`;
        ctx.fillStyle = layerType === "input" ? "#a7f3d0" : layerType === "output" ? "#fde68a" : "#c7d2fe";
        ctx.font = "10px monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText(`${layerSizes[l]} 个神经元 | ${actName}`, x, badgeY + badgeH + 4);
        ctx.restore();
      }

      // 6. Draw Neurons with Standard Artificial Neuron Iconography (McCulloch-Pitts Model Σ & σ)
      for (let l = 0; l < numLayers; l++) {
        const count = layerSizes[l];
        for (let i = 0; i < count; i++) {
          const pos = nodePositions[l][i];
          if (!pos) continue;

          // For collapsed visual nodes in large layers
          if (count > 8 && i === 4) {
            ctx.fillStyle = "#94a3b8";
            ctx.font = "bold 18px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("⋮", pos.x, pos.y);
            continue;
          }

          // Get activation and gradient value
          let activationVal = 0.5;
          let deltaVal = 0;
          if (networkState) {
            if (l === 0) {
              activationVal = networkState.fwd.layers[0]?.a[i] ?? 0.5;
            } else {
              activationVal = networkState.fwd.layers[l]?.a[i] ?? 0.5;
              deltaVal = networkState.bwd.layerGradients[l - 1]?.delta[i] ?? 0;
            }
          }

          const radius = l === 0 || l === numLayers - 1 ? 20 : 18;

          // Outer Glow based on activation
          const glowAlpha = Math.max(0.2, Math.min(0.95, Math.abs(activationVal)));
          ctx.beginPath();
          ctx.arc(pos.x, pos.y, radius + 4, 0, Math.PI * 2);
          ctx.fillStyle = l === 0
            ? `rgba(16, 185, 129, ${glowAlpha * 0.35})`
            : l === numLayers - 1
            ? `rgba(245, 158, 11, ${glowAlpha * 0.4})`
            : `rgba(99, 102, 241, ${glowAlpha * 0.4})`;
          ctx.fill();

          // Node Body (Dark sleek container)
          ctx.beginPath();
          ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
          ctx.fillStyle = "#0f172a";
          ctx.fill();

          // Node Inner Fill Meter based on activation level
          const fillRatio = Math.max(0, Math.min(1, activationVal));
          ctx.beginPath();
          ctx.arc(pos.x, pos.y, (radius - 2) * Math.sqrt(fillRatio), 0, Math.PI * 2);
          ctx.fillStyle = l === 0
            ? "rgba(16, 185, 129, 0.45)"
            : l === numLayers - 1
            ? "rgba(245, 158, 11, 0.45)"
            : "rgba(99, 102, 241, 0.45)";
          ctx.fill();

          // Node Border Ring
          ctx.beginPath();
          ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
          ctx.lineWidth = 2;
          if (l === 0) {
            ctx.strokeStyle = "#10b981"; // Emerald
          } else if (l === numLayers - 1) {
            ctx.strokeStyle = "#f59e0b"; // Amber
          } else {
            ctx.strokeStyle = "#818cf8"; // Indigo
          }
          ctx.stroke();

          // ── ARTIFICIAL NEURON ICONOGRAPHY (McCulloch-Pitts Dual Chamber: Σ & σ) ──
          if (l === 0) {
            // Input Neuron Icon: x_i input symbol with index
            ctx.fillStyle = "#34d399";
            ctx.font = "bold 10px monospace";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(`x${i + 1}`, pos.x, pos.y - 3);

            // Input value below
            ctx.fillStyle = "#ffffff";
            ctx.font = "8px monospace";
            ctx.fillText(`${activationVal.toFixed(2)}`, pos.x, pos.y + 7);
          } else {
            // Hidden / Output Neuron: Classic Artificial Neuron Icon
            // 1. Draw central vertical divider line for standard Σ | σ representation
            ctx.strokeStyle = "rgba(148, 163, 184, 0.4)";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(pos.x, pos.y - radius + 3);
            ctx.lineTo(pos.x, pos.y + radius - 3);
            ctx.stroke();

            // 2. Left side: Summation Σ icon (Weighted Linear Aggregator)
            ctx.fillStyle = "#93c5fd";
            ctx.font = "bold 9px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("Σ", pos.x - radius * 0.42, pos.y - 2);

            // 3. Right side: Activation σ / f curve icon (Non-linear Activation)
            ctx.fillStyle = l === numLayers - 1 ? "#fcd34d" : "#c4b5fd";
            ctx.font = "bold 9px sans-serif";
            ctx.fillText(l === numLayers - 1 ? "f" : "σ", pos.x + radius * 0.42, pos.y - 2);

            // 4. Value label at bottom of neuron
            ctx.fillStyle = "#ffffff";
            ctx.font = "bold 8px monospace";
            ctx.fillText(`${activationVal.toFixed(2)}`, pos.x, pos.y + radius * 0.45);
          }
        }
      }

      ctx.restore();

      animId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animId);
    };
  }, [nn, networkState, isPlaying]);

  // Handle Mouse Move for Node Hover Tooltip
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseX = ((e.clientX - rect.left) / rect.width) * canvas.clientWidth;
    const mouseY = ((e.clientY - rect.top) / rect.height) * canvas.clientHeight;

    const layerSizes = nn.layerSizes;
    const numLayers = layerSizes.length;
    const paddingX = 85;
    const paddingTop = 75;
    const paddingBottom = 45;
    const availableW = canvas.clientWidth - 2 * paddingX;
    const availableH = canvas.clientHeight - paddingTop - paddingBottom;

    let foundNode = null;
    for (let l = 0; l < numLayers; l++) {
      const count = layerSizes[l];
      const x = paddingX + (l / (numLayers - 1 || 1)) * availableW;
      for (let i = 0; i < count; i++) {
        let y = paddingTop + (i + 1) * (availableH / (count + 1));
        const dist = Math.hypot(mouseX - x, mouseY - y);
        if (dist <= 22) {
          foundNode = { layer: l, index: i, x: e.clientX - rect.left, y: e.clientY - rect.top };
          break;
        }
      }
      if (foundNode) break;
    }
    setHoveredNode(foundNode);
  };

  const activeFrame = stepTrace?.[currentStepIndex];

  return (
    <div ref={containerRef} className="flex flex-col h-full bg-white rounded-lg border border-slate-200 overflow-hidden shadow-sm">
      {/* Top Banner with Streamlined Topology Selector, Pulse Legends & MCP Anatomy Toggle */}
      <div className="flex flex-wrap items-center justify-between px-3.5 py-2 bg-slate-50 border-b border-slate-200 gap-2.5 text-xs">
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-2">
            <div className="w-2.5 h-2.5 rounded-full bg-blue-600 animate-pulse" />
            <span className="font-bold text-slate-800 text-xs sm:text-sm">
              多层前馈神经网络层级拓扑与脉冲流 (MLP Standard Hierarchy)
            </span>
          </div>

          {/* Quick Topology Switcher Pills */}
          {onApplyTopology && (
            <div className="flex items-center gap-1 bg-white border border-slate-200 rounded-lg p-0.5 shadow-2xs font-mono text-[11px]">
              <button
                onClick={() => onApplyTopology([2, 4, 1], ["leaky_relu", "linear"])}
                className={`px-2 py-0.5 rounded transition cursor-pointer font-semibold ${
                  is241
                    ? "bg-blue-600 text-white shadow-xs"
                    : "text-slate-600 hover:text-blue-600 hover:bg-blue-50"
                }`}
                title="2-4-1 经典初始拓扑"
              >
                2 → 4 → 1 (初始)
              </button>
              <button
                onClick={() => onApplyTopology([2, 8, 4, 1], ["leaky_relu", "leaky_relu", "linear"])}
                className={`px-2 py-0.5 rounded transition cursor-pointer font-semibold ${
                  is2841
                    ? "bg-blue-600 text-white shadow-xs"
                    : "text-slate-600 hover:text-blue-600 hover:bg-blue-50"
                }`}
                title="2-8-4-1 深度双隐藏层拓扑"
              >
                2 → 8 → 4 → 1
              </button>
              <button
                onClick={() => onApplyTopology([2, 6, 1], ["leaky_relu", "linear"])}
                className={`px-2 py-0.5 rounded transition cursor-pointer font-semibold ${
                  is261
                    ? "bg-blue-600 text-white shadow-xs"
                    : "text-slate-600 hover:text-blue-600 hover:bg-blue-50"
                }`}
                title="2-6-1 单隐藏层拓扑"
              >
                2 → 6 → 1
              </button>
            </div>
          )}

          {!onApplyTopology && (
            <span className="text-slate-700 font-mono text-[11px] bg-white px-2 py-0.5 rounded border border-slate-200 font-bold shadow-xs">
              {topologyStr}
            </span>
          )}
        </div>

        {/* Legend Slices & Action Controls */}
        <div className="flex items-center flex-wrap gap-2.5 text-[11px] font-mono">
          <div className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-sky-400" />
            <span className="text-slate-600">前向 a^(l) →</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-amber-500" />
            <span className="text-slate-600">← 误差 δ^(l)</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded bg-indigo-500" />
            <span className="text-slate-500">W&gt;0</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded bg-rose-500" />
            <span className="text-slate-500">W&lt;0</span>
          </div>

          <button
            onClick={() => setShowNeuronAnatomy(!showNeuronAnatomy)}
            className="flex items-center gap-1 px-2.5 py-1 bg-blue-50 hover:bg-blue-100 text-blue-700 rounded-md border border-blue-200 font-sans text-xs font-semibold transition cursor-pointer shadow-2xs"
          >
            <Cpu className="w-3.5 h-3.5 text-blue-600" />
            <span>人工神经元 (Σ+f)</span>
          </button>
        </div>
      </div>

      {/* Main Canvas Area */}
      <div className="relative flex-1 min-h-[380px] bg-[#090d16] flex items-center justify-center overflow-hidden">
        <canvas
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHoveredNode(null)}
          className="w-full h-full object-contain cursor-crosshair min-h-[380px]"
        />

        {/* Hovered Node Tooltip Card */}
        {hoveredNode && networkState && (
          <div
            className="absolute z-20 pointer-events-none bg-slate-900/95 border border-slate-700 rounded-lg p-3 shadow-2xl text-xs font-mono backdrop-blur text-slate-100"
            style={{
              left: Math.min(window.innerWidth - 260, hoveredNode.x + 15),
              top: Math.max(10, hoveredNode.y - 40),
            }}
          >
            <div className="flex items-center justify-between pb-1.5 mb-1.5 border-b border-slate-700 text-blue-400 font-bold">
              <span className="flex items-center gap-1">
                <Cpu className="w-3.5 h-3.5 text-blue-400" />
                神经元 L_{hoveredNode.layer} [节点 #{hoveredNode.index + 1}]
              </span>
              <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                hoveredNode.layer === 0
                  ? "bg-emerald-950 text-emerald-300 border border-emerald-700"
                  : hoveredNode.layer === nn.layerSizes.length - 1
                  ? "bg-amber-950 text-amber-300 border border-amber-700"
                  : "bg-indigo-950 text-indigo-300 border border-indigo-700"
              }`}>
                {hoveredNode.layer === 0
                  ? "输入层"
                  : hoveredNode.layer === nn.layerSizes.length - 1
                  ? "输出层"
                  : "隐藏层"}
              </span>
            </div>
            <div className="space-y-1 text-[11px]">
              <div className="flex justify-between gap-4">
                <span className="text-slate-400">输出激活值 a:</span>
                <span className="text-emerald-400 font-bold">
                  {(
                    networkState.fwd.layers[hoveredNode.layer]?.a[
                        hoveredNode.index
                      ] ?? 0
                  ).toFixed(5)}
                </span>
              </div>
              {hoveredNode.layer > 0 && (
                <>
                  <div className="flex justify-between gap-4">
                    <span className="text-slate-400">加权求和净输入 z:</span>
                    <span className="text-sky-300 font-medium">
                      {(
                        networkState.fwd.layers[hoveredNode.layer]?.z[
                            hoveredNode.index
                          ] ?? 0
                      ).toFixed(5)}
                    </span>
                  </div>
                  <div className="flex justify-between gap-4">
                    <span className="text-slate-400">偏置量 b:</span>
                    <span className="text-slate-300">
                      {(
                        nn.biases[hoveredNode.layer - 1]?.[
                            hoveredNode.index
                          ] ?? 0
                      ).toFixed(4)}
                    </span>
                  </div>
                  <div className="flex justify-between gap-4">
                    <span className="text-slate-400">误差梯度敏感度 δ:</span>
                    <span className="text-amber-400 font-bold">
                      {(
                        networkState.bwd.layerGradients[
                            hoveredNode.layer - 1
                          ]?.delta[hoveredNode.index] ?? 0
                      ).toFixed(6)}
                    </span>
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        {/* Artificial Neuron Mathematical Anatomy Floating Card */}
        {showNeuronAnatomy && (
          <div className="absolute top-3 right-3 z-30 w-80 bg-slate-900/95 border border-blue-500/50 rounded-xl p-3.5 shadow-2xl backdrop-blur text-slate-100 text-xs animate-fade-in">
            <div className="flex items-center justify-between pb-2 border-b border-slate-700">
              <div className="flex items-center gap-1.5 font-bold text-blue-400">
                <Cpu className="w-4 h-4 text-blue-400" />
                <span>经典人工神经元模型 (MCP Neuron)</span>
              </div>
              <button
                onClick={() => setShowNeuronAnatomy(false)}
                className="text-slate-400 hover:text-white text-sm font-bold cursor-pointer"
              >
                ✕
              </button>
            </div>

            {/* Neuron Anatomy Schematic */}
            <div className="my-2.5 p-2 bg-slate-950 rounded-lg border border-slate-800 space-y-2">
              <div className="flex items-center justify-between text-[11px] font-mono text-center">
                <div className="p-1 rounded bg-emerald-950/80 border border-emerald-600/50 text-emerald-300">
                  <div className="text-[9px] text-emerald-400">输入信号</div>
                  <div>x₁, x₂, ...</div>
                </div>
                <span className="text-slate-500">→ w_i →</span>
                <div className="p-1.5 rounded bg-indigo-950/80 border border-indigo-500 text-indigo-300 font-bold">
                  <div className="text-[9px] text-indigo-400">线性累加</div>
                  <div>Σ w_i·x_i + b</div>
                </div>
                <span className="text-slate-500">→ z →</span>
                <div className="p-1.5 rounded bg-amber-950/80 border border-amber-500 text-amber-300 font-bold">
                  <div className="text-[9px] text-amber-400">非线性激活</div>
                  <div>f(z) / σ</div>
                </div>
                <span className="text-slate-500">→</span>
                <div className="p-1 rounded bg-blue-950/80 border border-blue-500 text-blue-300">
                  <div className="text-[9px] text-blue-400">轴突输出</div>
                  <div>a</div>
                </div>
              </div>
            </div>

            <p className="text-[11px] text-slate-300 leading-relaxed">
              每个隐藏/输出神经元内部均由两大部分构成：
              <br />
              1. <strong className="text-sky-300">Σ 线性求和器</strong>：接收上一层突触传来的加权信号并加上偏置 $z = \sum w_i a_i + b$；
              <br />
              2. <strong className="text-amber-300">σ 激活发生器</strong>：通过非线性函数映射输出 $a = f(z)$。
            </p>
          </div>
        )}
      </div>

      {/* Step-by-Step Pedagogical Walkthrough Slice */}
      {stepTrace && stepTrace.length > 0 && (
        <div className="border-t border-slate-200 bg-slate-50 p-3">
          <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-200 text-[11px] font-mono font-semibold">
                单步推演 ({currentStepIndex + 1}/{stepTrace.length})
              </span>
              <h4 className="text-xs font-bold text-slate-800">
                {activeFrame?.title}
              </h4>
            </div>

            {/* Step Navigation Controls */}
            <div className="flex items-center gap-1.5">
              <button
                onClick={() =>
                  onSetStepIndex(
                    Math.max(0, currentStepIndex - 1)
                  )
                }
                disabled={currentStepIndex === 0}
                className="px-2.5 py-1 bg-white hover:bg-slate-50 disabled:opacity-40 text-slate-700 rounded border border-slate-300 text-xs font-medium transition cursor-pointer"
              >
                上一步
              </button>
              <button
                onClick={onNextStep}
                disabled={currentStepIndex === stepTrace.length - 1}
                className="flex items-center gap-1 px-3 py-1 bg-blue-600 hover:bg-blue-700 disabled:opacity-40 text-white rounded text-xs font-semibold shadow-sm transition cursor-pointer"
              >
                <span>下一步推演</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {/* Step Formula & Math Slice */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5 bg-white rounded-lg p-2.5 border border-slate-200 text-xs">
            <div className="md:col-span-1 border-r border-slate-200 pr-2">
              <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider block mb-1">
                运算形式化公式 (Formula):
              </span>
              {activeFrame && (
                <div className="bg-slate-50 p-1.5 rounded border border-slate-200 text-center text-blue-700">
                  <InlineMath math={activeFrame.formula} />
                </div>
              )}
            </div>

            <div className="md:col-span-1 border-r border-slate-200 pr-2">
              <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider block mb-1">
                运筹过程解析 (Description):
              </span>
              <p className="text-slate-600 leading-relaxed text-[11px]">
                {activeFrame?.description}
              </p>
            </div>

            <div className="md:col-span-1 font-mono text-[11px] space-y-0.5">
              <span className="text-[10px] text-slate-400 font-bold uppercase tracking-wider block mb-1 font-sans">数值计算细节:</span>
              {activeFrame?.mathDetails &&
                Object.entries(activeFrame.mathDetails).map(([k, v]) => (
                  <div key={k} className="flex justify-between text-slate-700">
                    <span className="text-slate-400">{k}:</span>
                    <span className="text-blue-700 font-semibold truncate max-w-[140px]" title={String(v)}>
                      {String(v)}
                    </span>
                  </div>
                ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
