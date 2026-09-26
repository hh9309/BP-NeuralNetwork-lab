import React, { useRef, useEffect, useState, useMemo, useCallback } from "react";
import {
  Grid,
  RotateCcw,
  PlusCircle,
  Sparkles,
  Layers,
  MousePointer,
  HelpCircle,
  CheckCircle2,
  AlertCircle,
  Sliders,
  Eye,
  Info,
  Maximize2,
  Flame,
} from "lucide-react";
import { NeuralNetwork } from "../lib/neural-engine";
import { TrainingSample } from "../types";
import {
  generateCirclesDataset,
  generateMoonsDataset,
  generateSpiralsDataset,
  generateXORDataset,
} from "../lib/datasets";

interface DecisionBoundary2DProps {
  nn: NeuralNetwork;
  samples: TrainingSample[];
  onUpdateSamples: (newSamples: TrainingSample[]) => void;
  epoch: number;
}

export const DecisionBoundary2D: React.FC<DecisionBoundary2DProps> = ({
  nn,
  samples,
  onUpdateSamples,
  epoch,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Dataset State
  const [datasetType, setDatasetType] = useState<"xor" | "circles" | "moons" | "spirals">("xor");
  const [xorSubPreset, setXorSubPreset] = useState<"standard" | "canonical4" | "dense" | "saddle">("standard");
  const [gridResolution, setGridResolution] = useState<number>(50);
  const [addMode, setAddMode] = useState<"class1" | "class0" | "view">("view");
  const [showConfidenceIsolines, setShowConfidenceIsolines] = useState<boolean>(true);
  const [showQuadrantBadges, setShowQuadrantBadges] = useState<boolean>(true);
  const [showTheoreticalInsight, setShowTheoreticalInsight] = useState<boolean>(false);

  // Mouse hover probing state
  const [hoverCoord, setHoverCoord] = useState<{ x: number; y: number; px: number; py: number } | null>(null);
  const [hoverInference, setHoverInference] = useState<{ pred: number; label: number; probPct: string } | null>(null);

  // Viewport Domain (centered with generous margin around [0, 1] so XOR points are fully visible)
  const domain = useMemo(() => {
    if (datasetType === "xor") {
      return { minX: -0.25, maxX: 1.25, minY: -0.25, maxY: 1.25 };
    }
    // Dynamic / standard domain for other 2D datasets
    return { minX: -0.2, maxX: 1.2, minY: -0.2, maxY: 1.2 };
  }, [datasetType]);

  // Generate XOR dataset based on sub-preset
  const getXORSamples = useCallback((preset: "standard" | "canonical4" | "dense" | "saddle") => {
    switch (preset) {
      case "canonical4":
        return [
          { input: [0, 0], target: [0], label: "负类 (0,0) → 0" },
          { input: [0, 1], target: [1], label: "正类 (0,1) → 1" },
          { input: [1, 0], target: [1], label: "正类 (1,0) → 1" },
          { input: [1, 1], target: [0], label: "负类 (1,1) → 0" },
        ];
      case "dense":
        return generateXORDataset(0.06, 120);
      case "saddle":
        // Generate continuous diagonal saddle distribution
        const saddleSamples: TrainingSample[] = [];
        for (let i = 0; i < 80; i++) {
          const x1 = Math.random();
          const x2 = Math.random();
          // XOR-like target: if x1 and x2 are in same polarity relative to 0.5
          const target = (x1 > 0.5 && x2 > 0.5) || (x1 <= 0.5 && x2 <= 0.5) ? 0 : 1;
          saddleSamples.push({
            input: [x1, x2],
            target: [target],
            label: target === 1 ? "正类 (Class 1)" : "负类 (Class 0)",
          });
        }
        return saddleSamples;
      case "standard":
      default:
        return generateXORDataset(0.04, 52);
    }
  }, []);

  // Switch toy benchmark dataset
  const handleDatasetChange = (type: "xor" | "circles" | "moons" | "spirals") => {
    setDatasetType(type);
    let newSamples: TrainingSample[] = [];
    if (type === "xor") newSamples = getXORSamples(xorSubPreset);
    else if (type === "circles") newSamples = generateCirclesDataset(0.05, 80);
    else if (type === "moons") newSamples = generateMoonsDataset(0.05, 80);
    else if (type === "spirals") newSamples = generateSpiralsDataset(0.03, 100);
    onUpdateSamples(newSamples);
  };

  // Change XOR sub-preset
  const handleXorPresetChange = (preset: "standard" | "canonical4" | "dense" | "saddle") => {
    setXorSubPreset(preset);
    if (datasetType === "xor") {
      onUpdateSamples(getXORSamples(preset));
    }
  };

  // Canonical XOR 4-point verification status calculation
  const xorCanonicalMetrics = useMemo(() => {
    const canonicals = [
      { input: [0, 0], target: 0, label: "(0, 0) → 0", quadrant: "左下 Q3 (负类)" },
      { input: [0, 1], target: 1, label: "(0, 1) → 1", quadrant: "左上 Q2 (正类)" },
      { input: [1, 0], target: 1, label: "(1, 0) → 1", quadrant: "右下 Q4 (正类)" },
      { input: [1, 1], target: 0, label: "(1, 1) → 0", quadrant: "右上 Q1 (负类)" },
    ];

    let correctCount = 0;
    const evaluated = canonicals.map((c) => {
      const fwd = nn.forward(c.input);
      const rawPred = fwd.output[0] ?? 0.5;
      const predictedClass = rawPred >= 0.5 ? 1 : 0;
      const isCorrect = predictedClass === c.target;
      if (isCorrect) correctCount++;
      return {
        ...c,
        rawPred,
        predictedClass,
        isCorrect,
        confidencePct: (predictedClass === 1 ? rawPred : 1 - rawPred) * 100,
      };
    });

    const isFullySeparated = correctCount === 4;
    return {
      evaluated,
      correctCount,
      accuracyPct: Math.round((correctCount / 4) * 100),
      isFullySeparated,
    };
  }, [nn, epoch]);

  // Add sample by clicking on canvas
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (addMode === "view") return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const clickRatioX = (e.clientX - rect.left) / rect.width;
    const clickRatioY = (e.clientY - rect.top) / rect.height;

    // Convert canvas click ratio back to math domain coordinates
    const mathX = domain.minX + clickRatioX * (domain.maxX - domain.minX);
    const mathY = domain.maxY - clickRatioY * (domain.maxY - domain.minY);

    const targetVal = addMode === "class1" ? 1 : 0;
    const newSample: TrainingSample = {
      input: [Number(mathX.toFixed(3)), Number(mathY.toFixed(3))],
      target: [targetVal],
      label: addMode === "class1" ? "正类 (Class 1)" : "负类 (Class 0)",
    };

    onUpdateSamples([...samples, newSample]);
  };

  // Handle Canvas Mouse Move for Real-time Probing
  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const ratioX = Math.max(0, Math.min(1, px / rect.width));
    const ratioY = Math.max(0, Math.min(1, py / rect.height));

    const mathX = domain.minX + ratioX * (domain.maxX - domain.minX);
    const mathY = domain.maxY - ratioY * (domain.maxY - domain.minY);

    const fwd = nn.forward([mathX, mathY]);
    const pred = fwd.output[0] ?? 0.5;
    const label = pred >= 0.5 ? 1 : 0;
    const probPct = (pred * 100).toFixed(1);

    setHoverCoord({ x: mathX, y: mathY, px, py });
    setHoverInference({ pred, label, probPct });
  };

  const handleCanvasMouseLeave = () => {
    setHoverCoord(null);
    setHoverInference(null);
  };

  // Render Grid Heatmap and Samples
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const res = gridResolution;
    const cellW = width / res;
    const cellH = height / res;
    const { minX, maxX, minY, maxY } = domain;

    ctx.clearRect(0, 0, width, height);

    // 1. Compute grid forward inference values
    const gridPredictions: number[][] = [];
    for (let r = 0; r < res; r++) {
      const row: number[] = [];
      const mathY = maxY - (r / (res - 1)) * (maxY - minY);
      for (let c = 0; c < res; c++) {
        const mathX = minX + (c / (res - 1)) * (maxX - minX);
        const fwd = nn.forward([mathX, mathY]);
        const pred = fwd.output[0] ?? 0.5;
        row.push(pred);
      }
      gridPredictions.push(row);
    }

    // 2. Render Heatmap Cells with Rich High-Contrast Gradient
    for (let r = 0; r < res; r++) {
      for (let c = 0; c < res; c++) {
        const prob = gridPredictions[r][c]; // 0 to 1

        let rColor: number, gColor: number, bColor: number;
        if (prob >= 0.5) {
          // Positive Region: Deep navy (0.5) -> Electric Sky Blue / Cyan (1.0)
          const t = Math.pow((prob - 0.5) * 2, 0.85); // 0 to 1
          rColor = Math.round(15 + t * (14 - 15));
          gColor = Math.round(23 + t * (165 - 23));
          bColor = Math.round(42 + t * (233 - 42));
        } else {
          // Negative Region: Deep navy (0.5) -> Vibrant Rose / Crimson (0.0)
          const t = Math.pow((0.5 - prob) * 2, 0.85); // 0 to 1
          rColor = Math.round(15 + t * (225 - 15));
          gColor = Math.round(23 + t * (29 - 23));
          bColor = Math.round(42 + t * (72 - 42));
        }

        ctx.fillStyle = `rgb(${rColor}, ${gColor}, ${bColor})`;
        ctx.fillRect(c * cellW, r * cellH, cellW + 0.6, cellH + 0.6);
      }
    }

    // Helper: Map math coord to canvas pixel
    const toCanvasX = (mx: number) => ((mx - minX) / (maxX - minX)) * width;
    const toCanvasY = (my: number) => height - ((my - minY) / (maxY - minY)) * height;

    // 3. Draw Unit-Square [0, 1] x [0, 1] Bounding Box & Quadrants (Essential for XOR!)
    if (datasetType === "xor") {
      const uX0 = toCanvasX(0);
      const uY0 = toCanvasY(0);
      const uX1 = toCanvasX(1);
      const uY1 = toCanvasY(1);

      // Unit Square Boundary (Subtle dashed line)
      ctx.strokeStyle = "rgba(255, 255, 255, 0.28)";
      ctx.lineWidth = 1.2;
      ctx.setLineDash([4, 3]);
      ctx.strokeRect(uX0, uY1, uX1 - uX0, uY0 - uY1);

      // Center Divider Crosshair (x = 0.5, y = 0.5)
      const uXMid = toCanvasX(0.5);
      const uYMid = toCanvasY(0.5);

      ctx.strokeStyle = "rgba(255, 255, 255, 0.18)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(uXMid, uY1);
      ctx.lineTo(uXMid, uY0);
      ctx.moveTo(uX0, uYMid);
      ctx.lineTo(uX1, uYMid);
      ctx.stroke();
      ctx.setLineDash([]);

      // Corner Region Indicators / Quadrant Watermarks
      if (showQuadrantBadges) {
        ctx.font = "bold 10px monospace";
        ctx.textAlign = "center";

        // (0,1) Top-Left -> Class 1
        ctx.fillStyle = "rgba(14, 165, 233, 0.85)";
        ctx.fillText("(0,1) 正类 1", uX0, uY1 - 8);

        // (1,1) Top-Right -> Class 0
        ctx.fillStyle = "rgba(244, 63, 94, 0.85)";
        ctx.fillText("(1,1) 负类 0", uX1, uY1 - 8);

        // (0,0) Bottom-Left -> Class 0
        ctx.fillStyle = "rgba(244, 63, 94, 0.85)";
        ctx.fillText("(0,0) 负类 0", uX0, uY0 + 16);

        // (1,0) Bottom-Right -> Class 1
        ctx.fillStyle = "rgba(14, 165, 233, 0.85)";
        ctx.fillText("(1,0) 正类 1", uX1, uY0 + 16);
      }
    }

    // 4. Draw Mathematical Axes (X=0 and Y=0 lines)
    const zeroX = toCanvasX(0);
    const zeroY = toCanvasY(0);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.22)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    if (zeroX >= 0 && zeroX <= width) {
      ctx.moveTo(zeroX, 0);
      ctx.lineTo(zeroX, height);
    }
    if (zeroY >= 0 && zeroY <= height) {
      ctx.moveTo(0, zeroY);
      ctx.lineTo(width, zeroY);
    }
    ctx.stroke();

    // 5. Draw Confidence Isolines (0.25 and 0.75 probability bands)
    if (showConfidenceIsolines) {
      const drawIsoContour = (threshold: number, color: string, dash: number[]) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.2;
        ctx.setLineDash(dash);
        for (let r = 0; r < res - 1; r++) {
          for (let c = 0; c < res - 1; c++) {
            const v0 = gridPredictions[r][c] - threshold;
            const v1 = gridPredictions[r][c + 1] - threshold;
            const v2 = gridPredictions[r + 1][c + 1] - threshold;
            const v3 = gridPredictions[r + 1][c] - threshold;

            const x = c * cellW;
            const y = r * cellH;

            if (v0 * v1 <= 0 && v0 !== v1) {
              const t = Math.abs(v0) / (Math.abs(v0) + Math.abs(v1));
              const cx = x + t * cellW;
              const cy = y;

              if (v3 * v2 <= 0 && v3 !== v2) {
                const tb = Math.abs(v3) / (Math.abs(v3) + Math.abs(v2));
                ctx.beginPath();
                ctx.moveTo(cx, cy);
                ctx.lineTo(x + tb * cellW, y + cellH);
                ctx.stroke();
              }
              if (v0 * v3 <= 0 && v0 !== v3) {
                const tl = Math.abs(v0) / (Math.abs(v0) + Math.abs(v3));
                ctx.beginPath();
                ctx.moveTo(cx, cy);
                ctx.lineTo(x, y + tl * cellH);
                ctx.stroke();
              }
              if (v1 * v2 <= 0 && v1 !== v2) {
                const tr = Math.abs(v1) / (Math.abs(v1) + Math.abs(v2));
                ctx.beginPath();
                ctx.moveTo(cx, cy);
                ctx.lineTo(x + cellW, y + tr * cellH);
                ctx.stroke();
              }
            }
          }
        }
      };

      // Draw P=0.25 and P=0.75 isolines
      drawIsoContour(0.25, "rgba(244, 63, 94, 0.45)", [3, 3]);
      drawIsoContour(0.75, "rgba(56, 189, 248, 0.45)", [3, 3]);
      ctx.setLineDash([]);
    }

    // 6. Draw Primary Decision Boundary Contour (P = 0.5) with Glowing Neon Stroke
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2.8;
    ctx.shadowColor = "#ffffff";
    ctx.shadowBlur = 9;

    for (let r = 0; r < res - 1; r++) {
      for (let c = 0; c < res - 1; c++) {
        const v0 = gridPredictions[r][c] - 0.5;
        const v1 = gridPredictions[r][c + 1] - 0.5;
        const v2 = gridPredictions[r + 1][c + 1] - 0.5;
        const v3 = gridPredictions[r + 1][c] - 0.5;

        const x = c * cellW;
        const y = r * cellH;

        if (v0 * v1 <= 0 && v0 !== v1) {
          const t = Math.abs(v0) / (Math.abs(v0) + Math.abs(v1));
          const cx = x + t * cellW;
          const cy = y;

          if (v3 * v2 <= 0 && v3 !== v2) {
            const tb = Math.abs(v3) / (Math.abs(v3) + Math.abs(v2));
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(x + tb * cellW, y + cellH);
            ctx.stroke();
          }
          if (v0 * v3 <= 0 && v0 !== v3) {
            const tl = Math.abs(v0) / (Math.abs(v0) + Math.abs(v3));
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(x, y + tl * cellH);
            ctx.stroke();
          }
          if (v1 * v2 <= 0 && v1 !== v2) {
            const tr = Math.abs(v1) / (Math.abs(v1) + Math.abs(v2));
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(x + cellW, y + tr * cellH);
            ctx.stroke();
          }
        }
      }
    }
    ctx.shadowBlur = 0; // reset glow

    // 7. Draw Training Sample Scatter Points with High Visibility
    for (const s of samples) {
      if (s.input.length < 2) continue;
      const sx = toCanvasX(s.input[0]);
      const sy = toCanvasY(s.input[1]);
      const isClass1 = (s.target[0] ?? 0) >= 0.5;

      // Outer border halo (depth)
      ctx.beginPath();
      ctx.arc(sx, sy, 6.5, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(0, 0, 0, 0.75)";
      ctx.fill();

      // Sample core
      ctx.beginPath();
      ctx.arc(sx, sy, 4.8, 0, Math.PI * 2);
      if (isClass1) {
        ctx.fillStyle = "#38bdf8"; // Electric Sky Blue
        ctx.strokeStyle = "#ffffff";
      } else {
        ctx.fillStyle = "#f43f5e"; // Vivid Rose Red
        ctx.strokeStyle = "#ffffff";
      }
      ctx.lineWidth = 1.6;
      ctx.fill();
      ctx.stroke();
    }

    // 8. Draw Hover Crosshair Probe
    if (hoverCoord) {
      ctx.strokeStyle = "rgba(255, 255, 255, 0.6)";
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);

      ctx.beginPath();
      ctx.moveTo(hoverCoord.px, 0);
      ctx.lineTo(hoverCoord.px, height);
      ctx.moveTo(0, hoverCoord.py);
      ctx.lineTo(width, hoverCoord.py);
      ctx.stroke();
      ctx.setLineDash([]);

      // Probe Reticle Circle
      ctx.beginPath();
      ctx.arc(hoverCoord.px, hoverCoord.py, 6, 0, Math.PI * 2);
      ctx.strokeStyle = hoverInference?.label === 1 ? "#38bdf8" : "#f43f5e";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }, [
    nn,
    samples,
    epoch,
    gridResolution,
    domain,
    datasetType,
    showConfidenceIsolines,
    showQuadrantBadges,
    hoverCoord,
    hoverInference,
  ]);

  return (
    <div
      ref={containerRef}
      className="bg-white border border-slate-200 rounded-lg p-3.5 sm:p-4 shadow-sm space-y-3"
    >
      {/* Header Banner & Dataset Switcher */}
      <div className="flex flex-wrap items-center justify-between gap-2.5 pb-2.5 border-b border-slate-200">
        <div className="flex items-center gap-2.5">
          <div className="w-2.5 h-2.5 rounded-full bg-blue-600 animate-pulse" />
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-xs sm:text-sm font-bold text-slate-900">
                分类决策边界 2D 拓扑平滑演化
              </h3>
              {datasetType === "xor" && (
                <span
                  className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-bold ${
                    xorCanonicalMetrics.isFullySeparated
                      ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                      : "bg-amber-50 text-amber-700 border border-amber-200"
                  }`}
                >
                  {xorCanonicalMetrics.isFullySeparated ? (
                    <>
                      <CheckCircle2 className="w-3 h-3 text-emerald-600" />
                      <span>XOR 4/4 完美分离</span>
                    </>
                  ) : (
                    <>
                      <AlertCircle className="w-3 h-3 text-amber-600" />
                      <span>
                        XOR 分离度: {xorCanonicalMetrics.correctCount}/4 (
                        {xorCanonicalMetrics.accuracyPct}%)
                      </span>
                    </>
                  )}
                </span>
              )}
            </div>
            <p className="text-[10px] text-slate-500 font-medium mt-0.5">
              观察隐藏层神经元如何通过非线性空间扭曲折叠对角分布，实现非线性分类
            </p>
          </div>
        </div>

        {/* Dataset Quick Switcher */}
        <div className="flex items-center gap-1 bg-slate-100 p-0.5 rounded-md border border-slate-200 text-xs">
          <button
            onClick={() => handleDatasetChange("xor")}
            className={`flex items-center gap-1 px-2.5 py-1 rounded font-semibold transition cursor-pointer ${
              datasetType === "xor"
                ? "bg-white text-blue-700 shadow-sm"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            <Sparkles className="w-3 h-3 text-blue-600" />
            <span>异或 (XOR)</span>
          </button>
          <button
            onClick={() => handleDatasetChange("circles")}
            className={`px-2.5 py-1 rounded font-semibold transition cursor-pointer ${
              datasetType === "circles"
                ? "bg-white text-blue-700 shadow-sm"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            同心圆 (Circles)
          </button>
          <button
            onClick={() => handleDatasetChange("moons")}
            className={`px-2.5 py-1 rounded font-semibold transition cursor-pointer ${
              datasetType === "moons"
                ? "bg-white text-blue-700 shadow-sm"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            双月牙 (Moons)
          </button>
          <button
            onClick={() => handleDatasetChange("spirals")}
            className={`px-2.5 py-1 rounded font-semibold transition cursor-pointer ${
              datasetType === "spirals"
                ? "bg-white text-blue-700 shadow-sm"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            双螺旋 (Spirals)
          </button>
        </div>
      </div>

      {/* XOR Dedicated Sub-Presets & Topology Controls */}
      {datasetType === "xor" && (
        <div className="bg-blue-50/60 border border-blue-100 rounded-lg p-2.5 flex flex-wrap items-center justify-between gap-2 text-xs">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-[11px] font-bold text-blue-900 flex items-center gap-1">
              <Flame className="w-3 h-3 text-blue-600" />
              <span>XOR 形态预设:</span>
            </span>
            <div className="flex items-center gap-1 bg-white p-0.5 rounded border border-blue-200">
              <button
                onClick={() => handleXorPresetChange("standard")}
                className={`px-2 py-0.5 rounded text-[11px] font-medium transition cursor-pointer ${
                  xorSubPreset === "standard"
                    ? "bg-blue-600 text-white font-bold"
                    : "text-slate-700 hover:bg-slate-50"
                }`}
              >
                标准 4 簇 (带微扰)
              </button>
              <button
                onClick={() => handleXorPresetChange("canonical4")}
                className={`px-2 py-0.5 rounded text-[11px] font-medium transition cursor-pointer ${
                  xorSubPreset === "canonical4"
                    ? "bg-blue-600 text-white font-bold"
                    : "text-slate-700 hover:bg-slate-50"
                }`}
              >
                纯净 4 极点 (真值表)
              </button>
              <button
                onClick={() => handleXorPresetChange("dense")}
                className={`px-2 py-0.5 rounded text-[11px] font-medium transition cursor-pointer ${
                  xorSubPreset === "dense"
                    ? "bg-blue-600 text-white font-bold"
                    : "text-slate-700 hover:bg-slate-50"
                }`}
              >
                密集高斯分布 (120点)
              </button>
              <button
                onClick={() => handleXorPresetChange("saddle")}
                className={`px-2 py-0.5 rounded text-[11px] font-medium transition cursor-pointer ${
                  xorSubPreset === "saddle"
                    ? "bg-blue-600 text-white font-bold"
                    : "text-slate-700 hover:bg-slate-50"
                }`}
              >
                连续对角鞍面
              </button>
            </div>
          </div>

          {/* Quick Toggle Controls */}
          <div className="flex items-center gap-3 text-[11px] text-slate-700">
            <label className="flex items-center gap-1 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={showQuadrantBadges}
                onChange={(e) => setShowQuadrantBadges(e.target.checked)}
                className="rounded border-slate-300 text-blue-600 focus:ring-0 cursor-pointer"
              />
              <span>象限真值角标</span>
            </label>
            <label className="flex items-center gap-1 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={showConfidenceIsolines}
                onChange={(e) => setShowConfidenceIsolines(e.target.checked)}
                className="rounded border-slate-300 text-blue-600 focus:ring-0 cursor-pointer"
              />
              <span>置信度等高线 (25%/75%)</span>
            </label>
            <button
              onClick={() => setShowTheoreticalInsight((prev) => !prev)}
              className="text-blue-700 hover:text-blue-800 font-semibold flex items-center gap-0.5 cursor-pointer underline decoration-dotted"
            >
              <HelpCircle className="w-3 h-3" />
              <span>{showTheoreticalInsight ? "收起原理解析" : "为什么单层无法解决 XOR?"}</span>
            </button>
          </div>
        </div>
      )}

      {/* Theoretical Insight Callout (Collapsible) */}
      {datasetType === "xor" && showTheoreticalInsight && (
        <div className="bg-slate-900 text-slate-200 rounded-lg p-3 text-xs space-y-1.5 border border-slate-800 animate-fade-in font-sans">
          <div className="flex items-center gap-1.5 text-sky-400 font-bold">
            <Info className="w-3.5 h-3.5" />
            <span>Minsky & Papert (1969) 异或门不可分性与多层感知机破解机制</span>
          </div>
          <p className="text-slate-300 text-[11px] leading-relaxed">
            1. <strong>线性不可分性</strong>：单层感知机仅包含一条超平面 $w_1 x_1 + w_2 x_2 + b = 0$。对于对角分布的 $(0,0)=0, (1,1)=0$ 和 $(0,1)=1, (1,0)=1$，无论如何旋转平移单条直线，都无法同时将对角两组点完全隔开。
          </p>
          <p className="text-slate-300 text-[11px] leading-relaxed">
            2. <strong>隐藏层空间折叠</strong>：包含 2 个隐藏神经元的网络分别学习两条超平面（如 $h_1 = \sigma(x_1 + x_2 - 0.5)$ 与 $h_2 = \sigma(x_1 + x_2 - 1.5)$），将原始 2D 坐标映射到隐空间 $(h_1, h_2)$，在隐空间中将 $(0,1)$ 与 $(1,0)$ 推向同一侧，使最终输出层只需一条线性分割线即可达成 100% 精度。
          </p>
        </div>
      )}

      {/* Interactive Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs bg-slate-50 p-2 rounded-lg border border-slate-200">
        {/* Sample Creation Mode */}
        <div className="flex items-center gap-2">
          <span className="text-[11px] text-slate-400 font-bold uppercase tracking-wider">
            交互探针 / 加点:
          </span>
          <div className="flex items-center bg-white rounded p-0.5 border border-slate-200">
            <button
              onClick={() => setAddMode("view")}
              className={`flex items-center gap-1 px-2 py-0.5 rounded text-xs transition cursor-pointer ${
                addMode === "view"
                  ? "bg-slate-100 text-slate-900 font-bold shadow-xs"
                  : "text-slate-500 hover:text-slate-800"
              }`}
            >
              <MousePointer className="w-3 h-3" />
              <span>仅探测 (Hover 探针)</span>
            </button>
            <button
              onClick={() => setAddMode("class1")}
              className={`flex items-center gap-1 px-2 py-0.5 rounded text-xs transition cursor-pointer ${
                addMode === "class1"
                  ? "bg-sky-50 text-sky-700 font-bold border border-sky-200"
                  : "text-slate-600 hover:text-sky-600"
              }`}
            >
              <span className="w-2 h-2 rounded-full bg-sky-500" />
              <span>+ 正样本 (Class 1)</span>
            </button>
            <button
              onClick={() => setAddMode("class0")}
              className={`flex items-center gap-1 px-2 py-0.5 rounded text-xs transition cursor-pointer ${
                addMode === "class0"
                  ? "bg-rose-50 text-rose-700 font-bold border border-rose-200"
                  : "text-slate-600 hover:text-rose-600"
              }`}
            >
              <span className="w-2 h-2 rounded-full bg-rose-500" />
              <span>+ 负样本 (Class 0)</span>
            </button>
          </div>
        </div>

        {/* Resolution selector & Clear */}
        <div className="flex items-center gap-2.5">
          <div className="flex items-center gap-1.5">
            <span className="text-[11px] text-slate-400 font-bold uppercase tracking-wider">
              网格细粒度:
            </span>
            <select
              value={gridResolution}
              onChange={(e) => setGridResolution(Number(e.target.value))}
              className="bg-white border border-slate-200 text-slate-800 rounded px-2 py-0.5 outline-none cursor-pointer text-xs"
            >
              <option value={35}>35 × 35 (极速)</option>
              <option value={50}>50 × 50 (标准高清)</option>
              <option value={70}>70 × 70 (超精细流形)</option>
            </select>
          </div>

          <button
            onClick={() => handleDatasetChange(datasetType)}
            className="flex items-center gap-1 px-2 py-0.5 bg-white border border-slate-200 hover:bg-slate-100 text-slate-700 rounded text-xs transition cursor-pointer"
            title="重置当前数据集"
          >
            <RotateCcw className="w-3 h-3 text-slate-500" />
            <span>重置分布</span>
          </button>
        </div>
      </div>

      {/* Main Boundary Canvas and HUD Display */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-3">
        {/* Canvas Area (9 cols on large screen) */}
        <div className="lg:col-span-9 relative w-full h-[360px] bg-slate-950 rounded-lg overflow-hidden border border-slate-800 flex items-center justify-center shadow-inner">
          <canvas
            ref={canvasRef}
            width={640}
            height={360}
            onClick={handleCanvasClick}
            onMouseMove={handleCanvasMouseMove}
            onMouseLeave={handleCanvasMouseLeave}
            className={`w-full h-full object-contain ${
              addMode !== "view" ? "cursor-crosshair" : "cursor-crosshair"
            }`}
          />

          {/* Real-time Hover Probe Floating HUD */}
          {hoverCoord && hoverInference && (
            <div className="absolute top-2.5 left-2.5 bg-slate-900/90 backdrop-blur border border-slate-700 rounded-md p-2 text-[11px] font-mono text-slate-200 shadow-lg pointer-events-none space-y-0.5">
              <div className="flex items-center justify-between gap-3 text-slate-400">
                <span>坐标探针</span>
                <span>
                  X₁: {hoverCoord.x.toFixed(2)}, X₂: {hoverCoord.y.toFixed(2)}
                </span>
              </div>
              <div className="flex items-center gap-2 pt-0.5 border-t border-slate-800">
                <span
                  className={`w-2 h-2 rounded-full ${
                    hoverInference.label === 1 ? "bg-sky-400" : "bg-rose-400"
                  }`}
                />
                <span className="font-bold text-white">
                  判决: {hoverInference.label === 1 ? "正类 (Class 1)" : "负类 (Class 0)"}
                </span>
                <span className="text-slate-400">
                  (P={hoverInference.probPct}%)
                </span>
              </div>
            </div>
          )}

          {/* Dynamic Legend Floating Badge */}
          <div className="absolute bottom-2.5 right-2.5 bg-slate-900/90 backdrop-blur border border-slate-700 rounded-md p-2 text-[10px] font-mono space-y-1 shadow-md pointer-events-none text-slate-300">
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-1 bg-white rounded-full shadow-[0_0_6px_#fff]" />
              <span className="font-bold text-white">主决策边界 (P=0.5)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-sky-400 border border-white/50" />
              <span className="text-sky-300 font-medium">正类流形 (P &gt; 0.5)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-rose-500 border border-white/50" />
              <span className="text-rose-300 font-medium">负类流形 (P &lt; 0.5)</span>
            </div>
          </div>
        </div>

        {/* Side Panel: XOR Truth Table Status & Manifold Diagnostic (3 cols) */}
        <div className="lg:col-span-3 flex flex-col justify-between bg-slate-50 rounded-lg p-3 border border-slate-200 space-y-2.5">
          {datasetType === "xor" ? (
            <>
              <div>
                <div className="flex items-center justify-between pb-1.5 border-b border-slate-200">
                  <h4 className="text-xs font-bold text-slate-900 flex items-center gap-1">
                    <Grid className="w-3.5 h-3.5 text-blue-600" />
                    <span>XOR 4点真值表诊断</span>
                  </h4>
                  <span
                    className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                      xorCanonicalMetrics.isFullySeparated
                        ? "bg-emerald-100 text-emerald-800"
                        : "bg-amber-100 text-amber-800"
                    }`}
                  >
                    {xorCanonicalMetrics.accuracyPct}%
                  </span>
                </div>

                {/* 4 Canonical Points List */}
                <div className="mt-2 space-y-1.5">
                  {xorCanonicalMetrics.evaluated.map((item, idx) => (
                    <div
                      key={idx}
                      className={`p-1.5 rounded border text-[11px] font-mono flex items-center justify-between ${
                        item.isCorrect
                          ? "bg-white border-emerald-200 text-slate-800"
                          : "bg-rose-50/70 border-rose-200 text-rose-900"
                      }`}
                    >
                      <div className="flex items-center gap-1.5">
                        <span
                          className={`w-2 h-2 rounded-full ${
                            item.target === 1 ? "bg-sky-500" : "bg-rose-500"
                          }`}
                        />
                        <span className="font-bold">{item.label}</span>
                      </div>
                      <div className="flex items-center gap-1 text-[10px]">
                        <span className="text-slate-500">
                          ŷ={item.rawPred.toFixed(2)}
                        </span>
                        {item.isCorrect ? (
                          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                        ) : (
                          <AlertCircle className="w-3.5 h-3.5 text-rose-500" />
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Status Note */}
              <div className="p-2 bg-white rounded border border-slate-200 text-[10px] text-slate-600 leading-relaxed">
                <span className="font-bold text-slate-800">流形状态评估：</span>
                {xorCanonicalMetrics.isFullySeparated ? (
                  <p className="text-emerald-700 font-medium mt-0.5">
                    🎉 神经网络已成功构建双重超平面边界，4 个异或逻辑极点全部正确分类！
                  </p>
                ) : (
                  <p className="text-amber-700 font-medium mt-0.5">
                    ⏳ 当前决策面仍为简单切分或欠拟合，请点击顶栏「▶ 开始训练」或调大迭代速度加速空间折叠。
                  </p>
                )}
              </div>
            </>
          ) : (
            <>
              <div>
                <div className="flex items-center justify-between pb-1.5 border-b border-slate-200">
                  <h4 className="text-xs font-bold text-slate-900 flex items-center gap-1">
                    <Layers className="w-3.5 h-3.5 text-blue-600" />
                    <span>2D 拓扑流形特征</span>
                  </h4>
                  <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-blue-100 text-blue-800">
                    {datasetType.toUpperCase()}
                  </span>
                </div>

                <div className="mt-2 space-y-1.5 text-[11px] text-slate-600">
                  <div className="flex justify-between py-1 border-b border-slate-100">
                    <span>当前样本数</span>
                    <span className="font-mono font-bold text-slate-800">
                      {samples.length} 个
                    </span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-slate-100">
                    <span>正样本比例 (Class 1)</span>
                    <span className="font-mono font-bold text-sky-600">
                      {(
                        (samples.filter((s) => (s.target[0] ?? 0) >= 0.5).length /
                          (samples.length || 1)) *
                        100
                      ).toFixed(0)}
                      %
                    </span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-slate-100">
                    <span>当前训练轮次</span>
                    <span className="font-mono font-bold text-slate-800">
                      Epoch {epoch}
                    </span>
                  </div>
                </div>
              </div>

              <div className="p-2 bg-white rounded border border-slate-200 text-[10px] text-slate-500 leading-relaxed">
                💡 <strong>提示</strong>：可以在上方选择「+ 正样本」或「+ 负样本」直接在画布任意位置点击加点，实时测试决策边界的拟合与泛化形变。
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
