import React, { useRef, useEffect, useState } from "react";
import {
  TrendingDown,
  Activity,
  Layers,
  Sparkles,
  BarChart3,
  BookmarkPlus,
  Trash2,
  Sliders,
} from "lucide-react";
import { TrainingMetrics } from "../types";

interface SavedRun {
  id: string;
  name: string;
  color: string;
  metrics: TrainingMetrics[];
  lr: number;
  optimizer: string;
  activation: string;
}

interface LossTrajectory2DProps {
  metricsHistory: TrainingMetrics[];
  currentEpoch: number;
  currentLoss: number;
  valLoss?: number;
  learningRate: number;
  optimizer: string;
  activation: string;
}

export const LossTrajectory2D: React.FC<LossTrajectory2DProps> = ({
  metricsHistory,
  currentEpoch,
  currentLoss,
  valLoss,
  learningRate,
  optimizer,
  activation,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [scaleMode, setScaleMode] = useState<"linear" | "log">("linear");
  const [savedRuns, setSavedRuns] = useState<SavedRun[]>([]);
  const [showGradNorm, setShowGradNorm] = useState(false);

  // Save current run for comparative overlay
  const handleSaveRun = () => {
    if (metricsHistory.length < 2) return;
    const colors = ["#ec4899", "#8b5cf6", "#06b6d4", "#f97316", "#10b981"];
    const color = colors[savedRuns.length % colors.length];
    const newRun: SavedRun = {
      id: Date.now().toString(),
      name: `Run #${savedRuns.length + 1} (η=${learningRate}, ${optimizer.toUpperCase()})`,
      color,
      metrics: [...metricsHistory],
      lr: learningRate,
      optimizer,
      activation,
    };
    setSavedRuns([...savedRuns, newRun]);
  };

  const handleClearSavedRuns = () => {
    setSavedRuns([]);
  };

  // Canvas Drawing
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;
    const padLeft = 60;
    const padRight = 30;
    const padTop = 30;
    const padBottom = 40;
    const plotW = width - padLeft - padRight;
    const plotH = height - padTop - padBottom;

    // Background
    ctx.fillStyle = "#090d16";
    ctx.fillRect(0, 0, width, height);

    // Compute bounds
    let maxEpoch = Math.max(50, currentEpoch, ...savedRuns.map((r) => r.metrics.length));
    let minLoss = 0;
    let maxLoss = 1.0;

    const allMetrics = [
      ...metricsHistory,
      ...savedRuns.flatMap((r) => r.metrics),
    ];

    if (allMetrics.length > 0) {
      const losses = allMetrics
        .flatMap((m) => [
          isNaN(m.loss) ? 1.0 : m.loss,
          m.valLoss !== undefined && !isNaN(m.valLoss) ? m.valLoss : m.loss,
        ])
        .filter((l) => isFinite(l));
      if (losses.length > 0) {
        maxLoss = Math.max(0.5, Math.min(10, Math.max(...losses) * 1.15));
        minLoss = Math.max(0, Math.min(...losses));
      }
    }

    // Grid and Axes
    ctx.strokeStyle = "rgba(30, 41, 59, 0.6)";
    ctx.lineWidth = 1;

    // Horizontal grid lines (Loss)
    const ySteps = 5;
    for (let i = 0; i <= ySteps; i++) {
      const yNorm = i / ySteps;
      const y = padTop + (1 - yNorm) * plotH;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(width - padRight, y);
      ctx.stroke();

      let valText = "";
      if (scaleMode === "log") {
        const minLog = Math.log10(Math.max(1e-4, minLoss || 1e-4));
        const maxLog = Math.log10(Math.max(1e-3, maxLoss));
        const curLog = minLog + yNorm * (maxLog - minLog);
        valText = Math.pow(10, curLog).toFixed(4);
      } else {
        const val = minLoss + yNorm * (maxLoss - minLoss);
        valText = val.toFixed(3);
      }

      ctx.fillStyle = "#64748b";
      ctx.font = "10px monospace";
      ctx.textAlign = "right";
      ctx.fillText(valText, padLeft - 8, y + 3);
    }

    // Vertical grid lines (Epoch)
    const xSteps = 5;
    for (let i = 0; i <= xSteps; i++) {
      const xNorm = i / xSteps;
      const x = padLeft + xNorm * plotW;
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, height - padBottom);
      ctx.stroke();

      const epochVal = Math.round(xNorm * maxEpoch);
      ctx.fillStyle = "#64748b";
      ctx.font = "10px monospace";
      ctx.textAlign = "center";
      ctx.fillText(epochVal.toString(), x, height - padBottom + 16);
    }

    // Axes labels
    ctx.fillStyle = "#94a3b8";
    ctx.font = "11px 'Noto Sans SC', sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("训练轮次 (Epochs)", padLeft + plotW / 2, height - 10);

    ctx.save();
    ctx.translate(16, padTop + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(
      scaleMode === "log" ? "损失函数 log₁₀(Loss)" : "损失值 J(W,b) (MSE / MAE / Cross-Entropy)",
      0,
      0
    );
    ctx.restore();

    // Function to map (epoch, loss) to canvas coordinates
    const mapPoint = (ep: number, val: number) => {
      const cleanVal = isNaN(val) || !isFinite(val) ? maxLoss : Math.max(0, val);
      const x = padLeft + (ep / maxEpoch) * plotW;
      let yNorm = 0;
      if (scaleMode === "log") {
        const minLog = Math.log10(Math.max(1e-4, minLoss || 1e-4));
        const maxLog = Math.log10(Math.max(1e-3, maxLoss));
        const curLog = Math.log10(Math.max(1e-4, cleanVal));
        yNorm = (curLog - minLog) / (maxLog - minLog || 1);
      } else {
        yNorm = (cleanVal - minLoss) / (maxLoss - minLoss || 1);
      }
      yNorm = Math.max(0, Math.min(1, yNorm));
      const y = padTop + (1 - yNorm) * plotH;
      return { x, y };
    };

    // Draw Saved Runs (Comparative Overlay)
    for (const run of savedRuns) {
      if (run.metrics.length < 2) continue;
      ctx.beginPath();
      ctx.strokeStyle = run.color;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);

      run.metrics.forEach((m, idx) => {
        const pt = mapPoint(m.epoch, m.loss);
        if (idx === 0) ctx.moveTo(pt.x, pt.y);
        else ctx.lineTo(pt.x, pt.y);
      });
      ctx.stroke();
      ctx.setLineDash([]); // reset
    }

    // Draw Gradient Norm Curve if enabled
    if (showGradNorm && metricsHistory.length >= 2) {
      ctx.beginPath();
      ctx.strokeStyle = "#f59e0b"; // Amber
      ctx.lineWidth = 1.5;
      metricsHistory.forEach((m, idx) => {
        const pt = mapPoint(m.epoch, m.gradNorm * 0.5);
        if (idx === 0) ctx.moveTo(pt.x, pt.y);
        else ctx.lineTo(pt.x, pt.y);
      });
      ctx.stroke();
    }

    // Draw Current Run Active Loss Curves
    if (metricsHistory.length >= 2) {
      // 1. Shaded area under 80% Train Loss curve
      ctx.beginPath();
      const firstPt = mapPoint(metricsHistory[0].epoch, metricsHistory[0].loss);
      ctx.moveTo(firstPt.x, height - padBottom);
      metricsHistory.forEach((m) => {
        const pt = mapPoint(m.epoch, m.loss);
        ctx.lineTo(pt.x, pt.y);
      });
      const lastPt = mapPoint(
        metricsHistory[metricsHistory.length - 1].epoch,
        metricsHistory[metricsHistory.length - 1].loss
      );
      ctx.lineTo(lastPt.x, height - padBottom);
      ctx.closePath();

      const gradient = ctx.createLinearGradient(0, padTop, 0, height - padBottom);
      gradient.addColorStop(0, "rgba(99, 102, 241, 0.25)");
      gradient.addColorStop(1, "rgba(99, 102, 241, 0.0)");
      ctx.fillStyle = gradient;
      ctx.fill();

      // 2. 80% Train Loss Stroke (Indigo)
      ctx.beginPath();
      ctx.strokeStyle = "#818cf8"; // Indigo 400
      ctx.lineWidth = 2.2;
      ctx.shadowColor = "#6366f1";
      ctx.shadowBlur = 6;

      metricsHistory.forEach((m, idx) => {
        const pt = mapPoint(m.epoch, m.loss);
        if (idx === 0) ctx.moveTo(pt.x, pt.y);
        else ctx.lineTo(pt.x, pt.y);
      });
      ctx.stroke();
      ctx.shadowBlur = 0; // reset

      // 3. 20% Validation Loss Curve (Emerald Green)
      const hasValLoss = metricsHistory.some((m) => m.valLoss !== undefined);
      if (hasValLoss) {
        ctx.beginPath();
        ctx.strokeStyle = "#34d399"; // Emerald 400
        ctx.lineWidth = 2.2;
        ctx.shadowColor = "#10b981";
        ctx.shadowBlur = 6;

        metricsHistory.forEach((m, idx) => {
          const vLoss = m.valLoss !== undefined ? m.valLoss : m.loss;
          const pt = mapPoint(m.epoch, vLoss);
          if (idx === 0) ctx.moveTo(pt.x, pt.y);
          else ctx.lineTo(pt.x, pt.y);
        });
        ctx.stroke();
        ctx.shadowBlur = 0;

        // Current Validation Point Pulsing Indicator
        const lastValLossVal =
          metricsHistory[metricsHistory.length - 1].valLoss ??
          metricsHistory[metricsHistory.length - 1].loss;
        const curValPt = mapPoint(
          metricsHistory[metricsHistory.length - 1].epoch,
          lastValLossVal
        );
        ctx.beginPath();
        ctx.arc(curValPt.x, curValPt.y, 4, 0, Math.PI * 2);
        ctx.fillStyle = "#34d399";
        ctx.shadowColor = "#34d399";
        ctx.shadowBlur = 8;
        ctx.fill();
        ctx.shadowBlur = 0;
      }

      // Current Train Point Pulsing Indicator
      const curPt = mapPoint(
        metricsHistory[metricsHistory.length - 1].epoch,
        metricsHistory[metricsHistory.length - 1].loss
      );
      ctx.beginPath();
      ctx.arc(curPt.x, curPt.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = "#818cf8";
      ctx.shadowColor = "#818cf8";
      ctx.shadowBlur = 8;
      ctx.fill();
      ctx.shadowBlur = 0;
    }
  }, [metricsHistory, currentEpoch, currentLoss, valLoss, savedRuns, scaleMode, showGradNorm]);

  // Derived Statistics
  const initialLoss = metricsHistory[0]?.loss ?? 0;
  const latestTrainLoss = metricsHistory[metricsHistory.length - 1]?.loss ?? currentLoss;
  const latestValLoss =
    metricsHistory[metricsHistory.length - 1]?.valLoss ?? valLoss ?? latestTrainLoss;
  const bestLoss =
    metricsHistory.length > 0
      ? Math.min(...metricsHistory.map((m) => (isNaN(m.loss) ? 999 : m.loss)))
      : 0;
  const lossDelta = initialLoss - latestTrainLoss;
  const convergenceRate =
    currentEpoch > 0 ? ((lossDelta / (currentEpoch || 1)) * 100).toFixed(4) : "0.0000";

  // Generalization Gap |L_val - L_train|
  const genGap = Math.abs(latestValLoss - latestTrainLoss);

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-3.5 shadow-sm space-y-3">
      {/* Header with Switchers & Legend */}
      <div className="flex flex-wrap items-center justify-between gap-2 pb-2.5 border-b border-slate-200">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-blue-600" />
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-xs sm:text-sm font-bold text-slate-900">
                训练与验证损失演化轨迹 (Train vs Val Loss)
              </h3>
              <span className="text-[10px] px-1.5 py-0.2 rounded bg-slate-100 text-slate-600 font-mono font-semibold border border-slate-200">
                80% 训练 / 20% 验证
              </span>
            </div>
            <p className="text-[10px] text-slate-500 font-medium">
              实时监测 80% 训练集反向传播收敛与 20% 验证集未见样本泛化误差
            </p>
          </div>
        </div>

        {/* Controls Slice */}
        <div className="flex items-center flex-wrap gap-1.5 text-xs">
          {/* Linear / Log scale */}
          <div className="flex items-center bg-slate-100 p-0.5 rounded border border-slate-200">
            <button
              onClick={() => setScaleMode("linear")}
              className={`px-2 py-0.5 rounded text-xs font-semibold transition cursor-pointer ${
                scaleMode === "linear" ? "bg-white text-blue-700 shadow-sm" : "text-slate-600 hover:text-slate-900"
              }`}
            >
              线性刻度
            </button>
            <button
              onClick={() => setScaleMode("log")}
              className={`px-2 py-0.5 rounded text-xs font-semibold transition cursor-pointer ${
                scaleMode === "log" ? "bg-white text-blue-700 shadow-sm" : "text-slate-600 hover:text-slate-900"
              }`}
            >
              对数 (Log)
            </button>
          </div>

          {/* Toggle Grad Norm */}
          <button
            onClick={() => setShowGradNorm(!showGradNorm)}
            className={`px-2 py-0.5 rounded border text-xs font-semibold transition cursor-pointer ${
              showGradNorm
                ? "bg-amber-50 border-amber-300 text-amber-700"
                : "bg-white border-slate-300 text-slate-700 hover:bg-slate-50"
            }`}
          >
            {showGradNorm ? "✓ 梯度范数" : "+ 梯度范数"}
          </button>

          {/* Snapshot Comparison Button */}
          <button
            onClick={handleSaveRun}
            disabled={metricsHistory.length < 5}
            className="flex items-center gap-1 px-2.5 py-0.5 bg-white hover:bg-slate-50 border border-slate-300 text-slate-700 disabled:opacity-40 rounded text-xs font-medium transition cursor-pointer"
            title="保存当前轨迹用于与新参数对比"
          >
            <BookmarkPlus className="w-3 h-3 text-blue-600" />
            <span>保存快照</span>
          </button>

          {savedRuns.length > 0 && (
            <button
              onClick={handleClearSavedRuns}
              className="p-1 text-slate-400 hover:text-rose-600 rounded hover:bg-slate-100 transition cursor-pointer"
              title="清除保存的历史快照"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Main Canvas Chart with Top-Right Floating Legend */}
      <div className="relative w-full h-[260px] bg-slate-900 rounded-lg overflow-hidden border border-slate-800">
        <canvas ref={canvasRef} width={760} height={260} className="w-full h-full object-contain" />

        {/* Legend Chips */}
        <div className="absolute top-2.5 right-3 flex items-center gap-3 bg-slate-950/80 px-2.5 py-1 rounded-md border border-slate-800 text-[11px] font-mono backdrop-blur-xs">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-indigo-400 shadow-xs" />
            <span className="text-slate-300">0.8 训练损失</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 shadow-xs" />
            <span className="text-slate-300">0.2 验证损失</span>
          </div>
          {showGradNorm && (
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-amber-400" />
              <span className="text-slate-300">梯度范数</span>
            </div>
          )}
        </div>
      </div>

      {/* Saved Runs Badge Slices */}
      {savedRuns.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5 text-xs">
          <span className="text-[11px] text-slate-400 font-bold uppercase tracking-wider">对比快照:</span>
          {savedRuns.map((r) => (
            <div
              key={r.id}
              className="flex items-center gap-1 px-2 py-0.5 rounded border text-[10px] font-mono bg-slate-50"
              style={{ borderColor: r.color, color: r.color }}
            >
              <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: r.color }} />
              <span>{r.name}</span>
            </div>
          ))}
        </div>
      )}

      {/* 4 Metric Stats Summary Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono text-xs">
        <div className="bg-slate-50 border border-slate-200 rounded p-2">
          <span className="text-slate-400 block text-[9px] font-bold uppercase tracking-wider font-sans">0.8 训练损失 (Train Loss):</span>
          <span className="text-indigo-600 font-bold text-xs">
            {latestTrainLoss ? latestTrainLoss.toFixed(5) : "0.00000"}
          </span>
        </div>
        <div className="bg-slate-50 border border-slate-200 rounded p-2">
          <span className="text-slate-400 block text-[9px] font-bold uppercase tracking-wider font-sans">0.2 验证损失 (Val Loss):</span>
          <span className="text-emerald-600 font-bold text-xs">
            {latestValLoss ? latestValLoss.toFixed(5) : "0.00000"}
          </span>
        </div>
        <div className="bg-slate-50 border border-slate-200 rounded p-2">
          <span className="text-slate-400 block text-[9px] font-bold uppercase tracking-wider font-sans">泛化差距 |L_val - L_train|:</span>
          <span
            className={`font-bold text-xs ${
              genGap < 0.05
                ? "text-emerald-600"
                : genGap < 0.15
                ? "text-amber-600"
                : "text-rose-600"
            }`}
          >
            {genGap.toFixed(5)}
          </span>
        </div>
        <div className="bg-slate-50 border border-slate-200 rounded p-2">
          <span className="text-slate-400 block text-[9px] font-bold uppercase tracking-wider font-sans">下降速率 ΔL/Epoch:</span>
          <span className="text-blue-700 font-bold text-xs">{convergenceRate}</span>
        </div>
      </div>
    </div>
  );
};
