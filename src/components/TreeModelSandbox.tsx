import React, { useState } from "react";
import {
  GitBranch,
  Layers,
  Plus,
  Trash2,
  Sliders,
  CheckCircle2,
  ArrowRight,
  Sparkles,
  Calculator,
  Compass,
  Cpu,
  Activity,
  BarChart3,
  TrendingDown,
  Info,
} from "lucide-react";
import { ActivationType, LossFunctionType, NeuralTopology } from "../types";
import { BlockMath, InlineMath } from "../lib/katex-helper";
import { NeuralNetwork } from "../lib/neural-engine";

interface TreeModelSandboxProps {
  currentTopology: number[];
  currentActivations: ActivationType[];
  onApplyTopology: (layers: number[], activations: ActivationType[]) => void;
  nn: NeuralNetwork;
  currentLossType?: LossFunctionType;
  onLossTypeChange?: (loss: LossFunctionType, huberDelta?: number) => void;
}

export const TreeModelSandbox: React.FC<TreeModelSandboxProps> = ({
  currentTopology,
  currentActivations,
  onApplyTopology,
  nn,
  currentLossType,
  onLossTypeChange,
}) => {
  const [selectedArchPreset, setSelectedArchPreset] = useState<"multi" | "single" | "deep" | "custom">("multi");
  const [customLayers, setCustomLayers] = useState<number[]>([...currentTopology]);
  const [customActs, setCustomActs] = useState<ActivationType[]>([...currentActivations]);

  // Loss function state in Sandbox
  const [selectedLoss, setSelectedLoss] = useState<LossFunctionType>(
    currentLossType || nn.defaultLossType || "mse"
  );
  const [huberDelta, setHuberDelta] = useState<number>(nn.huberDelta || 1.0);

  // Interactive probe residual for loss/derivative dynamic curve inspection
  const [probeResidual, setProbeResidual] = useState<number>(1.2);
  const [probeActivation, setProbeActivation] = useState<ActivationType>("linear");
  const [probeZ, setProbeZ] = useState<number>(1.0);

  // Synchronize Loss Function Change
  const handleLossChange = (newLoss: LossFunctionType) => {
    setSelectedLoss(newLoss);
    nn.defaultLossType = newLoss;
    if (onLossTypeChange) {
      onLossTypeChange(newLoss, huberDelta);
    }
  };

  const handleHuberDeltaChange = (delta: number) => {
    setHuberDelta(delta);
    nn.huberDelta = delta;
    if (onLossTypeChange) {
      onLossTypeChange(selectedLoss, delta);
    }
  };

  // Preset Configurations
  const handleSelectPreset = (type: "multi" | "single" | "deep") => {
    setSelectedArchPreset(type);
    if (type === "multi") {
      // 3 Inputs -> 4 Hidden 1 -> 4 Hidden 2 -> 2 Outputs (Multi-output / 2-class probability)
      const layers = [3, 4, 4, 2];
      const acts: ActivationType[] = ["tanh", "relu", "softmax"];
      setCustomLayers(layers);
      setCustomActs(acts);
      onApplyTopology(layers, acts);
    } else if (type === "single") {
      // 3 Inputs -> 5 Hidden -> 1 Output (Single continuous/binary output)
      const layers = [3, 5, 1];
      const acts: ActivationType[] = ["relu", "sigmoid"];
      setCustomLayers(layers);
      setCustomActs(acts);
      onApplyTopology(layers, acts);
    } else if (type === "deep") {
      // 3 Inputs -> 6 -> 6 -> 4 -> 2 Outputs (Deep hierarchical feature extraction)
      const layers = [3, 6, 6, 4, 2];
      const acts: ActivationType[] = ["leaky_relu", "relu", "tanh", "softmax"];
      setCustomLayers(layers);
      setCustomActs(acts);
      onApplyTopology(layers, acts);
    }
  };

  const handleUpdateLayerNeurons = (index: number, val: number) => {
    const next = [...customLayers];
    next[index] = Math.max(1, Math.min(16, val));
    setCustomLayers(next);
    setSelectedArchPreset("custom");
    onApplyTopology(next, customActs);
  };

  const handleUpdateLayerAct = (index: number, act: ActivationType) => {
    const next = [...customActs];
    next[index] = act;
    setCustomActs(next);
    setSelectedArchPreset("custom");
    onApplyTopology(customLayers, next);
  };

  const handleAddHiddenLayer = () => {
    if (customLayers.length >= 6) return;
    const nextLayers = [...customLayers];
    const nextActs = [...customActs];
    // Insert before output layer
    const insertIdx = nextLayers.length - 1;
    nextLayers.splice(insertIdx, 0, 4);
    nextActs.splice(insertIdx - 1, 0, "relu");
    setCustomLayers(nextLayers);
    setCustomActs(nextActs);
    setSelectedArchPreset("custom");
    onApplyTopology(nextLayers, nextActs);
  };

  const handleRemoveHiddenLayer = (index: number) => {
    if (customLayers.length <= 2) return;
    const nextLayers = [...customLayers];
    const nextActs = [...customActs];
    nextLayers.splice(index, 1);
    nextActs.splice(index - 1, 1);
    setCustomLayers(nextLayers);
    setCustomActs(nextActs);
    setSelectedArchPreset("custom");
    onApplyTopology(nextLayers, nextActs);
  };

  // Compute Probe Values
  const probeA = NeuralNetwork.activate(probeZ, probeActivation);
  const probeActDeriv = NeuralNetwork.activateDerivative(probeZ, probeA, probeActivation);
  const probeLossVal = NeuralNetwork.computeLossValue(probeResidual, 0, selectedLoss, huberDelta);
  const probeGradVal = NeuralNetwork.computeLossGradient(probeResidual, 0, selectedLoss, huberDelta);
  const probeDelta = probeGradVal * probeActDeriv;

  // Comparison curves points for SVG visualizer (-3 to +3)
  const curvePoints = Array.from({ length: 61 }, (_, i) => {
    const e = -3 + i * 0.1;
    const mseLoss = 0.5 * e * e;
    const mseGrad = e;

    const maeLoss = Math.abs(e);
    const maeGrad = e > 0 ? 1 : e < 0 ? -1 : 0;

    const absE = Math.abs(e);
    const huberLoss = absE <= huberDelta ? 0.5 * e * e : huberDelta * (absE - 0.5 * huberDelta);
    const huberGrad = Math.max(-huberDelta, Math.min(huberDelta, e));

    return { e, mseLoss, mseGrad, maeLoss, maeGrad, huberLoss, huberGrad };
  });

  // SVG dimensions for curve plotter
  const svgWidth = 460;
  const svgHeight = 170;
  const margin = { top: 15, right: 20, bottom: 25, left: 35 };
  const plotW = svgWidth - margin.left - margin.right;
  const plotH = svgHeight - margin.top - margin.bottom;

  const xScale = (e: number) => margin.left + ((e + 3) / 6) * plotW;
  const yScaleLoss = (loss: number) => margin.top + plotH - (Math.min(4.5, loss) / 4.5) * plotH;
  const yScaleGrad = (grad: number) => margin.top + plotH / 2 - (grad / 3.2) * (plotH / 2);

  const makePath = (getY: (p: (typeof curvePoints)[0]) => number) => {
    return curvePoints
      .map((p, idx) => `${idx === 0 ? "M" : "L"} ${xScale(p.e).toFixed(1)} ${getY(p).toFixed(1)}`)
      .join(" ");
  };

  return (
    <div className="space-y-4">
      {/* 1. Overview & Formal Math Banner */}
      <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-3 pb-3 mb-3 border-b border-slate-200">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-blue-600" />
            <div>
              <h2 className="text-xs sm:text-sm font-bold text-slate-900">
                形式化运筹建模与代数映射体系
              </h2>
              <p className="text-[10px] text-slate-500 font-medium">
                矩阵向量化正向映射与多元链式求导公式
              </p>
            </div>
          </div>

          {/* Quick Preset Selector */}
          <div className="flex items-center gap-1 bg-slate-100 p-0.5 rounded border border-slate-200 text-xs">
            <button
              onClick={() => handleSelectPreset("multi")}
              className={`px-2.5 py-0.5 rounded font-semibold transition cursor-pointer ${
                selectedArchPreset === "multi"
                  ? "bg-white text-blue-700 shadow-sm"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              多输出 (3→4→4→2)
            </button>
            <button
              onClick={() => handleSelectPreset("single")}
              className={`px-2.5 py-0.5 rounded font-semibold transition cursor-pointer ${
                selectedArchPreset === "single"
                  ? "bg-white text-blue-700 shadow-sm"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              单输出 (3→5→1)
            </button>
            <button
              onClick={() => handleSelectPreset("deep")}
              className={`px-2.5 py-0.5 rounded font-semibold transition cursor-pointer ${
                selectedArchPreset === "deep"
                  ? "bg-white text-blue-700 shadow-sm"
                  : "text-slate-600 hover:text-slate-900"
              }`}
            >
              深层多分支 (3→6→6→4→2)
            </button>
          </div>
        </div>

        {/* 4 Core Mathematical Pillars Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3 text-xs font-mono">
          {/* Pillar 1 */}
          <div className="bg-slate-50 border border-slate-200 rounded p-2.5 space-y-1.5">
            <div className="flex items-center justify-between text-blue-700 font-bold">
              <span>① 线性加权前向映射</span>
              <span className="text-[10px] text-slate-500 font-sans font-normal">仿射变换</span>
            </div>
            <div className="bg-white p-1.5 rounded text-center text-slate-800 border border-slate-200">
              <InlineMath math="z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}" />
            </div>
            <p className="text-slate-600 text-[10px] font-sans leading-relaxed">
              输入向量 <InlineMath math="a^{(l-1)} \in \mathbb{R}^{d_{l-1}}" /> 通过权重矩阵 <InlineMath math="W^{(l)}" /> 与偏置 <InlineMath math="b^{(l)}" /> 产生净输入。
            </p>
          </div>

          {/* Pillar 2 */}
          <div className="bg-slate-50 border border-slate-200 rounded p-2.5 space-y-1.5">
            <div className="flex items-center justify-between text-emerald-700 font-bold">
              <span>② 非线性流形激活</span>
              <span className="text-[10px] text-slate-500 font-sans font-normal">空间弯曲</span>
            </div>
            <div className="bg-white p-1.5 rounded text-center text-slate-800 border border-slate-200">
              <InlineMath math="a^{(l)} = \sigma(z^{(l)})" />
            </div>
            <p className="text-slate-600 text-[10px] font-sans leading-relaxed">
              通过 Sigmoid / ReLU / Tanh 等激活算子赋予网络拟合任意复杂决策曲面的能力。
            </p>
          </div>

          {/* Pillar 3 */}
          <div className="bg-slate-50 border border-slate-200 rounded p-2.5 space-y-1.5">
            <div className="flex items-center justify-between text-amber-700 font-bold">
              <span>③ 敏感度逆流传递</span>
              <span className="text-[10px] text-slate-500 font-sans font-normal">链式法则</span>
            </div>
            <div className="bg-white p-1.5 rounded text-center text-slate-800 border border-slate-200">
              <InlineMath math="\delta^{(l)} = ((W^{(l+1)})^T \delta^{(l+1)}) \odot \sigma'(z^{(l)})" />
            </div>
            <p className="text-slate-600 text-[10px] font-sans leading-relaxed">
              下一层误差项 <InlineMath math="\delta^{(l+1)}" /> 逆向乘权重转置矩阵后与当前层导数哈达玛积。
            </p>
          </div>

          {/* Pillar 4 */}
          <div className="bg-slate-50 border border-slate-200 rounded p-2.5 space-y-1.5">
            <div className="flex items-center justify-between text-rose-700 font-bold">
              <span>④ 梯度矩阵外积更新</span>
              <span className="text-[10px] text-slate-500 font-sans font-normal">参数演化</span>
            </div>
            <div className="bg-white p-1.5 rounded text-center text-slate-800 border border-slate-200">
              <InlineMath math="\frac{\partial J}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^T" />
            </div>
            <p className="text-slate-600 text-[10px] font-sans leading-relaxed">
              权重梯度等于当前层误差与前一层输出的外积，通过 <InlineMath math="W \leftarrow W - \eta \nabla W" /> 驱动演化。
            </p>
          </div>
        </div>
      </div>

      {/* 2. Loss Function Selection & Analytical Derivative Comparison Module */}
      <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-sm space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-200">
          <div className="flex items-center gap-2">
            <div className="w-2.5 h-2.5 rounded-full bg-indigo-600 animate-pulse" />
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-xs sm:text-sm font-bold text-slate-900">
                  损失函数切换与反向传播求导数学控制台
                </h3>
                <span className="text-[10px] px-2 py-0.5 rounded bg-indigo-50 text-indigo-700 font-mono font-bold border border-indigo-200">
                  Loss Derivative & Backprop Engine
                </span>
              </div>
              <p className="text-[10px] text-slate-500 font-medium">
                在 MSE、MAE 与 Huber Loss 之间灵活切换，动态解析输出敏感度项 <InlineMath math="\delta^{(L)}" /> 的反向求导机制
              </p>
            </div>
          </div>

          {/* Core Dropdown Selector for Loss Function */}
          <div className="flex items-center gap-2">
            <label className="text-xs font-bold text-slate-700 flex items-center gap-1">
              <Sliders className="w-3.5 h-3.5 text-indigo-600" />
              <span>目标损失函数 (Loss Function):</span>
            </label>
            <select
              value={selectedLoss}
              onChange={(e) => handleLossChange(e.target.value as LossFunctionType)}
              className="bg-indigo-50/80 hover:bg-indigo-100/80 border border-indigo-300 text-indigo-900 text-xs font-bold rounded px-3 py-1.5 outline-none cursor-pointer transition shadow-sm"
            >
              <option value="mse">MSE (均方误差, Mean Squared Error)</option>
              <option value="mae">MAE (平均绝对误差, Mean Absolute Error)</option>
              <option value="huber">Huber Loss (鲁棒平滑损失 / Smooth L1)</option>
            </select>
          </div>
        </div>

        {/* Huber Delta Parameter Slider (Appears when Huber Loss is selected) */}
        {selectedLoss === "huber" && (
          <div className="bg-indigo-50/50 border border-indigo-200 rounded p-3 flex flex-wrap items-center justify-between gap-3 text-xs">
            <div className="flex items-center gap-2">
              <span className="w-5 h-5 rounded-full bg-indigo-600 text-white flex items-center justify-center font-bold text-[10px]">
                δ
              </span>
              <div>
                <span className="font-bold text-slate-900">Huber 阈值超参数 (δ Threshold):</span>
                <p className="text-[10px] text-slate-500">
                  残差 <InlineMath math="|e| \le \delta" /> 时退化为二次 MSE；<InlineMath math="|e| > \delta" /> 时过渡为线性 MAE 抑制梯度爆炸。
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <input
                type="range"
                min="0.1"
                max="3.0"
                step="0.1"
                value={huberDelta}
                onChange={(e) => handleHuberDeltaChange(parseFloat(e.target.value))}
                className="w-32 sm:w-48 accent-indigo-600 cursor-pointer"
              />
              <span className="font-mono font-bold text-indigo-700 bg-white border border-indigo-200 px-2 py-0.5 rounded text-xs">
                δ = {huberDelta.toFixed(1)}
              </span>
            </div>
          </div>
        )}

        {/* 3-Column Mathematical Formulation Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 font-mono text-xs">
          {/* Card 1: MSE */}
          <div
            onClick={() => handleLossChange("mse")}
            className={`p-3 rounded-lg border transition-all cursor-pointer ${
              selectedLoss === "mse"
                ? "bg-blue-50/70 border-blue-400 ring-2 ring-blue-500/20 shadow-sm"
                : "bg-slate-50/60 border-slate-200 hover:border-slate-300 opacity-80"
            }`}
          >
            <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-200 text-blue-700 font-bold">
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-blue-600" />
                MSE (均方误差)
              </span>
              {selectedLoss === "mse" && (
                <span className="text-[9px] bg-blue-600 text-white px-1.5 py-0.2 rounded font-sans">
                  当前激活
                </span>
              )}
            </div>

            <div className="space-y-2 text-slate-800 text-[11px]">
              <div>
                <span className="text-slate-400 text-[10px] block font-sans">① 目标损失函数公式:</span>
                <div className="bg-white p-1 rounded border border-slate-200 text-center">
                  <InlineMath math="J = \frac{1}{2} (\hat{y} - y)^2" />
                </div>
              </div>

              <div>
                <span className="text-slate-400 text-[10px] block font-sans">② 对输出预测偏导:</span>
                <div className="bg-white p-1 rounded border border-slate-200 text-center text-blue-700 font-bold">
                  <InlineMath math="\frac{\partial J}{\partial \hat{y}} = \hat{y} - y = e" />
                </div>
              </div>

              <div>
                <span className="text-slate-400 text-[10px] block font-sans">③ 输出层敏感度误差项 (Delta):</span>
                <div className="bg-white p-1 rounded border border-slate-200 text-center text-emerald-700 font-bold">
                  <InlineMath math="\delta^{(L)} = (\hat{y} - y) \cdot \sigma'(z^{(L)})" />
                </div>
              </div>

              <div className="pt-1 text-[10px] font-sans text-slate-600 leading-relaxed">
                <span className="font-bold text-slate-700">特点:</span> 处处光滑可微，对大残差呈二次方惩罚；但对异常离群点极度敏感。
              </div>
            </div>
          </div>

          {/* Card 2: MAE */}
          <div
            onClick={() => handleLossChange("mae")}
            className={`p-3 rounded-lg border transition-all cursor-pointer ${
              selectedLoss === "mae"
                ? "bg-amber-50/70 border-amber-400 ring-2 ring-amber-500/20 shadow-sm"
                : "bg-slate-50/60 border-slate-200 hover:border-slate-300 opacity-80"
            }`}
          >
            <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-200 text-amber-700 font-bold">
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-amber-600" />
                MAE (平均绝对误差 / L1)
              </span>
              {selectedLoss === "mae" && (
                <span className="text-[9px] bg-amber-600 text-white px-1.5 py-0.2 rounded font-sans">
                  当前激活
                </span>
              )}
            </div>

            <div className="space-y-2 text-slate-800 text-[11px]">
              <div>
                <span className="text-slate-400 text-[10px] block font-sans">① 目标损失函数公式:</span>
                <div className="bg-white p-1 rounded border border-slate-200 text-center">
                  <InlineMath math="J = |\hat{y} - y| = |e|" />
                </div>
              </div>

              <div>
                <span className="text-slate-400 text-[10px] block font-sans">② 对输出预测偏导:</span>
                <div className="bg-white p-1 rounded border border-slate-200 text-center text-amber-700 font-bold">
                  <InlineMath math="\frac{\partial J}{\partial \hat{y}} = \text{sgn}(\hat{y} - y)" />
                </div>
              </div>

              <div>
                <span className="text-slate-400 text-[10px] block font-sans">③ 输出层敏感度误差项 (Delta):</span>
                <div className="bg-white p-1 rounded border border-slate-200 text-center text-emerald-700 font-bold">
                  <InlineMath math="\delta^{(L)} = \text{sgn}(\hat{y} - y) \cdot \sigma'(z^{(L)})" />
                </div>
              </div>

              <div className="pt-1 text-[10px] font-sans text-slate-600 leading-relaxed">
                <span className="font-bold text-slate-700">特点:</span> 梯度恒为 ±1，对离群值鲁棒；但零点不可导且零点邻域易发生震荡。
              </div>
            </div>
          </div>

          {/* Card 3: Huber Loss */}
          <div
            onClick={() => handleLossChange("huber")}
            className={`p-3 rounded-lg border transition-all cursor-pointer ${
              selectedLoss === "huber"
                ? "bg-purple-50/70 border-purple-400 ring-2 ring-purple-500/20 shadow-sm"
                : "bg-slate-50/60 border-slate-200 hover:border-slate-300 opacity-80"
            }`}
          >
            <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-200 text-purple-700 font-bold">
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-purple-600" />
                Huber Loss (鲁棒平滑损失)
              </span>
              {selectedLoss === "huber" && (
                <span className="text-[9px] bg-purple-600 text-white px-1.5 py-0.2 rounded font-sans">
                  当前激活
                </span>
              )}
            </div>

            <div className="space-y-2 text-slate-800 text-[11px]">
              <div>
                <span className="text-slate-400 text-[10px] block font-sans">① 目标损失函数分段公式:</span>
                <div className="bg-white p-1 rounded border border-slate-200 text-center text-[10px]">
                  <InlineMath math="J = \begin{cases} \frac{1}{2} e^2 & |e| \le \delta \\ \delta(|e| - \frac{1}{2}\delta) & |e| > \delta \end{cases}" />
                </div>
              </div>

              <div>
                <span className="text-slate-400 text-[10px] block font-sans">② 对输出预测偏导:</span>
                <div className="bg-white p-1 rounded border border-slate-200 text-center text-purple-700 font-bold text-[10px]">
                  <InlineMath math="\frac{\partial J}{\partial \hat{y}} = \text{clip}(e, -\delta, \delta)" />
                </div>
              </div>

              <div>
                <span className="text-slate-400 text-[10px] block font-sans">③ 输出层敏感度误差项 (Delta):</span>
                <div className="bg-white p-1 rounded border border-slate-200 text-center text-emerald-700 font-bold text-[10px]">
                  <InlineMath math="\delta^{(L)} = \text{clip}(e, -\delta, \delta) \cdot \sigma'(z^{(L)})" />
                </div>
              </div>

              <div className="pt-1 text-[10px] font-sans text-slate-600 leading-relaxed">
                <span className="font-bold text-slate-700">特点:</span> 结合 MSE 近零点平滑收敛与 MAE 远端抗离群值优势，梯度自动截断。
              </div>
            </div>
          </div>
        </div>

        {/* 3. Interactive Loss Curve & Derivative Live Plotter */}
        <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-slate-200">
            <div className="flex items-center gap-1.5 text-xs font-bold text-slate-800">
              <Activity className="w-4 h-4 text-indigo-600" />
              <span>实时残差与导数动态演算探针 (Interactive Residual Probe & Gradient Visualizer)</span>
            </div>
            <div className="flex items-center gap-2 text-[11px]">
              <span className="text-slate-500">输出层激活函数 σ:</span>
              <select
                value={probeActivation}
                onChange={(e) => setProbeActivation(e.target.value as ActivationType)}
                className="bg-white border border-slate-300 rounded px-2 py-0.5 text-slate-800 text-xs font-mono font-bold"
              >
                <option value="linear">Linear (恒等, σ'(z)=1.0)</option>
                <option value="sigmoid">Sigmoid (S型, σ'(z)=a(1-a))</option>
                <option value="tanh">Tanh (双曲正切, σ'(z)=1-a²)</option>
                <option value="relu">ReLU (线性整流, z&gt;0?1:0)</option>
                <option value="leaky_relu">LeakyReLU (带泄露ReLU)</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-center">
            {/* SVG Visualizer (7 cols) */}
            <div className="lg:col-span-7 bg-white p-2.5 rounded border border-slate-200 shadow-inner flex flex-col items-center">
              <div className="w-full flex items-center justify-between text-[10px] text-slate-500 font-mono pb-1 border-b border-slate-100">
                <div className="flex items-center gap-3">
                  <span className="flex items-center gap-1">
                    <span className="w-3 h-0.5 bg-blue-600 inline-block" /> 损失曲线 J(e)
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-3 h-0.5 bg-rose-600 inline-block" /> 损失导数 ∂J/∂ŷ
                  </span>
                  <span className="flex items-center gap-1 text-slate-400">
                    <span className="w-3 h-0.5 border-t border-dashed border-slate-400 inline-block" /> MSE/MAE/Huber对比
                  </span>
                </div>
                <span>残差范围 e ∈ [-3, +3]</span>
              </div>

              <svg
                viewBox={`0 0 ${svgWidth} ${svgHeight}`}
                className="w-full h-auto max-h-[170px] select-none"
              >
                {/* Axes and Grid Lines */}
                <line
                  x1={margin.left}
                  y1={margin.top + plotH / 2}
                  x2={margin.left + plotW}
                  y2={margin.top + plotH / 2}
                  stroke="#cbd5e1"
                  strokeWidth="1"
                  strokeDasharray="3 3"
                />
                <line
                  x1={margin.left + plotW / 2}
                  y1={margin.top}
                  x2={margin.left + plotW / 2}
                  y2={margin.top + plotH}
                  stroke="#cbd5e1"
                  strokeWidth="1"
                  strokeDasharray="3 3"
                />

                {/* Reference Curves for Comparison (Dashed) */}
                <path
                  d={makePath((p) => yScaleLoss(p.mseLoss))}
                  fill="none"
                  stroke="#3b82f6"
                  strokeWidth="1"
                  strokeDasharray="2 2"
                  opacity="0.35"
                />
                <path
                  d={makePath((p) => yScaleLoss(p.maeLoss))}
                  fill="none"
                  stroke="#f59e0b"
                  strokeWidth="1"
                  strokeDasharray="2 2"
                  opacity="0.35"
                />
                <path
                  d={makePath((p) => yScaleLoss(p.huberLoss))}
                  fill="none"
                  stroke="#a855f7"
                  strokeWidth="1"
                  strokeDasharray="2 2"
                  opacity="0.35"
                />

                {/* Active Loss Curve (Thick Solid) */}
                <path
                  d={makePath((p) =>
                    yScaleLoss(
                      selectedLoss === "mse"
                        ? p.mseLoss
                        : selectedLoss === "mae"
                        ? p.maeLoss
                        : p.huberLoss
                    )
                  )}
                  fill="none"
                  stroke="#2563eb"
                  strokeWidth="2.5"
                />

                {/* Active Derivative Curve (Rose Solid) */}
                <path
                  d={makePath((p) =>
                    yScaleGrad(
                      selectedLoss === "mse"
                        ? p.mseGrad
                        : selectedLoss === "mae"
                        ? p.maeGrad
                        : p.huberGrad
                    )
                  )}
                  fill="none"
                  stroke="#e11d48"
                  strokeWidth="2"
                />

                {/* Probe Vertical Crosshair Line */}
                <line
                  x1={xScale(probeResidual)}
                  y1={margin.top}
                  x2={xScale(probeResidual)}
                  y2={margin.top + plotH}
                  stroke="#4f46e5"
                  strokeWidth="1.5"
                  strokeDasharray="4 2"
                />

                {/* Probe Points on Active Curves */}
                <circle
                  cx={xScale(probeResidual)}
                  cy={yScaleLoss(probeLossVal)}
                  r="4.5"
                  fill="#2563eb"
                  stroke="#ffffff"
                  strokeWidth="2"
                />
                <circle
                  cx={xScale(probeResidual)}
                  cy={yScaleGrad(probeGradVal)}
                  r="4.5"
                  fill="#e11d48"
                  stroke="#ffffff"
                  strokeWidth="2"
                />

                {/* Coordinate Markers */}
                <text x={margin.left} y={margin.top + plotH + 16} fontSize="9" fill="#64748b" textAnchor="middle">
                  -3.0
                </text>
                <text x={margin.left + plotW / 2} y={margin.top + plotH + 16} fontSize="9" fill="#64748b" textAnchor="middle">
                  0.0 (e=0)
                </text>
                <text x={margin.left + plotW} y={margin.top + plotH + 16} fontSize="9" fill="#64748b" textAnchor="middle">
                  +3.0
                </text>

                <text x={margin.left - 6} y={margin.top + 8} fontSize="9" fill="#2563eb" textAnchor="end">
                  J=4.5
                </text>
                <text x={margin.left - 6} y={margin.top + plotH / 2 + 3} fontSize="9" fill="#64748b" textAnchor="end">
                  0.0
                </text>
              </svg>
            </div>

            {/* Probe Controls & Live Values Panel (5 cols) */}
            <div className="lg:col-span-5 space-y-2 font-mono text-xs">
              {/* Residual Slider */}
              <div className="bg-white p-2 rounded border border-slate-200 space-y-1">
                <div className="flex items-center justify-between text-[11px]">
                  <span className="font-bold text-slate-700">残差探针 e = ŷ - y:</span>
                  <span className="font-bold text-indigo-700 font-mono">
                    {probeResidual > 0 ? `+${probeResidual.toFixed(2)}` : probeResidual.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min="-3.0"
                  max="3.0"
                  step="0.05"
                  value={probeResidual}
                  onChange={(e) => setProbeResidual(parseFloat(e.target.value))}
                  className="w-full accent-indigo-600 cursor-pointer"
                />
              </div>

              {/* Live Numerical Readout Grid */}
              <div className="grid grid-cols-2 gap-1.5 text-[11px]">
                <div className="bg-white p-2 rounded border border-slate-200">
                  <span className="text-slate-400 text-[10px] block font-sans">损失计算值 J(e):</span>
                  <span className="font-bold text-blue-700 text-sm">
                    {probeLossVal.toFixed(4)}
                  </span>
                </div>

                <div className="bg-white p-2 rounded border border-slate-200">
                  <span className="text-slate-400 text-[10px] block font-sans">输出损失偏导 ∂J/∂ŷ:</span>
                  <span className="font-bold text-rose-700 text-sm">
                    {probeGradVal > 0 ? `+${probeGradVal.toFixed(4)}` : probeGradVal.toFixed(4)}
                  </span>
                </div>

                <div className="bg-white p-2 rounded border border-slate-200">
                  <span className="text-slate-400 text-[10px] block font-sans">激活导数 σ'(z):</span>
                  <span className="font-bold text-emerald-700 text-sm">
                    {probeActDeriv.toFixed(4)}
                  </span>
                </div>

                <div className="bg-indigo-50/80 p-2 rounded border border-indigo-200">
                  <span className="text-indigo-900 text-[10px] block font-sans font-bold">
                    反向敏感度 δ^(L):
                  </span>
                  <span className="font-bold text-indigo-700 text-sm">
                    {probeDelta > 0 ? `+${probeDelta.toFixed(4)}` : probeDelta.toFixed(4)}
                  </span>
                </div>
              </div>

              <div className="bg-white p-2 rounded border border-slate-200 text-[10px] font-sans text-slate-500">
                <span className="font-bold text-slate-700">链式法则反向求导关系:</span>
                <p className="font-mono text-slate-700 mt-0.5">
                  δ^(L) = (∂J/∂ŷ) · σ'(z) = ({probeGradVal.toFixed(3)}) · ({probeActDeriv.toFixed(3)}) ={" "}
                  <strong className="text-indigo-700">{probeDelta.toFixed(4)}</strong>
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 4. Interactive Tree Topology Builder & Matrix Parameter Calculator */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Left Column: Visual Tree Layer Stack (7 cols) */}
        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-lg p-4 shadow-sm space-y-3">
          <div className="flex items-center justify-between pb-2 border-b border-slate-200">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-blue-600" />
              <h3 className="text-xs sm:text-sm font-bold text-slate-900">
                可交互树状拓扑分叉构建器
              </h3>
            </div>
            <button
              onClick={handleAddHiddenLayer}
              disabled={customLayers.length >= 6}
              className="flex items-center gap-1 px-2.5 py-1 bg-blue-600 hover:bg-blue-700 disabled:opacity-40 text-white text-xs font-semibold rounded shadow-sm transition cursor-pointer"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>添加隐藏层</span>
            </button>
          </div>

          {/* Layer Cards Stack */}
          <div className="space-y-2">
            {customLayers.map((count, idx) => {
              const isInput = idx === 0;
              const isOutput = idx === customLayers.length - 1;
              const act = idx === 0 ? "identity" : customActs[idx - 1] || "sigmoid";

              return (
                <div
                  key={idx}
                  className={`p-2.5 rounded border transition-all ${
                    isInput
                      ? "bg-emerald-50/60 border-emerald-200"
                      : isOutput
                      ? "bg-amber-50/60 border-amber-200"
                      : "bg-slate-50 border-slate-200"
                  }`}
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    {/* Layer Tag & Role */}
                    <div className="flex items-center gap-2">
                      <span
                        className={`w-6 h-6 rounded flex items-center justify-center text-xs font-mono font-bold ${
                          isInput
                            ? "bg-emerald-100 text-emerald-800 border border-emerald-300"
                            : isOutput
                            ? "bg-amber-100 text-amber-800 border border-amber-300"
                            : "bg-blue-100 text-blue-800 border border-blue-300"
                        }`}
                      >
                        L{idx}
                      </span>
                      <div>
                        <div className="text-xs font-bold text-slate-800">
                          {isInput
                            ? "输入特征层 (Input Layer)"
                            : isOutput
                            ? "输出判决层 (Output Layer)"
                            : `隐藏层 #${idx} (Hidden Feature Extraction)`}
                        </div>
                        <span className="text-[10px] text-slate-500 font-mono">
                          张量维度: <InlineMath math={`\\mathbb{R}^{${count}}`} />
                        </span>
                      </div>
                    </div>

                    {/* Neurons Count Stepper */}
                    <div className="flex items-center gap-1.5">
                      <span className="text-[11px] text-slate-500 font-mono">神经元:</span>
                      <div className="flex items-center bg-white border border-slate-300 rounded font-mono text-xs">
                        <button
                          onClick={() => handleUpdateLayerNeurons(idx, count - 1)}
                          disabled={count <= 1}
                          className="px-2 py-0.5 text-slate-600 hover:text-slate-900 disabled:opacity-30 cursor-pointer font-bold"
                        >
                          -
                        </button>
                        <span className="px-2 font-bold text-blue-700 min-w-[20px] text-center">
                          {count}
                        </span>
                        <button
                          onClick={() => handleUpdateLayerNeurons(idx, count + 1)}
                          disabled={count >= 16}
                          className="px-2 py-0.5 text-slate-600 hover:text-slate-900 disabled:opacity-30 cursor-pointer font-bold"
                        >
                          +
                        </button>
                      </div>
                    </div>

                    {/* Activation Function Selection for Hidden/Output */}
                    {!isInput && (
                      <div className="flex items-center gap-1.5">
                        <span className="text-[11px] text-slate-500">激活:</span>
                        <select
                          value={act}
                          onChange={(e) =>
                            handleUpdateLayerAct(idx - 1, e.target.value as ActivationType)
                          }
                          className="bg-white border border-slate-300 text-slate-800 text-xs rounded px-2 py-0.5 outline-none cursor-pointer"
                        >
                          <option value="sigmoid">Sigmoid</option>
                          <option value="tanh">Tanh</option>
                          <option value="relu">ReLU</option>
                          <option value="leaky_relu">LeakyReLU</option>
                          <option value="linear">Linear</option>
                          <option value="softmax">Softmax</option>
                        </select>
                      </div>
                    )}

                    {/* Remove button for hidden layers */}
                    {!isInput && !isOutput && customLayers.length > 2 && (
                      <button
                        onClick={() => handleRemoveHiddenLayer(idx)}
                        className="p-1 text-slate-400 hover:text-rose-600 hover:bg-rose-50 rounded transition cursor-pointer"
                        title="删除该隐藏层"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right Column: Real-time Formal Parameter Matrix Breakdown (5 cols) */}
        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-lg p-4 shadow-sm space-y-3">
          <div className="flex items-center gap-2 pb-2 border-b border-slate-200">
            <div className="w-2 h-2 rounded-full bg-emerald-600" />
            <h3 className="text-xs sm:text-sm font-bold text-slate-900">
              网络参数矩阵维度推演
            </h3>
          </div>

          <div className="space-y-2 font-mono text-xs">
            {customLayers.map((_, l) => {
              if (l === customLayers.length - 1) return null;
              const inDim = customLayers[l];
              const outDim = customLayers[l + 1];
              const act = customActs[l] || "sigmoid";
              const weightCount = inDim * outDim;
              const biasCount = outDim;

              return (
                <div
                  key={l}
                  className="bg-slate-50 border border-slate-200 rounded p-2.5 space-y-1"
                >
                  <div className="flex items-center justify-between text-blue-700 font-bold">
                    <span>第 {l + 1} 变换层 (L_{l} → L_{l + 1})</span>
                    <span className="text-[10px] text-slate-500 font-sans font-normal">
                      {weightCount + biasCount} 参数
                    </span>
                  </div>

                  <div className="text-slate-700 text-[11px] space-y-0.5">
                    <div className="flex justify-between">
                      <span className="text-slate-400">权重矩阵 W^({l + 1}):</span>
                      <span className="text-emerald-700 font-semibold">
                        <InlineMath math={`\\mathbb{R}^{${outDim} \\times ${inDim}}`} /> ({weightCount} 个权重)
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">偏置向量 b^({l + 1}):</span>
                      <span className="text-amber-700 font-semibold">
                        <InlineMath math={`\\mathbb{R}^{${outDim} \\times 1}`} /> ({biasCount} 个偏置)
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">激活算子 σ:</span>
                      <span className="text-blue-700 font-bold uppercase">{act}</span>
                    </div>
                  </div>
                </div>
              );
            })}

            {/* Total Parameters Summary Pill */}
            <div className="bg-blue-50 border border-blue-200 rounded p-2.5 flex items-center justify-between">
              <span className="text-blue-900 font-sans font-semibold text-xs">网络总可学习参数量:</span>
              <span className="text-blue-700 font-bold text-sm">
                {customLayers.reduce((acc, curr, idx) => {
                  if (idx === customLayers.length - 1) return acc;
                  const next = customLayers[idx + 1];
                  return acc + curr * next + next;
                }, 0)}{" "}
                Params
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
