import React, { useState } from "react";
import {
  Cpu,
  Zap,
  BookOpen,
  GitCommit,
  BarChart2,
  Compass,
  Sliders,
  Flame,
  ShieldCheck,
  TrendingDown,
  ArrowRight,
  Sparkles,
  Info,
  Layers,
  Activity,
} from "lucide-react";
import { BlockMath, InlineMath } from "../lib/katex-helper";
import { ActivationType } from "../types";

export const KnowledgeWorkspace: React.FC = () => {
  // Interactive Artificial Neuron Live Simulation State
  const [x1, setX1] = useState<number>(0.8);
  const [x2, setX2] = useState<number>(-0.5);
  const [x3, setX3] = useState<number>(0.4);
  const [w1, setW1] = useState<number>(1.2);
  const [w2, setW2] = useState<number>(-1.5);
  const [w3, setW3] = useState<number>(0.7);
  const [bias, setBias] = useState<number>(-0.2);
  const [selectedActivation, setSelectedActivation] = useState<ActivationType>("sigmoid");

  // Calculations
  const term1 = x1 * w1;
  const term2 = x2 * w2;
  const term3 = x3 * w3;
  const netInputZ = term1 + term2 + term3 + bias;

  // Activation computation
  const computeActivation = (z: number, act: ActivationType): { val: number; formula: string; desc: string } => {
    switch (act) {
      case "sigmoid":
        return {
          val: 1 / (1 + Math.exp(-Math.max(-50, Math.min(50, z)))),
          formula: "\\sigma(z) = \\frac{1}{1 + e^{-z}}",
          desc: "S 型平滑激活，将净输入压缩至 (0, 1) 开区间，具备良好的概率解释性。",
        };
      case "relu":
        return {
          val: Math.max(0, z),
          formula: "f(z) = \\max(0, z)",
          desc: "单侧抑制激活，在正半区保持单位斜率，避免梯度饱和消失，计算极度高效。",
        };
      case "leaky_relu":
        return {
          val: z > 0 ? z : 0.01 * z,
          formula: "f(z) = \\begin{cases} z & z > 0 \\\\ 0.01z & z \\le 0 \\end{cases}",
          desc: "带泄露的 ReLU，在负半区保留微小梯度 (0.01)，彻底消除神经元死亡问题。",
        };
      case "tanh":
        return {
          val: Math.tanh(z),
          formula: "\\tanh(z) = \\frac{e^z - e^{-z}}{e^z + e^{-z}}",
          desc: "双曲正切函数，零中心化 (-1, 1)，收敛速度通常快于标准 Sigmoid。",
        };
      case "linear":
        return {
          val: z,
          formula: "f(z) = z",
          desc: "恒等恒线性变换，常用于连续值回归任务的最终输出层。",
        };
      default:
        return {
          val: 1 / (1 + Math.exp(-z)),
          formula: "\\sigma(z) = \\frac{1}{1 + e^{-z}}",
          desc: "标准激活函数映射。",
        };
    }
  };

  const actResult = computeActivation(netInputZ, selectedActivation);

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Header Banner */}
      <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-sm flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-lg bg-emerald-600 text-white shadow-sm flex items-center justify-center">
            <BookOpen className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-sm sm:text-base font-bold text-slate-900">
              8. 知识导引 (Neural Network Knowledge Guidance)
            </h2>
            <p className="text-xs text-slate-500 font-medium mt-0.5">
              从生物神经突触到数学人工神经元 · 几何超平面、链式反向传播与通用近似定理深度剖析
            </p>
          </div>
        </div>
        <div className="flex items-center gap-1.5 px-3 py-1 bg-emerald-50 border border-emerald-200 rounded-md text-xs font-semibold text-emerald-700">
          <ShieldCheck className="w-3.5 h-3.5" />
          <span>理论与几何全景解析</span>
        </div>
      </div>

      {/* ── CORE HIGHLIGHT: Interactive Artificial Neuron Architecture & Mechanism Visualizer ── */}
      <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-sm space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-100">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-blue-50 text-blue-700 border border-blue-200">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <span>经典人工神经元模型结构图与运算机制 (McCulloch-Pitts Neuron)</span>
                <span className="px-2 py-0.5 rounded-full text-[10px] bg-blue-100 text-blue-800 font-semibold">
                  交互图解 & 实时运算
                </span>
              </h3>
              <p className="text-xs text-slate-500">
                直观解构「输入信号 → 突触加权 → 线性求和累加 $\Sigma$ → 非线性激活 $\sigma(z)$ → 轴突输出 $a$」的全流程数学机制
              </p>
            </div>
          </div>
        </div>

        {/* 1. Vector Schematic Canvas (High-Clarity Diagram) */}
        <div className="bg-slate-950 rounded-xl p-4 sm:p-6 border border-slate-800 text-slate-100 shadow-inner overflow-x-auto">
          <div className="min-w-[680px] max-w-4xl mx-auto">
            {/* Stage Headers */}
            <div className="grid grid-cols-5 text-center text-xs font-semibold mb-4 text-slate-400">
              <div className="text-emerald-400">① 输入特征 (Inputs)</div>
              <div className="text-indigo-400">② 突触权值 (Weights)</div>
              <div className="text-sky-300">③ 累加求和器 (Sum $\Sigma$)</div>
              <div className="text-amber-400">④ 激活函数 (Activation $f$)</div>
              <div className="text-emerald-300">⑤ 轴突输出 (Output $a$)</div>
            </div>

            {/* Graphical Architecture */}
            <div className="relative flex items-center justify-between py-6 px-4 bg-slate-900/80 rounded-xl border border-slate-800">
              {/* Inputs column */}
              <div className="flex flex-col space-y-4 z-10">
                {/* Input 1 */}
                <div className="flex items-center gap-2 bg-emerald-950/80 border border-emerald-600/60 px-3 py-2 rounded-lg text-xs shadow-sm">
                  <span className="font-mono text-emerald-400 font-bold">x₁</span>
                  <span className="font-mono text-white text-[11px] font-semibold bg-emerald-900/60 px-1.5 py-0.5 rounded">
                    {x1.toFixed(2)}
                  </span>
                </div>

                {/* Input 2 */}
                <div className="flex items-center gap-2 bg-emerald-950/80 border border-emerald-600/60 px-3 py-2 rounded-lg text-xs shadow-sm">
                  <span className="font-mono text-emerald-400 font-bold">x₂</span>
                  <span className="font-mono text-white text-[11px] font-semibold bg-emerald-900/60 px-1.5 py-0.5 rounded">
                    {x2.toFixed(2)}
                  </span>
                </div>

                {/* Input 3 */}
                <div className="flex items-center gap-2 bg-emerald-950/80 border border-emerald-600/60 px-3 py-2 rounded-lg text-xs shadow-sm">
                  <span className="font-mono text-emerald-400 font-bold">x₃</span>
                  <span className="font-mono text-white text-[11px] font-semibold bg-emerald-900/60 px-1.5 py-0.5 rounded">
                    {x3.toFixed(2)}
                  </span>
                </div>

                {/* Bias Node */}
                <div className="flex items-center gap-2 bg-purple-950/80 border border-purple-600/60 px-3 py-1.5 rounded-lg text-xs shadow-sm">
                  <span className="font-mono text-purple-400 font-bold">b (偏置)</span>
                  <span className="font-mono text-white text-[11px] font-semibold bg-purple-900/60 px-1.5 py-0.5 rounded">
                    {bias.toFixed(2)}
                  </span>
                </div>
              </div>

              {/* Connecting Synapse Lines & Weights */}
              <div className="flex flex-col justify-around h-44 z-10 px-2">
                <div className="flex items-center gap-1 text-[11px] font-mono bg-indigo-950/90 border border-indigo-500/60 px-2 py-1 rounded text-indigo-300">
                  <span>× w₁=</span>
                  <span className="font-bold text-white">{w1.toFixed(2)}</span>
                  <span className="text-slate-500">→</span>
                  <span className="text-emerald-300 font-semibold">{term1.toFixed(2)}</span>
                </div>

                <div className="flex items-center gap-1 text-[11px] font-mono bg-indigo-950/90 border border-indigo-500/60 px-2 py-1 rounded text-indigo-300">
                  <span>× w₂=</span>
                  <span className="font-bold text-white">{w2.toFixed(2)}</span>
                  <span className="text-slate-500">→</span>
                  <span className="text-emerald-300 font-semibold">{term2.toFixed(2)}</span>
                </div>

                <div className="flex items-center gap-1 text-[11px] font-mono bg-indigo-950/90 border border-indigo-500/60 px-2 py-1 rounded text-indigo-300">
                  <span>× w₃=</span>
                  <span className="font-bold text-white">{w3.toFixed(2)}</span>
                  <span className="text-slate-500">→</span>
                  <span className="text-emerald-300 font-semibold">{term3.toFixed(2)}</span>
                </div>

                <div className="flex items-center gap-1 text-[11px] font-mono bg-purple-950/90 border border-purple-500/60 px-2 py-0.5 rounded text-purple-300">
                  <span>+ b=</span>
                  <span className="font-bold text-white">{bias.toFixed(2)}</span>
                </div>
              </div>

              {/* The Biological Soma Body (Σ + Activation Function Dual Chamber) */}
              <div className="relative z-10 p-1 bg-gradient-to-r from-blue-600 via-indigo-600 to-amber-500 rounded-2xl shadow-xl">
                <div className="bg-slate-900 rounded-[14px] p-4 flex items-center gap-4">
                  {/* Left Chamber: Linear Summation Sigma */}
                  <div className="flex flex-col items-center justify-center w-24 h-24 rounded-xl bg-blue-950/90 border border-blue-500/50 p-2 text-center">
                    <div className="text-xl font-bold font-serif text-blue-400">Σ</div>
                    <div className="text-[10px] text-slate-400 mt-1">加权求和</div>
                    <div className="text-[11px] font-mono font-bold text-sky-300 mt-0.5">
                      z = {netInputZ.toFixed(2)}
                    </div>
                  </div>

                  {/* Inter-chamber flow arrow */}
                  <div className="text-slate-400 font-mono font-bold text-sm">→ z →</div>

                  {/* Right Chamber: Non-linear Activation Function */}
                  <div className="flex flex-col items-center justify-center w-24 h-24 rounded-xl bg-amber-950/90 border border-amber-500/50 p-2 text-center">
                    <div className="text-lg font-bold font-serif text-amber-400">f(z)</div>
                    <div className="text-[10px] text-slate-400 mt-1 uppercase font-mono">{selectedActivation}</div>
                    <div className="text-[11px] font-mono font-bold text-amber-300 mt-0.5">
                      a = {actResult.val.toFixed(4)}
                    </div>
                  </div>
                </div>
              </div>

              {/* Output Axon Arrow */}
              <div className="flex items-center gap-2 z-10">
                <div className="h-0.5 w-10 bg-gradient-to-r from-amber-500 to-emerald-400 animate-pulse" />
                <div className="flex flex-col items-center bg-emerald-950/90 border border-emerald-500/80 px-4 py-3 rounded-xl text-center shadow-lg">
                  <span className="text-[10px] text-emerald-400 font-bold uppercase tracking-wider">输出激活 a</span>
                  <span className="font-mono text-base font-bold text-white mt-0.5">
                    {actResult.val.toFixed(4)}
                  </span>
                  <span className="text-[9px] text-slate-400 mt-0.5">传递至下一层</span>
                </div>
              </div>
            </div>

            {/* Formula Strip below diagram */}
            <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3 text-xs bg-slate-900 p-3 rounded-lg border border-slate-800">
              <div className="flex items-center gap-2">
                <span className="text-slate-400 shrink-0 font-semibold">1. 净输入公式:</span>
                <div className="text-blue-300 font-mono text-[11px] overflow-x-auto">
                  <InlineMath math={`z = (${w1.toFixed(1)} \\times ${x1.toFixed(1)}) + (${w2.toFixed(1)} \\times ${x2.toFixed(1)}) + (${w3.toFixed(1)} \\times ${x3.toFixed(1)}) + (${bias.toFixed(1)}) = ${netInputZ.toFixed(3)}`} />
                </div>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-slate-400 shrink-0 font-semibold">2. 激活输出:</span>
                <div className="text-amber-300 font-mono text-[11px] overflow-x-auto">
                  <InlineMath math={`a = ${selectedActivation}(${netInputZ.toFixed(3)}) = ${actResult.val.toFixed(4)}`} />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* 2. Interactive Parameter Playground & Activation Switcher */}
        <div className="bg-slate-50 rounded-xl p-4 border border-slate-200 space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
              <Sliders className="w-4 h-4 text-blue-600" />
              人工神经元参数实时推演控制台 (Live Neuron Tuning)
            </span>
            <span className="text-[11px] text-slate-500">
              调节输入特征 $x_i$、权值 $w_i$ 或激活函数，观察神经元输出突触变化
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {/* Input 1 & Weight 1 */}
            <div className="p-2.5 bg-white rounded-lg border border-slate-200 space-y-2">
              <div className="flex justify-between text-xs font-semibold">
                <span className="text-emerald-700">特征输入 x₁:</span>
                <span className="font-mono text-emerald-800">{x1.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="-2.0"
                max="2.0"
                step="0.1"
                value={x1}
                onChange={(e) => setX1(Number(e.target.value))}
                className="w-full accent-emerald-600 cursor-pointer"
              />
              <div className="flex justify-between text-xs font-semibold pt-1 border-t border-slate-100">
                <span className="text-indigo-700">突触权值 w₁:</span>
                <span className="font-mono text-indigo-800">{w1.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="-3.0"
                max="3.0"
                step="0.1"
                value={w1}
                onChange={(e) => setW1(Number(e.target.value))}
                className="w-full accent-indigo-600 cursor-pointer"
              />
            </div>

            {/* Input 2 & Weight 2 */}
            <div className="p-2.5 bg-white rounded-lg border border-slate-200 space-y-2">
              <div className="flex justify-between text-xs font-semibold">
                <span className="text-emerald-700">特征输入 x₂:</span>
                <span className="font-mono text-emerald-800">{x2.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="-2.0"
                max="2.0"
                step="0.1"
                value={x2}
                onChange={(e) => setX2(Number(e.target.value))}
                className="w-full accent-emerald-600 cursor-pointer"
              />
              <div className="flex justify-between text-xs font-semibold pt-1 border-t border-slate-100">
                <span className="text-indigo-700">突触权值 w₂:</span>
                <span className="font-mono text-indigo-800">{w2.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="-3.0"
                max="3.0"
                step="0.1"
                value={w2}
                onChange={(e) => setW2(Number(e.target.value))}
                className="w-full accent-indigo-600 cursor-pointer"
              />
            </div>

            {/* Input 3 & Weight 3 */}
            <div className="p-2.5 bg-white rounded-lg border border-slate-200 space-y-2">
              <div className="flex justify-between text-xs font-semibold">
                <span className="text-emerald-700">特征输入 x₃:</span>
                <span className="font-mono text-emerald-800">{x3.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="-2.0"
                max="2.0"
                step="0.1"
                value={x3}
                onChange={(e) => setX3(Number(e.target.value))}
                className="w-full accent-emerald-600 cursor-pointer"
              />
              <div className="flex justify-between text-xs font-semibold pt-1 border-t border-slate-100">
                <span className="text-indigo-700">突触权值 w₃:</span>
                <span className="font-mono text-indigo-800">{w3.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="-3.0"
                max="3.0"
                step="0.1"
                value={w3}
                onChange={(e) => setW3(Number(e.target.value))}
                className="w-full accent-indigo-600 cursor-pointer"
              />
            </div>

            {/* Bias & Activation Switch */}
            <div className="p-2.5 bg-white rounded-lg border border-slate-200 space-y-2">
              <div className="flex justify-between text-xs font-semibold">
                <span className="text-purple-700">基底偏置 b:</span>
                <span className="font-mono text-purple-800">{bias.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="-2.0"
                max="2.0"
                step="0.1"
                value={bias}
                onChange={(e) => setBias(Number(e.target.value))}
                className="w-full accent-purple-600 cursor-pointer"
              />
              <div className="pt-1 border-t border-slate-100">
                <label className="text-[11px] font-semibold text-slate-600 block mb-1">选择激活函数:</label>
                <select
                  value={selectedActivation}
                  onChange={(e) => setSelectedActivation(e.target.value as ActivationType)}
                  className="w-full px-2 py-1 text-xs bg-slate-50 border border-slate-300 rounded font-semibold text-slate-800 cursor-pointer"
                >
                  <option value="sigmoid">Sigmoid (0 ~ 1)</option>
                  <option value="leaky_relu">Leaky ReLU (带泄露)</option>
                  <option value="relu">ReLU (线性整流)</option>
                  <option value="tanh">Tanh (-1 ~ 1)</option>
                  <option value="linear">Linear (恒等输出)</option>
                </select>
              </div>
            </div>
          </div>

          {/* Activation Detail Note */}
          <div className="p-3 bg-blue-50/60 rounded-lg border border-blue-200 text-xs text-blue-900 flex items-start gap-2.5">
            <Info className="w-4 h-4 text-blue-600 shrink-0 mt-0.5" />
            <div className="leading-relaxed">
              <strong>当前激活函数特性：</strong> {actResult.desc} 
              <span className="ml-2 font-mono bg-white px-2 py-0.5 rounded border border-blue-200 font-semibold text-blue-800">
                <InlineMath math={actResult.formula} />
              </span>
            </div>
          </div>
        </div>

        {/* 3. Detailed Mechanism 5-Stage Breakdown Table */}
        <div className="space-y-2">
          <h4 className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
            <Activity className="w-4 h-4 text-emerald-600" />
            人工神经元运行机制 5 阶段精细化映射
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-5 gap-2.5 text-xs">
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-1">
              <div className="font-bold text-emerald-700">1. 特征信号输入</div>
              <p className="text-[11px] text-slate-600 leading-relaxed">
                对应生物神经元的树突末梢，接收外部样本特征向量 <InlineMath math={"X = [x_1, x_2, \\dots, x_n]^T"} />。
              </p>
            </div>

            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-1">
              <div className="font-bold text-indigo-700">2. 突触权值缩放</div>
              <p className="text-[11px] text-slate-600 leading-relaxed">
                每个突触对应一个权重 <InlineMath math={"w_i"} />，代表连接强度。<InlineMath math={"w_i > 0"} /> 为兴奋性突触，<InlineMath math={"w_i < 0"} /> 为抑制性突触。
              </p>
            </div>

            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-1">
              <div className="font-bold text-sky-700">3. 细胞体代数累加</div>
              <p className="text-[11px] text-slate-600 leading-relaxed">
                神经元核心累加器计算加权净输入：<InlineMath math={"z = \\sum_{i=1}^n w_i x_i + b"} />，基底偏置 <InlineMath math={"b"} /> 决定静息兴奋门槛。
              </p>
            </div>

            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-1">
              <div className="font-bold text-amber-700">4. 非线性激发映射</div>
              <p className="text-[11px] text-slate-600 leading-relaxed">
                通过非线性函数 <InlineMath math={"a = f(z)"} /> 将连续标量映射为输出激活电位，赋予全网突破线性边界的能力。
              </p>
            </div>

            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 space-y-1">
              <div className="font-bold text-emerald-700">5. 轴突脉冲输出</div>
              <p className="text-[11px] text-slate-600 leading-relaxed">
                激活值 <InlineMath math={"a"} /> 作为后续层所有神经元的输入信号，或直接作为整个网络的最终预测概率/回归值。
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Module 1: Biological vs Mathematical Neurons */}
      <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-sm space-y-4">
        <div className="flex items-center gap-2.5 pb-3 border-b border-slate-100">
          <div className="p-2 rounded-lg bg-emerald-50 text-emerald-700 border border-emerald-200">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-900">
              一、生物与数学神经元对照 (Biological vs Artificial Neurons)
            </h3>
            <p className="text-xs text-slate-500">
              解构生物神经系统到人工神经元的几何与代数精确映射
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Biological Component */}
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 space-y-2.5">
            <div className="flex items-center gap-2 text-xs font-bold text-slate-800">
              <span className="w-2 h-2 rounded-full bg-emerald-500" />
              <span>生物神经元 (Biological Neuron)</span>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed">
              人脑包含约 860 亿个生物神经元，彼此通过突触连接构成复杂神经网络：
            </p>
            <ul className="space-y-1.5 text-xs text-slate-600 list-disc pl-4 leading-relaxed">
              <li>
                <b>树突 (Dendrites)</b>：从成千上万个前驱神经元接收电化学输入信号。
              </li>
              <li>
                <b>突触连接强度 (Synaptic Strengths)</b>：化学突触间隙的可塑性决定了信号传递效率（对应学习与记忆）。
              </li>
              <li>
                <b>细胞体 (Soma)</b>：对所有树突输入的离子电位进行空间与时间上的加权代数累加整合。
              </li>
              <li>
                <b>轴突与动作电位 (Axon & Action Potential)</b>：当膜电位超过兴奋阈值（All-or-None 原则）时产生脉冲放电，向后续突触传递。
              </li>
            </ul>
          </div>

          {/* Artificial Component */}
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 space-y-2.5">
            <div className="flex items-center gap-2 text-xs font-bold text-slate-800">
              <span className="w-2 h-2 rounded-full bg-blue-500" />
              <span>人工神经元 (Mathematical MCP Neuron)</span>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed">
              麦卡洛克-皮茨 (McCulloch-Pitts) 模型与现代人工神经元的代数形式：
            </p>
            <div className="p-2.5 bg-white border border-slate-200 rounded font-mono text-xs text-slate-900">
              <BlockMath math={"z = \\sum_{i=1}^n w_i x_i + b = W^T x + b, \\quad a = \\sigma(z)"} />
            </div>
            <ul className="space-y-1.5 text-xs text-slate-600 list-disc pl-4 leading-relaxed">
              <li>
                <b>输入向量 <InlineMath math={"x \\in \\mathbb{R}^n"} /></b>：对应生物树突接收的多维特征信号。
              </li>
              <li>
                <b>权重矩阵 <InlineMath math={"W \\in \\mathbb{R}^n"} /> 与偏置 <InlineMath math={"b"} /></b>：对应突触连接强度与放电静息阈值。
              </li>
              <li>
                <b>激活函数 <InlineMath math={"\\sigma(\\cdot)"} /></b>：提供非线性阈值激发能力（如 ReLU, Sigmoid, Tanh, GELU），赋予网络高维非线性拟合能力。
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* Module 2: Geometric Interpretation */}
      <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-sm space-y-4">
        <div className="flex items-center gap-2.5 pb-3 border-b border-slate-100">
          <div className="p-2 rounded-lg bg-blue-50 text-blue-700 border border-blue-200">
            <Compass className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-900">
              二、人工神经元的几何意义 (Geometric Interpretation)
            </h3>
            <p className="text-xs text-slate-500">
              从超平面划分到高维特征空间流形折叠
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-3.5 space-y-2">
            <span className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
              <Sliders className="w-4 h-4 text-blue-600" />
              1. 决策超平面 (Hyperplane)
            </span>
            <p className="text-xs text-slate-600 leading-relaxed">
              每个隐藏神经元定义了特征空间中的一个超平面 <InlineMath math={"w_1 x_1 + w_2 x_2 + \\dots + b = 0"} />。法向量 <InlineMath math={"W"} /> 决定了倾斜方向，偏置 <InlineMath math={"b"} /> 控制截距平移。
            </p>
          </div>

          <div className="bg-slate-50 border border-slate-200 rounded-lg p-3.5 space-y-2">
            <span className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
              <Flame className="w-4 h-4 text-amber-600" />
              2. 半空间赋权 (Half-space)
            </span>
            <p className="text-xs text-slate-600 leading-relaxed">
              激活函数 <InlineMath math={"\\sigma(z)"} /> 将空间平滑划分为正负两个半空间，形成阶跃或平滑的概率过渡带（如决策边界两旁的渐变区域）。
            </p>
          </div>

          <div className="bg-slate-50 border border-slate-200 rounded-lg p-3.5 space-y-2">
            <span className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
              <Zap className="w-4 h-4 text-emerald-600" />
              3. 高维凸多面体与流形折叠
            </span>
            <p className="text-xs text-slate-600 leading-relaxed">
              多层隐藏层通过仿射变换与非线性折叠，将原本线性不可分的同心圆/双螺旋数据映射至新的隐空间，使其转变为线性可分流形。
            </p>
          </div>
        </div>
      </div>

      {/* Module 3: Key Theory Cards */}
      <div className="bg-white border border-slate-200 rounded-lg p-5 shadow-sm space-y-4">
        <div className="flex items-center gap-2.5 pb-3 border-b border-slate-100">
          <div className="p-2 rounded-lg bg-indigo-50 text-indigo-700 border border-indigo-200">
            <GitCommit className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-900">
              三、反向传播与深度学习核心理论图谱 (Backpropagation & UAT)
            </h3>
            <p className="text-xs text-slate-500">
              多元微积分链式法则、损失函数凸性与通用近似定理
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-800">
                1. 链式法则与误差敏感度递推 (Chain Rule & Delta)
              </span>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed">
              对于损失函数 <InlineMath math={"E"} />，输出层误差敏感度 <InlineMath math={"\\delta^{(L)}"} /> 与隐藏层误差敏感度 <InlineMath math={"\\delta^{(l)}"} /> 的递推关系为：
            </p>
            <div className="p-2.5 bg-white border border-slate-200 rounded font-mono text-xs text-slate-900">
              <BlockMath math={"\\delta_j^{(l)} = \\left( \\sum_{k} \\delta_k^{(l+1)} w_{kj}^{(l+1)} \\right) \\cdot \\sigma'\\left(z_j^{(l)}\\right), \\quad \\frac{\\partial E}{\\partial w_{ji}^{(l)}} = \\delta_j^{(l)} a_i^{(l-1)}"} />
            </div>
            <p className="text-[11px] text-slate-500 leading-relaxed">
              若使用 Sigmoid 激活函数，由于 <InlineMath math={"\\sigma'(z) \\le 0.25"} />，多层连乘会导致浅层梯度呈指数级衰减至 0，即产生<b>梯度消失 (Vanishing Gradient)</b>。
            </p>
          </div>

          <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-800">
                2. 通用近似定理 (Universal Approximation Theorem)
              </span>
            </div>
            <p className="text-xs text-slate-600 leading-relaxed">
              设 <InlineMath math={"\\sigma"} /> 为任意非常数、有界且单调递增的连续激活函数。对于任意紧集 <InlineMath math={"K \\subset \\mathbb{R}^n"} /> 上的连续函数 <InlineMath math={"f(x)"} /> 及任意误差 <InlineMath math={"\\varepsilon > 0"} />，存在有限个神经元数量 <InlineMath math={"m"} />、权重 <InlineMath math={"w_i, v_i"} /> 与偏置 <InlineMath math={"b_i"} />，使得网络：
            </p>
            <div className="p-2.5 bg-white border border-slate-200 rounded font-mono text-xs text-slate-900">
              <BlockMath math={"F(x) = \\sum_{i=1}^m v_i \\sigma(w_i^T x + b_i), \\quad \\sup_{x \\in K} |F(x) - f(x)| < \\varepsilon"} />
            </div>
            <p className="text-[11px] text-slate-500 leading-relaxed">
              该定理证明了<b>单隐藏层前馈神经网络具备以任意精度逼近任意连续非线性函数的能力</b>。
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

