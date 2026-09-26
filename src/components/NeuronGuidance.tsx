import React, { useState } from "react";
import {
  Compass,
  BookOpen,
  Zap,
  Layers,
  Sparkles,
  TrendingDown,
  ArrowRight,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
  Play,
  RotateCcw,
} from "lucide-react";
import { BlockMath, InlineMath } from "../lib/katex-helper";
import { ActivationType, OptimizerType } from "../types";

interface NeuronGuidanceProps {
  onLoadExperiment: (
    topology: number[],
    activation: ActivationType,
    lr: number,
    optimizer: OptimizerType,
    caseId?: string
  ) => void;
}

export const NeuronGuidance: React.FC<NeuronGuidanceProps> = ({
  onLoadExperiment,
}) => {
  const [activeTab, setActiveTab] = useState<"levels" | "activations" | "optimizers">("levels");
  const [selectedLevel, setSelectedLevel] = useState<number>(1);

  const levels = [
    {
      id: 1,
      title: "关卡 1：单层感知机局限与 XOR 异或困境",
      subtitle: "理解非线性流形映射与隐藏层激活函数的本质突破",
      concept: "1969年 Minsky 证明单层感知机无法划分异或 (XOR) 问题。本关卡演示通过引入隐藏层非线性神经元，将原本在 2D 空间线性不可分的点映射至高维特征空间，实现完美非线性分割。",
      formula: "y = \\sigma(W_2 \\cdot \\sigma(W_1 x + b_1) + b_2)",
      recommended: {
        topology: [2, 4, 1],
        activation: "sigmoid" as ActivationType,
        lr: 0.8,
        optimizer: "adam" as OptimizerType,
        caseId: "xor_gate",
      },
      insight: "重点观察：单层网络只能画一条直线，而引入 4 个隐藏层节点后，网络学会了用两条相交的弯曲决策边界包裹正样本！",
    },
    {
      id: 2,
      title: "关卡 2：梯度消失 (Vanishing Gradient) 与激活函数革命",
      subtitle: "探究 Sigmoid 导数上限 0.25 导致的链式连乘雪崩",
      concept: "Sigmoid 激活函数的导数最大仅为 0.25 (当 z=0 时)。在多层深层网络中，多层链式法则反向连乘 (0.25)^L 会迅速趋近于 0，导致靠近输入层的浅层权重梯度几乎为零停滞不前。而 ReLU 导数为 1，彻底解决了梯度消失。",
      formula: "\\delta^{(1)} = \\left( \\prod_{k=2}^{L} W^{(k)} \\sigma'(z^{(k)}) \\right) \\delta^{(L)}, \\quad \\sigma'(z) \\le 0.25",
      recommended: {
        topology: [3, 8, 8, 8, 2],
        activation: "sigmoid" as ActivationType,
        lr: 0.1,
        optimizer: "sgd" as OptimizerType,
      },
      insight: "重点观察：在多层深层网络下使用 Sigmoid，浅层误差项 δ 迅速衰减至 10⁻⁵ 以下；切换为 ReLU 后，梯度常态传递，收敛提速 10 倍以上！",
    },
    {
      id: 3,
      title: "关卡 3：优化器进化史 (SGD → Momentum → RMSProp → Adam)",
      subtitle: "一阶二阶动量自适应梯度更新与峡谷震荡平抑",
      concept: "传统 SGD 在狭长峡谷地形中容易剧烈横向震荡且纵向推进缓慢。Momentum 引入物理动量惯性冲过局部极小值；RMSProp 通过历史梯度的二阶矩平滑各维度步长；Adam 融合一阶动量与二阶自适应学习率，兼备快速逃逸鞍点与平滑收敛。",
      formula: "m_t = \\beta_1 m_{t-1} + (1-\\beta_1)g_t, \\quad v_t = \\beta_2 v_{t-1} + (1-\\beta_2)g_t^2, \\quad \\theta_{t+1} = \\theta_t - \\frac{\\eta}{\\sqrt{\\hat{v}_t} + \\epsilon} \\hat{m}_t",
      recommended: {
        topology: [2, 6, 6, 1],
        activation: "relu" as ActivationType,
        lr: 0.05,
        optimizer: "adam" as OptimizerType,
        caseId: "nonlinear_wave",
      },
      insight: "重点观察：切换不同优化器对比损失轨迹。Adam 往往以极平滑的下潜曲线直奔全局最优！",
    },
    {
      id: 4,
      title: "关卡 4：学习率 η 的临界效应 (发散 vs 鞍点停滞)",
      subtitle: "观察学习率过大引起的梯度爆炸、损失振荡与学习率衰减",
      concept: "学习率 η 决定了在梯度曲面上的单步跨度。若 η 过大（如 η=5.0），参数会在最优解两侧剧烈震荡甚至发散至 NaN；若 η 过小（如 η=0.0001），网络陷入平台鞍点停滞不前。",
      formula: "W_{new} = W_{old} - \\eta \\nabla_W J(W)",
      recommended: {
        topology: [3, 5, 2],
        activation: "tanh" as ActivationType,
        lr: 3.5,
        optimizer: "sgd" as OptimizerType,
      },
      insight: "重点观察：当 η 设为 3.5 以上时，损失曲线会出现剧烈的高耸波峰！",
    },
  ];

  const currentLevelData = levels.find((l) => l.id === selectedLevel) || levels[0];

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-sm space-y-3">
      {/* Header Banner */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-200">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-blue-600" />
          <div>
            <h3 className="text-xs sm:text-sm font-bold text-slate-900">
              神经元导引系统与探究式实验关卡
            </h3>
            <p className="text-[10px] text-slate-500 font-medium">
              从异或危机到梯度消失，深入理解反向传播演化机理
            </p>
          </div>
        </div>

        {/* Tab switchers */}
        <div className="flex items-center bg-slate-100 p-0.5 rounded border border-slate-200 text-xs">
          <button
            onClick={() => setActiveTab("levels")}
            className={`px-2.5 py-1 rounded font-semibold transition cursor-pointer ${
              activeTab === "levels"
                ? "bg-white text-blue-700 shadow-sm"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            关卡挑战 (4 大实验)
          </button>
          <button
            onClick={() => setActiveTab("activations")}
            className={`px-2.5 py-1 rounded font-semibold transition cursor-pointer ${
              activeTab === "activations"
                ? "bg-white text-blue-700 shadow-sm"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            激活函数及其导数全景
          </button>
          <button
            onClick={() => setActiveTab("optimizers")}
            className={`px-2.5 py-1 rounded font-semibold transition cursor-pointer ${
              activeTab === "optimizers"
                ? "bg-white text-blue-700 shadow-sm"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            四大优化器机理对比
          </button>
        </div>
      </div>

      {/* Mode 1: Levels */}
      {activeTab === "levels" && (
        <div className="space-y-3">
          {/* Level Selector Pills */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2">
            {levels.map((lvl) => (
              <button
                key={lvl.id}
                onClick={() => setSelectedLevel(lvl.id)}
                className={`p-2.5 rounded border text-left transition cursor-pointer ${
                  selectedLevel === lvl.id
                    ? "bg-blue-50/60 border-blue-600 shadow-sm ring-1 ring-blue-500"
                    : "bg-slate-50 border-slate-200 hover:bg-slate-100/80"
                }`}
              >
                <div className="flex items-center justify-between text-xs mb-0.5">
                  <span
                    className={`font-mono font-bold text-xs ${
                      selectedLevel === lvl.id ? "text-blue-700" : "text-slate-500"
                    }`}
                  >
                    关卡 0{lvl.id}
                  </span>
                  {selectedLevel === lvl.id && (
                    <span className="w-1.5 h-1.5 rounded-full bg-blue-600" />
                  )}
                </div>
                <h4 className="text-xs font-bold text-slate-800 line-clamp-1">
                  {lvl.title.split("：")[1]}
                </h4>
              </button>
            ))}
          </div>

          {/* Active Level Detailed Card */}
          <div className="bg-slate-50 border border-slate-200 rounded p-3.5 space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h4 className="text-xs sm:text-sm font-bold text-slate-900">{currentLevelData.title}</h4>
                <p className="text-xs text-blue-700 font-semibold mt-0.5">{currentLevelData.subtitle}</p>
              </div>

              <button
                onClick={() =>
                  onLoadExperiment(
                    currentLevelData.recommended.topology,
                    currentLevelData.recommended.activation,
                    currentLevelData.recommended.lr,
                    currentLevelData.recommended.optimizer,
                    currentLevelData.recommended.caseId
                  )
                }
                className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded text-xs font-semibold shadow-sm transition cursor-pointer"
              >
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>一键载入本关参数并重演</span>
              </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 text-xs">
              <div className="space-y-1.5">
                <span className="text-slate-600 block font-bold text-[11px]">原理精讲:</span>
                <p className="text-slate-700 leading-relaxed bg-white p-2.5 rounded border border-slate-200 text-[11px]">
                  {currentLevelData.concept}
                </p>
                <div className="bg-amber-50 p-2.5 rounded border border-amber-200 text-amber-900 text-xs font-medium">
                  💡 {currentLevelData.insight}
                </div>
              </div>

              <div className="space-y-1.5">
                <span className="text-slate-600 block font-bold text-[11px]">核心代数公式:</span>
                <div className="bg-white p-2.5 rounded border border-slate-200 text-center text-blue-800 font-mono text-xs">
                  <BlockMath math={currentLevelData.formula} />
                </div>

                <div className="bg-white p-2.5 rounded border border-slate-200 flex items-center justify-between font-mono text-xs">
                  <span className="text-slate-500">预设拓扑超参:</span>
                  <span className="text-emerald-700 font-bold">
                    [{currentLevelData.recommended.topology.join("-")}] /{" "}
                    {currentLevelData.recommended.activation.toUpperCase()} / η=
                    {currentLevelData.recommended.lr} /{" "}
                    {currentLevelData.recommended.optimizer.toUpperCase()}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Mode 2: Activations */}
      {activeTab === "activations" && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs font-mono">
          <div className="bg-slate-50 p-3 rounded border border-slate-200 space-y-1.5">
            <div className="flex justify-between text-blue-700 font-bold">
              <span>Sigmoid 激活算子</span>
              <span className="text-[10px] text-slate-500 font-sans font-normal">导数最大 0.25</span>
            </div>
            <div className="bg-white p-2 rounded text-center text-slate-800 border border-slate-200">
              <InlineMath math="\sigma(z) = \frac{1}{1 + e^{-z}}, \quad \sigma'(z) = \sigma(z)(1 - \sigma(z))" />
            </div>
            <p className="text-slate-600 font-sans text-[11px] leading-relaxed">
              输出范围 (0, 1)，易用于概率估算。缺点是两端导数饱和归零，引发深层网络梯度消失。
            </p>
          </div>

          <div className="bg-slate-50 p-3 rounded border border-slate-200 space-y-1.5">
            <div className="flex justify-between text-emerald-700 font-bold">
              <span>ReLU (修正线性单元)</span>
              <span className="text-[10px] text-slate-500 font-sans font-normal">单侧抑制 / 导数恒为 1</span>
            </div>
            <div className="bg-white p-2 rounded text-center text-slate-800 border border-slate-200">
              <InlineMath math="\text{ReLU}(z) = \max(0, z), \quad \text{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \le 0 \end{cases}" />
            </div>
            <p className="text-slate-600 font-sans text-[11px] leading-relaxed">
              正半轴无饱和区，极大加速收敛且计算极快。现代深度学习的黄金标准激活函数。
            </p>
          </div>

          <div className="bg-slate-50 p-3 rounded border border-slate-200 space-y-1.5">
            <div className="flex justify-between text-amber-700 font-bold">
              <span>Tanh (双曲正切)</span>
              <span className="text-[10px] text-slate-500 font-sans font-normal">零均值 (Zero-Centered)</span>
            </div>
            <div className="bg-white p-2 rounded text-center text-slate-800 border border-slate-200">
              <InlineMath math="\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}, \quad \tanh'(z) = 1 - \tanh^2(z)" />
            </div>
            <p className="text-slate-600 font-sans text-[11px] leading-relaxed">
              输出范围 (-1, 1)，零均值特性优于 Sigmoid，但依然存在两端梯度饱和问题。
            </p>
          </div>

          <div className="bg-slate-50 p-3 rounded border border-slate-200 space-y-1.5">
            <div className="flex justify-between text-purple-700 font-bold">
              <span>Softmax 归一化指数</span>
              <span className="text-[10px] text-slate-500 font-sans font-normal">多分类概率分布</span>
            </div>
            <div className="bg-white p-2 rounded text-center text-slate-800 border border-slate-200">
              <InlineMath math="\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}, \quad \sum p_i = 1" />
            </div>
            <p className="text-slate-600 font-sans text-[11px] leading-relaxed">
              将输出层各节点未归一化对数概率转化为和为 1 的离散多分类概率分布。
            </p>
          </div>
        </div>
      )}

      {/* Mode 3: Optimizers */}
      {activeTab === "optimizers" && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs font-mono">
          <div className="bg-slate-50 p-3 rounded border border-slate-200 space-y-1.5">
            <span className="text-blue-700 font-bold block">1. SGD (随机梯度下降)</span>
            <div className="bg-white p-2 rounded text-center text-slate-800 border border-slate-200">
              <InlineMath math="\theta_{t+1} = \theta_t - \eta \nabla J(\theta_t)" />
            </div>
            <p className="text-slate-600 font-sans text-[11px]">
              基础基石算法，容易在峡谷地形剧烈振荡且遇到鞍点速度极慢。
            </p>
          </div>

          <div className="bg-slate-50 p-3 rounded border border-slate-200 space-y-1.5">
            <span className="text-emerald-700 font-bold block">2. Momentum (动量法)</span>
            <div className="bg-white p-2 rounded text-center text-slate-800 border border-slate-200">
              <InlineMath math="v_t = \gamma v_{t-1} + \eta \nabla J(\theta), \quad \theta \leftarrow \theta - v_t" />
            </div>
            <p className="text-slate-600 font-sans text-[11px]">
              模拟物理滚球惯性，累加历史梯度方向，有效平抑震荡并加速冲过平坦地带。
            </p>
          </div>

          <div className="bg-slate-50 p-3 rounded border border-slate-200 space-y-1.5">
            <span className="text-amber-700 font-bold block">3. RMSProp (均方根传递)</span>
            <div className="bg-white p-2 rounded text-center text-slate-800 border border-slate-200">
              <InlineMath math="s_t = \beta s_{t-1} + (1-\beta)g_t^2, \quad \theta \leftarrow \theta - \frac{\eta}{\sqrt{s_t + \epsilon}} g_t" />
            </div>
            <p className="text-slate-600 font-sans text-[11px]">
              根据梯度平方指数加权移动平均，自适应缩放各维度的有效更新步长。
            </p>
          </div>

          <div className="bg-slate-50 p-3 rounded border border-slate-200 space-y-1.5">
            <span className="text-rose-700 font-bold block">4. Adam (自适应矩估计)</span>
            <div className="bg-white p-2 rounded text-center text-slate-800 border border-slate-200">
              <InlineMath math="m_t = \text{一阶矩(动量)}, \quad v_t = \text{二阶矩(自适应方差)}" />
            </div>
            <p className="text-slate-600 font-sans text-[11px]">
              集大成者，结合 Momentum 与 RMSProp 优势，对初始超参鲁棒性极高。
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
