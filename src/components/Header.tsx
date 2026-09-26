import React from "react";
import {
  Play,
  Pause,
  RotateCcw,
  StepForward,
  Sparkles,
  Zap,
  Activity,
  Layers,
  Code2,
  BookOpen,
  GitBranch,
  Grid,
  FileText,
} from "lucide-react";
import { ActivationType, OptimizerType } from "../types";

export type NavTabType =
  | "lab"
  | "tree"
  | "boundary"
  | "cases"
  | "code"
  | "guidance"
  | "ai_assistant"
  | "knowledge_guidance"
  | "report_export";

interface HeaderProps {
  activeTab: NavTabType;
  onTabChange: (tab: NavTabType) => void;
  isPlaying: boolean;
  onTogglePlay: () => void;
  onStepOnce: () => void;
  onReset: () => void;
  epoch: number;
  currentLoss: number;
  valLoss?: number;
  gradNorm?: number;
  accuracy?: number;
  learningRate: number;
  onLearningRateChange: (lr: number) => void;
  activation: ActivationType;
  onActivationChange: (act: ActivationType) => void;
  optimizer: OptimizerType;
  onOptimizerChange: (opt: OptimizerType) => void;
  simSpeed: number;
  onSimSpeedChange: (speed: number) => void;
  onOpenAiModal?: () => void;
  isAiDiagnosing?: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  onTabChange,
  isPlaying,
  onTogglePlay,
  onStepOnce,
  onReset,
  epoch,
  currentLoss,
  valLoss,
  gradNorm = 0,
  accuracy,
  learningRate,
  onLearningRateChange,
  activation,
  onActivationChange,
  optimizer,
  onOptimizerChange,
  simSpeed,
  onSimSpeedChange,
  onOpenAiModal,
  isAiDiagnosing,
}) => {
  const tabs: { id: NavTabType; label: string; icon: React.FC<{ className?: string }> }[] = [
    { id: "lab", label: "1. 核心拓扑实验", icon: Activity },
    { id: "tree", label: "2. 树状模型沙盒", icon: GitBranch },
    { id: "boundary", label: "3. 2D 边界演化", icon: Grid },
    { id: "cases", label: "4. 六大经典案例", icon: Layers },
    { id: "code", label: "5. 代码引擎与沙箱", icon: Code2 },
    { id: "guidance", label: "6. 理论与关卡", icon: BookOpen },
    { id: "ai_assistant", label: "7. AI诊断助手", icon: Sparkles },
    { id: "knowledge_guidance", label: "8. 知识导引", icon: BookOpen },
    { id: "report_export", label: "9. 报告导出", icon: FileText },
  ];

  return (
    <header className="bg-white border-b border-slate-200 shadow-sm sticky top-0 z-40">
      {/* Top Banner */}
      <div className="max-w-7xl mx-auto px-3 sm:px-4 py-2.5 flex flex-wrap items-center justify-between gap-3">
        {/* Title & Brand */}
        <div className="flex items-center gap-2.5">
          <div className="bg-blue-600 p-2 rounded-lg text-white shadow-sm flex items-center justify-center">
            <Zap className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base sm:text-lg font-bold tracking-tight text-slate-900">
                BP 神经网络与反向传播演化实验室
              </h1>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-50 text-blue-700 border border-blue-200 font-mono font-semibold">
                High Density v2.5
              </span>
            </div>
            <p className="text-[10px] text-slate-500 uppercase tracking-widest font-medium">
              BP Neural Network Evolution & Dynamics Lab
            </p>
          </div>
        </div>

        {/* Real-time Metrics High-Density Readout */}
        <div className="flex items-center gap-3 bg-slate-50 border border-slate-200 rounded-lg px-3 py-1.5 font-mono text-xs">
          <div className="flex flex-col items-end border-r pr-3 border-slate-200">
            <span className="text-[9px] text-slate-400 font-bold uppercase tracking-wider">Epoch 迭代</span>
            <span className="text-sm sm:text-base font-bold text-slate-800 leading-none">
              {epoch.toString().padStart(6, "0")}
            </span>
          </div>
          <div className="flex flex-col items-end border-r pr-3 border-slate-200">
            <span className="text-[9px] text-slate-400 font-bold uppercase tracking-wider">
              {valLoss !== undefined ? "Train / Val 损失" : "Loss 损失"}
            </span>
            <div className="flex items-center gap-1.5 leading-none">
              <span
                className={`text-xs sm:text-sm font-bold ${
                  isNaN(currentLoss)
                    ? "text-rose-600"
                    : currentLoss < 0.05
                    ? "text-emerald-600"
                    : "text-blue-600"
                }`}
                title="80% 训练集损失"
              >
                {isNaN(currentLoss) ? "NaN" : currentLoss.toFixed(4)}
              </span>
              {valLoss !== undefined && (
                <>
                  <span className="text-[10px] text-slate-300">/</span>
                  <span
                    className={`text-xs sm:text-sm font-bold ${
                      isNaN(valLoss)
                        ? "text-rose-600"
                        : valLoss < 0.05
                        ? "text-emerald-600"
                        : "text-teal-600"
                    }`}
                    title="20% 验证集损失"
                  >
                    {isNaN(valLoss) ? "NaN" : valLoss.toFixed(4)}
                  </span>
                </>
              )}
            </div>
          </div>
          <div className="flex flex-col items-end border-r pr-3 border-slate-200">
            <span className="text-[9px] text-slate-400 font-bold uppercase tracking-wider">||∇W|| 梯度</span>
            <span className="text-xs sm:text-sm font-semibold text-slate-700 leading-none">
              {isNaN(gradNorm) ? "0.0000" : gradNorm.toFixed(4)}
            </span>
          </div>
          <div className="flex flex-col items-end">
            <span className="text-[9px] text-slate-400 font-bold uppercase tracking-wider">LR 学习率</span>
            <span className="text-xs sm:text-sm font-semibold text-blue-700 leading-none">
              {learningRate.toFixed(3)}
            </span>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-1.5">
          <button
            onClick={onTogglePlay}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-semibold shadow-sm transition-all cursor-pointer ${
              isPlaying
                ? "bg-amber-600 hover:bg-amber-700 text-white"
                : "bg-blue-600 hover:bg-blue-700 text-white"
            }`}
            title={isPlaying ? "暂停迭代训练" : "开始连续训练"}
          >
            {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
            <span>{isPlaying ? "暂停训练" : "开始训练"}</span>
          </button>
          <button
            onClick={onStepOnce}
            disabled={isPlaying}
            className="flex items-center gap-1 px-2.5 py-1.5 bg-white hover:bg-slate-50 border border-slate-300 text-slate-700 disabled:opacity-40 rounded text-xs font-medium transition cursor-pointer"
            title="单步推演前向与反向梯度"
          >
            <StepForward className="w-3.5 h-3.5 text-slate-600" />
            <span>单步</span>
          </button>
          <button
            onClick={onReset}
            className="flex items-center gap-1 px-2.5 py-1.5 bg-white hover:bg-slate-50 border border-slate-300 text-slate-700 rounded text-xs font-medium transition cursor-pointer"
            title="重置网络参数与损失"
          >
            <RotateCcw className="w-3.5 h-3.5 text-slate-600" />
            <span>重置</span>
          </button>
        </div>
      </div>

      {/* Row 2: Navigation Tabs (第 2 行：居中展示) */}
      <div className="border-t border-slate-200 bg-slate-50/90 px-3 sm:px-4 py-1.5 flex items-center justify-center">
        <nav className="flex items-center justify-center gap-1 overflow-x-auto scrollbar-none py-0.5 max-w-full">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => onTabChange(tab.id)}
                className={`flex items-center gap-1.5 px-3 py-1 rounded text-xs transition-colors cursor-pointer whitespace-nowrap ${
                  isActive
                    ? "bg-white text-blue-700 border border-slate-200 shadow-sm font-bold"
                    : "text-slate-600 hover:text-slate-900 hover:bg-white/60 font-medium"
                }`}
              >
                <Icon className={`w-3.5 h-3.5 ${isActive ? "text-blue-600" : "text-slate-400"}`} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Row 3: Hyperparameter Quick Tuning Bar (第 3 行：居中展示) */}
      <div className="border-t border-slate-200/70 bg-slate-100/60 px-3 sm:px-4 py-1.5 flex items-center justify-center">
        <div className="flex items-center justify-center flex-wrap gap-2.5 sm:gap-3 text-xs max-w-full">
          {/* Speed Selector */}
          <div className="flex items-center gap-1 bg-white border border-slate-200 rounded px-2.5 py-0.5 shadow-2xs">
            <span className="text-[11px] text-slate-400 font-medium">倍速:</span>
            <select
              value={simSpeed}
              onChange={(e) => onSimSpeedChange(Number(e.target.value))}
              className="bg-transparent text-slate-800 font-mono font-medium outline-none cursor-pointer text-xs"
            >
              <option value={1}>1.0x</option>
              <option value={2}>2.0x</option>
              <option value={5}>5.0x</option>
              <option value={10}>10x 极速</option>
            </select>
          </div>

          {/* Learning Rate Slider */}
          <div className="flex items-center gap-1.5 bg-white border border-slate-200 rounded px-2.5 py-0.5 shadow-2xs">
            <span className="text-[11px] text-slate-400 font-medium">η 学习率:</span>
            <input
              type="range"
              min="0.001"
              max="1.5"
              step="0.005"
              value={learningRate}
              onChange={(e) => onLearningRateChange(parseFloat(e.target.value))}
              className="w-16 sm:w-20 accent-blue-600 cursor-pointer"
            />
            <span className="font-mono font-bold text-blue-700 w-10 text-right text-xs">
              {learningRate.toFixed(3)}
            </span>
          </div>

          {/* Activation Selector */}
          <div className="flex items-center gap-1 bg-white border border-slate-200 rounded px-2.5 py-0.5 shadow-2xs">
            <span className="text-[11px] text-slate-400 font-medium">激活:</span>
            <select
              value={activation}
              onChange={(e) => onActivationChange(e.target.value as ActivationType)}
              className="bg-transparent text-slate-800 font-medium outline-none cursor-pointer text-xs"
            >
              <option value="sigmoid">Sigmoid (σ)</option>
              <option value="tanh">Tanh (双曲正切)</option>
              <option value="relu">ReLU (线性整流)</option>
              <option value="leaky_relu">LeakyReLU</option>
              <option value="linear">Linear (恒等)</option>
            </select>
          </div>

          {/* Optimizer Selector */}
          <div className="flex items-center gap-1 bg-white border border-slate-200 rounded px-2.5 py-0.5 shadow-2xs">
            <span className="text-[11px] text-slate-400 font-medium">优化器:</span>
            <select
              value={optimizer}
              onChange={(e) => onOptimizerChange(e.target.value as OptimizerType)}
              className="bg-transparent text-slate-800 font-medium outline-none cursor-pointer text-xs"
            >
              <option value="adam">Adam (自适应矩)</option>
              <option value="sgd">SGD (梯度下降)</option>
              <option value="momentum">Momentum (动量)</option>
              <option value="rmsprop">RMSProp</option>
            </select>
          </div>
        </div>
      </div>
    </header>
  );
};
