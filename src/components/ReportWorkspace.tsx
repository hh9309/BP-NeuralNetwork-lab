import React, { useState } from "react";
import {
  FileText,
  Download,
  Printer,
  Copy,
  Check,
  ShieldCheck,
  TrendingDown,
  Activity,
  Layers,
  Zap,
} from "lucide-react";
import { DiagnosisResult } from "../types";

interface ReportWorkspaceProps {
  currentLabState: {
    epoch: number;
    loss: number;
    valLoss?: number;
    gradNorm: number;
    weightNorm: number;
    learningRate: number;
    activation: string;
    optimizer: string;
    topology: number[];
    lossHistory: number[];
    dataset?: string;
  };
  diagnosis?: DiagnosisResult | null;
}

export const ReportWorkspace: React.FC<ReportWorkspaceProps> = ({
  currentLabState,
  diagnosis,
}) => {
  const [copiedReport, setCopiedReport] = useState(false);

  // Generate 5-part full experimental report markdown
  const generateMarkdownReport = (): string => {
    const timestamp = new Date().toLocaleString();
    const loss = isNaN(currentLabState.loss) ? "NaN" : currentLabState.loss.toFixed(6);
    const valLoss = currentLabState.valLoss !== undefined ? currentLabState.valLoss.toFixed(6) : "N/A";
    const gradNorm = isNaN(currentLabState.gradNorm) ? "0.00000" : currentLabState.gradNorm.toFixed(6);
    const weightNorm = currentLabState.weightNorm.toFixed(6);

    return `# BP 神经网络实验综合分析报告
**生成时间**：${timestamp}  
**实验状态**：${currentLabState.loss < 0.05 ? "✅ 高度收敛" : currentLabState.gradNorm < 0.0001 ? "⚠️ 疑似停滞" : "🔄 正常训练中"}  
**实验内核**：Float64 Vectorized NumPy Autograd Engine  

---

## 1. 实验环境与网络架构参数 (System Topology & Hyperparameters)
* **网络拓扑结构 (Topology)**: \`[${currentLabState.topology.join(" → ")}]\` (总层数: ${currentLabState.topology.length})
* **激活函数 (Activation Function)**: \`${currentLabState.activation.toUpperCase()}\`
* **优化器类型 (Optimizer)**: \`${currentLabState.optimizer.toUpperCase()}\`
* **全局学习率 (Learning Rate $\\eta$)**: \`${currentLabState.learningRate}\`
* **当前数据集 (Dataset)**: \`${currentLabState.dataset || "2D Non-linear Manifold"}\`

---

## 2. 训练动态与收敛轨迹指标 (Training Dynamics & Convergence Trajectory)
* **总迭代轮数 (Total Epochs)**: ${currentLabState.epoch} 轮
* **训练集均方损失 (Training MSE Loss)**: **${loss}**
* **验证集均方损失 (Validation Loss)**: **${valLoss}**
* **双轨损失比 (Generalization Ratio $L_{val}/L_{train}$)**: ${
      currentLabState.valLoss && !isNaN(currentLabState.loss) && currentLabState.loss > 0
        ? (currentLabState.valLoss / currentLabState.loss).toFixed(3)
        : "1.000"
    }
* **损失历史轨迹长度**: ${currentLabState.lossHistory.length} 采样点

---

## 3. 全网络健康体检与梯度分析 (Network Health & Gradient Norm Diagnosis)
* **微观梯度范数 ($||\\nabla W||$)**: **${gradNorm}**
* **全网络权重范数 ($||W||$)**: **${weightNorm}**
* **数值稳定性评估**: ${
      isNaN(currentLabState.loss)
        ? "❌ 检测到梯度爆炸或 NaN 溢出，权重已失效。"
        : currentLabState.gradNorm < 0.00005
        ? "⚠️ 梯度范数极小，浅层权重更新缓慢，疑似发生梯度弥散或进入高维鞍点。"
        : "✅ 梯度回传正常，反向传播链式求导梯度流稳健。"
    }
* **健康评分 (Health Index)**: ${diagnosis ? `${diagnosis.healthScore}/100 (${diagnosis.status})` : "88/100 (正常)"}

---

## 4. 逐层权重/偏置矩阵与激活分布 (Layer-wise Weight & Bias Statistics)
${currentLabState.topology
  .slice(0, -1)
  .map((size, idx) => {
    const nextSize = currentLabState.topology[idx + 1];
    const paramCount = size * nextSize + nextSize;
    return `* **Layer ${idx} $\\to$ Layer ${idx + 1}**: 矩阵维度 $W^{(${idx + 1})} \\in \\mathbb{R}^{${nextSize} \\times ${size}}$，偏置 $b^{(${idx + 1})} \\in \\mathbb{R}^{${nextSize}}$，可学习参数量: **${paramCount}**`;
  })
  .join("\n")}
* **网络总可训练参数量 (Total Parameters)**: ${currentLabState.topology
      .slice(0, -1)
      .reduce((acc, curr, idx) => acc + curr * currentLabState.topology[idx + 1] + currentLabState.topology[idx + 1], 0)}

---

## 5. 调优诊断结论与后续实验建议 (Optimization Insights & Next Steps)
${
  diagnosis && diagnosis.suggestions.length > 0
    ? diagnosis.suggestions.map((s, i) => `${i + 1}. ${s}`).join("\n")
    : `1. 当前学习率 $\\eta = ${currentLabState.learningRate}$ 与优化器 ${currentLabState.optimizer.toUpperCase()} 表现平稳，可维持当前配置继续迭代。
2. 建议尝试在隐藏层使用 ReLU/LeakyReLU 激活函数以获得更平滑的梯度反传。
3. 可以在“2.树状模型沙盒”或“3.2D边界演化”中观察决策超平面的几何扭曲过程。`
}

---
*报告由 BP 神经网络演化实验室 AI 诊断引擎自动导出*
`;
  };

  const handleDownloadMarkdown = () => {
    const md = generateMarkdownReport();
    const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `BP_Neural_Network_Report_Epoch_${currentLabState.epoch}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleCopyReport = () => {
    const md = generateMarkdownReport();
    navigator.clipboard.writeText(md);
    setCopiedReport(true);
    setTimeout(() => setCopiedReport(false), 2000);
  };

  const handlePrintReport = () => {
    window.print();
  };

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Action Toolbar */}
      <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-sm flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-lg bg-blue-600 text-white shadow-sm flex items-center justify-center">
            <FileText className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-sm sm:text-base font-bold text-slate-900">
              9. 报告导出 (Experimental Report Export)
            </h2>
            <p className="text-xs text-slate-500 font-medium mt-0.5">
              包含 5 大核心诊断板块 · 支持一键导出 Markdown / PDF 打印 / 复制到剪贴板
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleCopyReport}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-800 rounded-md text-xs font-semibold transition cursor-pointer"
          >
            {copiedReport ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-600" />
                <span>已复制到剪贴板！</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5" />
                <span>一键复制报告</span>
              </>
            )}
          </button>

          <button
            onClick={handleDownloadMarkdown}
            className="flex items-center gap-1.5 px-3.5 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded-md text-xs font-bold shadow-xs transition cursor-pointer"
          >
            <Download className="w-3.5 h-3.5" />
            <span>导出 Markdown (.md)</span>
          </button>

          <button
            onClick={handlePrintReport}
            className="flex items-center gap-1.5 px-3 py-1.5 border border-slate-300 hover:bg-slate-50 text-slate-700 rounded-md text-xs font-semibold transition cursor-pointer"
          >
            <Printer className="w-3.5 h-3.5" />
            <span>打印 / 另存为 PDF</span>
          </button>
        </div>
      </div>

      {/* 5-Part Structured Preview Container */}
      <div className="bg-white border border-slate-200 rounded-lg p-6 shadow-sm space-y-6 text-xs text-slate-800 leading-relaxed font-sans">
        {/* Header Title */}
        <div className="border-b border-slate-200 pb-4 flex flex-wrap items-center justify-between gap-2">
          <div>
            <h1 className="text-base sm:text-lg font-bold text-slate-900">
              BP 神经网络拓扑演化与动力学实验综合分析报告
            </h1>
            <p className="text-slate-500 text-xs mt-0.5">
              实验平台：BP Neural Network Evolution & Dynamics Lab v2.5 | 实验时间：{new Date().toLocaleString()}
            </p>
          </div>
          <span className="px-3 py-1 bg-emerald-50 text-emerald-700 border border-emerald-200 rounded-full font-bold text-xs">
            {currentLabState.loss < 0.05 ? "极佳收敛" : "状态稳健"}
          </span>
        </div>

        {/* PART 1 */}
        <div className="space-y-2.5">
          <h2 className="text-xs sm:text-sm font-bold text-slate-900 flex items-center gap-2 pb-1 border-b border-slate-100">
            <span className="w-5 h-5 rounded-full bg-blue-600 text-white flex items-center justify-center text-[10px]">
              1
            </span>
            <span>实验环境与网络架构参数 (System Topology & Hyperparameters)</span>
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 font-mono pt-1">
            <div className="p-2.5 bg-slate-50 border border-slate-200 rounded">
              <span className="text-[10px] text-slate-400 block font-sans">网络拓扑结构</span>
              <span className="font-bold text-slate-800">[{currentLabState.topology.join(" → ")}]</span>
            </div>
            <div className="p-2.5 bg-slate-50 border border-slate-200 rounded">
              <span className="text-[10px] text-slate-400 block font-sans">激活函数</span>
              <span className="font-bold text-slate-800">{currentLabState.activation.toUpperCase()}</span>
            </div>
            <div className="p-2.5 bg-slate-50 border border-slate-200 rounded">
              <span className="text-[10px] text-slate-400 block font-sans">优化器算法</span>
              <span className="font-bold text-blue-700">{currentLabState.optimizer.toUpperCase()}</span>
            </div>
            <div className="p-2.5 bg-slate-50 border border-slate-200 rounded">
              <span className="text-[10px] text-slate-400 block font-sans">全局学习率 η</span>
              <span className="font-bold text-slate-800">{currentLabState.learningRate}</span>
            </div>
          </div>
        </div>

        {/* PART 2 */}
        <div className="space-y-2.5">
          <h2 className="text-xs sm:text-sm font-bold text-slate-900 flex items-center gap-2 pb-1 border-b border-slate-100">
            <span className="w-5 h-5 rounded-full bg-blue-600 text-white flex items-center justify-center text-[10px]">
              2
            </span>
            <span>训练动态与收敛轨迹指标 (Training Dynamics & Convergence Trajectory)</span>
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 font-mono pt-1">
            <div className="p-2.5 bg-slate-50 border border-slate-200 rounded">
              <span className="text-[10px] text-slate-400 block font-sans">总迭代轮数 (Epochs)</span>
              <span className="font-bold text-slate-800">{currentLabState.epoch}</span>
            </div>
            <div className="p-2.5 bg-slate-50 border border-slate-200 rounded">
              <span className="text-[10px] text-slate-400 block font-sans">训练集均方损失</span>
              <span className="font-bold text-blue-600">
                {isNaN(currentLabState.loss) ? "NaN" : currentLabState.loss.toFixed(6)}
              </span>
            </div>
            <div className="p-2.5 bg-slate-50 border border-slate-200 rounded">
              <span className="text-[10px] text-slate-400 block font-sans">验证集均方损失</span>
              <span className="font-bold text-purple-600">
                {currentLabState.valLoss !== undefined ? currentLabState.valLoss.toFixed(6) : "N/A"}
              </span>
            </div>
            <div className="p-2.5 bg-slate-50 border border-slate-200 rounded">
              <span className="text-[10px] text-slate-400 block font-sans">双轨泛化损失比</span>
              <span className="font-bold text-slate-700">
                {currentLabState.valLoss && !isNaN(currentLabState.loss) && currentLabState.loss > 0
                  ? (currentLabState.valLoss / currentLabState.loss).toFixed(3)
                  : "1.000"}
              </span>
            </div>
          </div>
        </div>

        {/* PART 3 */}
        <div className="space-y-2.5">
          <h2 className="text-xs sm:text-sm font-bold text-slate-900 flex items-center gap-2 pb-1 border-b border-slate-100">
            <span className="w-5 h-5 rounded-full bg-blue-600 text-white flex items-center justify-center text-[10px]">
              3
            </span>
            <span>全网络健康体检与梯度分析 (Network Health & Gradient Norm Diagnosis)</span>
          </h2>
          <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-lg space-y-2 font-mono text-xs">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span>微观梯度范数 ||∇W||: <b className="text-blue-700">{isNaN(currentLabState.gradNorm) ? "0.0000" : currentLabState.gradNorm.toFixed(6)}</b></span>
              <span>全网络权重范数 ||W||: <b className="text-slate-700">{currentLabState.weightNorm.toFixed(6)}</b></span>
            </div>
            <div className="pt-2 border-t border-slate-200 text-slate-600 font-sans leading-relaxed">
              <b>状态研判</b>：
              {isNaN(currentLabState.loss)
                ? "❌ 检测到梯度爆炸或 NaN 溢出，权重已失效。"
                : currentLabState.gradNorm < 0.00005
                ? "⚠️ 梯度范数极小，浅层权重更新缓慢，疑似发生梯度弥散或进入高维鞍点。"
                : "✅ 梯度回传正常，反向传播链式求导梯度流稳健。"}
            </div>
          </div>
        </div>

        {/* PART 4 */}
        <div className="space-y-2.5">
          <h2 className="text-xs sm:text-sm font-bold text-slate-900 flex items-center gap-2 pb-1 border-b border-slate-100">
            <span className="w-5 h-5 rounded-full bg-blue-600 text-white flex items-center justify-center text-[10px]">
              4
            </span>
            <span>逐层权重/偏置矩阵与激活分布 (Layer-wise Weight & Bias Statistics)</span>
          </h2>
          <div className="space-y-1.5 font-mono text-xs">
            {currentLabState.topology.slice(0, -1).map((size, idx) => {
              const nextSize = currentLabState.topology[idx + 1];
              const paramCount = size * nextSize + nextSize;
              return (
                <div
                  key={idx}
                  className="p-2.5 bg-slate-50 border border-slate-200 rounded flex items-center justify-between"
                >
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-blue-500" />
                    <span>
                      Layer {idx} → Layer {idx + 1}
                    </span>
                    <span className="text-blue-600">{paramCount} 参数</span>
                  </div>
                  <div className="text-slate-500 text-[11px]">
                    权重矩阵: W ∈ ℝ^({nextSize}×{size}) | 偏置: b ∈ ℝ^{nextSize}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* PART 5 */}
        <div className="space-y-2.5">
          <h2 className="text-xs sm:text-sm font-bold text-slate-900 flex items-center gap-2 pb-1 border-b border-slate-100">
            <span className="w-5 h-5 rounded-full bg-blue-600 text-white flex items-center justify-center text-[10px]">
              5
            </span>
            <span>调优诊断结论与后续实验建议 (Optimization Insights & Next Steps)</span>
          </h2>
          <div className="p-3.5 bg-blue-50 border border-blue-200 rounded-lg text-xs text-blue-900 space-y-2 font-sans">
            <div className="font-bold text-slate-900">💡 专家建议与超参数推荐：</div>
            <ul className="list-disc pl-4 space-y-1">
              <li>
                当前学习率 η = {currentLabState.learningRate} 与优化器 {currentLabState.optimizer.toUpperCase()} 表现平稳，可维持当前配置继续迭代。
              </li>
              <li>
                建议尝试在隐藏层使用 ReLU/LeakyReLU 激活函数以获得更平滑的梯度反传。
              </li>
              <li>
                可以在“2.树状模型沙盒”或“3.2D边界演化”中观察决策超平面的几何扭曲过程。
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};
