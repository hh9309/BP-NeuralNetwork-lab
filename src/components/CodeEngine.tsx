import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  Play,
  RotateCcw,
  Terminal,
  LineChart,
  Table as TableIcon,
  Check,
  Copy,
  Download,
  AlertCircle,
  FileCode,
  CheckCircle2,
  Trash2,
  Award,
} from "lucide-react";
import { ActivationType, OptimizerType, LossFunctionType, TrainingSample } from "../types";
import { NeuralNetwork } from "../lib/neural-engine";
import {
  generateXORDataset,
  generateCirclesDataset,
  generateMoonsDataset,
  generateSpiralsDataset,
  generateThermalComfortDataset,
  generateLinearIdentityDataset,
  getNonlinearWaveDataset,
  getIrisDataset,
  getHousePriceRegressionDataset,
  getCustomerChurnDataset,
  getMNISTDigitSamples,
  splitTrainTest,
} from "../lib/datasets";

interface CodeEngineProps {
  topology: number[];
  activations: ActivationType[];
  learningRate: number;
  optimizer: OptimizerType;
  lossFunction?: LossFunctionType;
}

interface LogEntry {
  id: string;
  type: "log" | "info" | "success" | "warn" | "error" | "plot";
  message: string;
  timestamp: string;
}

interface PlotPayload {
  type: "decision_boundary" | "loss_curve" | "regression" | "bar_chart";
  title?: string;
  data?: any;
}

interface TablePayload {
  title?: string;
  headers: string[];
  rows: (string | number)[][];
}

// -------------------------------------------------------------
// 四个实验关卡专属精炼脚本 (Four Challenge Levels Dedicated Scripts)
// -------------------------------------------------------------
const FOUR_LEVEL_TEMPLATES: Record<
  string,
  {
    level: number;
    title: string;
    badge: string;
    desc: string;
    defaultCode: string;
  }
> = {
  level_1_xor: {
    level: 1,
    title: "关卡 1：单层感知机局限与 XOR 异或突破",
    badge: "非线性分类 / 2D云图",
    desc: "突破单层线性不可分瓶颈，构建 [2, 4, 1] 隐藏层网络，迭代训练并在画布上实时绘制 2D 决策流形包裹正样本。",
    defaultCode: [
      "// =================================================================",
      "// 关卡 1：单层感知机局限与 XOR 异或非线性分类突破",
      "// 核心目标：构建 [2, 4, 1] 隐藏层网络，突破单层线性不可分瓶颈",
      "// =================================================================",
      "",
      'print("🚀 [关卡 1] 载入非线性 XOR 异或数据集 (80 样本, 0.08 噪声)...");',
      'const rawSamples = getDataset("xor", { count: 80, noise: 0.08 });',
      "const { train, test } = splitTrainTest(rawSamples, 0.8);",
      "",
      'print("📊 样本划分完成: 训练集 " + train.length + " 条 | 验证集 " + test.length + " 条");',
      "",
      "// 实例化前馈神经网络: 输入 2 维 -> 隐藏层 4 节点 (ReLU) -> 输出 1 节点 (Sigmoid)",
      'const net = new NeuralNetwork([2, 4, 1], ["relu", "sigmoid"], 0.15, "adam", "mse");',
      "",
      "const epochs = 600;",
      "const trainLossHistory = [];",
      "const valLossHistory = [];",
      "",
      'print("🔄 开始前向传播与反向传播迭代训练 (" + epochs + " 轮)...");',
      "",
      "for (let ep = 1; ep <= epochs; ep++) {",
      '  const tr = net.trainEpoch(train, 0.15, "adam", "mse");',
      '  const val = net.evaluate(test, "mse");',
      "",
      "  if (ep % 20 === 0 || ep === 1) {",
      "    trainLossHistory.push(tr.loss);",
      "    valLossHistory.push(val.loss);",
      "  }",
      "",
      "  if (ep % 150 === 0 || ep === epochs) {",
      '    print("Epoch [" + ep + "/" + epochs + "] -> Train Loss: " + tr.loss.toFixed(5) + " | Val Loss: " + val.loss.toFixed(5) + " | Val Acc: " + val.accuracy.toFixed(1) + "%");',
      "  }",
      "}",
      "",
      'const finalEval = net.evaluate(test, "mse");',
      'print("✅ [关卡 1 达成] 最终验证集准确率: " + finalEval.accuracy.toFixed(2) + "% (非线性决策边界已完全包裹异或正样本)");',
      "",
      "// 绘制 2D 决策流形分类云图",
      "plotDecisionBoundary(net, rawSamples, {",
      '  title: "关卡 1: XOR 异或门非线性分类决策云图 (准确率: " + finalEval.accuracy.toFixed(1) + "%)",',
      "  resolution: 50",
      "});",
      "",
      "// 输出预测明细对照表",
      "displayTable({",
      '  title: "关卡 1: XOR 验证集逐条预测明细与残差",',
      '  headers: ["输入特征 (x1, x2)", "真实标签 y", "网络预测 ŷ", "判定类别", "绝对残差 |y - ŷ|"],',
      "  rows: test.map(s => {",
      "    const pred = net.forward(s.input).output[0];",
      "    return [",
      '      "[" + s.input.map(v => v.toFixed(2)).join(", ") + "]",',
      "      s.target[0].toFixed(1),",
      "      pred.toFixed(4),",
      '      pred >= 0.5 ? "正类 (Class 1)" : "负类 (Class 0)",',
      "      Math.abs(s.target[0] - pred).toFixed(4)",
      "    ];",
      "  })",
      "});",
    ].join("\n"),
  },

  level_2_vanishing_gradient: {
    level: 2,
    title: "关卡 2：深层网络梯度消失与激活函数革命",
    badge: "反向连乘 / 梯度流诊断",
    desc: "在 5 层深层网络 [2, 6, 6, 6, 1] 中深入对比 Sigmoid 饱和导数连乘与 LeakyReLU 梯度传递，直观诊断浅层梯度消失现象。",
    defaultCode: [
      "// =================================================================",
      "// 关卡 2：深层网络梯度消失 (Vanishing Gradient) 与激活函数革命",
      "// 核心目标：对比 5 层深层网络中 Sigmoid 导数饱和连乘 vs LeakyReLU 梯度传递",
      "// =================================================================",
      "",
      'print("🔬 [关卡 2] 构建 5 层深层网络拓扑 [2, 6, 6, 6, 1]...");',
      "const topology = [2, 6, 6, 6, 1];",
      "",
      "// 1. 构建全 Sigmoid 深层网络 vs LeakyReLU 深层网络",
      'const netSigmoid = new NeuralNetwork(topology, ["sigmoid", "sigmoid", "sigmoid", "sigmoid"], 0.1, "sgd", "mse");',
      'const netReLU = new NeuralNetwork(topology, ["leaky_relu", "leaky_relu", "leaky_relu", "sigmoid"], 0.1, "sgd", "mse");',
      "",
      "const sample = { input: [0.8, -0.6], target: [1.0] };",
      "",
      "// 计算单步解析反向传播",
      'const bwdSigmoid = netSigmoid.backward(sample.input, sample.target, "mse");',
      'const bwdReLU = netReLU.backward(sample.input, sample.target, "mse");',
      "",
      'print("📉 [单步梯度流对比] 各层权重梯度范数 ||dW^(l)||:");',
      'print("  • Sigmoid: 输入浅层 ||dW^(1)|| = " + bwdSigmoid.layerGradients[0].norm.toExponential(4) + " | 输出深层 ||dW^(4)|| = " + bwdSigmoid.layerGradients[3].norm.toExponential(4));',
      'print("  • LeakyReLU: 输入浅层 ||dW^(1)|| = " + bwdReLU.layerGradients[0].norm.toExponential(4) + " | 输出深层 ||dW^(4)|| = " + bwdReLU.layerGradients[3].norm.toExponential(4));',
      "",
      "// 2. 在双月牙数据集上迭代 400 轮，记录收敛对比",
      'const rawSamples = getDataset("moons", { count: 80, noise: 0.1 });',
      "const { train, test } = splitTrainTest(rawSamples, 0.8);",
      "",
      "const epochs = 400;",
      "const sigmoidLosses = [];",
      "const reluLosses = [];",
      "",
      "for (let ep = 1; ep <= epochs; ep++) {",
      '  const trSig = netSigmoid.trainEpoch(train, 0.1, "sgd", "mse");',
      '  const trReLU = netReLU.trainEpoch(train, 0.1, "sgd", "mse");',
      "",
      "  if (ep % 15 === 0 || ep === 1) {",
      "    sigmoidLosses.push(trSig.loss);",
      "    reluLosses.push(trReLU.loss);",
      "  }",
      "}",
      "",
      'const evalSig = netSigmoid.evaluate(test, "mse");',
      'const evalReLU = netReLU.evaluate(test, "mse");',
      "",
      'print("✅ [关卡 2 达成] 训练结束! 最终准确率: Sigmoid = " + evalSig.accuracy.toFixed(1) + "% | LeakyReLU = " + evalReLU.accuracy.toFixed(1) + "%");',
      "",
      "// 绘制损失收敛轨迹对比",
      "plotLoss({",
      '  title: "关卡 2: 5层网络收敛轨迹 (紫线: Sigmoid梯度衰减停滞 vs 绿线: LeakyReLU稳健收敛)",',
      "  trainLoss: sigmoidLosses,",
      "  valLoss: reluLosses",
      "});",
      "",
      "// 输出逐层梯度衰减对比表",
      "displayTable({",
      '  title: "关卡 2: 5层深层网络各层梯度敏感度 ||dW^(l)|| 衰减对比",',
      '  headers: ["网络层级", "Sigmoid 梯度范数", "LeakyReLU 梯度范数", "梯度倍率优势", "现象诊断"],',
      "  rows: bwdSigmoid.layerGradients.map((g, idx) => {",
      "    const sNorm = g.norm;",
      "    const rNorm = bwdReLU.layerGradients[idx].norm;",
      "    const ratio = (rNorm / (sNorm + 1e-12)).toFixed(1);",
      "    return [",
      '      "第 " + (idx + 1) + " 层 (" + (idx === 0 ? "输入浅层" : idx === 3 ? "输出深层" : "隐藏层") + ")",',
      "      sNorm.toExponential(4),",
      "      rNorm.toExponential(4),",
      '      ratio + "x",',
      '      sNorm < 1e-4 ? "⚠️ 发生梯度消失 (停滞)" : "✅ 梯度健康传导"',
      "    ];",
      "  })",
      "});",
    ].join("\n"),
  },

  level_3_optimizers: {
    level: 3,
    title: "关卡 3：优化器进化史与收敛竞赛",
    badge: "SGD vs Momentum vs RMSprop vs Adam",
    desc: "在相同初值拓扑下让四大优化器同台竞技，记录多维收敛速度、峡谷震荡平抑与最终精度排行。",
    defaultCode: [
      "// =================================================================",
      "// 关卡 3：优化器进化史与收敛竞赛 (SGD vs Momentum vs RMSprop vs Adam)",
      "// 核心目标：在相同初值拓扑下对比四大优化器的收敛速度与稳定性",
      "// =================================================================",
      "",
      'print("🏁 [关卡 3] 初始化四大优化器同台竞速试验...");',
      "",
      'const dataset = getDataset("moons", { count: 100, noise: 0.1 });',
      "const { train, test } = splitTrainTest(dataset, 0.8);",
      "",
      'const optimizers = ["sgd", "momentum", "rmsprop", "adam"];',
      "const netConfig = [2, 6, 4, 1];",
      'const actConfig = ["leaky_relu", "leaky_relu", "sigmoid"];',
      "",
      "const networks = {",
      '  sgd: new NeuralNetwork(netConfig, actConfig, 0.08, "sgd", "mse"),',
      '  momentum: new NeuralNetwork(netConfig, actConfig, 0.08, "momentum", "mse"),',
      '  rmsprop: new NeuralNetwork(netConfig, actConfig, 0.08, "rmsprop", "mse"),',
      '  adam: new NeuralNetwork(netConfig, actConfig, 0.08, "adam", "mse"),',
      "};",
      "",
      "const epochs = 300;",
      "const lossHistories = { sgd: [], momentum: [], rmsprop: [], adam: [] };",
      "",
      "for (let ep = 1; ep <= epochs; ep++) {",
      "  optimizers.forEach(opt => {",
      '    const tr = networks[opt].trainEpoch(train, 0.08, opt, "mse");',
      "    if (ep % 10 === 0 || ep === 1) {",
      "      lossHistories[opt].push(tr.loss);",
      "    }",
      "  });",
      "}",
      "",
      "// 汇总测试集评估指标",
      "const summary = optimizers.map(opt => {",
      '  const ev = networks[opt].evaluate(test, "mse");',
      "  const initL = lossHistories[opt][0];",
      "  const finalL = lossHistories[opt][lossHistories[opt].length - 1];",
      "  const dropPct = (((initL - finalL) / initL) * 100).toFixed(1);",
      "  return {",
      "    opt: opt.toUpperCase(),",
      "    finalLoss: finalL,",
      "    accuracy: ev.accuracy,",
      '    drop: dropPct + "%"',
      "  };",
      "});",
      "",
      'print("🏆 [关卡 3 达成] 四大优化器竞速排行榜:");',
      "summary.forEach((s, i) => {",
      '  print("  第 " + (i + 1) + " 名: " + s.opt.padEnd(8) + " -> 最终损失: " + s.finalLoss.toFixed(5) + ", 验证准确率: " + s.accuracy.toFixed(1) + "% (降幅 " + s.drop + ")");',
      "});",
      "",
      "// 绘制 SGD (紫线) vs Adam (绿线) 损失曲线",
      "plotLoss({",
      '  title: "关卡 3: 优化器收敛轨迹对比 (紫线: SGD 震荡迟缓 vs 绿线: Adam 快速自适应逼近)",',
      "  trainLoss: lossHistories.sgd,",
      "  valLoss: lossHistories.adam",
      "});",
      "",
      "// 渲染四大优化器详细评测表",
      "displayTable({",
      '  title: "关卡 3: 四大优化器 300 轮训练效能综合对照",',
      '  headers: ["优化器类型", "初始损失", "最终损失", "损失降幅", "测试准确率", "收敛特性评估"],',
      "  rows: summary.map((s, idx) => {",
      "    const optKey = optimizers[idx];",
      "    const initL = lossHistories[optKey][0].toFixed(5);",
      "    const finalL = s.finalLoss.toFixed(5);",
      "    const notes = optKey === 'adam'",
      '      ? "⭐ 一阶动量+二阶自适应学习率，收敛最快最稳"',
      "      : optKey === 'rmsprop'",
      '      ? "⚡ 梯度均方根归一化，自适应平抑陡峭维度震荡"',
      "      : optKey === 'momentum'",
      '      ? "💨 增加惯性动量，加速冲过平坦鞍点"',
      '      : "🐢 传统梯度下降，在狭长峡谷中容易剧烈横向震荡";',
      '    return [s.opt, initL, finalL, s.drop, s.accuracy.toFixed(1) + "%", notes];',
      "  })",
      "});",
    ].join("\n"),
  },

  level_4_learning_rate_uat: {
    level: 4,
    title: "关卡 4：学习率临界效应与 UAT 波形拟合",
    badge: "UAT连续回归 / 拟合优度 R²",
    desc: "探究通用近似定理 (UAT)，使用连续隐藏层高保真逼近复合高频非线性波形，检验拟合优度 R² 与残差分布。",
    defaultCode: [
      "// =================================================================",
      "// 关卡 4：学习率 η 临界效应与连续非线性函数逼近 (UAT 通用近似定理)",
      "// 核心目标：验证连续单隐藏层对复合高频波形 f(x) 的高精度回归逼近",
      "// =================================================================",
      "",
      'print("🌊 [关卡 4] 载入连续非线性复合波形数据集 f(x) = sin(2πx) + 0.3cos(4πx)...");',
      'const waveData = getDataset("wave", { count: 80, noise: 0.03 });',
      "const { train, test } = splitTrainTest(waveData, 0.8);",
      "",
      'print("📊 样本数: 训练 " + train.length + " 点 | 验证 " + test.length + " 点");',
      "",
      "// 构建回归神经网络: [1 -> 10 -> 6 -> 1] (隐藏层 Tanh/LeakyReLU, 输出层 Linear)",
      'const net = new NeuralNetwork([1, 10, 6, 1], ["tanh", "leaky_relu", "linear"], 0.05, "adam", "mse");',
      "",
      "const epochs = 800;",
      'print("🔄 开始回归拟合训练 (" + epochs + " 轮)...");',
      "",
      "for (let ep = 1; ep <= epochs; ep++) {",
      '  const tr = net.trainEpoch(train, 0.05, "adam", "mse");',
      "  if (ep % 200 === 0 || ep === epochs) {",
      '    const val = net.evaluate(test, "mse");',
      '    print("Epoch [" + ep + "/" + epochs + "] -> Train MSE: " + tr.loss.toFixed(6) + " | Val MSE: " + val.loss.toFixed(6));',
      "  }",
      "}",
      "",
      "// 提取测试点绘制连续波形拟合曲线",
      "const sortedSamples = [...waveData].sort((a, b) => a.input[0] - b.input[0]);",
      "const xVals = [];",
      "const yTrue = [];",
      "const yPred = [];",
      "",
      "sortedSamples.forEach(s => {",
      "  xVals.push(s.input[0]);",
      "  yTrue.push(s.target[0]);",
      "  const p = net.forward(s.input).output[0];",
      "  yPred.push(p);",
      "});",
      "",
      "// 计算决定系数 R²",
      "const meanY = yTrue.reduce((a, b) => a + b, 0) / yTrue.length;",
      "const ssTot = yTrue.reduce((acc, y) => acc + Math.pow(y - meanY, 2), 0);",
      "const ssRes = yTrue.reduce((acc, y, idx) => acc + Math.pow(y - yPred[idx], 2), 0);",
      "const r2Score = (1 - ssRes / (ssTot + 1e-8)).toFixed(4);",
      "",
      'print("🏆 [关卡 4 达成] 回归拟合优度 R² = " + r2Score + " (越接近 1.0 表示通用近似拟合越完美)");',
      "",
      "// 绘制真实目标曲线 vs 神经网络预测拟合曲线",
      "plotRegression({",
      '  title: "关卡 4: 连续非线性波形回归拟合 (R² 拟合优度: " + r2Score + ")",',
      "  xValues: xVals,",
      "  yTrue: yTrue,",
      "  yPred: yPred",
      "});",
      "",
      "// 输出测试集样本残差对照表",
      "displayTable({",
      '  title: "关卡 4: 连续波形样本点预测与残差对照 (抽样 10 点)",',
      '  headers: ["输入自变量 x", "真实函数值 f(x)", "网络回归预测 ŷ", "绝对残差 |y - ŷ|", "拟合评级"],',
      "  rows: sortedSamples.filter((_, i) => i % 8 === 0).map(s => {",
      "    const p = net.forward(s.input).output[0];",
      "    const diff = Math.abs(s.target[0] - p);",
      "    return [",
      "      s.input[0].toFixed(3),",
      "      s.target[0].toFixed(4),",
      "      p.toFixed(4),",
      "      diff.toFixed(4),",
      '      diff < 0.05 ? "⭐ 极高精度" : diff < 0.12 ? "✅ 良好逼近" : "⚠️ 存在偏差"',
      "    ];",
      "  })",
      "});",
    ].join("\n"),
  },
};

export const CodeEngine: React.FC<CodeEngineProps> = ({
  topology,
  activations,
  learningRate,
  optimizer,
  lossFunction = "mse",
}) => {
  // Currently Selected Level Template
  const [selectedLevelKey, setSelectedLevelKey] = useState<string>("level_1_xor");

  // Output Window Sub-tabs: Plots vs Console Logs vs Metrics Table
  const [outputTab, setOutputTab] = useState<"plot" | "terminal" | "table">("plot");

  // Running & Execution States
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [executionTimeMs, setExecutionTimeMs] = useState<number | null>(null);
  const [executionStatus, setExecutionStatus] = useState<"idle" | "success" | "error">("idle");
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [currentPlot, setCurrentPlot] = useState<PlotPayload | null>(null);
  const [currentTable, setCurrentTable] = useState<TablePayload | null>(null);

  const [copiedCode, setCopiedCode] = useState(false);
  const [copiedLogs, setCopiedLogs] = useState(false);

  // Plot Canvas Reference
  const plotCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const consoleBottomRef = useRef<HTMLDivElement | null>(null);

  // Editable Code State
  const [editorCode, setEditorCode] = useState<string>(
    FOUR_LEVEL_TEMPLATES.level_1_xor.defaultCode
  );

  // When template changes, update editor content
  const handleSelectLevel = (key: string) => {
    setSelectedLevelKey(key);
    if (FOUR_LEVEL_TEMPLATES[key]) {
      setEditorCode(FOUR_LEVEL_TEMPLATES[key].defaultCode);
      setLogs([]);
      setCurrentPlot(null);
      setCurrentTable(null);
      setExecutionStatus("idle");
    }
  };

  // Reset current template code
  const handleResetCode = () => {
    if (FOUR_LEVEL_TEMPLATES[selectedLevelKey]) {
      setEditorCode(FOUR_LEVEL_TEMPLATES[selectedLevelKey].defaultCode);
      setLogs((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          type: "info",
          message: "已重置为当前关卡的精炼标准源码。",
          timestamp: new Date().toLocaleTimeString(),
        },
      ]);
    }
  };

  // -------------------------------------------------------------
  // RUNNER EXECUTION SANDBOX (In-Browser Native Safe Runner)
  // -------------------------------------------------------------
  const handleRunCode = useCallback(async () => {
    if (isRunning) return;

    setIsRunning(true);
    setExecutionStatus("idle");
    const capturedLogs: LogEntry[] = [];
    let detectedPlot: PlotPayload | null = null;
    let detectedTable: TablePayload | null = null;

    const startTime = performance.now();

    const addLog = (
      type: "log" | "info" | "success" | "warn" | "error" | "plot",
      msg: any
    ) => {
      const formatted =
        typeof msg === "object" ? JSON.stringify(msg, null, 2) : String(msg);
      capturedLogs.push({
        id: Math.random().toString(36).substring(2, 9),
        type,
        message: formatted,
        timestamp: new Date().toLocaleTimeString(),
      });
    };

    try {
      addLog("info", `🚀 启动沙箱执行环境: ${FOUR_LEVEL_TEMPLATES[selectedLevelKey]?.title || "实验脚本"}...`);

      // Helper functions injected into sandbox
      const sandboxGetDataset = (
        name: string,
        options: { count?: number; noise?: number } = {}
      ): TrainingSample[] => {
        const count = options.count ?? 80;
        const noise = options.noise ?? 0.08;
        switch (name.toLowerCase()) {
          case "xor":
            return generateXORDataset(noise, count);
          case "moons":
          case "moon":
            return generateMoonsDataset(noise, count);
          case "circles":
          case "circle":
            return generateCirclesDataset(noise, count);
          case "spiral":
          case "spirals":
            return generateSpiralsDataset(noise, count);
          case "thermal":
          case "comfort":
            return generateThermalComfortDataset(count, noise);
          case "linear":
            return generateLinearIdentityDataset(count, noise);
          case "wave":
            return getNonlinearWaveDataset(count);
          case "iris":
            return getIrisDataset();
          case "housing":
            return getHousePriceRegressionDataset();
          case "churn":
            return getCustomerChurnDataset();
          case "digits":
            return getMNISTDigitSamples().samples.slice(0, count);
          default:
            return generateXORDataset(noise, count);
        }
      };

      const sandboxPlotDecisionBoundary = (
        nnInstance: NeuralNetwork,
        samplesData: TrainingSample[],
        opts: { title?: string; resolution?: number } = {}
      ) => {
        detectedPlot = {
          type: "decision_boundary",
          title: opts.title || "2D 决策边界分类云图",
          data: {
            nn: nnInstance,
            samples: samplesData,
            resolution: opts.resolution || 48,
          },
        };
        addLog("plot", `📊 成功生成 2D 决策边界云图 [${opts.title || "Decision Boundary"}]`);
      };

      const sandboxPlotLoss = (opts: {
        epochs?: number[];
        trainLoss?: number[];
        valLoss?: number[];
        title?: string;
      }) => {
        detectedPlot = {
          type: "loss_curve",
          title: opts.title || "训练与验证损失收敛轨迹",
          data: opts,
        };
        addLog("plot", `📈 成功生成损失收敛曲线 [${opts.title || "Loss Curve"}]`);
      };

      const sandboxPlotRegression = (opts: {
        xValues: number[];
        yTrue: number[];
        yPred: number[];
        title?: string;
      }) => {
        detectedPlot = {
          type: "regression",
          title: opts.title || "回归拟合对比曲线",
          data: opts,
        };
        addLog("plot", `🌊 成功生成连续回归拟合曲线 [${opts.title || "Regression Fit"}]`);
      };

      const sandboxDisplayTable = (payload: TablePayload) => {
        detectedTable = payload;
        addLog("info", `📋 成功生成结构化指标表: ${payload.title || "数据明细"}`);
      };

      // Create sandboxed function
      const sandboxGlobals = {
        NeuralNetwork,
        getDataset: sandboxGetDataset,
        splitTrainTest,
        plotDecisionBoundary: sandboxPlotDecisionBoundary,
        plotLoss: sandboxPlotLoss,
        plotRegression: sandboxPlotRegression,
        displayTable: sandboxDisplayTable,
        print: (...args: any[]) => {
          const text = args.map((a) => (typeof a === "object" ? JSON.stringify(a) : String(a))).join(" ");
          addLog("log", text);
        },
        console: {
          log: (...args: any[]) => addLog("log", args.join(" ")),
          info: (...args: any[]) => addLog("info", args.join(" ")),
          warn: (...args: any[]) => addLog("warn", args.join(" ")),
          error: (...args: any[]) => addLog("error", args.join(" ")),
        },
        Math,
        Array,
        Object,
        Number,
        String,
        Boolean,
        JSON,
        Date,
        parseInt,
        parseFloat,
        isNaN,
        isFinite,
      };

      // Wrap user code in async function
      const paramNames = Object.keys(sandboxGlobals);
      const paramValues = Object.values(sandboxGlobals);

      // Async wrapper
      const runnableFn = new Function(
        ...paramNames,
        `"use strict";\nreturn (async () => {\n${editorCode}\n})();`
      );

      await runnableFn(...paramValues);

      const endTime = performance.now();
      const elapsed = Math.round(endTime - startTime);
      setExecutionTimeMs(elapsed);
      setExecutionStatus("success");
      addLog("success", `✨ 关卡脚本执行成功，沙盒总耗时: ${elapsed} ms`);

      if (detectedPlot) {
        setCurrentPlot(detectedPlot);
        setOutputTab("plot");
      } else if (detectedTable) {
        setCurrentTable(detectedTable);
        setOutputTab("table");
      } else {
        setOutputTab("terminal");
      }

      if (detectedTable) setCurrentTable(detectedTable);
    } catch (err: any) {
      const endTime = performance.now();
      const elapsed = Math.round(endTime - startTime);
      setExecutionTimeMs(elapsed);
      setExecutionStatus("error");
      addLog("error", `❌ 代码运行异常: ${err?.message || String(err)}`);
      if (err?.stack) {
        addLog("warn", `Stack Trace: ${err.stack.split("\n").slice(0, 3).join("\n")}`);
      }
      setOutputTab("terminal");
    } finally {
      setIsRunning(false);
      setLogs(capturedLogs);
    }
  }, [editorCode, isRunning, selectedLevelKey]);

  // Handle Ctrl+Enter / Cmd+Enter Shortcut inside editor
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      handleRunCode();
    } else if (e.key === "Tab") {
      e.preventDefault();
      const textarea = e.currentTarget;
      const start = textarea.selectionStart;
      const end = textarea.selectionEnd;
      const val = textarea.value;
      textarea.value = val.substring(0, start) + "  " + val.substring(end);
      textarea.selectionStart = textarea.selectionEnd = start + 2;
      setEditorCode(textarea.value);
    }
  };

  // -------------------------------------------------------------
  // CANVAS PLOT RENDERING EFFECT
  // -------------------------------------------------------------
  useEffect(() => {
    if (!currentPlot || outputTab !== "plot") return;

    let animId: number;
    const renderCanvas = () => {
      if (!plotCanvasRef.current) return;
      const canvas = plotCanvasRef.current;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const width = canvas.width;
      const height = canvas.height;

      ctx.clearRect(0, 0, width, height);

      if (currentPlot.type === "decision_boundary" && currentPlot.data) {
        const { nn, samples, resolution = 48 } = currentPlot.data;
        if (!nn) return;

        // Calculate dynamic bounding box
        let minX = -0.2;
        let maxX = 1.2;
        let minY = -0.2;
        let maxY = 1.2;

        if (samples && Array.isArray(samples) && samples.length > 0) {
          const xs = samples.map((s: TrainingSample) => s.input[0]).filter((v) => typeof v === "number" && !isNaN(v));
          const ys = samples.map((s: TrainingSample) => s.input[1]).filter((v) => typeof v === "number" && !isNaN(v));
          if (xs.length > 0 && ys.length > 0) {
            const dataMinX = Math.min(...xs);
            const dataMaxX = Math.max(...xs);
            const dataMinY = Math.min(...ys);
            const dataMaxY = Math.max(...ys);
            const spanX = Math.max(0.3, dataMaxX - dataMinX);
            const spanY = Math.max(0.3, dataMaxY - dataMinY);
            minX = dataMinX - spanX * 0.15;
            maxX = dataMaxX + spanX * 0.15;
            minY = dataMinY - spanY * 0.15;
            maxY = dataMaxY + spanY * 0.15;
          }
        }

        const cellW = width / resolution;
        const cellH = height / resolution;

        // 1. Draw 2D contour grid
        for (let i = 0; i < resolution; i++) {
          for (let j = 0; j < resolution; j++) {
            const x = minX + (i / resolution) * (maxX - minX);
            const y = maxY - (j / resolution) * (maxY - minY);

            const out = nn.forward([x, y]).output[0] ?? 0.5;

            // High contrast boundary: Blue (0) vs Red (1)
            if (out > 0.5) {
              const alpha = Math.min(0.85, (out - 0.5) * 1.8);
              ctx.fillStyle = `rgba(239, 68, 68, ${alpha})`; // Red
            } else {
              const alpha = Math.min(0.85, (0.5 - out) * 1.8);
              ctx.fillStyle = `rgba(59, 130, 246, ${alpha})`; // Blue
            }
            ctx.fillRect(i * cellW, j * cellH, cellW + 1, cellH + 1);
          }
        }

        // 2. Draw Grid Ticks & Axes
        ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        const zeroX = ((0 - minX) / (maxX - minX || 1)) * width;
        const zeroY = height - ((0 - minY) / (maxY - minY || 1)) * height;
        if (zeroX >= 0 && zeroX <= width) {
          ctx.moveTo(zeroX, 0);
          ctx.lineTo(zeroX, height);
        }
        if (zeroY >= 0 && zeroY <= height) {
          ctx.moveTo(0, zeroY);
          ctx.lineTo(width, zeroY);
        }
        ctx.stroke();

        // 3. Draw Scatter Points
        if (samples && Array.isArray(samples)) {
          samples.forEach((s: TrainingSample) => {
            const sx = ((s.input[0] - minX) / (maxX - minX || 1)) * width;
            const sy = height - ((s.input[1] - minY) / (maxY - minY || 1)) * height;
            const targetClass = s.target[0] >= 0.5 ? 1 : 0;

            ctx.beginPath();
            ctx.arc(sx, sy, 5, 0, Math.PI * 2);
            ctx.fillStyle = targetClass === 1 ? "#ef4444" : "#3b82f6";
            ctx.fill();
            ctx.strokeStyle = "#ffffff";
            ctx.lineWidth = 1.5;
            ctx.stroke();
          });
        }

        // 4. Legend Overlay
        ctx.fillStyle = "rgba(15, 23, 42, 0.88)";
        ctx.fillRect(width - 150, 10, 140, 58);
        ctx.strokeStyle = "rgba(255, 255, 255, 0.2)";
        ctx.strokeRect(width - 150, 10, 140, 58);

        ctx.fillStyle = "#ffffff";
        ctx.font = "11px sans-serif";
        ctx.fillText("类别 0 (Blue)", width - 115, 28);
        ctx.fillText("类别 1 (Red)", width - 115, 50);

        ctx.beginPath();
        ctx.arc(width - 130, 24, 4.5, 0, Math.PI * 2);
        ctx.fillStyle = "#3b82f6";
        ctx.fill();

        ctx.beginPath();
        ctx.arc(width - 130, 46, 4.5, 0, Math.PI * 2);
        ctx.fillStyle = "#ef4444";
        ctx.fill();
      } else if (currentPlot.type === "regression" && currentPlot.data) {
        const { xValues, yTrue, yPred } = currentPlot.data;
        if (!xValues || xValues.length === 0) return;

        const pad = 40;
        const plotW = width - pad * 2;
        const plotH = height - pad * 2;

        ctx.fillStyle = "#0f172a";
        ctx.fillRect(0, 0, width, height);

        ctx.strokeStyle = "#334155";
        ctx.lineWidth = 1;
        ctx.strokeRect(pad, pad, plotW, plotH);

        const minX = Math.min(...xValues);
        const maxX = Math.max(...xValues);
        const allY = [...yTrue, ...yPred];
        const minY = Math.min(...allY) - 0.15;
        const maxY = Math.max(...allY) + 0.15;

        const mapX = (x: number) => pad + ((x - minX) / (maxX - minX || 1)) * plotW;
        const mapY = (y: number) => pad + plotH - ((y - minY) / (maxY - minY || 1)) * plotH;

        // 1. Draw True Curve (Cyan dashed)
        ctx.beginPath();
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 3]);
        xValues.forEach((x: number, idx: number) => {
          const px = mapX(x);
          const py = mapY(yTrue[idx]);
          if (idx === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();
        ctx.setLineDash([]);

        // 2. Draw Predicted Curve (Amber solid)
        ctx.beginPath();
        ctx.strokeStyle = "#f59e0b";
        ctx.lineWidth = 2.5;
        xValues.forEach((x: number, idx: number) => {
          const px = mapX(x);
          const py = mapY(yPred[idx]);
          if (idx === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();

        // 3. Draw Scatter Points
        xValues.forEach((x: number, idx: number) => {
          const px = mapX(x);
          const py = mapY(yTrue[idx]);
          ctx.beginPath();
          ctx.arc(px, py, 2.5, 0, Math.PI * 2);
          ctx.fillStyle = "#38bdf8";
          ctx.fill();
        });

        // Legend
        ctx.fillStyle = "rgba(15, 23, 42, 0.88)";
        ctx.fillRect(width - 170, 15, 155, 52);
        ctx.strokeStyle = "#334155";
        ctx.strokeRect(width - 170, 15, 155, 52);

        ctx.fillStyle = "#38bdf8";
        ctx.font = "11px monospace";
        ctx.fillText("--- 真实目标 f(x)", width - 155, 33);

        ctx.fillStyle = "#f59e0b";
        ctx.fillText("── 神经网络预测 ŷ", width - 155, 52);
      } else if (currentPlot.type === "loss_curve" && currentPlot.data) {
        const { trainLoss = [], valLoss = [] } = currentPlot.data;
        const pad = 40;
        const plotW = width - pad * 2;
        const plotH = height - pad * 2;

        ctx.fillStyle = "#0f172a";
        ctx.fillRect(0, 0, width, height);

        const maxLoss = Math.max(0.5, ...trainLoss, ...valLoss);
        const count = Math.max(trainLoss.length, valLoss.length);

        const mapX = (i: number) => pad + (i / (count - 1 || 1)) * plotW;
        const mapY = (l: number) => pad + plotH - (l / maxLoss) * plotH;

        // Grid Lines
        ctx.strokeStyle = "#1e293b";
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
          const gy = pad + (i / 4) * plotH;
          ctx.beginPath();
          ctx.moveTo(pad, gy);
          ctx.lineTo(pad + plotW, gy);
          ctx.stroke();
        }

        // Train / Contrast Loss Line 1 (Indigo / Purple)
        if (trainLoss.length > 1) {
          ctx.beginPath();
          ctx.strokeStyle = "#a855f7";
          ctx.lineWidth = 2.2;
          trainLoss.forEach((l: number, idx: number) => {
            const px = mapX(idx);
            const py = mapY(l);
            if (idx === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
          });
          ctx.stroke();
        }

        // Val / Contrast Loss Line 2 (Emerald / Green)
        if (valLoss.length > 1) {
          ctx.beginPath();
          ctx.strokeStyle = "#10b981";
          ctx.lineWidth = 2.2;
          valLoss.forEach((l: number, idx: number) => {
            const px = mapX(idx);
            const py = mapY(l);
            if (idx === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
          });
          ctx.stroke();
        }

        // Legend
        ctx.fillStyle = "rgba(15, 23, 42, 0.88)";
        ctx.fillRect(width - 180, 15, 165, 52);
        ctx.strokeStyle = "#334155";
        ctx.strokeRect(width - 180, 15, 165, 52);

        ctx.fillStyle = "#a855f7";
        ctx.font = "11px monospace";
        ctx.fillText("── 对照组 A (紫线)", width - 165, 33);

        ctx.fillStyle = "#10b981";
        ctx.fillText("── 对照组 B (绿线)", width - 165, 52);
      }
    };

    animId = requestAnimationFrame(renderCanvas);
    const timer = setTimeout(renderCanvas, 50);

    return () => {
      cancelAnimationFrame(animId);
      clearTimeout(timer);
    };
  }, [currentPlot, outputTab]);

  // Copy active editor code
  const handleCopyCode = () => {
    navigator.clipboard.writeText(editorCode);
    setCopiedCode(true);
    setTimeout(() => setCopiedCode(false), 2000);
  };

  // Copy terminal logs
  const handleCopyLogs = () => {
    const text = logs.map((l) => `[${l.timestamp}] [${l.type.toUpperCase()}] ${l.message}`).join("\n");
    navigator.clipboard.writeText(text);
    setCopiedLogs(true);
    setTimeout(() => setCopiedLogs(false), 2000);
  };

  // Download script file
  const handleDownloadCode = () => {
    const filename = `level_${FOUR_LEVEL_TEMPLATES[selectedLevelKey]?.level || 1}_script.js`;
    const blob = new Blob([editorCode], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(url);
  };

  const activeLevel = FOUR_LEVEL_TEMPLATES[selectedLevelKey] || FOUR_LEVEL_TEMPLATES.level_1_xor;

  return (
    <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-sm space-y-3.5">
      {/* Top Banner: Four Levels Navigation Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-200">
        <div className="flex items-center gap-2.5">
          <div className="w-2.5 h-2.5 rounded-full bg-blue-600 animate-pulse" />
          <div>
            <h3 className="text-xs sm:text-sm font-bold text-slate-900 flex items-center gap-2">
              <span>代码引擎与沙箱切片 (Code Engine & Sandbox Slices)</span>
              <span className="text-[10px] px-2 py-0.5 rounded bg-blue-50 text-blue-700 font-bold border border-blue-200 font-mono">
                四大关卡精炼脚本 (4 Levels In-Browser Native)
              </span>
            </h3>
            <p className="text-[11px] text-slate-500 font-medium">
              围绕四大实验关卡组织算法脚本 · 在项目原生浏览器沙盒中单键运行与即时可视化诊断
            </p>
          </div>
        </div>

        {/* 4 Levels Selector Pills */}
        <div className="flex flex-wrap items-center gap-1.5 bg-slate-100 p-1 rounded-lg border border-slate-200 text-xs font-mono">
          {Object.entries(FOUR_LEVEL_TEMPLATES).map(([key, t]) => (
            <button
              key={key}
              onClick={() => handleSelectLevel(key)}
              className={`px-3 py-1.5 rounded-md font-semibold flex items-center gap-1.5 transition cursor-pointer ${
                selectedLevelKey === key
                  ? "bg-blue-600 text-white shadow-xs"
                  : "text-slate-700 hover:bg-slate-200"
              }`}
              title={t.desc}
            >
              <Award className="w-3.5 h-3.5" />
              <span>关卡 {t.level}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Level Summary Banner */}
      <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 bg-blue-50/60 border border-blue-100 rounded-lg text-xs">
        <div className="flex items-center gap-2">
          <span className="font-bold text-blue-900">{activeLevel.title}</span>
          <span className="text-[10px] px-2 py-0.5 rounded bg-white text-blue-700 border border-blue-200 font-mono font-semibold">
            {activeLevel.badge}
          </span>
        </div>
        <p className="text-[11px] text-slate-600 flex-1 sm:text-right">
          {activeLevel.desc}
        </p>
      </div>

      {/* Action Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-2.5 bg-slate-50 p-2.5 rounded-lg border border-slate-200">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold text-slate-700 font-sans flex items-center gap-1">
            <FileCode className="w-3.5 h-3.5 text-blue-600" />
            当前关卡脚本:
          </span>
          <span className="text-xs font-mono font-bold text-slate-800 bg-white px-2 py-0.5 rounded border border-slate-300">
            level_{activeLevel.level}_script.js
          </span>
        </div>

        {/* Execution & Action Buttons */}
        <div className="flex items-center gap-2">
          <button
            onClick={handleRunCode}
            disabled={isRunning}
            className="flex items-center gap-1.5 px-3.5 py-1.5 bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 text-white rounded text-xs font-bold shadow-sm transition cursor-pointer"
            title="运行当前关卡沙盒代码 (快捷键: Ctrl + Enter / Cmd + Enter)"
          >
            {isRunning ? (
              <>
                <RotateCcw className="w-3.5 h-3.5 animate-spin" />
                <span>执行中...</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>▶ 运行关卡代码 (Run)</span>
              </>
            )}
          </button>

          <button
            onClick={handleResetCode}
            className="flex items-center gap-1 px-2.5 py-1.5 bg-white hover:bg-slate-100 text-slate-700 rounded text-xs font-semibold border border-slate-300 shadow-xs transition cursor-pointer"
            title="重置为当前关卡初始精炼代码"
          >
            <RotateCcw className="w-3.5 h-3.5 text-slate-500" />
            <span>重置代码</span>
          </button>

          <button
            onClick={handleCopyCode}
            className="flex items-center gap-1 px-2.5 py-1.5 bg-white hover:bg-slate-100 text-slate-700 rounded text-xs font-semibold border border-slate-300 shadow-xs transition cursor-pointer"
            title="复制代码到剪贴板"
          >
            {copiedCode ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-600" />
                <span className="text-emerald-700 font-bold">已复制</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5 text-slate-500" />
                <span>复制代码</span>
              </>
            )}
          </button>

          <button
            onClick={handleDownloadCode}
            className="flex items-center gap-1 px-2.5 py-1.5 bg-white hover:bg-slate-100 text-slate-700 rounded text-xs font-semibold border border-slate-300 shadow-xs transition cursor-pointer"
            title="下载关卡脚本文件"
          >
            <Download className="w-3.5 h-3.5 text-slate-500" />
            <span>导出脚本</span>
          </button>
        </div>
      </div>

      {/* Dual Column Layout: Code Editor (Left) & Output Window (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-3.5">
        {/* Left: Code Editor (6 cols on lg) */}
        <div className="lg:col-span-6 flex flex-col bg-slate-950 rounded-lg border border-slate-800 overflow-hidden shadow-sm">
          <div className="px-3 py-2 bg-slate-900 border-b border-slate-800 flex items-center justify-between text-xs font-mono text-slate-300">
            <div className="flex items-center gap-2">
              <FileCode className="w-3.5 h-3.5 text-blue-400" />
              <span className="font-bold">level_{activeLevel.level}_script.js</span>
              <span className="text-[10px] text-slate-500">
                ({editorCode.split("\n").length} 行代码)
              </span>
            </div>
            <div className="text-[10px] text-slate-400 font-sans">
              按 <kbd className="px-1 py-0.5 bg-slate-800 rounded text-slate-300">Ctrl + Enter</kbd> 快速运行
            </div>
          </div>

          <div className="relative flex-1 min-h-[400px] max-h-[480px] p-2 flex font-mono text-xs overflow-hidden">
            {/* Line Numbers Column */}
            <div className="select-none pr-3 pl-1 text-slate-600 text-right font-mono text-xs leading-relaxed border-r border-slate-800/80">
              {editorCode.split("\n").map((_, i) => (
                <div key={i}>{i + 1}</div>
              ))}
            </div>

            {/* Editable Code Textarea */}
            <textarea
              value={editorCode}
              onChange={(e) => setEditorCode(e.target.value)}
              onKeyDown={handleKeyDown}
              spellCheck={false}
              className="flex-1 bg-transparent text-sky-100 placeholder-slate-600 font-mono text-xs leading-relaxed pl-3 outline-none resize-none overflow-y-auto whitespace-pre tab-2"
              placeholder="// 在此处编写 JavaScript / TypeScript 关卡算法脚本..."
            />
          </div>
        </div>

        {/* Right: Output Window (6 cols on lg) */}
        <div className="lg:col-span-6 flex flex-col bg-slate-900 rounded-lg border border-slate-800 overflow-hidden shadow-sm">
          {/* Output Window Header & Sub-tabs */}
          <div className="px-3 py-2 bg-slate-950 border-b border-slate-800 flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-1.5">
              <button
                onClick={() => setOutputTab("plot")}
                className={`px-2.5 py-1 rounded text-xs font-semibold flex items-center gap-1 transition cursor-pointer ${
                  outputTab === "plot"
                    ? "bg-blue-600 text-white shadow-xs"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <LineChart className="w-3.5 h-3.5" />
                <span>📊 可视化图表</span>
              </button>

              <button
                onClick={() => setOutputTab("terminal")}
                className={`px-2.5 py-1 rounded text-xs font-semibold flex items-center gap-1 transition cursor-pointer ${
                  outputTab === "terminal"
                    ? "bg-blue-600 text-white shadow-xs"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                <Terminal className="w-3.5 h-3.5" />
                <span>🖥️ 控制台输出 ({logs.length})</span>
              </button>

              {currentTable && (
                <button
                  onClick={() => setOutputTab("table")}
                  className={`px-2.5 py-1 rounded text-xs font-semibold flex items-center gap-1 transition cursor-pointer ${
                    outputTab === "table"
                      ? "bg-blue-600 text-white shadow-xs"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  <TableIcon className="w-3.5 h-3.5" />
                  <span>📋 评估明细</span>
                </button>
              )}
            </div>

            {/* Status Indicator */}
            <div className="flex items-center gap-2 text-xs font-mono">
              {executionStatus === "success" && (
                <span className="text-emerald-400 font-bold flex items-center gap-1">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  <span>耗时: {executionTimeMs}ms</span>
                </span>
              )}
              {executionStatus === "error" && (
                <span className="text-rose-400 font-bold flex items-center gap-1">
                  <AlertCircle className="w-3.5 h-3.5" />
                  <span>执行出错</span>
                </span>
              )}
              {logs.length > 0 && (
                <button
                  onClick={() => {
                    setLogs([]);
                    setCurrentPlot(null);
                    setCurrentTable(null);
                    setExecutionStatus("idle");
                  }}
                  className="text-slate-500 hover:text-slate-300 transition cursor-pointer"
                  title="清空输出与图表"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          </div>

          {/* Output Content Body */}
          <div className="p-3 min-h-[400px] max-h-[480px] flex flex-col justify-center overflow-y-auto">
            {/* 1. Plot Canvas Tab */}
            {outputTab === "plot" && (
              <div className="flex flex-col items-center justify-center h-full w-full space-y-2">
                {currentPlot ? (
                  <div className="w-full h-full flex flex-col space-y-1.5">
                    <div className="flex items-center justify-between text-xs text-slate-300 font-bold">
                      <span>{currentPlot.title}</span>
                      <span className="text-[10px] text-slate-500 font-mono">
                        {currentPlot.type.toUpperCase()}
                      </span>
                    </div>
                    <div className="relative w-full h-[330px] bg-slate-950 rounded-lg overflow-hidden border border-slate-800 flex items-center justify-center">
                      <canvas
                        ref={plotCanvasRef}
                        width={540}
                        height={330}
                        className="w-full h-full object-contain"
                      />
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12 space-y-2 text-slate-500">
                    <LineChart className="w-10 h-10 mx-auto text-slate-600 opacity-60" />
                    <p className="text-xs font-medium">尚未渲染关卡图表</p>
                    <p className="text-[11px] text-slate-600 max-w-xs">
                      点击上方 “▶ 运行关卡代码” 按钮，脚本中的 2D 决策流形、收敛轨迹或回归拟合曲线将在此实时呈现！
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* 2. Console Logs Tab */}
            {outputTab === "terminal" && (
              <div className="h-full flex flex-col space-y-2 font-mono text-xs">
                <div className="flex items-center justify-between pb-1 border-b border-slate-800 text-[11px] text-slate-400">
                  <span>Console Standard Output (stdout):</span>
                  {logs.length > 0 && (
                    <button
                      onClick={handleCopyLogs}
                      className="flex items-center gap-1 text-slate-400 hover:text-slate-200 transition cursor-pointer"
                    >
                      {copiedLogs ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                      <span>{copiedLogs ? "已复制日志" : "复制全部日志"}</span>
                    </button>
                  )}
                </div>

                <div className="flex-1 overflow-y-auto space-y-1 pr-1 max-h-[360px]">
                  {logs.length === 0 ? (
                    <div className="text-slate-600 text-center py-12">
                      控制台为空。点击 “▶ 运行关卡代码” 执行并查看推导输出。
                    </div>
                  ) : (
                    logs.map((log) => (
                      <div
                        key={log.id}
                        className={`flex items-start gap-2 py-0.5 leading-relaxed break-all ${
                          log.type === "error"
                            ? "text-rose-400 bg-rose-950/20 px-1.5 rounded"
                            : log.type === "success"
                            ? "text-emerald-300 font-bold"
                            : log.type === "warn"
                            ? "text-amber-300"
                            : log.type === "plot"
                            ? "text-sky-300 font-semibold"
                            : "text-slate-200"
                        }`}
                      >
                        <span className="text-[10px] text-slate-600 select-none">
                          [{log.timestamp}]
                        </span>
                        <span className="whitespace-pre-wrap">{log.message}</span>
                      </div>
                    ))
                  )}
                  <div ref={consoleBottomRef} />
                </div>
              </div>
            )}

            {/* 3. Table Output Tab */}
            {outputTab === "table" && currentTable && (
              <div className="h-full flex flex-col space-y-2 overflow-y-auto">
                <h4 className="text-xs font-bold text-slate-200">
                  {currentTable.title || "关卡评估明细"}
                </h4>
                <div className="overflow-x-auto border border-slate-800 rounded-lg bg-slate-950 max-h-[350px]">
                  <table className="w-full text-left font-mono text-[11px] text-slate-300">
                    <thead className="bg-slate-800/80 text-slate-400 uppercase text-[10px] sticky top-0">
                      <tr>
                        {currentTable.headers.map((h, i) => (
                          <th key={i} className="px-3 py-2 border-b border-slate-700">
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800">
                      {currentTable.rows.map((row, rIdx) => (
                        <tr key={rIdx} className="hover:bg-slate-800/40">
                          {row.map((cell, cIdx) => (
                            <td key={cIdx} className="px-3 py-1.5 whitespace-nowrap">
                              {cell}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
