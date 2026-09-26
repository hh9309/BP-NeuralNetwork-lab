import React, { useState, useMemo, useEffect, useRef } from "react";
import {
  LineChart,
  Binary,
  Home,
  Users,
  Flower,
  Activity,
  CheckCircle2,
  AlertCircle,
  Play,
  RotateCcw,
  Sparkles,
  Zap,
  Sliders,
  Database,
  Target,
  RefreshCw,
  Layers,
  ArrowRight,
  TrendingUp,
  Table,
  Download,
  FileSpreadsheet,
  FileJson,
  Copy,
  Check,
  Eye,
  X,
  ThermometerSun,
  Wind,
} from "lucide-react";
import {
  ALL_CASE_STUDIES,
  splitTrainTest,
  generateThermalComfortDataset,
  generateLinearIdentityDataset,
  getMNISTDigitSamples,
  getHousePriceRegressionDataset,
  getCustomerChurnDataset,
  getIrisDataset,
  getNonlinearWaveDataset,
  exportCaseStudyToCSV,
  exportCaseStudyToJSON,
  triggerFileDownload,
} from "../lib/datasets";
import { CaseStudy, TrainingSample, ActivationType, OptimizerType } from "../types";
import { NeuralNetwork } from "../lib/neural-engine";
import { InlineMath } from "../lib/katex-helper";

interface CaseStudiesGalleryProps {
  currentCaseId: string;
  onSelectCase: (caseStudy: CaseStudy) => void;
  nn: NeuralNetwork;
}

export const CaseStudiesGallery: React.FC<CaseStudiesGalleryProps> = ({
  currentCaseId,
  onSelectCase,
  nn: globalNN,
}) => {
  const [selectedCase, setSelectedCase] = useState<CaseStudy>(
    ALL_CASE_STUDIES.find((c) => c.id === currentCaseId) || ALL_CASE_STUDIES[0]
  );

  // Workflow Phase: "train" (0.8 训练学习) vs "predict" (0.2 预测集评估) vs "interactive" (单样本在线推断)
  const [workflowTab, setWorkflowTab] = useState<"train" | "predict" | "interactive">("train");

  // Local Case Study Neural Network Engine (for dedicated isolated training/evaluating)
  const caseNNRef = useRef<NeuralNetwork>(
    new NeuralNetwork(
      selectedCase.recommendedTopology,
      Array(selectedCase.recommendedTopology.length - 1).fill(selectedCase.recommendedActivation).map((act, i, arr) =>
        i === arr.length - 1 && selectedCase.outputDim > 1 ? "softmax" : act
      ),
      selectedCase.recommendedLr,
      selectedCase.recommendedOptimizer || "adam"
    )
  );

  // Training state inside case study
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [trainEpoch, setTrainEpoch] = useState<number>(0);
  const [trainLoss, setTrainLoss] = useState<number>(0.5);
  const [lossHistory, setLossHistory] = useState<{ epoch: number; loss: number }[]>([]);

  // Dataset state for selected case (with split)
  const [caseSamples, setCaseSamples] = useState<TrainingSample[]>(selectedCase.samples);
  const [yNoise, setYNoise] = useState<number>(0.06);

  // 0.8 Train and 0.2 Test Split
  const { train: trainSet, test: testSet } = useMemo(() => {
    return splitTrainTest(caseSamples, 0.8);
  }, [caseSamples]);

  // Hovered item in scatter plot or test list
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  // Interactive Live Single-Sample Inference Inputs per Case
  const [thermalInputs, setThermalInputs] = useState({ temp: 28.0, rh: 65.0 });
  const [linearXInput, setLinearXInput] = useState<number>(0.65);
  const [houseInputs, setHouseInputs] = useState({ area: 110, rooms: 3, loc: 8.0 });
  const [churnInputs, setChurnInputs] = useState({ tenure: 12, charge: 75, calls: 3 });
  const [irisInputs, setIrisInputs] = useState({ sl: 5.8, sw: 3.0, pl: 4.2, pw: 1.3 });
  const [waveXInput, setWaveXInput] = useState<number>(0.45);
  const [selectedDigitIdx, setSelectedDigitIdx] = useState<number>(0);

  // Raw Data Export States
  const [isExportModalOpen, setIsExportModalOpen] = useState<boolean>(false);
  const [exportSubset, setExportSubset] = useState<"all" | "train" | "val">("all");
  const [exportFormat, setExportFormat] = useState<"csv" | "json">("csv");
  const [copied, setCopied] = useState<boolean>(false);

  const handleDownloadDataset = (format: "csv" | "json", subset: "all" | "train" | "val" = "all") => {
    if (format === "csv") {
      const csvStr = exportCaseStudyToCSV(selectedCase, subset, caseSamples);
      const filename = `${selectedCase.id}_dataset_${subset}.csv`;
      triggerFileDownload(csvStr, filename, "text/csv;charset=utf-8;");
    } else {
      const jsonStr = exportCaseStudyToJSON(selectedCase, subset, caseSamples);
      const filename = `${selectedCase.id}_dataset_${subset}.json`;
      triggerFileDownload(jsonStr, filename, "application/json;charset=utf-8;");
    }
  };

  const handleCopyExportText = () => {
    const text =
      exportFormat === "csv"
        ? exportCaseStudyToCSV(selectedCase, exportSubset, caseSamples)
        : exportCaseStudyToJSON(selectedCase, exportSubset, caseSamples);
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const mnistData = useMemo(() => getMNISTDigitSamples(), []);

  // Re-instantiate local NN when case changes
  useEffect(() => {
    const acts: ActivationType[] = Array(selectedCase.recommendedTopology.length - 1).fill(
      selectedCase.recommendedActivation
    );
    if (selectedCase.outputDim > 1) {
      acts[acts.length - 1] = "softmax";
    }

    const newNet = new NeuralNetwork(
      selectedCase.recommendedTopology,
      acts,
      selectedCase.recommendedLr,
      selectedCase.recommendedOptimizer || "adam"
    );
    caseNNRef.current = newNet;
    setTrainEpoch(0);
    setTrainLoss(0.5);
    setLossHistory([]);
    setIsTraining(false);
    setCaseSamples(selectedCase.samples);
  }, [selectedCase]);

  // Handle Case switch
  const handleCaseSelect = (cs: CaseStudy) => {
    setSelectedCase(cs);
    onSelectCase(cs);
  };

  // Regenerate Case 1 random thermal comfort / apparent temp data
  const handleRegenerateThermalData = () => {
    if (selectedCase.id === "thermal_comfort") {
      const newSamples = generateThermalComfortDataset(60, yNoise);
      setCaseSamples(newSamples);
      handleResetCaseNetwork();
    } else if (selectedCase.id === "linear_identity") {
      const newSamples = generateLinearIdentityDataset(60, yNoise);
      setCaseSamples(newSamples);
      handleResetCaseNetwork();
    }
  };

  // Reset local network
  const handleResetCaseNetwork = () => {
    setIsTraining(false);
    const acts: ActivationType[] = Array(selectedCase.recommendedTopology.length - 1).fill(
      selectedCase.recommendedActivation
    );
    if (selectedCase.outputDim > 1) {
      acts[acts.length - 1] = "softmax";
    }
    caseNNRef.current = new NeuralNetwork(
      selectedCase.recommendedTopology,
      acts,
      selectedCase.recommendedLr,
      selectedCase.recommendedOptimizer || "adam"
    );
    setTrainEpoch(0);
    setTrainLoss(0.5);
    setLossHistory([]);
  };

  // Train N steps on 0.8 Train Set
  const runTrainSteps = (epochsToRun: number) => {
    if (trainSet.length === 0) return;
    const net = caseNNRef.current;
    let lastLoss = trainLoss;
    const newHistory = [...lossHistory];

    for (let e = 0; e < epochsToRun; e++) {
      const metrics = net.trainEpoch(trainSet);
      lastLoss = metrics.loss;
      const curE = trainEpoch + e + 1;
      if (curE % 5 === 0 || e === epochsToRun - 1) {
        newHistory.push({ epoch: curE, loss: Number(lastLoss.toFixed(5)) });
      }
    }

    setTrainEpoch((prev) => prev + epochsToRun);
    setTrainLoss(lastLoss);
    setLossHistory(newHistory.slice(-60));
  };

  // Auto-train loop
  useEffect(() => {
    if (!isTraining) return;
    const interval = setInterval(() => {
      runTrainSteps(4);
    }, 60);
    return () => clearInterval(interval);
  }, [isTraining, trainEpoch, trainLoss, trainSet]);

  // Apply to Global Network
  const handleSyncToGlobal = () => {
    onSelectCase(selectedCase);
  };

  // Inference on 0.2 Test Set for Evaluation
  const testPredictions = useMemo(() => {
    const net = caseNNRef.current;
    return testSet.map((sample, idx) => {
      const fwd = net.forward(sample.input);
      const pred = fwd.output;
      return {
        index: idx,
        input: sample.input,
        target: sample.target,
        prediction: pred,
        label: sample.label,
      };
    });
  }, [testSet, trainEpoch, trainLoss]);

  // Inference on 0.8 Train Set (for regression comparison scatter)
  const trainPredictions = useMemo(() => {
    const net = caseNNRef.current;
    return trainSet.map((sample, idx) => {
      const fwd = net.forward(sample.input);
      return {
        index: idx,
        input: sample.input,
        target: sample.target,
        prediction: fwd.output,
        label: sample.label,
      };
    });
  }, [trainSet, trainEpoch, trainLoss]);

  // Calculate Evaluation Metrics on 0.2 Test Set
  const testMetrics = useMemo(() => {
    if (testPredictions.length === 0) return { loss: 0, r2: 0, mae: 0, acc: 0, correctCount: 0 };
    const n = testPredictions.length;

    if (selectedCase.category === "regression" || selectedCase.category === "approximation") {
      let sumSqErr = 0;
      let sumAbsErr = 0;
      let sumTarget = 0;

      testPredictions.forEach((item) => {
        const yTrue = item.target[0] ?? 0;
        const yPred = item.prediction[0] ?? 0;
        const diff = yPred - yTrue;
        sumSqErr += diff * diff;
        sumAbsErr += Math.abs(diff);
        sumTarget += yTrue;
      });

      const meanTarget = sumTarget / n;
      let sumTotSq = 0;
      testPredictions.forEach((item) => {
        const yTrue = item.target[0] ?? 0;
        sumTotSq += (yTrue - meanTarget) ** 2;
      });

      const mse = sumSqErr / n;
      const mae = sumAbsErr / n;
      const r2 = sumTotSq > 1e-8 ? Math.max(-1, Math.min(1, 1 - sumSqErr / sumTotSq)) : 1.0;

      return { loss: mse, r2, mae, acc: 0, correctCount: 0 };
    } else {
      // Classification (Binary / Multiclass)
      let correct = 0;
      let totalLoss = 0;

      testPredictions.forEach((item) => {
        if (selectedCase.outputDim === 1) {
          const yTrue = item.target[0] >= 0.5 ? 1 : 0;
          const yPred = item.prediction[0] >= 0.5 ? 1 : 0;
          if (yTrue === yPred) correct++;
          const p = Math.max(1e-7, Math.min(1 - 1e-7, item.prediction[0]));
          totalLoss -= yTrue * Math.log(p) + (1 - yTrue) * Math.log(1 - p);
        } else {
          // Multiclass argmax
          const maxTargetIdx = item.target.indexOf(Math.max(...item.target));
          const maxPredIdx = item.prediction.indexOf(Math.max(...item.prediction));
          if (maxTargetIdx === maxPredIdx) correct++;
          const p = Math.max(1e-7, item.prediction[maxTargetIdx] || 1e-7);
          totalLoss -= Math.log(p);
        }
      });

      return {
        loss: totalLoss / n,
        r2: 0,
        mae: 0,
        acc: (correct / n) * 100,
        correctCount: correct,
      };
    }
  }, [testPredictions, selectedCase]);

  // Live Interactive Single Sample Calculations
  const livePredictions = useMemo(() => {
    const net = caseNNRef.current;

    // Case 1: Thermal Comfort (2 in, 1 out: Temp & Humidity -> Apparent Temp)
    const normT = Math.max(0, Math.min(1, (thermalInputs.temp - 16) / 22));
    const normRH = Math.max(0, Math.min(1, (thermalInputs.rh - 20) / 70));
    const thermalRawPred = net.forward([normT, normRH]).output[0] ?? 0.5;
    const thermalApparentPred = (thermalRawPred * 30 + 15).toFixed(1);

    const deltaT = Math.max(0, thermalInputs.temp - 15);
    const humFactor = (thermalInputs.rh - 40) / 100;
    const thermalPhysicsTruth = (
      thermalInputs.temp + humFactor * (0.4 * deltaT + 0.03 * deltaT * deltaT)
    ).toFixed(1);

    // Case 1 (Legacy fallback): y = x
    const yEqualsXPred = (net.forward([linearXInput]).output[0] ?? 0).toFixed(4);

    // Case 2: MNIST
    const flat = mnistData.digitPatterns[selectedDigitIdx]?.flat() ?? [];
    const mnistPred = flat.length ? net.forward(flat).output : [];

    // Case 3: House
    const normHouse = [
      (houseInputs.area - 50) / 200,
      (houseInputs.rooms - 1) / 5,
      houseInputs.loc / 10,
    ];
    const housePriceVal = ((net.forward(normHouse).output[0] ?? 0.5) * 800 + 100).toFixed(1);

    // Case 4: Churn
    const normChurn = [churnInputs.tenure / 72, (churnInputs.charge - 20) / 100, churnInputs.calls / 9];
    const churnProbVal = ((net.forward(normChurn).output[0] ?? 0.5) * 100).toFixed(1);

    // Case 5: Iris
    const normIris = [
      (irisInputs.sl - 4.0) / 4.0,
      (irisInputs.sw - 2.0) / 2.5,
      (irisInputs.pl - 1.0) / 6.0,
      irisInputs.pw / 2.5,
    ];
    const irisPred = net.forward(normIris).output;

    // Case 6: Wave
    const waveTrueVal = (
      (Math.sin(2 * Math.PI * waveXInput) * Math.exp(-0.4 * waveXInput) +
        0.3 * Math.cos(4 * Math.PI * waveXInput) +
        1.2) /
      2.4
    ).toFixed(4);
    const wavePredVal = (net.forward([waveXInput]).output[0] ?? 0).toFixed(4);

    return {
      thermalApparentPred,
      thermalPhysicsTruth,
      yEqualsXPred,
      mnistPred,
      housePriceVal,
      churnProbVal,
      irisPred,
      waveTrueVal,
      wavePredVal,
    };
  }, [
    trainEpoch,
    thermalInputs,
    linearXInput,
    houseInputs,
    churnInputs,
    irisInputs,
    waveXInput,
    selectedDigitIdx,
    mnistData,
  ]);

  // Icon Resolver
  const getIcon = (iconName: string) => {
    switch (iconName) {
      case "ThermometerSun":
        return <ThermometerSun className="w-4 h-4" />;
      case "LineChart":
        return <LineChart className="w-4 h-4" />;
      case "Binary":
        return <Binary className="w-4 h-4" />;
      case "Home":
        return <Home className="w-4 h-4" />;
      case "Users":
        return <Users className="w-4 h-4" />;
      case "Flower":
        return <Flower className="w-4 h-4" />;
      case "Activity":
        return <Activity className="w-4 h-4" />;
      default:
        return <Layers className="w-4 h-4" />;
    }
  };

  return (
    <div className="space-y-4">
      {/* 6 Classic Cases Bento Selection Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {ALL_CASE_STUDIES.map((cs) => {
          const isCurrent = selectedCase.id === cs.id;
          return (
            <div
              key={cs.id}
              onClick={() => handleCaseSelect(cs)}
              className={`p-3 rounded-lg border transition-all cursor-pointer flex flex-col justify-between ${
                isCurrent
                  ? "bg-blue-50/50 border-blue-600 shadow-sm ring-1 ring-blue-500"
                  : "bg-white border-slate-200 hover:border-slate-300 hover:bg-slate-50/80 shadow-sm"
              }`}
            >
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <div
                    className={`w-7 h-7 rounded-md flex items-center justify-center ${
                      isCurrent
                        ? "bg-blue-600 text-white shadow-xs"
                        : "bg-slate-100 text-slate-700 border border-slate-200"
                    }`}
                  >
                    {getIcon(cs.iconName)}
                  </div>

                  <div className="flex items-center gap-1.5">
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-100 text-slate-600 font-semibold border border-slate-200">
                      0.8 训练 / 0.2 预测
                    </span>
                    {isCurrent && (
                      <span className="flex items-center gap-1 text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-800 font-bold border border-emerald-300">
                        <CheckCircle2 className="w-3 h-3" />
                        <span>当前探索</span>
                      </span>
                    )}
                  </div>
                </div>

                <div>
                  <h3 className="text-xs font-bold text-slate-900 leading-tight">{cs.title}</h3>
                  <p className="text-[11px] text-blue-700 font-semibold mt-0.5 leading-snug line-clamp-1">
                    {cs.subtitle}
                  </p>
                </div>

                <p className="text-[11px] text-slate-600 leading-relaxed line-clamp-2">
                  {cs.description}
                </p>
              </div>

              {/* Specs footer */}
              <div className="pt-2 mt-2 border-t border-slate-100 flex items-center justify-between text-[10px] font-mono text-slate-400">
                <span>拓扑 [{cs.recommendedTopology.join("-")}]</span>
                <span className="text-slate-600 font-semibold uppercase">{cs.recommendedActivation}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Main Interactive Learning & Prediction Workspace */}
      <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-sm space-y-4">
        {/* Header & Global Sync Action */}
        <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-200">
          <div className="space-y-0.5">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-blue-600 animate-pulse" />
              <h3 className="text-sm font-bold text-slate-900">
                经典案例全流程演练：{selectedCase.title}
              </h3>
            </div>
            <p className="text-xs text-slate-500">
              数据划分：<strong className="text-slate-700">80% 训练集 ({trainSet.length} 样本)</strong> 先行学习收敛，再在{" "}
              <strong className="text-slate-700">20% 预测集 ({testSet.length} 样本)</strong> 上检验模型泛化与误差表现。
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsExportModalOpen(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-white hover:bg-slate-50 text-slate-700 border border-slate-300 text-xs font-semibold shadow-xs transition cursor-pointer"
              title="导出当前案例的原始数据集 (CSV/JSON)"
            >
              <Download className="w-3.5 h-3.5 text-blue-600" />
              <span>导出原始数据</span>
            </button>

            <button
              onClick={handleSyncToGlobal}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold shadow-xs transition cursor-pointer"
            >
              <Zap className="w-3.5 h-3.5" />
              <span>载入至主实验室画布</span>
            </button>
          </div>
        </div>

        {/* 3-Stage Workflow Navigation Tabs */}
        <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-lg border border-slate-200">
          <button
            onClick={() => setWorkflowTab("train")}
            className={`flex-1 py-1.5 px-3 rounded-md text-xs font-bold transition flex items-center justify-center gap-1.5 cursor-pointer ${
              workflowTab === "train"
                ? "bg-white text-blue-700 shadow-xs border border-slate-200/80"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            <Layers className="w-3.5 h-3.5 text-blue-600" />
            <span>第一阶段：0.8 训练集模型学习 (Epoch {trainEpoch})</span>
          </button>

          <button
            onClick={() => setWorkflowTab("predict")}
            className={`flex-1 py-1.5 px-3 rounded-md text-xs font-bold transition flex items-center justify-center gap-1.5 cursor-pointer ${
              workflowTab === "predict"
                ? "bg-white text-blue-700 shadow-xs border border-slate-200/80"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            <TrendingUp className="w-3.5 h-3.5 text-emerald-600" />
            <span>第二阶段：0.2 预测集对比与散点图评估</span>
          </button>

          <button
            onClick={() => setWorkflowTab("interactive")}
            className={`flex-1 py-1.5 px-3 rounded-md text-xs font-bold transition flex items-center justify-center gap-1.5 cursor-pointer ${
              workflowTab === "interactive"
                ? "bg-white text-blue-700 shadow-xs border border-slate-200/80"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            <Sliders className="w-3.5 h-3.5 text-violet-600" />
            <span>第三阶段：在线实时单样本参数推断</span>
          </button>
        </div>

        {/* ================= STAGE 1: 0.8 TRAIN SET LEARNING ================= */}
        {workflowTab === "train" && (
          <div className="space-y-4">
            {/* Control Strip & Metrics Banner */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {/* Training Action Box */}
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 space-y-2.5">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                    <Zap className="w-3.5 h-3.5 text-amber-500" />
                    训练控制器 (80% 训练集)
                  </span>
                  <button
                    onClick={handleResetCaseNetwork}
                    title="重置网络权重"
                    className="p-1 text-slate-400 hover:text-slate-600 hover:bg-slate-200 rounded transition cursor-pointer"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                  </button>
                </div>

                <div className="flex gap-2">
                  <button
                    onClick={() => setIsTraining(!isTraining)}
                    className={`flex-1 py-1.5 px-3 rounded-md text-xs font-bold flex items-center justify-center gap-1.5 shadow-xs transition cursor-pointer ${
                      isTraining
                        ? "bg-amber-600 hover:bg-amber-700 text-white"
                        : "bg-blue-600 hover:bg-blue-700 text-white"
                    }`}
                  >
                    <Play className={`w-3.5 h-3.5 ${isTraining ? "animate-spin" : ""}`} />
                    <span>{isTraining ? "暂停学习" : "启动连续学习"}</span>
                  </button>

                  <button
                    onClick={() => runTrainSteps(10)}
                    disabled={isTraining}
                    className="py-1.5 px-2.5 rounded-md border border-slate-300 bg-white hover:bg-slate-100 text-slate-700 text-xs font-bold disabled:opacity-50 transition cursor-pointer"
                  >
                    +10 轮
                  </button>

                  <button
                    onClick={() => runTrainSteps(50)}
                    disabled={isTraining}
                    className="py-1.5 px-2.5 rounded-md border border-slate-300 bg-white hover:bg-slate-100 text-slate-700 text-xs font-bold disabled:opacity-50 transition cursor-pointer"
                  >
                    +50 轮
                  </button>
                </div>

                {/* Optional Case 1 Data Generation noise slider */}
                {(selectedCase.id === "thermal_comfort" || selectedCase.id === "linear_identity") && (
                  <div className="pt-2 border-t border-slate-200 space-y-1.5">
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="text-slate-500 font-medium">温湿度传感器扰动噪声强度:</span>
                      <span className="font-mono font-bold text-blue-700">{yNoise.toFixed(2)}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <input
                        type="range"
                        min="0.01"
                        max="0.15"
                        step="0.01"
                        value={yNoise}
                        onChange={(e) => setYNoise(Number(e.target.value))}
                        className="flex-1 accent-blue-600 cursor-pointer"
                      />
                      <button
                        onClick={handleRegenerateThermalData}
                        className="px-2 py-1 rounded bg-slate-200 hover:bg-slate-300 text-slate-700 text-[11px] font-bold flex items-center gap-1 cursor-pointer"
                      >
                        <RefreshCw className="w-3 h-3" />
                        <span>重新生成</span>
                      </button>
                    </div>
                  </div>
                )}
              </div>

              {/* Live Loss Metric Display */}
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 flex flex-col justify-between">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-bold text-slate-700">当前训练轮次与损失:</span>
                  <span className="font-mono text-[11px] px-2 py-0.5 rounded bg-blue-100 text-blue-800 font-bold">
                    Epoch {trainEpoch}
                  </span>
                </div>

                <div className="space-y-1 my-1">
                  <div className="text-2xl font-mono font-bold text-slate-900">
                    {trainLoss.toFixed(5)}
                  </div>
                  <div className="text-[11px] text-slate-500">
                    损失函数: <span className="font-mono text-slate-700 font-semibold">{selectedCase.recommendedLoss.toUpperCase()}</span>
                  </div>
                </div>

                {/* Micro sparkline */}
                <div className="h-8 bg-white rounded border border-slate-200 p-1 flex items-end gap-0.5 overflow-hidden">
                  {lossHistory.length === 0 ? (
                    <span className="text-[10px] text-slate-400 m-auto">点击启动学习后绘制损失轨迹</span>
                  ) : (
                    lossHistory.map((item, idx) => {
                      const maxL = Math.max(...lossHistory.map((h) => h.loss), 0.1);
                      const hPct = Math.max(10, Math.min(100, (item.loss / maxL) * 100));
                      return (
                        <div
                          key={idx}
                          style={{ height: `${hPct}%` }}
                          className="flex-1 bg-blue-500 rounded-t-[1px] hover:bg-blue-700 transition-all"
                          title={`Epoch ${item.epoch}: Loss ${item.loss}`}
                        />
                      );
                    })
                  )}
                </div>
              </div>

              {/* 80% Dataset Split Card */}
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-3 space-y-1.5">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-bold text-slate-700">数据集 80/20 分割规范:</span>
                  <span className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-800 font-bold">
                    总计 {caseSamples.length} 样本
                  </span>
                </div>

                <div className="space-y-1 text-xs">
                  <div className="flex justify-between p-1.5 bg-white rounded border border-slate-200">
                    <span className="text-slate-600">训练集 (0.8 Train):</span>
                    <span className="font-mono font-bold text-blue-700">{trainSet.length} 条样本 (用于梯度更新)</span>
                  </div>
                  <div className="flex justify-between p-1.5 bg-white rounded border border-slate-200">
                    <span className="text-slate-600">预测集 (0.2 Test):</span>
                    <span className="font-mono font-bold text-emerald-700">{testSet.length} 条样本 (用于评估泛化)</span>
                  </div>
                </div>

                <div className="text-[10px] text-slate-500 leading-tight">
                  提示：可在上方启动连续学习，观察训练损失平稳下降，随后切换至「第二阶段」查看预测集散点图对比。
                </div>
              </div>
            </div>

            {/* 80% Training Samples Preview Table */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <h4 className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                  <Table className="w-3.5 h-3.5 text-blue-600" />
                  0.8 训练集样本列表 (正在参与模型梯度反向传播):
                </h4>
                <div className="flex items-center gap-2">
                  <span className="text-[11px] text-slate-500 font-mono">共 {trainSet.length} 条训练记录</span>
                  <button
                    onClick={() => handleDownloadDataset("csv", "train")}
                    className="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold bg-slate-100 hover:bg-blue-50 text-slate-700 hover:text-blue-700 border border-slate-200 transition cursor-pointer"
                    title="下载当前 80% 训练集 CSV"
                  >
                    <Download className="w-3 h-3 text-blue-600" />
                    <span>导出此训练集 (CSV)</span>
                  </button>
                </div>
              </div>

              <div className="max-h-48 overflow-y-auto border border-slate-200 rounded-lg bg-white">
                <table className="w-full text-left text-xs border-collapse font-mono">
                  <thead className="bg-slate-50 border-b border-slate-200 sticky top-0 text-[11px] text-slate-600">
                    <tr>
                      <th className="p-2 font-semibold">序号</th>
                      <th className="p-2 font-semibold">输入向量 X (特征)</th>
                      <th className="p-2 font-semibold">真实标签/目标 Y</th>
                      <th className="p-2 font-semibold">样本业务描述</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100 text-[11px]">
                    {trainSet.slice(0, 15).map((s, idx) => (
                      <tr key={idx} className="hover:bg-slate-50">
                        <td className="p-2 text-slate-400">#{idx + 1}</td>
                        <td className="p-2 text-slate-800 font-bold">
                          [{s.input.map((v) => v.toFixed(3)).join(", ")}]
                        </td>
                        <td className="p-2 text-blue-700 font-bold">
                          [{s.target.map((v) => v.toFixed(3)).join(", ")}]
                        </td>
                        <td className="p-2 text-slate-600">{s.label ?? "--"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {trainSet.length > 15 && (
                  <div className="p-1.5 text-center text-[10px] text-slate-400 bg-slate-50 border-t border-slate-100">
                    ... 已省略展示其余 {trainSet.length - 15} 条训练样本
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ================= STAGE 2: 0.2 TEST PREDICTION & SCATTER PLOT ================= */}
        {workflowTab === "predict" && (
          <div className="space-y-4">
            {/* Evaluation Scoreboard */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
              <div className="bg-slate-50 border border-slate-200 rounded-lg p-2.5">
                <div className="text-[11px] text-slate-500">0.2 预测集样本数</div>
                <div className="text-lg font-mono font-bold text-slate-900">{testSet.length} 条</div>
                <div className="text-[10px] text-emerald-600 font-medium">模型未参与训练的未见数据</div>
              </div>

              {selectedCase.category === "regression" || selectedCase.category === "approximation" ? (
                <>
                  <div className="bg-slate-50 border border-slate-200 rounded-lg p-2.5">
                    <div className="text-[11px] text-slate-500">决定系数 (R² 拟合优度)</div>
                    <div
                      className={`text-lg font-mono font-bold ${
                        testMetrics.r2 > 0.85
                          ? "text-emerald-600"
                          : testMetrics.r2 > 0.5
                          ? "text-amber-600"
                          : "text-slate-700"
                      }`}
                    >
                      {testMetrics.r2.toFixed(4)}
                    </div>
                    <div className="text-[10px] text-slate-500">接近 1.0 表示拟合极其精确</div>
                  </div>

                  <div className="bg-slate-50 border border-slate-200 rounded-lg p-2.5">
                    <div className="text-[11px] text-slate-500">预测均方误差 (Test MSE)</div>
                    <div className="text-lg font-mono font-bold text-blue-700">
                      {testMetrics.loss.toFixed(5)}
                    </div>
                    <div className="text-[10px] text-slate-500">0.2 预测集上的均方误差</div>
                  </div>

                  <div className="bg-slate-50 border border-slate-200 rounded-lg p-2.5">
                    <div className="text-[11px] text-slate-500">平均绝对误差 (MAE)</div>
                    <div className="text-lg font-mono font-bold text-violet-700">
                      {testMetrics.mae.toFixed(4)}
                    </div>
                    <div className="text-[10px] text-slate-500">单点平均偏离真值的距离</div>
                  </div>
                </>
              ) : (
                <>
                  <div className="bg-slate-50 border border-slate-200 rounded-lg p-2.5">
                    <div className="text-[11px] text-slate-500">测试集分类准确率 (Accuracy)</div>
                    <div
                      className={`text-lg font-mono font-bold ${
                        testMetrics.acc >= 85
                          ? "text-emerald-600"
                          : testMetrics.acc >= 60
                          ? "text-amber-600"
                          : "text-rose-600"
                      }`}
                    >
                      {testMetrics.acc.toFixed(1)}%
                    </div>
                    <div className="text-[10px] text-slate-500">
                      正确 {testMetrics.correctCount} / 总计 {testSet.length}
                    </div>
                  </div>

                  <div className="bg-slate-50 border border-slate-200 rounded-lg p-2.5">
                    <div className="text-[11px] text-slate-500">测试集交叉熵 (Test Loss)</div>
                    <div className="text-lg font-mono font-bold text-blue-700">
                      {testMetrics.loss.toFixed(4)}
                    </div>
                    <div className="text-[10px] text-slate-500">Softmax / BCE 损失</div>
                  </div>

                  <div className="bg-slate-50 border border-slate-200 rounded-lg p-2.5">
                    <div className="text-[11px] text-slate-500">分类状态评估</div>
                    <div className="text-xs font-bold text-slate-800 mt-1">
                      {testMetrics.acc >= 80 ? "🟢 泛化性能优异" : "🟡 建议增加训练轮次"}
                    </div>
                    <div className="text-[10px] text-slate-500">已学习 {trainEpoch} 轮</div>
                  </div>
                </>
              )}
            </div>

            {/* Specialized True vs. Predicted Scatter Plot for Case 1 (and Regression Cases) */}
            {(selectedCase.id === "thermal_comfort" ||
              selectedCase.id === "linear_identity" ||
              selectedCase.id === "house_price" ||
              selectedCase.id === "nonlinear_wave") && (
              <div className="space-y-2 bg-slate-50 p-3.5 rounded-lg border border-slate-200">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div>
                    <h4 className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
                      <TrendingUp className="w-4 h-4 text-blue-600" />
                      真实值 (Ground Truth y) vs 预测值 (Predicted ŷ) 对比散点图:
                    </h4>
                    <p className="text-[11px] text-slate-500">
                      包含 80% 训练集散点与 20% 测试集预测对比。点越贴近 45° 红色理想参考线 (y = ŷ)，代表预测越精准。
                    </p>
                  </div>

                  <div className="flex items-center gap-3 text-[11px] font-medium">
                    <span className="flex items-center gap-1">
                      <span className="w-2.5 h-2.5 rounded-full bg-slate-400 inline-block" />
                      0.8 训练集散点
                    </span>
                    <span className="flex items-center gap-1">
                      <span className="w-3 h-3 rounded-full bg-emerald-500 ring-2 ring-emerald-300 inline-block" />
                      0.2 预测集样本
                    </span>
                    <span className="flex items-center gap-1">
                      <span className="w-4 h-0.5 bg-rose-500 inline-block" />
                      理想恒等线 y = ŷ
                    </span>
                  </div>
                </div>

                {/* SVG Scatter Plot Canvas */}
                <div className="relative bg-white border border-slate-200 rounded-lg p-3">
                  <svg viewBox="0 0 500 320" className="w-full h-64 select-none">
                    {/* Grid lines */}
                    {[0, 0.25, 0.5, 0.75, 1.0].map((v, i) => {
                      const pos = 30 + v * 240;
                      const yPos = 270 - v * 240;
                      return (
                        <g key={i}>
                          {/* Horizontal grid */}
                          <line x1="30" y1={yPos} x2="470" y2={yPos} stroke="#f1f5f9" strokeWidth="1" />
                          <text x="24" y={yPos + 3} textAnchor="end" fontSize="9" fill="#94a3b8" fontFamily="monospace">
                            {v.toFixed(2)}
                          </text>

                          {/* Vertical grid */}
                          <line x1={pos} y1="30" x2={pos} y2="270" stroke="#f1f5f9" strokeWidth="1" />
                          <text x={pos} y="284" textAnchor="middle" fontSize="9" fill="#94a3b8" fontFamily="monospace">
                            {v.toFixed(2)}
                          </text>
                        </g>
                      );
                    })}

                    {/* Coordinate axes */}
                    <line x1="30" y1="270" x2="470" y2="270" stroke="#cbd5e1" strokeWidth="1.5" />
                    <line x1="30" y1="30" x2="30" y2="270" stroke="#cbd5e1" strokeWidth="1.5" />

                    {/* Axis Labels */}
                    <text x="250" y="305" textAnchor="middle" fontSize="11" fill="#475569" fontWeight="bold">
                      真实真实值 y (Ground Truth)
                    </text>
                    <text
                      x="12"
                      y="150"
                      textAnchor="middle"
                      fontSize="11"
                      fill="#475569"
                      fontWeight="bold"
                      transform="rotate(-90 12 150)"
                    >
                      模型预测值 ŷ (Predicted)
                    </text>

                    {/* 45° Ideal Reference Line: y = x */}
                    <line
                      x1="30"
                      y1="270"
                      x2="270"
                      y2="30"
                      stroke="#f43f5e"
                      strokeWidth="1.5"
                      strokeDasharray="4 3"
                    />

                    {/* 0.8 Train Set Scatter Dots (Soft background context) */}
                    {trainPredictions.map((pt, i) => {
                      const yTrue = Math.max(0, Math.min(1, pt.target[0] ?? 0));
                      const yPred = Math.max(0, Math.min(1, pt.prediction[0] ?? 0));
                      const cx = 30 + yTrue * 240;
                      const cy = 270 - yPred * 240;
                      return (
                        <circle
                          key={`train-${i}`}
                          cx={cx}
                          cy={cy}
                          r="2.5"
                          fill="#94a3b8"
                          opacity="0.6"
                        />
                      );
                    })}

                    {/* 0.2 Test Set Scatter Dots + Residual Error Lines */}
                    {testPredictions.map((pt, i) => {
                      const yTrue = Math.max(0, Math.min(1, pt.target[0] ?? 0));
                      const yPred = Math.max(0, Math.min(1, pt.prediction[0] ?? 0));
                      const cx = 30 + yTrue * 240;
                      const cy = 270 - yPred * 240;
                      const idealCy = 270 - yTrue * 240;
                      const isHovered = hoveredIndex === i;

                      return (
                        <g
                          key={`test-${i}`}
                          onMouseEnter={() => setHoveredIndex(i)}
                          onMouseLeave={() => setHoveredIndex(null)}
                          className="cursor-pointer"
                        >
                          {/* Residual line to 45 degree line */}
                          <line
                            x1={cx}
                            y1={cy}
                            x2={cx}
                            y2={idealCy}
                            stroke={Math.abs(yPred - yTrue) < 0.05 ? "#10b981" : "#f59e0b"}
                            strokeWidth={isHovered ? 2 : 1}
                            strokeDasharray="2 2"
                            opacity={isHovered ? 1 : 0.6}
                          />

                          {/* Test point circle */}
                          <circle
                            cx={cx}
                            cy={cy}
                            r={isHovered ? 6 : 4.5}
                            fill="#10b981"
                            stroke="#065f46"
                            strokeWidth="1.5"
                            className="transition-all duration-150"
                          />

                          {/* Highlight pulse if hovered */}
                          {isHovered && (
                            <circle
                              cx={cx}
                              cy={cy}
                              r="10"
                              fill="none"
                              stroke="#10b981"
                              strokeWidth="1.5"
                              opacity="0.5"
                            />
                          )}
                        </g>
                      );
                    })}
                  </svg>

                  {/* Active Point Floating Tooltip */}
                  {hoveredIndex !== null && testPredictions[hoveredIndex] && (
                    <div className="mt-2 p-2 bg-slate-900 text-white rounded text-[11px] font-mono flex flex-wrap items-center justify-between gap-2 shadow-md">
                      <div>
                        <span className="text-emerald-400 font-bold">测试样本 #{hoveredIndex + 1}: </span>
                        <span>{testPredictions[hoveredIndex].label ?? ""}</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <span>真实值 y: {testPredictions[hoveredIndex].target[0].toFixed(4)}</span>
                        <span>预测值 ŷ: {testPredictions[hoveredIndex].prediction[0].toFixed(4)}</span>
                        <span className="text-amber-300 font-bold">
                          残差 e: {(testPredictions[hoveredIndex].prediction[0] - testPredictions[hoveredIndex].target[0]).toFixed(4)}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* 0.2 Test Set Results Table / Cards */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <h4 className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                  <Target className="w-3.5 h-3.5 text-emerald-600" />
                  0.2 预测集样本逐条验证与误差对比明细:
                </h4>
                <div className="flex items-center gap-2">
                  <span className="text-[11px] text-slate-500 font-mono">共 {testPredictions.length} 条预测记录</span>
                  <button
                    onClick={() => handleDownloadDataset("csv", "val")}
                    className="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-semibold bg-slate-100 hover:bg-emerald-50 text-slate-700 hover:text-emerald-700 border border-slate-200 transition cursor-pointer"
                    title="下载当前 20% 验证/预测集 CSV"
                  >
                    <Download className="w-3 h-3 text-emerald-600" />
                    <span>导出此预测集 (CSV)</span>
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
                {testPredictions.map((item, idx) => {
                  const isReg = selectedCase.category === "regression" || selectedCase.category === "approximation";
                  let isAccurate = false;
                  let errorDiff = 0;

                  if (isReg) {
                    errorDiff = Math.abs((item.prediction[0] ?? 0) - (item.target[0] ?? 0));
                    isAccurate = errorDiff < 0.08;
                  } else {
                    if (selectedCase.outputDim === 1) {
                      isAccurate = (item.prediction[0] >= 0.5 ? 1 : 0) === (item.target[0] >= 0.5 ? 1 : 0);
                    } else {
                      const maxT = item.target.indexOf(Math.max(...item.target));
                      const maxP = item.prediction.indexOf(Math.max(...item.prediction));
                      isAccurate = maxT === maxP;
                    }
                  }

                  return (
                    <div
                      key={idx}
                      onMouseEnter={() => setHoveredIndex(idx)}
                      onMouseLeave={() => setHoveredIndex(null)}
                      className={`p-2.5 rounded-lg border transition-all ${
                        hoveredIndex === idx
                          ? "ring-2 ring-emerald-500 bg-emerald-50/50 border-emerald-400"
                          : "bg-white border-slate-200 hover:border-slate-300"
                      }`}
                    >
                      <div className="flex items-center justify-between text-xs pb-1.5 border-b border-slate-100">
                        <span className="font-mono font-bold text-slate-700">测试样本 #{idx + 1}</span>
                        <span
                          className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${
                            isAccurate
                              ? "bg-emerald-100 text-emerald-800"
                              : "bg-rose-100 text-rose-800"
                          }`}
                        >
                          {isAccurate ? "✓ 预测精准" : "⚠ 存在偏差"}
                        </span>
                      </div>

                      <div className="mt-1.5 space-y-1 text-xs font-mono">
                        <div className="text-[11px] text-slate-500 truncate" title={String(item.label)}>
                          {item.label || `输入: [${item.input.map((v) => v.toFixed(2)).join(", ")}]`}
                        </div>

                        <div className="flex justify-between">
                          <span className="text-slate-500">真实目标 y:</span>
                          <span className="font-bold text-slate-800">
                            [{item.target.map((v) => v.toFixed(3)).join(", ")}]
                          </span>
                        </div>

                        <div className="flex justify-between">
                          <span className="text-slate-500">模型推断 ŷ:</span>
                          <span className="font-bold text-blue-700">
                            [{item.prediction.map((v) => v.toFixed(3)).join(", ")}]
                          </span>
                        </div>

                        {isReg && (
                          <div className="flex justify-between text-[10px] pt-1 border-t border-slate-100">
                            <span className="text-slate-400">绝对残差 |ŷ - y|:</span>
                            <span className={errorDiff < 0.05 ? "text-emerald-600 font-bold" : "text-amber-600 font-bold"}>
                              {errorDiff.toFixed(4)}
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        )}

        {/* ================= STAGE 3: LIVE SINGLE-SAMPLE INFERENCE ================= */}
        {workflowTab === "interactive" && (
          <div className="space-y-4">
            <div className="bg-slate-50 p-3 rounded-lg border border-slate-200 space-y-1">
              <h4 className="text-xs font-bold text-slate-900 flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-violet-600" />
                在线单样本实时推断实验室：{selectedCase.title}
              </h4>
              <p className="text-[11px] text-slate-500">
                通过随意调节下方特征控件，实时触发已训练神经网络的前向推断 (Forward Pass)，检验模型对任意输入点预测的平滑性与鲁棒性。
              </p>
            </div>

            {/* Case 1: Thermal Comfort (2-In, 1-Out) Interactive */}
            {selectedCase.id === "thermal_comfort" && (
              <div className="bg-white p-4 rounded-lg border border-slate-200 space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-800 flex items-center gap-1.5">
                    <ThermometerSun className="w-4 h-4 text-orange-500" />
                    双维度环境特征实时推断调节：
                  </span>
                  <span className="text-[11px] text-slate-500">
                    实时期望体感温度与智能空调策略联动
                  </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Feature 1: Temperature */}
                  <div className="p-3 bg-orange-50/50 rounded-lg border border-orange-200/60 space-y-2">
                    <div className="flex justify-between items-center text-xs">
                      <span className="font-semibold text-slate-700">① 环境测量温度 T (16 ~ 38°C):</span>
                      <span className="font-mono text-sm font-bold text-orange-600">{thermalInputs.temp.toFixed(1)} °C</span>
                    </div>
                    <input
                      type="range"
                      min="16.0"
                      max="38.0"
                      step="0.5"
                      value={thermalInputs.temp}
                      onChange={(e) => setThermalInputs((prev) => ({ ...prev, temp: Number(e.target.value) }))}
                      className="w-full accent-orange-500 cursor-pointer"
                    />
                    <div className="flex justify-between text-[10px] text-slate-500">
                      <span>16.0°C (凉爽)</span>
                      <span>26.0°C (常温)</span>
                      <span>38.0°C (酷暑)</span>
                    </div>
                  </div>

                  {/* Feature 2: Humidity */}
                  <div className="p-3 bg-cyan-50/50 rounded-lg border border-cyan-200/60 space-y-2">
                    <div className="flex justify-between items-center text-xs">
                      <span className="font-semibold text-slate-700">② 相对环境湿度 RH (20 ~ 90%):</span>
                      <span className="font-mono text-sm font-bold text-cyan-700">{thermalInputs.rh.toFixed(0)} %</span>
                    </div>
                    <input
                      type="range"
                      min="20"
                      max="90"
                      step="1"
                      value={thermalInputs.rh}
                      onChange={(e) => setThermalInputs((prev) => ({ ...prev, rh: Number(e.target.value) }))}
                      className="w-full accent-cyan-600 cursor-pointer"
                    />
                    <div className="flex justify-between text-[10px] text-slate-500">
                      <span>20% (干燥)</span>
                      <span>50% (适宜)</span>
                      <span>90% (高湿桑拿)</span>
                    </div>
                  </div>
                </div>

                {/* Output Comparison & Status */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1">
                  <div className="p-2.5 bg-slate-50 rounded border border-slate-200 text-center">
                    <div className="text-[11px] text-slate-500">气象/热工理论体感值</div>
                    <div className="text-base font-mono font-bold text-slate-800">
                      {livePredictions.thermalPhysicsTruth} °C
                    </div>
                  </div>

                  <div className="p-2.5 bg-blue-50 rounded border border-blue-200 text-center">
                    <div className="text-[11px] text-blue-700">BP 神经网络实时预测 ŷ</div>
                    <div className="text-base font-mono font-bold text-blue-700">
                      {livePredictions.thermalApparentPred} °C
                    </div>
                  </div>

                  <div className="p-2.5 bg-emerald-50 rounded border border-emerald-200 text-center">
                    <div className="text-[11px] text-emerald-700">预测残差 |ŷ - T_true|</div>
                    <div className="text-base font-mono font-bold text-emerald-700">
                      {Math.abs(Number(livePredictions.thermalApparentPred) - Number(livePredictions.thermalPhysicsTruth)).toFixed(2)} °C
                    </div>
                  </div>
                </div>

                {/* HVAC Smart Decision Suggestion */}
                <div className={`p-3 rounded-lg border flex items-center justify-between text-xs ${
                  Number(livePredictions.thermalApparentPred) > 36
                    ? "bg-rose-50 border-rose-200 text-rose-800"
                    : Number(livePredictions.thermalApparentPred) > 28
                    ? "bg-amber-50 border-amber-200 text-amber-800"
                    : Number(livePredictions.thermalApparentPred) >= 20
                    ? "bg-emerald-50 border-emerald-200 text-emerald-800"
                    : "bg-blue-50 border-blue-200 text-blue-800"
                }`}>
                  <div className="flex items-center gap-2">
                    <Wind className="w-4 h-4 shrink-0" />
                    <div>
                      <span className="font-bold">
                        {Number(livePredictions.thermalApparentPred) > 36
                          ? "体感评级：极度闷热 (中暑危险)"
                          : Number(livePredictions.thermalApparentPred) > 28
                          ? "体感评级：闷热潮湿 (热负荷高)"
                          : Number(livePredictions.thermalApparentPred) >= 20
                          ? "体感评级：黄金舒适区 (宜人)"
                          : "体感评级：清凉微寒"}
                      </span>
                      <span className="ml-2 text-[11px] opacity-80">
                        {Number(livePredictions.thermalApparentPred) > 36
                          ? "智能空调控制：开启强劲制冷 (24°C) + 深度除湿"
                          : Number(livePredictions.thermalApparentPred) > 28
                          ? "智能空调控制：开启变频恒温 (26°C) + 低风挡除湿"
                          : Number(livePredictions.thermalApparentPred) >= 20
                          ? "智能空调控制：自然风循环节能模式"
                          : "智能空调控制：微风送风或待机"}
                      </span>
                    </div>
                  </div>
                  <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-white/80 border border-current shadow-xs shrink-0">
                    实时联动
                  </span>
                </div>
              </div>
            )}

            {/* Case 1 (Legacy Fallback): y = x Interactive */}
            {selectedCase.id === "linear_identity" && (
              <div className="bg-white p-4 rounded-lg border border-slate-200 space-y-3">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-bold text-slate-700">输入自变量 X (0.0 ~ 1.0):</span>
                  <span className="font-mono text-sm font-bold text-blue-700">{linearXInput.toFixed(3)}</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.01"
                  value={linearXInput}
                  onChange={(e) => setLinearXInput(Number(e.target.value))}
                  className="w-full accent-blue-600 cursor-pointer"
                />

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2">
                  <div className="p-2.5 bg-slate-50 rounded border border-slate-200 text-center">
                    <div className="text-[11px] text-slate-500">理想恒等值 y=x</div>
                    <div className="text-base font-mono font-bold text-slate-800">{linearXInput.toFixed(4)}</div>
                  </div>
                  <div className="p-2.5 bg-blue-50 rounded border border-blue-200 text-center">
                    <div className="text-[11px] text-blue-700">神经网络预测值 ŷ</div>
                    <div className="text-base font-mono font-bold text-blue-700">{livePredictions.yEqualsXPred}</div>
                  </div>
                  <div className="p-2.5 bg-emerald-50 rounded border border-emerald-200 text-center">
                    <div className="text-[11px] text-emerald-700">拟合偏差 |ŷ - x|</div>
                    <div className="text-base font-mono font-bold text-emerald-700">
                      {Math.abs(Number(livePredictions.yEqualsXPred) - linearXInput).toFixed(4)}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Case 2: MNIST Interactive */}
            {selectedCase.id === "mnist_digit" && (
              <div className="bg-white p-4 rounded-lg border border-slate-200 space-y-3">
                <h5 className="text-xs font-bold text-slate-800">选择手写字符并观察 Softmax 5分类输出:</h5>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <div className="flex gap-2">
                      {[0, 1, 2, 3, 4].map((d) => (
                        <button
                          key={d}
                          onClick={() => setSelectedDigitIdx(d)}
                          className={`w-9 h-9 rounded font-mono text-xs font-bold transition cursor-pointer border ${
                            selectedDigitIdx === d
                              ? "bg-blue-600 text-white border-blue-600 shadow-xs"
                              : "bg-white text-slate-700 border-slate-300 hover:bg-slate-100"
                          }`}
                        >
                          {d}
                        </button>
                      ))}
                    </div>
                    <div className="p-2 bg-slate-900 rounded border border-slate-800 inline-block">
                      <div className="grid grid-cols-8 gap-0.5 w-24 h-24">
                        {mnistData.digitPatterns[selectedDigitIdx]?.flat().map((pixel, i) => (
                          <div
                            key={i}
                            className={`w-full h-full rounded-[1px] ${
                              pixel ? "bg-sky-400" : "bg-slate-950"
                            }`}
                          />
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="space-y-1.5">
                    <span className="text-xs text-slate-500 font-semibold">输出层 Softmax 概率分布:</span>
                    <div className="space-y-1 font-mono text-xs">
                      {livePredictions.mnistPred &&
                        livePredictions.mnistPred.map((prob, idx) => {
                          const pct = Math.max(0, Math.min(100, prob * 100));
                          const isWinner = prob === Math.max(...livePredictions.mnistPred);
                          return (
                            <div key={idx} className="space-y-0.5">
                              <div className="flex justify-between text-[10px]">
                                <span className={isWinner ? "text-blue-700 font-bold" : "text-slate-500"}>
                                  数字 {idx} {isWinner ? "★ (Top 1 识别)" : ""}
                                </span>
                                <span className={isWinner ? "text-emerald-600 font-bold" : "text-slate-600"}>
                                  {pct.toFixed(2)}%
                                </span>
                              </div>
                              <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden">
                                <div
                                  className={`h-full transition-all duration-300 ${
                                    isWinner ? "bg-emerald-500" : "bg-blue-600"
                                  }`}
                                  style={{ width: `${pct}%` }}
                                />
                              </div>
                            </div>
                          );
                        })}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Case 3: House Price Interactive */}
            {selectedCase.id === "house_price" && (
              <div className="bg-white p-4 rounded-lg border border-slate-200 space-y-3">
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div className="space-y-1 bg-slate-50 p-2.5 rounded border border-slate-200">
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-500">建筑面积:</span>
                      <span className="font-mono text-blue-700 font-bold">{houseInputs.area} m²</span>
                    </div>
                    <input
                      type="range"
                      min="50"
                      max="250"
                      value={houseInputs.area}
                      onChange={(e) => setHouseInputs({ ...houseInputs, area: Number(e.target.value) })}
                      className="w-full accent-blue-600 cursor-pointer"
                    />
                  </div>

                  <div className="space-y-1 bg-slate-50 p-2.5 rounded border border-slate-200">
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-500">居室数量:</span>
                      <span className="font-mono text-blue-700 font-bold">{houseInputs.rooms} 室</span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="5"
                      value={houseInputs.rooms}
                      onChange={(e) => setHouseInputs({ ...houseInputs, rooms: Number(e.target.value) })}
                      className="w-full accent-blue-600 cursor-pointer"
                    />
                  </div>

                  <div className="space-y-1 bg-slate-50 p-2.5 rounded border border-slate-200">
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-500">地段综合分:</span>
                      <span className="font-mono text-blue-700 font-bold">{houseInputs.loc.toFixed(1)} 分</span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="10"
                      step="0.5"
                      value={houseInputs.loc}
                      onChange={(e) => setHouseInputs({ ...houseInputs, loc: Number(e.target.value) })}
                      className="w-full accent-blue-600 cursor-pointer"
                    />
                  </div>
                </div>

                <div className="p-3 bg-blue-50 rounded-lg flex items-center justify-between border border-blue-200">
                  <span className="text-xs text-blue-900 font-semibold">模型在线回归估值 (Forward Price):</span>
                  <span className="text-lg font-bold text-blue-700 font-mono">
                    {livePredictions.housePriceVal} 万元 CNY
                  </span>
                </div>
              </div>
            )}

            {/* Case 4: Customer Churn Interactive */}
            {selectedCase.id === "customer_churn" && (
              <div className="bg-white p-4 rounded-lg border border-slate-200 space-y-3">
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div className="space-y-1 bg-slate-50 p-2.5 rounded border border-slate-200">
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-500">在网时长:</span>
                      <span className="font-mono text-blue-700 font-bold">{churnInputs.tenure} 月</span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="72"
                      value={churnInputs.tenure}
                      onChange={(e) => setChurnInputs({ ...churnInputs, tenure: Number(e.target.value) })}
                      className="w-full accent-blue-600 cursor-pointer"
                    />
                  </div>

                  <div className="space-y-1 bg-slate-50 p-2.5 rounded border border-slate-200">
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-500">月套餐资费:</span>
                      <span className="font-mono text-blue-700 font-bold">${churnInputs.charge}</span>
                    </div>
                    <input
                      type="range"
                      min="20"
                      max="120"
                      value={churnInputs.charge}
                      onChange={(e) => setChurnInputs({ ...churnInputs, charge: Number(e.target.value) })}
                      className="w-full accent-blue-600 cursor-pointer"
                    />
                  </div>

                  <div className="space-y-1 bg-slate-50 p-2.5 rounded border border-slate-200">
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-500">投诉工单量:</span>
                      <span className="font-mono text-blue-700 font-bold">{churnInputs.calls} 次</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="9"
                      value={churnInputs.calls}
                      onChange={(e) => setChurnInputs({ ...churnInputs, calls: Number(e.target.value) })}
                      className="w-full accent-blue-600 cursor-pointer"
                    />
                  </div>
                </div>

                <div className="p-3 bg-slate-50 rounded-lg flex items-center justify-between border border-slate-200">
                  <span className="text-xs text-slate-700 font-semibold">客户流失风险概率预警:</span>
                  <div className="flex items-center gap-2">
                    <span
                      className={`text-base font-bold font-mono ${
                        Number(livePredictions.churnProbVal) > 50 ? "text-rose-600" : "text-emerald-600"
                      }`}
                    >
                      {livePredictions.churnProbVal}%
                    </span>
                    <span
                      className={`text-xs px-2 py-0.5 rounded font-bold ${
                        Number(livePredictions.churnProbVal) > 50
                          ? "bg-rose-100 text-rose-800"
                          : "bg-emerald-100 text-emerald-800"
                      }`}
                    >
                      {Number(livePredictions.churnProbVal) > 50 ? "⚠️ 高流失风险" : "🟢 稳定留存客户"}
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Case 5: Iris Interactive */}
            {selectedCase.id === "iris_species" && (
              <div className="bg-white p-4 rounded-lg border border-slate-200 space-y-3">
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                  <div className="space-y-1 bg-slate-50 p-2 rounded border border-slate-200">
                    <div className="flex justify-between text-[11px]">
                      <span className="text-slate-500">花萼长:</span>
                      <span className="font-mono font-bold text-blue-700">{irisInputs.sl.toFixed(1)}</span>
                    </div>
                    <input
                      type="range"
                      min="4.0"
                      max="8.0"
                      step="0.1"
                      value={irisInputs.sl}
                      onChange={(e) => setIrisInputs({ ...irisInputs, sl: Number(e.target.value) })}
                      className="w-full accent-blue-600 cursor-pointer"
                    />
                  </div>

                  <div className="space-y-1 bg-slate-50 p-2 rounded border border-slate-200">
                    <div className="flex justify-between text-[11px]">
                      <span className="text-slate-500">花萼宽:</span>
                      <span className="font-mono font-bold text-blue-700">{irisInputs.sw.toFixed(1)}</span>
                    </div>
                    <input
                      type="range"
                      min="2.0"
                      max="4.5"
                      step="0.1"
                      value={irisInputs.sw}
                      onChange={(e) => setIrisInputs({ ...irisInputs, sw: Number(e.target.value) })}
                      className="w-full accent-blue-600 cursor-pointer"
                    />
                  </div>

                  <div className="space-y-1 bg-slate-50 p-2 rounded border border-slate-200">
                    <div className="flex justify-between text-[11px]">
                      <span className="text-slate-500">花瓣长:</span>
                      <span className="font-mono font-bold text-blue-700">{irisInputs.pl.toFixed(1)}</span>
                    </div>
                    <input
                      type="range"
                      min="1.0"
                      max="7.0"
                      step="0.1"
                      value={irisInputs.pl}
                      onChange={(e) => setIrisInputs({ ...irisInputs, pl: Number(e.target.value) })}
                      className="w-full accent-blue-600 cursor-pointer"
                    />
                  </div>

                  <div className="space-y-1 bg-slate-50 p-2 rounded border border-slate-200">
                    <div className="flex justify-between text-[11px]">
                      <span className="text-slate-500">花瓣宽:</span>
                      <span className="font-mono font-bold text-blue-700">{irisInputs.pw.toFixed(1)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="2.5"
                      step="0.1"
                      value={irisInputs.pw}
                      onChange={(e) => setIrisInputs({ ...irisInputs, pw: Number(e.target.value) })}
                      className="w-full accent-blue-600 cursor-pointer"
                    />
                  </div>
                </div>

                <div className="space-y-1.5 pt-1">
                  <span className="text-xs text-slate-600 font-semibold">鸢尾花物种 Softmax 推断概率:</span>
                  <div className="grid grid-cols-3 gap-2">
                    {["山鸢尾 (Setosa)", "杂色鸢尾 (Versicolor)", "维吉尼亚 (Virginica)"].map((species, i) => {
                      const prob = livePredictions.irisPred[i] ?? 0;
                      const pct = (prob * 100).toFixed(1);
                      const isWinner = prob === Math.max(...livePredictions.irisPred);
                      return (
                        <div
                          key={i}
                          className={`p-2 rounded border text-center ${
                            isWinner
                              ? "bg-emerald-50 border-emerald-300 ring-1 ring-emerald-400"
                              : "bg-slate-50 border-slate-200"
                          }`}
                        >
                          <div className="text-[11px] font-bold text-slate-800">{species}</div>
                          <div className="text-sm font-mono font-bold text-blue-700 mt-0.5">{pct}%</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}

            {/* Case 6: Wave Interactive */}
            {selectedCase.id === "nonlinear_wave" && (
              <div className="bg-white p-4 rounded-lg border border-slate-200 space-y-3">
                <div className="flex justify-between items-center text-xs">
                  <span className="font-bold text-slate-700">自变量 X (0.0 ~ 1.0):</span>
                  <span className="font-mono text-sm font-bold text-blue-700">{waveXInput.toFixed(3)}</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.01"
                  value={waveXInput}
                  onChange={(e) => setWaveXInput(Number(e.target.value))}
                  className="w-full accent-blue-600 cursor-pointer"
                />

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2">
                  <div className="p-2.5 bg-slate-50 rounded border border-slate-200 text-center">
                    <div className="text-[11px] text-slate-500">真实复合波形解析解 Y</div>
                    <div className="text-base font-mono font-bold text-slate-800">{livePredictions.waveTrueVal}</div>
                  </div>
                  <div className="p-2.5 bg-blue-50 rounded border border-blue-200 text-center">
                    <div className="text-[11px] text-blue-700">神经网络逼近推断 Ŷ</div>
                    <div className="text-base font-mono font-bold text-blue-700">{livePredictions.wavePredVal}</div>
                  </div>
                  <div className="p-2.5 bg-emerald-50 rounded border border-emerald-200 text-center">
                    <div className="text-[11px] text-emerald-700">逼近残差 |Ŷ - Y|</div>
                    <div className="text-base font-mono font-bold text-emerald-700">
                      {Math.abs(Number(livePredictions.wavePredVal) - Number(livePredictions.waveTrueVal)).toFixed(4)}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ================= RAW DATA EXPORT MODAL ================= */}
      {isExportModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-xs animate-in fade-in duration-150">
          <div className="bg-white rounded-xl shadow-2xl border border-slate-200 w-full max-w-3xl flex flex-col max-h-[90vh] overflow-hidden">
            {/* Modal Header */}
            <div className="px-5 py-4 border-b border-slate-200 flex items-center justify-between bg-slate-50">
              <div className="flex items-center gap-2.5">
                <div className="p-2 rounded-lg bg-blue-100 text-blue-700">
                  <Download className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-slate-900">
                    原始数据集导出：{selectedCase.title}
                  </h3>
                  <p className="text-xs text-slate-500">
                    支持按 100% 全量样本、80% 训练集或 20% 验证预测集导出标准 CSV 与 JSON
                  </p>
                </div>
              </div>
              <button
                onClick={() => setIsExportModalOpen(false)}
                className="p-1.5 rounded-lg hover:bg-slate-200 text-slate-400 hover:text-slate-700 transition cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Controls & Options */}
            <div className="p-5 space-y-4 overflow-y-auto flex-1 text-xs">
              {/* Option Selectors */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 bg-slate-50 p-3.5 rounded-lg border border-slate-200">
                {/* 1. Subset Scope Selection */}
                <div className="space-y-1.5">
                  <label className="font-bold text-slate-700 block">选择数据导出范围：</label>
                  <div className="grid grid-cols-3 gap-1.5">
                    <button
                      onClick={() => setExportSubset("all")}
                      className={`py-1.5 px-2 rounded-md font-semibold text-center transition cursor-pointer border ${
                        exportSubset === "all"
                          ? "bg-blue-600 text-white border-blue-600 shadow-xs"
                          : "bg-white text-slate-700 border-slate-300 hover:bg-slate-100"
                      }`}
                    >
                      全部样本 ({caseSamples.length})
                    </button>
                    <button
                      onClick={() => setExportSubset("train")}
                      className={`py-1.5 px-2 rounded-md font-semibold text-center transition cursor-pointer border ${
                        exportSubset === "train"
                          ? "bg-blue-600 text-white border-blue-600 shadow-xs"
                          : "bg-white text-slate-700 border-slate-300 hover:bg-slate-100"
                      }`}
                    >
                      0.8 训练集 ({trainSet.length})
                    </button>
                    <button
                      onClick={() => setExportSubset("val")}
                      className={`py-1.5 px-2 rounded-md font-semibold text-center transition cursor-pointer border ${
                        exportSubset === "val"
                          ? "bg-blue-600 text-white border-blue-600 shadow-xs"
                          : "bg-white text-slate-700 border-slate-300 hover:bg-slate-100"
                      }`}
                    >
                      0.2 预测集 ({testSet.length})
                    </button>
                  </div>
                </div>

                {/* 2. Format Selection */}
                <div className="space-y-1.5">
                  <label className="font-bold text-slate-700 block">导出文件格式：</label>
                  <div className="grid grid-cols-2 gap-1.5">
                    <button
                      onClick={() => setExportFormat("csv")}
                      className={`py-1.5 px-3 rounded-md font-semibold flex items-center justify-center gap-1.5 transition cursor-pointer border ${
                        exportFormat === "csv"
                          ? "bg-emerald-600 text-white border-emerald-600 shadow-xs"
                          : "bg-white text-slate-700 border-slate-300 hover:bg-slate-100"
                      }`}
                    >
                      <FileSpreadsheet className="w-3.5 h-3.5" />
                      <span>CSV 表格文件 (.csv)</span>
                    </button>
                    <button
                      onClick={() => setExportFormat("json")}
                      className={`py-1.5 px-3 rounded-md font-semibold flex items-center justify-center gap-1.5 transition cursor-pointer border ${
                        exportFormat === "json"
                          ? "bg-emerald-600 text-white border-emerald-600 shadow-xs"
                          : "bg-white text-slate-700 border-slate-300 hover:bg-slate-100"
                      }`}
                    >
                      <FileJson className="w-3.5 h-3.5" />
                      <span>JSON 结构体 (.json)</span>
                    </button>
                  </div>
                </div>
              </div>

              {/* Data Schema Specs Preview */}
              <div className="bg-blue-50/60 border border-blue-200 rounded-lg p-3 space-y-1.5 text-[11px]">
                <div className="font-bold text-blue-900 flex items-center justify-between">
                  <span>包含数据字段规范 (Schema Columns):</span>
                  <span className="font-mono text-blue-700">
                    输入特征: {selectedCase.inputDim} 维 | 输出目标: {selectedCase.outputDim} 维
                  </span>
                </div>
                <div className="text-slate-600">
                  <strong>特征列 (Features): </strong>
                  {selectedCase.featureNames.slice(0, 6).join(", ")}
                  {selectedCase.featureNames.length > 6 && ` ... (共 ${selectedCase.featureNames.length} 维)`}
                </div>
                <div className="text-slate-600">
                  <strong>目标列 (Targets): </strong>
                  {selectedCase.targetNames.join(", ")}
                </div>
              </div>

              {/* Raw Text Preview Code Block */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="font-bold text-slate-700 flex items-center gap-1.5">
                    <Eye className="w-3.5 h-3.5 text-slate-500" />
                    原始数据内容预览 (前 20 条记录):
                  </span>
                  <button
                    onClick={handleCopyExportText}
                    className="flex items-center gap-1 px-2 py-0.5 rounded bg-slate-100 hover:bg-slate-200 text-slate-700 transition cursor-pointer text-[11px]"
                  >
                    {copied ? <Check className="w-3 h-3 text-emerald-600" /> : <Copy className="w-3 h-3 text-slate-500" />}
                    <span>{copied ? "已复制到剪贴板" : "复制全部文本"}</span>
                  </button>
                </div>

                <div className="bg-slate-900 text-slate-200 font-mono text-[11px] p-3 rounded-lg max-h-52 overflow-auto border border-slate-800 leading-relaxed whitespace-pre">
                  {exportFormat === "csv"
                    ? exportCaseStudyToCSV(selectedCase, exportSubset, caseSamples)
                        .split("\n")
                        .slice(0, 22)
                        .join("\n") +
                      (caseSamples.length > 20 ? "\n... (更多样本已包含在下载文件中)" : "")
                    : exportCaseStudyToJSON(selectedCase, exportSubset, caseSamples)}
                </div>
              </div>
            </div>

            {/* Modal Footer Actions */}
            <div className="px-5 py-3.5 border-t border-slate-200 bg-slate-50 flex items-center justify-between">
              <div className="text-[11px] text-slate-500">
                文件命名：<code className="font-mono text-slate-700 font-bold">{selectedCase.id}_dataset_{exportSubset}.{exportFormat}</code>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setIsExportModalOpen(false)}
                  className="px-3 py-1.5 rounded-md text-xs font-semibold text-slate-600 hover:bg-slate-200 transition cursor-pointer"
                >
                  关闭
                </button>
                <button
                  onClick={() => handleDownloadDataset(exportFormat, exportSubset)}
                  className="flex items-center gap-1.5 px-4 py-1.5 rounded-md bg-blue-600 hover:bg-blue-700 text-white text-xs font-bold shadow-sm transition cursor-pointer"
                >
                  <Download className="w-4 h-4" />
                  <span>立即下载导出文件 (.{exportFormat.toUpperCase()})</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
