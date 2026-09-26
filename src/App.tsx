import React, { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { Header } from "./components/Header";
import { TopologyPulseCanvas } from "./components/TopologyPulseCanvas";
import { TreeModelSandbox } from "./components/TreeModelSandbox";
import { LossTrajectory2D } from "./components/LossTrajectory2D";
import { DecisionBoundary2D } from "./components/DecisionBoundary2D";
import { CaseStudiesGallery } from "./components/CaseStudiesGallery";
import { CodeEngine } from "./components/CodeEngine";
import { NeuronGuidance } from "./components/NeuronGuidance";
import { AiAssistantModal } from "./components/AiAssistantModal";
import { AiDiagnosticWorkspace } from "./components/AiDiagnosticWorkspace";
import { KnowledgeWorkspace } from "./components/KnowledgeWorkspace";
import { ReportWorkspace } from "./components/ReportWorkspace";
import { NeuralNetwork } from "./lib/neural-engine";
import {
  ALL_CASE_STUDIES,
  generateThermalComfortDataset,
  generateLinearIdentityDataset,
  generateXORDataset,
  splitTrainTest,
} from "./lib/datasets";
import {
  ActivationType,
  CaseStudy,
  LossFunctionType,
  OptimizerType,
  StepTraceFrame,
  TrainingMetrics,
  TrainingSample,
} from "./types";

export default function App() {
  // Navigation Tab State
  const [activeTab, setActiveTab] = useState<
    | "lab"
    | "tree"
    | "boundary"
    | "cases"
    | "code"
    | "guidance"
    | "ai_assistant"
    | "knowledge_guidance"
    | "report_export"
  >("lab");

  // Hyperparameters
  const [learningRate, setLearningRate] = useState<number>(0.05);
  const [activation, setActivation] = useState<ActivationType>("leaky_relu");
  const [optimizer, setOptimizer] = useState<OptimizerType>("adam");
  const [simSpeed, setSimSpeed] = useState<number>(2); // iterations per tick
  const [isPlaying, setIsPlaying] = useState<boolean>(false);

  // Network State
  const [layerSizes, setLayerSizes] = useState<number[]>([2, 4, 1]);
  const [activations, setActivations] = useState<ActivationType[]>(["leaky_relu", "linear"]);
  const [currentCaseId, setCurrentCaseId] = useState<string>("thermal_comfort");
  const [lossFunction, setLossFunction] = useState<LossFunctionType>("mse");
  const [huberDelta, setHuberDelta] = useState<number>(1.0);

  // Training & Dataset State
  const [samples, setSamples] = useState<TrainingSample[]>(() =>
    generateThermalComfortDataset(60, 0.03)
  );

  // Automatic 8:2 split into 80% Training Set and 20% Validation Set
  const { train: trainSamples, test: valSamples } = useMemo(() => {
    return splitTrainTest(samples, 0.8);
  }, [samples]);

  const [epoch, setEpoch] = useState<number>(0);
  const [currentLoss, setCurrentLoss] = useState<number>(0.5);
  const [currentValLoss, setCurrentValLoss] = useState<number>(0.5);
  const [metricsHistory, setMetricsHistory] = useState<TrainingMetrics[]>([]);

  // Pedagogical Step Inspector State
  const [stepTrace, setStepTrace] = useState<StepTraceFrame[]>([]);
  const [currentStepIndex, setCurrentStepIndex] = useState<number>(0);

  // AI Modal
  const [isAiModalOpen, setIsAiModalOpen] = useState<boolean>(false);

  // Neural Network Engine Ref
  const nnRef = useRef<NeuralNetwork>(
    new NeuralNetwork([2, 4, 1], ["leaky_relu", "linear"], 0.05, "adam", "mse", 1.0)
  );

  // Re-initialize network when topology or activation structure changes
  const reinitializeNetwork = useCallback(
    (
      newLayers: number[],
      newActs: ActivationType[],
      newLr?: number,
      newOpt?: OptimizerType,
      newLoss?: LossFunctionType,
      newHuberDelta?: number
    ) => {
      const lr = newLr ?? learningRate;
      const opt = newOpt ?? optimizer;
      const lossType = newLoss ?? lossFunction;
      const hDelta = newHuberDelta ?? huberDelta;
      const net = new NeuralNetwork(newLayers, newActs, lr, opt, lossType, hDelta);
      nnRef.current = net;
      setLayerSizes(newLayers);
      setActivations(newActs);
      setEpoch(0);
      setCurrentLoss(0.5);
      setCurrentValLoss(0.5);
      setMetricsHistory([]);
      setIsPlaying(false);

      // Generate initial step trace on the first training sample
      const firstSample = trainSamples[0] || samples[0];
      if (firstSample) {
        const trace = net.generateStepTrace(firstSample, undefined, lossType);
        setStepTrace(trace);
        setCurrentStepIndex(0);
      }
    },
    [learningRate, optimizer, lossFunction, huberDelta, trainSamples, samples]
  );

  // Sync hyperparameter changes to existing network
  useEffect(() => {
    nnRef.current.learningRate = learningRate;
    nnRef.current.optimizer = optimizer;
    nnRef.current.defaultLossType = lossFunction;
    nnRef.current.huberDelta = huberDelta;
  }, [learningRate, optimizer, lossFunction, huberDelta]);

  // Initial step trace generation on load
  useEffect(() => {
    const firstSample = trainSamples[0] || samples[0];
    if (firstSample) {
      const trace = nnRef.current.generateStepTrace(firstSample, undefined, lossFunction);
      setStepTrace(trace);
    }
  }, [trainSamples, samples, lossFunction]);

  // Training Loop: Backprop ONLY on 80% trainSamples, Evaluate real-time on 20% valSamples
  useEffect(() => {
    if (!isPlaying || trainSamples.length === 0) return;

    const interval = setInterval(() => {
      const net = nnRef.current;
      let lastTrainLoss = currentLoss;
      let lastGradNorm = 0;
      let lastWeightNorm = 0;

      // 1. Backprop and gradient updates on 80% training set only
      for (let i = 0; i < simSpeed; i++) {
        const metrics = net.trainEpoch(trainSamples, learningRate, optimizer, lossFunction);
        lastTrainLoss = metrics.loss;
        lastGradNorm = metrics.gradNorm;
        lastWeightNorm = metrics.weightNorm;
      }

      // 2. Real-time evaluate loss and accuracy on remaining 20% validation set (forward pass only)
      let lastValLoss = lastTrainLoss;
      let lastValAcc = 100;
      if (valSamples.length > 0) {
        const valMetrics = net.evaluate(valSamples, lossFunction);
        lastValLoss = valMetrics.loss;
        lastValAcc = valMetrics.accuracy;
      }

      setEpoch((prev) => {
        const nextEpoch = prev + simSpeed;
        setMetricsHistory((hist) => {
          const newHist = [
            ...hist,
            {
              epoch: nextEpoch,
              loss: lastTrainLoss,
              valLoss: lastValLoss,
              accuracy: 1.0,
              valAccuracy: lastValAcc,
              gradNorm: lastGradNorm,
              weightNorm: lastWeightNorm,
              timestamp: Date.now(),
            },
          ];
          // Limit memory to 300 points
          return newHist.slice(-300);
        });
        return nextEpoch;
      });

      setCurrentLoss(lastTrainLoss);
      setCurrentValLoss(lastValLoss);
    }, 50);

    return () => clearInterval(interval);
  }, [isPlaying, trainSamples, valSamples, simSpeed, currentLoss, learningRate, optimizer, lossFunction]);

  // Single Step Training & Trace Update
  const handleStepOnce = () => {
    const net = nnRef.current;
    if (trainSamples.length === 0) return;

    // 1. Train only on 80% training set
    const metrics = net.trainEpoch(trainSamples, learningRate, optimizer, lossFunction);

    // 2. Evaluate on 20% validation set
    const valMetrics =
      valSamples.length > 0
        ? net.evaluate(valSamples, lossFunction)
        : { loss: metrics.loss, accuracy: 100, totalLoss: 0 };

    setEpoch((prev) => prev + 1);
    setCurrentLoss(metrics.loss);
    setCurrentValLoss(valMetrics.loss);
    setMetricsHistory((prev) => [
      ...prev,
      {
        epoch: epoch + 1,
        loss: metrics.loss,
        valLoss: valMetrics.loss,
        accuracy: 1.0,
        valAccuracy: valMetrics.accuracy,
        gradNorm: metrics.gradNorm,
        weightNorm: metrics.weightNorm,
        timestamp: Date.now(),
      },
    ]);

    // Update step trace on the first training sample
    const firstSample = trainSamples[0] || samples[0];
    if (firstSample) {
      const trace = net.generateStepTrace(firstSample, undefined, lossFunction);
      setStepTrace(trace);
    }
  };

  const handleReset = () => {
    reinitializeNetwork(layerSizes, activations, learningRate, optimizer);
  };

  // Case Study selection
  const handleSelectCase = (cs: CaseStudy) => {
    setCurrentCaseId(cs.id);
    const newActs: ActivationType[] = Array(cs.recommendedTopology.length - 1).fill(
      cs.recommendedActivation
    );
    // If output dimension is > 1 for classification (like MNIST or Iris), use softmax for output layer
    if (cs.recommendedTopology[cs.recommendedTopology.length - 1] > 1) {
      newActs[newActs.length - 1] = "softmax";
    }

    const opt = cs.recommendedOptimizer || "adam";
    setSamples(cs.samples);
    setLearningRate(cs.recommendedLr);
    setActivation(cs.recommendedActivation);
    setOptimizer(opt);
    reinitializeNetwork(
      cs.recommendedTopology,
      newActs,
      cs.recommendedLr,
      opt
    );
  };

  // Guidance level loader
  const handleLoadExperiment = (
    top: number[],
    act: ActivationType,
    lr: number,
    opt: OptimizerType,
    caseId?: string
  ) => {
    const newActs: ActivationType[] = Array(top.length - 1).fill(act);
    setLearningRate(lr);
    setActivation(act);
    setOptimizer(opt);
    if (caseId) {
      const found = ALL_CASE_STUDIES.find((c) => c.id === caseId);
      if (found) {
        setSamples(found.samples);
        setCurrentCaseId(found.id);
      }
    }
    reinitializeNetwork(top, newActs, lr, opt);
    setActiveTab("lab");
  };

  const latestGradNorm = metricsHistory[metricsHistory.length - 1]?.gradNorm ?? 0;

  return (
    <div className="min-h-screen bg-[#f8fafc] text-slate-800 flex flex-col font-sans selection:bg-blue-600 selection:text-white">
      {/* Top Header & Simulation Controls */}
      <Header
        isPlaying={isPlaying}
        onTogglePlay={() => setIsPlaying(!isPlaying)}
        onStepOnce={handleStepOnce}
        onReset={handleReset}
        epoch={epoch}
        currentLoss={currentLoss}
        valLoss={currentValLoss}
        gradNorm={latestGradNorm}
        learningRate={learningRate}
        onLearningRateChange={(val) => setLearningRate(val)}
        activation={activation}
        onActivationChange={(act) => {
          setActivation(act);
          const newActs: ActivationType[] = Array(layerSizes.length - 1).fill(act);
          reinitializeNetwork(layerSizes, newActs);
        }}
        optimizer={optimizer}
        onOptimizerChange={(opt) => setOptimizer(opt)}
        simSpeed={simSpeed}
        onSimSpeedChange={(spd) => setSimSpeed(spd)}
        activeTab={activeTab}
        onTabChange={(tab) => setActiveTab(tab)}
        onOpenAiModal={() => setIsAiModalOpen(true)}
      />

      {/* Main Content Workspace */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-3 sm:p-4 space-y-4">
        {/* TAB 1: 核心拓扑实验 (Core Lab) */}
        {activeTab === "lab" && (
          <div className="space-y-4">
            {/* Top Row: Topology Pulse 2D Canvas (Module 2) */}
            <div className="w-full">
              <TopologyPulseCanvas
                nn={nnRef.current}
                currentSample={trainSamples[0] || samples[0]}
                isPlaying={isPlaying}
                stepTrace={stepTrace}
                currentStepIndex={currentStepIndex}
                onSetStepIndex={(idx) => setCurrentStepIndex(idx)}
                onNextStep={() =>
                  setCurrentStepIndex((prev) =>
                    Math.min(stepTrace.length - 1, prev + 1)
                  )
                }
                epoch={epoch}
                onApplyTopology={(layers, acts) => {
                  reinitializeNetwork(layers, acts);
                }}
              />
            </div>

            {/* Bottom Row: 2D Loss Trajectory (Module 3) & 2D Decision Boundary (Module 4) */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Module 3: 2D Loss Trajectory */}
              <LossTrajectory2D
                metricsHistory={metricsHistory}
                currentEpoch={epoch}
                currentLoss={currentLoss}
                valLoss={currentValLoss}
                learningRate={learningRate}
                optimizer={optimizer}
                activation={activation}
              />

              {/* Module 4: 2D Decision Boundary Smooth Evolution */}
              <DecisionBoundary2D
                nn={nnRef.current}
                samples={samples}
                onUpdateSamples={(newS) => setSamples(newS)}
                epoch={epoch}
              />
            </div>
          </div>
        )}

        {/* TAB 2: 树状模型沙盒 (Module 1) */}
        {activeTab === "tree" && (
          <div className="animate-fade-in">
            <TreeModelSandbox
              currentTopology={layerSizes}
              currentActivations={activations}
              onApplyTopology={(layers, acts) => {
                reinitializeNetwork(layers, acts);
              }}
              nn={nnRef.current}
              currentLossType={lossFunction}
              onLossTypeChange={(newLoss, newDelta) => {
                setLossFunction(newLoss);
                if (newDelta !== undefined) setHuberDelta(newDelta);
                nnRef.current.defaultLossType = newLoss;
                if (newDelta !== undefined) nnRef.current.huberDelta = newDelta;
                // Regenerate step trace if sample exists
                const firstSample = trainSamples[0] || samples[0];
                if (firstSample) {
                  setStepTrace(nnRef.current.generateStepTrace(firstSample, undefined, newLoss));
                }
              }}
            />
          </div>
        )}

        {/* TAB 3: 2D 边界演化专用演播室 (Module 4 focused) */}
        {activeTab === "boundary" && (
          <div className="animate-fade-in">
            <DecisionBoundary2D
              nn={nnRef.current}
              samples={samples}
              onUpdateSamples={(newS) => setSamples(newS)}
              epoch={epoch}
            />
          </div>
        )}

        {/* TAB 4: 六大应用案例库 (Module 5) */}
        {activeTab === "cases" && (
          <div className="animate-fade-in">
            <CaseStudiesGallery
              currentCaseId={currentCaseId}
              onSelectCase={handleSelectCase}
              nn={nnRef.current}
            />
          </div>
        )}

        {/* TAB 5: Python / PyTorch 代码引擎 (Module 6) */}
        {activeTab === "code" && (
          <div className="animate-fade-in">
            <CodeEngine
              topology={layerSizes}
              activations={activations}
              learningRate={learningRate}
              optimizer={optimizer}
              lossFunction={lossFunction}
            />
          </div>
        )}

        {/* TAB 6: 神经元导引与实验关卡 (Module 8) */}
        {activeTab === "guidance" && (
          <div className="animate-fade-in">
            <NeuronGuidance onLoadExperiment={handleLoadExperiment} />
          </div>
        )}

        {/* TAB 7: AI诊断助手与导师沙箱 (Module 7) */}
        {activeTab === "ai_assistant" && (
          <div className="animate-fade-in">
            <AiDiagnosticWorkspace
              currentLabState={{
                epoch,
                loss: currentLoss,
                valLoss: currentValLoss,
                gradNorm: latestGradNorm,
                weightNorm: metricsHistory[metricsHistory.length - 1]?.weightNorm ?? 0,
                learningRate,
                activation,
                optimizer,
                topology: layerSizes,
                lossHistory: metricsHistory.map((m) => m.loss),
                dataset: currentCaseId,
              }}
              onApplyTuning={({ learningRate: lr, activation: act, optimizer: opt }) => {
                if (lr !== undefined) setLearningRate(lr);
                if (act !== undefined) {
                  setActivation(act);
                  setActivations(Array(layerSizes.length - 1).fill(act));
                }
                if (opt !== undefined) setOptimizer(opt);
              }}
            />
          </div>
        )}

        {/* TAB 8: 知识导引 (Module 8) */}
        {activeTab === "knowledge_guidance" && (
          <div className="animate-fade-in">
            <KnowledgeWorkspace />
          </div>
        )}

        {/* TAB 9: 报告导出 (Module 9) */}
        {activeTab === "report_export" && (
          <div className="animate-fade-in">
            <ReportWorkspace
              currentLabState={{
                epoch,
                loss: currentLoss,
                valLoss: currentValLoss,
                gradNorm: latestGradNorm,
                weightNorm: metricsHistory[metricsHistory.length - 1]?.weightNorm ?? 0,
                learningRate,
                activation,
                optimizer,
                topology: layerSizes,
                lossHistory: metricsHistory.map((m) => m.loss),
                dataset: currentCaseId,
              }}
            />
          </div>
        )}
      </main>

      {/* High Density Footer */}
      <footer className="border-t border-slate-200 bg-white px-4 py-2.5 text-[10px] font-mono text-slate-500 flex flex-wrap items-center justify-between gap-2 shadow-sm">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-500" />
          <span className="font-bold text-slate-700 uppercase">Status: ONLINE</span>
          <span className="text-slate-300">|</span>
          <span>Kernel: Vectorized NumPy / Float64 Autograd</span>
          <span className="text-slate-300">|</span>
          <span>Topology: [{layerSizes.join(" → ")}]</span>
        </div>
        <div className="flex items-center gap-3 text-slate-400 font-medium">
          <span className="text-blue-600 font-semibold">[通用近似定理 Universal Approximation]</span>
          <span>[多元链式法则 Chain Rule]</span>
          <span>[Google Gemini 3.7 AI 驱动]</span>
        </div>
      </footer>

      {/* AI Assistant & Diagnostic Modal (Module 7) */}
      <AiAssistantModal
        isOpen={isAiModalOpen}
        onClose={() => setIsAiModalOpen(false)}
        currentLabState={{
          epoch,
          loss: currentLoss,
          gradNorm: latestGradNorm,
          weightNorm: metricsHistory[metricsHistory.length - 1]?.weightNorm ?? 0,
          learningRate,
          activation,
          optimizer,
          topology: layerSizes,
          lossHistory: metricsHistory.map((m) => m.loss),
          dataset: currentCaseId,
        }}
      />
    </div>
  );
}
