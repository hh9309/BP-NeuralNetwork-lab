/**
 * Type definitions for BP Neural Network & Backpropagation Lab
 */

export type ActivationType = "sigmoid" | "tanh" | "relu" | "leaky_relu" | "linear" | "softmax";
export type OptimizerType = "sgd" | "momentum" | "adam" | "rmsprop";
export type LossFunctionType = "mse" | "mae" | "huber" | "cross_entropy" | "binary_cross_entropy";

export interface LayerConfig {
  id: string;
  neurons: number;
  activation: ActivationType;
  name?: string;
}

export interface NeuralTopology {
  id: string;
  name: string;
  description: string;
  layers: LayerConfig[];
  type: "single_output" | "multi_output" | "custom";
}

export interface TrainingSample {
  input: number[];
  target: number[];
  label?: string | number;
}

export interface ForwardLayerResult {
  layerIndex: number;
  z: number[]; // pre-activation inputs
  a: number[]; // post-activation outputs
}

export interface BackwardLayerResult {
  layerIndex: number;
  delta: number[]; // error gradient dJ/dz
  dW: number[][]; // weight gradients
  db: number[]; // bias gradients
  norm?: number; // Frobenius/L2 gradient norm for this layer
}

export interface ForwardPassResult {
  output: number[];
  layers: ForwardLayerResult[];
}

export interface BackwardPassResult {
  loss: number;
  layerGradients: BackwardLayerResult[];
  totalGradNorm: number;
}

export interface TrainingMetrics {
  epoch: number;
  loss: number; // 80% Train Set Loss
  valLoss?: number; // 20% Validation Set Loss
  accuracy?: number; // Train Accuracy
  valAccuracy?: number; // Validation Accuracy
  gradNorm: number;
  weightNorm: number;
  timestamp: number;
}

export interface CaseStudy {
  id: string;
  title: string;
  subtitle: string;
  category: "classification" | "regression" | "multiclass" | "logic" | "approximation" | "binary";
  iconName: string;
  inputDim: number;
  outputDim: number;
  recommendedTopology: number[];
  recommendedActivation: ActivationType;
  recommendedOutputActivation: ActivationType;
  recommendedLr: number;
  recommendedOptimizer?: OptimizerType;
  recommendedLoss: LossFunctionType;
  description: string;
  realWorldContext: string;
  featureNames: string[];
  targetNames: string[];
  samples: TrainingSample[];
  boundarySupport: boolean;
}

export interface StepTraceFrame {
  stepIndex: number;
  phase: "idle" | "forward_input" | "forward_hidden" | "forward_output" | "compute_loss" | "backward_output" | "backward_hidden" | "update_weights";
  title: string;
  formula: string;
  description: string;
  activeLayer: number;
  fromLayer?: number;
  toLayer?: number;
  activeNeurons?: number[];
  mathDetails?: Record<string, string | number>;
}

export interface ChatMessage {
  id: string;
  role: "user" | "model" | "system";
  content: string;
  timestamp: number;
  isAiDiagnostic?: boolean;
}

export interface DiagnosisResult {
  report: string;
  healthScore: number;
  status: string;
  risks?: string[];
  suggestions?: string[];
}
