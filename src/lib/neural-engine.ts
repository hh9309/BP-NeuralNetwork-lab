/**
 * Vectorized Neural Network & Backpropagation Engine for BP Lab
 */

import {
  ActivationType,
  OptimizerType,
  LossFunctionType,
  ForwardPassResult,
  BackwardPassResult,
  BackwardLayerResult,
  TrainingSample,
  StepTraceFrame,
} from "../types";

export class NeuralNetwork {
  public layerSizes: number[];
  public activations: ActivationType[];
  public weights: number[][][]; // weights[l][j][i] = weight from neuron i in layer l to neuron j in layer l+1
  public biases: number[][]; // biases[l][j] = bias for neuron j in layer l+1

  public learningRate: number = 0.1;
  public optimizer: OptimizerType = "adam";
  public defaultLossType: LossFunctionType = "mse";
  public huberDelta: number = 1.0;

  // Optimizer momentum / adaptive states
  public vWeights: number[][][];
  public vBiases: number[][];
  public sWeights: number[][][];
  public sBiases: number[][];
  public optimizerT: number = 0;

  constructor(
    layerSizes: number[],
    activations: ActivationType[],
    learningRate: number = 0.1,
    optimizer: OptimizerType = "adam",
    defaultLossType: LossFunctionType = "mse",
    huberDelta: number = 1.0
  ) {
    this.layerSizes = [...layerSizes];
    this.activations = [...activations];
    this.learningRate = learningRate;
    this.optimizer = optimizer;
    this.defaultLossType = defaultLossType;
    this.huberDelta = huberDelta;
    this.weights = [];
    this.biases = [];
    this.vWeights = [];
    this.vBiases = [];
    this.sWeights = [];
    this.sBiases = [];
    this.initParameters();
  }

  public initParameters(seed?: number) {
    this.weights = [];
    this.biases = [];
    this.vWeights = [];
    this.vBiases = [];
    this.sWeights = [];
    this.sBiases = [];
    this.optimizerT = 0;

    for (let l = 0; l < this.layerSizes.length - 1; l++) {
      const fanIn = this.layerSizes[l];
      const fanOut = this.layerSizes[l + 1];
      const act = this.activations[l] || "sigmoid";

      // He initialization for ReLU, Xavier for Sigmoid/Tanh
      let std = Math.sqrt(2.0 / fanIn);
      if (act === "sigmoid" || act === "tanh" || act === "linear" || act === "softmax") {
        std = Math.sqrt(2.0 / (fanIn + fanOut));
      }

      const layerW: number[][] = [];
      const vLayerW: number[][] = [];
      const sLayerW: number[][] = [];
      for (let j = 0; j < fanOut; j++) {
        const rowW: number[] = [];
        const vRowW: number[] = [];
        const sRowW: number[] = [];
        for (let i = 0; i < fanIn; i++) {
          rowW.push(randomNormal(0, std));
          vRowW.push(0);
          sRowW.push(0);
        }
        layerW.push(rowW);
        vLayerW.push(vRowW);
        sLayerW.push(sRowW);
      }
      this.weights.push(layerW);
      this.vWeights.push(vLayerW);
      this.sWeights.push(sLayerW);

      const layerB: number[] = [];
      const vLayerB: number[] = [];
      const sLayerB: number[] = [];
      for (let j = 0; j < fanOut; j++) {
        layerB.push(0.01);
        vLayerB.push(0);
        sLayerB.push(0);
      }
      this.biases.push(layerB);
      this.vBiases.push(vLayerB);
      this.sBiases.push(sLayerB);
    }
  }

  // Activation function evaluator
  public static activate(z: number, type: ActivationType): number {
    switch (type) {
      case "sigmoid":
        if (z > 40) return 1;
        if (z < -40) return 0;
        return 1.0 / (1.0 + Math.exp(-z));
      case "tanh":
        return Math.tanh(z);
      case "relu":
        return Math.max(0, z);
      case "leaky_relu":
        return z > 0 ? z : 0.05 * z;
      case "linear":
        return z;
      default:
        return z;
    }
  }

  // Activation derivative evaluator with respect to z
  public static activateDerivative(z: number, a: number, type: ActivationType): number {
    switch (type) {
      case "sigmoid":
        return a * (1.0 - a);
      case "tanh":
        return 1.0 - a * a;
      case "relu":
        return z > 0 ? 1.0 : 0.0;
      case "leaky_relu":
        return z > 0 ? 1.0 : 0.05;
      case "linear":
        return 1.0;
      default:
        return 1.0;
    }
  }

  // Softmax vector activation
  public static softmax(vec: number[]): number[] {
    if (!vec || !vec.length) return [1];
    const maxVal = Math.max(...vec);
    const exps = vec.map((v) => Math.exp(Math.min(50, Math.max(-50, v - maxVal))));
    const sum = exps.reduce((acc, curr) => acc + curr, 0) || 1e-9;
    return exps.map((e) => e / sum);
  }

  // Forward propagation for single input vector x
  public forward(x: number[]): ForwardPassResult {
    const inputDim = this.layerSizes[0] || 1;
    const safeX: number[] = Array.isArray(x)
      ? (x.length === inputDim
          ? [...x]
          : (x.length < inputDim
              ? [...x, ...new Array(inputDim - x.length).fill(0)]
              : x.slice(0, inputDim)))
      : new Array(inputDim).fill(0);

    const layers: { layerIndex: number; z: number[]; a: number[] }[] = [];
    let currentA = [...safeX];

    layers.push({
      layerIndex: 0,
      z: [...safeX],
      a: [...currentA],
    });

    for (let l = 0; l < this.weights.length; l++) {
      const W = this.weights[l];
      const b = this.biases[l];
      const act = this.activations[l] || "sigmoid";
      const fanOut = this.layerSizes[l + 1];

      const z: number[] = [];
      for (let j = 0; j < fanOut; j++) {
        let sum = b[j] ?? 0;
        for (let i = 0; i < currentA.length; i++) {
          sum += (W[j]?.[i] ?? 0) * currentA[i];
        }
        z.push(sum);
      }

      let a: number[];
      if (act === "softmax") {
        a = NeuralNetwork.softmax(z);
      } else {
        a = z.map((val) => NeuralNetwork.activate(val, act));
      }

      layers.push({
        layerIndex: l + 1,
        z,
        a,
      });

      currentA = a;
    }

    return {
      output: currentA,
      layers,
    };
  }

  // Backward propagation for single sample (x, y)
  public backwardSingle(
    x: number[],
    target: number[],
    lossType: LossFunctionType = "mse"
  ): BackwardPassResult {
    const inputDim = this.layerSizes[0] || 1;
    const outputDim = this.layerSizes[this.layerSizes.length - 1] || 1;
    const safeX: number[] = Array.isArray(x)
      ? (x.length === inputDim
          ? [...x]
          : (x.length < inputDim
              ? [...x, ...new Array(inputDim - x.length).fill(0)]
              : x.slice(0, inputDim)))
      : new Array(inputDim).fill(0);

    const safeTarget: number[] = Array.isArray(target)
      ? (target.length === outputDim
          ? [...target]
          : (target.length < outputDim
              ? [...target, ...new Array(outputDim - target.length).fill(0)]
              : target.slice(0, outputDim)))
      : new Array(outputDim).fill(0);

    const fwd = this.forward(safeX);
    const numLayers = this.layerSizes.length;
    const outputLayerIndex = numLayers - 1;
    const outputLayer = fwd.layers[outputLayerIndex];
    const yHat = outputLayer.a;
    const lastAct = this.activations[this.activations.length - 1] || "sigmoid";

    // 1. Compute Loss and Output Layer Delta
    let loss = 0;
    const outputDelta: number[] = [];
    const hDelta = this.huberDelta || 1.0;

    if (lossType === "cross_entropy" && lastAct === "softmax") {
      for (let j = 0; j < yHat.length; j++) {
        const y = safeTarget[j] ?? 0;
        loss -= y * Math.log(Math.max(1e-12, yHat[j]));
        outputDelta.push(yHat[j] - y);
      }
    } else if (lossType === "binary_cross_entropy") {
      for (let j = 0; j < yHat.length; j++) {
        const y = safeTarget[j] ?? 0;
        const p = Math.max(1e-12, Math.min(1 - 1e-12, yHat[j]));
        loss -= (y * Math.log(p) + (1 - y) * Math.log(1 - p)) / yHat.length;
        // For sigmoid + BCE, dJ/dz = a - y
        if (lastAct === "sigmoid") {
          outputDelta.push(yHat[j] - y);
        } else {
          const dLoss_da = -(y / p - (1 - y) / (1 - p));
          const da_dz = NeuralNetwork.activateDerivative(outputLayer.z[j], yHat[j], lastAct);
          outputDelta.push(dLoss_da * da_dz);
        }
      }
    } else if (lossType === "mae") {
      // MAE (L1 Loss): sum(|yHat_j - y_j|)
      for (let j = 0; j < yHat.length; j++) {
        const y = safeTarget[j] ?? 0;
        const diff = yHat[j] - y;
        loss += Math.abs(diff);
        const sgn = diff > 0 ? 1 : diff < 0 ? -1 : 0;
        const da_dz = NeuralNetwork.activateDerivative(outputLayer.z[j], yHat[j], lastAct);
        outputDelta.push(sgn * da_dz);
      }
    } else if (lossType === "huber") {
      // Huber Loss (Smooth L1 Loss)
      for (let j = 0; j < yHat.length; j++) {
        const y = safeTarget[j] ?? 0;
        const diff = yHat[j] - y;
        const absDiff = Math.abs(diff);
        if (absDiff <= hDelta) {
          loss += 0.5 * diff * diff;
        } else {
          loss += hDelta * (absDiff - 0.5 * hDelta);
        }
        // Derivative dJ/dyHat = clip(diff, -hDelta, hDelta)
        const dLoss_da = Math.max(-hDelta, Math.min(hDelta, diff));
        const da_dz = NeuralNetwork.activateDerivative(outputLayer.z[j], yHat[j], lastAct);
        outputDelta.push(dLoss_da * da_dz);
      }
    } else {
      // Default: MSE Loss: 0.5 * sum((y_j - yHat_j)^2)
      for (let j = 0; j < yHat.length; j++) {
        const y = safeTarget[j] ?? 0;
        const diff = yHat[j] - y;
        loss += 0.5 * diff * diff;
        const da_dz = NeuralNetwork.activateDerivative(outputLayer.z[j], yHat[j], lastAct);
        outputDelta.push(diff * da_dz);
      }
    }

    // 2. Backpropagate deltas through hidden layers
    const deltas: number[][] = new Array(numLayers);
    deltas[outputLayerIndex] = outputDelta;

    for (let l = numLayers - 2; l >= 1; l--) {
      const nextDelta = deltas[l + 1];
      const W_next = this.weights[l]; // connects layer l to layer l+1
      const act = this.activations[l - 1];
      const layerZ = fwd.layers[l].z;
      const layerA = fwd.layers[l].a;
      const currentDelta: number[] = [];

      for (let i = 0; i < this.layerSizes[l]; i++) {
        let sum = 0;
        for (let j = 0; j < this.layerSizes[l + 1]; j++) {
          sum += (W_next[j]?.[i] ?? 0) * nextDelta[j];
        }
        const da_dz = NeuralNetwork.activateDerivative(layerZ[i], layerA[i], act);
        currentDelta.push(sum * da_dz);
      }
      deltas[l] = currentDelta;
    }

    // 3. Compute gradients dW and db
    const layerGradients: BackwardLayerResult[] = [];
    let totalGradSq = 0;

    for (let l = 0; l < this.weights.length; l++) {
      const nextDelta = deltas[l + 1];
      const prevA = fwd.layers[l].a;
      const dW: number[][] = [];
      const db: number[] = [];
      let layerGradSq = 0;

      for (let j = 0; j < nextDelta.length; j++) {
        const rowDW: number[] = [];
        for (let i = 0; i < prevA.length; i++) {
          const grad = nextDelta[j] * prevA[i];
          rowDW.push(grad);
          layerGradSq += grad * grad;
          totalGradSq += grad * grad;
        }
        dW.push(rowDW);
        db.push(nextDelta[j]);
        layerGradSq += nextDelta[j] * nextDelta[j];
        totalGradSq += nextDelta[j] * nextDelta[j];
      }

      layerGradients.push({
        layerIndex: l,
        delta: nextDelta,
        dW,
        db,
        norm: Math.sqrt(layerGradSq),
      });
    }

    return {
      loss,
      layerGradients,
      totalGradNorm: Math.sqrt(totalGradSq),
    };
  }

  // Alias for backwardSingle to support standard backward(x, target, lossType) calls
  public backward(
    x: number[],
    target: number[],
    lossType: LossFunctionType = "mse"
  ): BackwardPassResult {
    return this.backwardSingle(x, target, lossType);
  }

  // Train on a dataset batch with selected optimizer
  public trainEpoch(
    dataset: TrainingSample[],
    learningRate: number = 0.1,
    optimizer: OptimizerType = "adam",
    lossType: LossFunctionType = "mse"
  ): { loss: number; accuracy: number; gradNorm: number; weightNorm: number } {
    if (!dataset.length) return { loss: 0, accuracy: 0, gradNorm: 0, weightNorm: 0 };

    this.optimizerT++;
    let totalLoss = 0;
    let correctCount = 0;

    // Accumulate gradients across dataset
    const accumDW: number[][][] = this.weights.map((layer) =>
      layer.map((row) => new Array(row.length).fill(0))
    );
    const accumDB: number[][] = this.biases.map((layer) => new Array(layer.length).fill(0));

    let batchGradNorm = 0;

    for (const sample of dataset) {
      const bwd = this.backwardSingle(sample.input, sample.target, lossType);
      totalLoss += bwd.loss;
      batchGradNorm += bwd.totalGradNorm;

      // Check accuracy for classification
      const fwd = this.forward(sample.input);
      if (sample.target.length === 1) {
        const pred = (fwd.output[0] ?? 0) >= 0.5 ? 1 : 0;
        const target = (sample.target[0] ?? 0) >= 0.5 ? 1 : 0;
        if (pred === target) correctCount++;
      } else {
        const predClass = argMax(fwd.output);
        const targetClass = argMax(sample.target);
        if (predClass === targetClass) correctCount++;
      }

      for (let l = 0; l < this.weights.length; l++) {
        const gradL = bwd.layerGradients[l];
        for (let j = 0; j < gradL.dW.length; j++) {
          for (let i = 0; i < gradL.dW[j].length; i++) {
            accumDW[l][j][i] += gradL.dW[j][i];
          }
          accumDB[l][j] += gradL.db[j];
        }
      }
    }

    const n = dataset.length;
    const avgLoss = totalLoss / n;
    const accuracy = (correctCount / n) * 100;
    const avgGradNorm = batchGradNorm / n;

    // Apply optimizer update
    const beta1 = 0.9;
    const beta2 = 0.999;
    const eps = 1e-8;

    for (let l = 0; l < this.weights.length; l++) {
      for (let j = 0; j < this.weights[l].length; j++) {
        // Biases update
        const gB = accumDB[l][j] / n;
        if (optimizer === "sgd") {
          this.biases[l][j] -= learningRate * gB;
        } else if (optimizer === "momentum") {
          this.vBiases[l][j] = beta1 * this.vBiases[l][j] + (1 - beta1) * gB;
          this.biases[l][j] -= learningRate * this.vBiases[l][j];
        } else if (optimizer === "rmsprop") {
          this.sBiases[l][j] = 0.9 * this.sBiases[l][j] + 0.1 * gB * gB;
          this.biases[l][j] -= (learningRate / (Math.sqrt(this.sBiases[l][j]) + eps)) * gB;
        } else if (optimizer === "adam") {
          this.vBiases[l][j] = beta1 * this.vBiases[l][j] + (1 - beta1) * gB;
          this.sBiases[l][j] = beta2 * this.sBiases[l][j] + (1 - beta2) * gB * gB;
          const mHat = this.vBiases[l][j] / (1 - Math.pow(beta1, this.optimizerT));
          const vHat = this.sBiases[l][j] / (1 - Math.pow(beta2, this.optimizerT));
          this.biases[l][j] -= (learningRate / (Math.sqrt(vHat) + eps)) * mHat;
        }

        // Weights update
        for (let i = 0; i < this.weights[l][j].length; i++) {
          const gW = accumDW[l][j][i] / n;
          if (optimizer === "sgd") {
            this.weights[l][j][i] -= learningRate * gW;
          } else if (optimizer === "momentum") {
            this.vWeights[l][j][i] = beta1 * this.vWeights[l][j][i] + (1 - beta1) * gW;
            this.weights[l][j][i] -= learningRate * this.vWeights[l][j][i];
          } else if (optimizer === "rmsprop") {
            this.sWeights[l][j][i] = 0.9 * this.sWeights[l][j][i] + 0.1 * gW * gW;
            this.weights[l][j][i] -= (learningRate / (Math.sqrt(this.sWeights[l][j][i]) + eps)) * gW;
          } else if (optimizer === "adam") {
            this.vWeights[l][j][i] = beta1 * this.vWeights[l][j][i] + (1 - beta1) * gW;
            this.sWeights[l][j][i] = beta2 * this.sWeights[l][j][i] + (1 - beta2) * gW * gW;
            const mHat = this.vWeights[l][j][i] / (1 - Math.pow(beta1, this.optimizerT));
            const vHat = this.sWeights[l][j][i] / (1 - Math.pow(beta2, this.optimizerT));
            this.weights[l][j][i] -= (learningRate / (Math.sqrt(vHat) + eps)) * mHat;
          }
        }
      }
    }

    // Calculate total weight norm
    let totalWeightSq = 0;
    for (const layer of this.weights) {
      for (const row of layer) {
        for (const w of row) {
          totalWeightSq += w * w;
        }
      }
    }

    return {
      loss: avgLoss,
      accuracy,
      gradNorm: avgGradNorm,
      weightNorm: Math.sqrt(totalWeightSq),
    };
  }

  // Pure forward evaluation on arbitrary dataset (e.g. 20% validation set) without mutating weights or momentum
  public evaluate(
    dataset: TrainingSample[],
    lossType: LossFunctionType = this.defaultLossType
  ): { loss: number; accuracy: number; totalLoss: number } {
    if (!dataset || dataset.length === 0) {
      return { loss: 0, accuracy: 0, totalLoss: 0 };
    }

    let totalLoss = 0;
    let correctCount = 0;
    const lastAct = this.activations[this.activations.length - 1] || "sigmoid";
    const hDelta = this.huberDelta || 1.0;

    for (const sample of dataset) {
      const fwd = this.forward(sample.input);
      const yHat = fwd.output;
      const target = sample.target;

      let sampleLoss = 0;
      if (lossType === "cross_entropy" && lastAct === "softmax") {
        for (let j = 0; j < yHat.length; j++) {
          const y = target[j] ?? 0;
          const eps = 1e-12;
          sampleLoss -= y * Math.log(Math.max(eps, yHat[j]));
        }
      } else if (lossType === "binary_cross_entropy") {
        for (let j = 0; j < yHat.length; j++) {
          const y = target[j] ?? 0;
          const p = Math.max(1e-12, Math.min(1 - 1e-12, yHat[j]));
          sampleLoss -= y * Math.log(p) + (1 - y) * Math.log(1 - p);
        }
        sampleLoss /= yHat.length || 1;
      } else if (lossType === "mae") {
        for (let j = 0; j < yHat.length; j++) {
          sampleLoss += Math.abs(yHat[j] - (target[j] ?? 0));
        }
      } else if (lossType === "huber") {
        for (let j = 0; j < yHat.length; j++) {
          const diff = yHat[j] - (target[j] ?? 0);
          const absDiff = Math.abs(diff);
          sampleLoss += absDiff <= hDelta ? 0.5 * diff * diff : hDelta * (absDiff - 0.5 * hDelta);
        }
      } else {
        for (let j = 0; j < yHat.length; j++) {
          const diff = (target[j] ?? 0) - yHat[j];
          sampleLoss += 0.5 * diff * diff;
        }
      }
      totalLoss += sampleLoss;

      // Accuracy evaluation
      if (target.length === 1) {
        const pred = (yHat[0] ?? 0) >= 0.5 ? 1 : 0;
        const tgt = (target[0] ?? 0) >= 0.5 ? 1 : 0;
        if (pred === tgt) correctCount++;
      } else {
        const predClass = argMax(yHat);
        const targetClass = argMax(target);
        if (predClass === targetClass) correctCount++;
      }
    }

    const avgLoss = totalLoss / dataset.length;
    const accuracy = (correctCount / dataset.length) * 100;

    return {
      loss: avgLoss,
      accuracy,
      totalLoss,
    };
  }

  // Generate Step-by-Step educational trace for single forward/backward pass
  public generateStepTrace(
    sampleOrInput: TrainingSample | number[],
    targetOrLossType?: number[] | LossFunctionType,
    optionalLossType: LossFunctionType = "mse"
  ): StepTraceFrame[] {
    let input: number[];
    let target: number[];
    let lossType: LossFunctionType = "mse";

    if (Array.isArray(sampleOrInput)) {
      input = sampleOrInput;
      target = Array.isArray(targetOrLossType) ? targetOrLossType : [0];
      if (typeof optionalLossType === "string") {
        lossType = optionalLossType;
      }
    } else if (sampleOrInput && typeof sampleOrInput === "object") {
      input = sampleOrInput.input || [];
      target = sampleOrInput.target || [];
      if (typeof targetOrLossType === "string") {
        lossType = targetOrLossType as LossFunctionType;
      }
    } else {
      input = new Array(this.layerSizes[0] || 1).fill(0);
      target = new Array(this.layerSizes[this.layerSizes.length - 1] || 1).fill(0);
    }

    const frames: StepTraceFrame[] = [];
    const fwd = this.forward(input);
    const bwd = this.backwardSingle(input, target, lossType);
    const numLayers = this.layerSizes.length;

    // Step 0: Input Injection
    frames.push({
      stepIndex: 0,
      phase: "forward_input",
      title: "第 0 步：输入层信号载入 (Input Loading)",
      formula: "a^{(0)} = x",
      description: `将样本输入向量 [${input.map((v) => v.toFixed(3)).join(", ")}] 载入输入层神经元。`,
      activeLayer: 0,
      activeNeurons: Array.from({ length: this.layerSizes[0] }, (_, i) => i),
      mathDetails: {
        "输入特征数": this.layerSizes[0],
        "输入向量": `[${input.map((v) => v.toFixed(2)).join(", ")}]`,
      },
    });

    // Hidden Forward Steps
    for (let l = 0; l < this.weights.length - 1; l++) {
      const act = this.activations[l];
      const z = fwd.layers[l + 1].z;
      const a = fwd.layers[l + 1].a;
      frames.push({
        stepIndex: frames.length,
        phase: "forward_hidden",
        title: `第 ${frames.length} 步：前向传播第 ${l + 1} 隐藏层 (Hidden Forward)`,
        formula: `z^{(${l + 1})} = W^{(${l + 1})} a^{(${l})} + b^{(${l + 1})}, \\quad a^{(${l + 1})} = \\sigma(z^{(${l + 1})})`,
        description: `利用突触权重矩阵做仿射加权求和，并通过 ${act} 非线性激活函数，映射得到隐藏特征向量。`,
        activeLayer: l + 1,
        fromLayer: l,
        toLayer: l + 1,
        mathDetails: {
          "激活函数": act,
          "净输入 z": `[${z.map((v) => v.toFixed(3)).join(", ")}]`,
          "激活输出 a": `[${a.map((v) => v.toFixed(3)).join(", ")}]`,
        },
      });
    }

    // Output Forward Step
    const outL = numLayers - 1;
    frames.push({
      stepIndex: frames.length,
      phase: "forward_output",
      title: `第 ${frames.length} 步：输出层前向映射与预测 (Output Forward)`,
      formula: `\\hat{y} = a^{(${outL})} = \\sigma_{out}(z^{(${outL})})`,
      description: `输出层产生模型推断值 [${fwd.output.map((v) => v.toFixed(3)).join(", ")}]，准备与真实标签 [${target.map((v) => v.toFixed(3)).join(", ")}] 对比。`,
      activeLayer: outL,
      fromLayer: outL - 1,
      toLayer: outL,
      mathDetails: {
        "预测值 y_hat": `[${fwd.output.map((v) => v.toFixed(4)).join(", ")}]`,
        "目标真实值 y": `[${target.map((v) => v.toFixed(4)).join(", ")}]`,
      },
    });

    // Compute Loss Step
    let lossFormula = `J(W, b) = \\frac{1}{2} \\sum_{k} (y_k - \\hat{y}_k)^2`;
    let outputDeltaFormula = `\\delta^{(L)} = \\frac{\\partial J}{\\partial z^{(L)}} = (\\hat{y} - y) \\odot \\sigma'(z^{(L)})`;

    if (lossType === "mae") {
      lossFormula = `J(W, b) = \\sum_{k} |y_k - \\hat{y}_k|`;
      outputDeltaFormula = `\\delta^{(L)} = \\frac{\\partial J}{\\partial z^{(L)}} = \\text{sgn}(\\hat{y} - y) \\odot \\sigma'(z^{(L)})`;
    } else if (lossType === "huber") {
      lossFormula = `J(W, b) = \\sum_{k} \\begin{cases} \\frac{1}{2}(y_k - \\hat{y}_k)^2 & |y_k - \\hat{y}_k| \\le \\delta \\\\ \\delta(|y_k - \\hat{y}_k| - \\frac{1}{2}\\delta) & |y_k - \\hat{y}_k| > \\delta \\end{cases}`;
      outputDeltaFormula = `\\delta^{(L)} = \\frac{\\partial J}{\\partial z^{(L)}} = \\text{clip}(\\hat{y} - y, -\\delta, \\delta) \\odot \\sigma'(z^{(L)})`;
    } else if (lossType === "cross_entropy") {
      lossFormula = `J(W, b) = -\\sum_{k} y_k \\ln(\\hat{y}_k)`;
      outputDeltaFormula = `\\delta^{(L)} = \\hat{y} - y`;
    } else if (lossType === "binary_cross_entropy") {
      lossFormula = `J(W, b) = -\\frac{1}{K} \\sum_{k} \\left[ y_k \\ln(\\hat{y}_k) + (1-y_k) \\ln(1-\\hat{y}_k) \\right]`;
      outputDeltaFormula = `\\delta^{(L)} = \\hat{y} - y`;
    }

    frames.push({
      stepIndex: frames.length,
      phase: "compute_loss",
      title: `第 ${frames.length} 步：计算目标损失 (Compute Objective Loss)`,
      formula: lossFormula,
      description: `评估当前预测与真实标签之间的距离误差，当前单样本损失 J = ${bwd.loss.toFixed(6)}。`,
      activeLayer: outL,
      mathDetails: {
        "损失函数类型": lossType.toUpperCase(),
        "当前损失值 J": bwd.loss.toFixed(6),
        "残差 (yHat - y)": `[${fwd.output.map((v, i) => (v - (target[i] ?? 0)).toFixed(4)).join(", ")}]`,
      },
    });

    // Output Error Gradient Delta Step
    frames.push({
      stepIndex: frames.length,
      phase: "backward_output",
      title: `第 ${frames.length} 步：反向求导输出层误差敏感度 (Output Delta \\delta^{(L)})`,
      formula: outputDeltaFormula,
      description: `由链式法则计算输出神经元的敏感度 \\delta^{(L)} = [${bwd.layerGradients[bwd.layerGradients.length - 1].delta.map((v) => v.toFixed(4)).join(", ")}]，误差由此开始逆流！`,
      activeLayer: outL,
      mathDetails: {
        "输出损失偏导 dJ/dyHat": `[${fwd.output.map((v, i) => NeuralNetwork.computeLossGradient(v, target[i] ?? 0, lossType, this.huberDelta).toFixed(4)).join(", ")}]`,
        "输出误差项 delta": `[${bwd.layerGradients[bwd.layerGradients.length - 1].delta.map((v) => v.toFixed(4)).join(", ")}]`,
      },
    });

    // Hidden Delta Propagation Steps
    for (let l = this.weights.length - 2; l >= 0; l--) {
      const gradL = bwd.layerGradients[l];
      frames.push({
        stepIndex: frames.length,
        phase: "backward_hidden",
        title: `第 ${frames.length} 步：误差项逆流至第 ${l + 1} 隐藏层 (Error Back-Flow)`,
        formula: `\\delta^{(${l + 1})} = \\left( (W^{(${l + 2})})^T \\delta^{(${l + 2})} \\right) \\odot \\sigma'(z^{(${l + 1})})`,
        description: `误差信号逆着突触权重反向回传，加权整合后乘以该层导数 \\sigma'(z^{(${l + 1})})。`,
        activeLayer: l + 1,
        fromLayer: l + 2,
        toLayer: l + 1,
        mathDetails: {
          "隐藏层 delta": `[${gradL.delta.map((v) => v.toFixed(4)).join(", ")}]`,
        },
      });
    }

    // Weight Update Step
    frames.push({
      stepIndex: frames.length,
      phase: "update_weights",
      title: `第 ${frames.length} 步：梯度下降参数更新 (Gradient Descent Step)`,
      formula: `W^{(l)} \\leftarrow W^{(l)} - \\eta \\cdot \\delta^{(l)} (a^{(l-1)})^T, \\quad b^{(l)} \\leftarrow b^{(l)} - \\eta \\cdot \\delta^{(l)}`,
      description: `计算权重偏导数 \\frac{\\partial J}{\\partial W}，沿负梯度方向平移参数，完成一轮演化迭代。`,
      activeLayer: 0,
      mathDetails: {
        "梯度范数 ||grad||": bwd.totalGradNorm.toFixed(6),
      },
    });

    return frames;
  }

  // Static evaluation helpers for Loss and Derivatives
  public static computeLossValue(
    yHat: number,
    y: number,
    lossType: LossFunctionType,
    huberDelta: number = 1.0
  ): number {
    const diff = yHat - y;
    if (lossType === "mae") {
      return Math.abs(diff);
    }
    if (lossType === "huber") {
      const absDiff = Math.abs(diff);
      return absDiff <= huberDelta
        ? 0.5 * diff * diff
        : huberDelta * (absDiff - 0.5 * huberDelta);
    }
    // Default MSE
    return 0.5 * diff * diff;
  }

  public static computeLossGradient(
    yHat: number,
    y: number,
    lossType: LossFunctionType,
    huberDelta: number = 1.0
  ): number {
    const diff = yHat - y;
    if (lossType === "mae") {
      return diff > 0 ? 1 : diff < 0 ? -1 : 0;
    }
    if (lossType === "huber") {
      return Math.max(-huberDelta, Math.min(huberDelta, diff));
    }
    // Default MSE dJ/dyHat = yHat - y
    return diff;
  }
}

// Helpers
function randomNormal(mean = 0, std = 1): number {
  let u1 = Math.random();
  let u2 = Math.random();
  while (u1 === 0) u1 = Math.random();
  while (u2 === 0) u2 = Math.random();
  const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  return mean + z * std;
}

function argMax(arr: number[]): number {
  if (!arr.length) return 0;
  let maxIdx = 0;
  let maxVal = arr[0];
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > maxVal) {
      maxVal = arr[i];
      maxIdx = i;
    }
  }
  return maxIdx;
}
