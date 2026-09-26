/**
 * Case Study Datasets & 2D Toy Boundary Distributions for BP Lab
 */

import { CaseStudy, TrainingSample } from "../types";

// Helper: Split dataset into 80% Train and 20% Test/Prediction set with deterministic distribution
export function splitTrainTest(
  samples: TrainingSample[],
  trainRatio: number = 0.8
): { train: TrainingSample[]; test: TrainingSample[] } {
  if (!samples || samples.length === 0) {
    return { train: [], test: [] };
  }

  // Uniform interleaved sampling to preserve class/value distribution
  const train: TrainingSample[] = [];
  const test: TrainingSample[] = [];
  const step = Math.round(1 / (1 - trainRatio)); // e.g. 5 for 0.8

  samples.forEach((sample, idx) => {
    if ((idx + 1) % step === 0) {
      test.push(sample);
    } else {
      train.push(sample);
    }
  });

  // Fallback check
  if (test.length === 0 && samples.length > 1) {
    test.push(train.pop()!);
  }

  return { train, test };
}

// 1. Case 1: 2-Input 1-Output Thermal Comfort / Apparent Temperature Dataset Generator
export function generateThermalComfortDataset(
  count: number = 60,
  noise: number = 0.03
): TrainingSample[] {
  const samples: TrainingSample[] = [];
  const rows = Math.max(6, Math.round(Math.sqrt(count * 1.1)));
  const cols = Math.max(6, Math.round(count / rows));

  let index = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (samples.length >= count) break;

      // Jittered grid covering temperature [16°C, 38°C] and relative humidity [20%, 90%]
      const tRatio = Math.max(0, Math.min(1, (r + Math.random() * 0.85) / rows));
      const rhRatio = Math.max(0, Math.min(1, (c + Math.random() * 0.85) / cols));

      const tempC = 16 + tRatio * 22; // 16 ~ 38 °C
      const rhPct = 20 + rhRatio * 70; // 20 ~ 90 %

      // Non-linear Apparent Temperature / Thermal Feel model (HVAC / Meteorological approximation)
      // When temperature is high, humidity dramatically impedes perspiration evaporation
      const deltaT = Math.max(0, tempC - 15);
      const humidityFactor = (rhPct - 40) / 100;
      const apparentTemp = tempC + humidityFactor * (0.4 * deltaT + 0.03 * deltaT * deltaT);

      // Add realistic sensor observation noise
      const noiseVal = (Math.random() - 0.5) * 2 * noise * 30;
      const noisyApparent = Math.max(14, Math.min(46, apparentTemp + noiseVal));

      // Normalized inputs and target [0, 1] for neural network
      const normT = Number(tRatio.toFixed(4));
      const normRH = Number(rhRatio.toFixed(4));
      const normTarget = Number(Math.max(0, Math.min(1, (noisyApparent - 15) / 30)).toFixed(4));

      let comfortStatus = "舒适宜人";
      if (noisyApparent < 18) comfortStatus = "偏凉爽";
      else if (noisyApparent > 38) comfortStatus = "极度闷热 (防暑预警)";
      else if (noisyApparent > 30) comfortStatus = "闷热潮湿";

      samples.push({
        input: [normT, normRH],
        target: [normTarget],
        label: `${tempC.toFixed(1)}°C, 湿度${rhPct.toFixed(0)}% → 体感${noisyApparent.toFixed(1)}°C (${comfortStatus})`,
      });
      index++;
    }
  }

  // Interleave and sort for clean distribution
  return samples;
}

// 1. Auxiliary: y = x Linear Identity Dataset Generator (with customizable noise)
export function generateLinearIdentityDataset(
  count: number = 60,
  noise: number = 0.06,
  minX: number = 0.0,
  maxX: number = 1.0
): TrainingSample[] {
  const samples: TrainingSample[] = [];
  const range = maxX - minX;

  for (let i = 0; i < count; i++) {
    // Generate evenly spaced base x with slight jitter for rich distribution
    const baseRatio = i / (count - 1);
    const x = minX + baseRatio * range + (Math.random() - 0.5) * (range / count) * 0.5;
    const clampedX = Math.max(minX, Math.min(maxX, x));

    // y = x + gaussian/uniform noise
    const noiseVal = (Math.random() - 0.5) * 2 * noise;
    const y = Math.max(0, Math.min(1.2, clampedX + noiseVal));

    samples.push({
      input: [Number(clampedX.toFixed(4))],
      target: [Number(y.toFixed(4))],
      label: `x = ${clampedX.toFixed(2)}, y = ${y.toFixed(2)}`,
    });
  }

  // Sort by x ascending for clean plotting and sequence comprehension
  return samples.sort((a, b) => a.input[0] - b.input[0]);
}

// 2. XOR Dataset Generator (retained as auxiliary 2D boundary dataset)
export function generateXORDataset(noiseOrCount: number = 0.05, countOrNoise: number = 60): TrainingSample[] {
  let noise = 0.05;
  let count = 60;
  if (noiseOrCount > 1 && countOrNoise <= 1) {
    count = Math.round(noiseOrCount);
    noise = countOrNoise;
  } else {
    noise = noiseOrCount;
    count = Math.round(countOrNoise);
  }

  const base = [
    { input: [0, 0], target: [0] },
    { input: [0, 1], target: [1] },
    { input: [1, 0], target: [1] },
    { input: [1, 1], target: [0] },
  ];

  const samples: TrainingSample[] = [];
  const perCorner = Math.max(1, Math.floor(count / 4));
  for (const b of base) {
    for (let i = 0; i < perCorner; i++) {
      const n1 = (Math.random() - 0.5) * noise * 2;
      const n2 = (Math.random() - 0.5) * noise * 2;
      samples.push({
        input: [Math.max(-0.2, Math.min(1.2, b.input[0] + n1)), Math.max(-0.2, Math.min(1.2, b.input[1] + n2))],
        target: [...b.target],
        label: b.target[0],
      });
    }
  }
  return samples;
}

// 2. Concentric Circles Dataset Generator (2D)
export function generateCirclesDataset(noiseOrCount: number = 0.05, countOrNoise: number = 100): TrainingSample[] {
  let noise = 0.05;
  let count = 100;
  if (noiseOrCount > 1 && countOrNoise <= 1) {
    count = Math.round(noiseOrCount);
    noise = countOrNoise;
  } else {
    noise = noiseOrCount;
    count = Math.round(countOrNoise);
  }
  const samples: TrainingSample[] = [];
  const half = Math.max(1, Math.floor(count / 2));

  // Inner circle: class 1
  for (let i = 0; i < half; i++) {
    const angle = Math.random() * 2 * Math.PI;
    const r = 0.3 * Math.sqrt(Math.random()) + (Math.random() - 0.5) * noise;
    const x1 = 0.5 + r * Math.cos(angle);
    const x2 = 0.5 + r * Math.sin(angle);
    samples.push({ input: [x1, x2], target: [1], label: 1 });
  }

  // Outer ring: class 0
  for (let i = 0; i < half; i++) {
    const angle = Math.random() * 2 * Math.PI;
    const r = 0.65 + 0.25 * Math.random() + (Math.random() - 0.5) * noise;
    const x1 = 0.5 + r * Math.cos(angle);
    const x2 = 0.5 + r * Math.sin(angle);
    samples.push({ input: [x1, x2], target: [0], label: 0 });
  }

  return samples;
}

// 3. Two Spirals Dataset Generator (2D)
export function generateSpiralsDataset(noiseOrCount: number = 0.03, countOrNoise: number = 120): TrainingSample[] {
  let noise = 0.03;
  let count = 120;
  if (noiseOrCount > 1 && countOrNoise <= 1) {
    count = Math.round(noiseOrCount);
    noise = countOrNoise;
  } else {
    noise = noiseOrCount;
    count = Math.round(countOrNoise);
  }
  const samples: TrainingSample[] = [];
  const n = Math.max(1, Math.floor(count / 2));

  for (let i = 0; i < n; i++) {
    const r = (i / n) * 0.45 + 0.05;
    const t = 1.75 * i * (2 * Math.PI / n);
    // Spiral 1 (Class 1)
    const x1_1 = 0.5 + r * Math.cos(t) + (Math.random() - 0.5) * noise;
    const x2_1 = 0.5 + r * Math.sin(t) + (Math.random() - 0.5) * noise;
    samples.push({ input: [x1_1, x2_1], target: [1], label: 1 });

    // Spiral 2 (Class 0)
    const x1_0 = 0.5 + r * Math.cos(t + Math.PI) + (Math.random() - 0.5) * noise;
    const x2_0 = 0.5 + r * Math.sin(t + Math.PI) + (Math.random() - 0.5) * noise;
    samples.push({ input: [x1_0, x2_0], target: [0], label: 0 });
  }

  return samples;
}

// 4. Two Moons Dataset Generator (2D)
export function generateMoonsDataset(noiseOrCount: number = 0.05, countOrNoise: number = 100): TrainingSample[] {
  let noise = 0.05;
  let count = 100;
  if (noiseOrCount > 1 && countOrNoise <= 1) {
    count = Math.round(noiseOrCount);
    noise = countOrNoise;
  } else {
    noise = noiseOrCount;
    count = Math.round(countOrNoise);
  }
  const samples: TrainingSample[] = [];
  const n = Math.max(1, Math.floor(count / 2));

  // Top moon (Class 1)
  for (let i = 0; i < n; i++) {
    const phi = (i / n) * Math.PI;
    const x1 = 0.35 + 0.35 * Math.cos(phi) + (Math.random() - 0.5) * noise;
    const x2 = 0.55 + 0.35 * Math.sin(phi) + (Math.random() - 0.5) * noise;
    samples.push({ input: [x1, x2], target: [1], label: 1 });
  }

  // Bottom moon (Class 0)
  for (let i = 0; i < n; i++) {
    const phi = (i / n) * Math.PI;
    const x1 = 0.65 - 0.35 * Math.cos(phi) + (Math.random() - 0.5) * noise;
    const x2 = 0.45 - 0.35 * Math.sin(phi) + (Math.random() - 0.5) * noise;
    samples.push({ input: [x1, x2], target: [0], label: 0 });
  }

  return samples;
}

// 5. MNIST Mini 8x8 Feature Set (Digits 0-4 / 0-9)
export function getMNISTDigitSamples(): { samples: TrainingSample[]; digitPatterns: number[][][] } {
  // 8x8 bitmap templates for digits 0, 1, 2, 3, 4
  const patterns: number[][][] = [
    // 0
    [
      [0, 0, 1, 1, 1, 1, 0, 0],
      [0, 1, 1, 0, 0, 1, 1, 0],
      [1, 1, 0, 0, 0, 0, 1, 1],
      [1, 1, 0, 0, 0, 0, 1, 1],
      [1, 1, 0, 0, 0, 0, 1, 1],
      [1, 1, 0, 0, 0, 0, 1, 1],
      [0, 1, 1, 0, 0, 1, 1, 0],
      [0, 0, 1, 1, 1, 1, 0, 0],
    ],
    // 1
    [
      [0, 0, 0, 1, 1, 0, 0, 0],
      [0, 0, 1, 1, 1, 0, 0, 0],
      [0, 1, 0, 1, 1, 0, 0, 0],
      [0, 0, 0, 1, 1, 0, 0, 0],
      [0, 0, 0, 1, 1, 0, 0, 0],
      [0, 0, 0, 1, 1, 0, 0, 0],
      [0, 0, 0, 1, 1, 0, 0, 0],
      [0, 1, 1, 1, 1, 1, 1, 0],
    ],
    // 2
    [
      [0, 1, 1, 1, 1, 1, 0, 0],
      [1, 1, 0, 0, 0, 1, 1, 0],
      [0, 0, 0, 0, 0, 1, 1, 0],
      [0, 0, 0, 0, 1, 1, 0, 0],
      [0, 0, 0, 1, 1, 0, 0, 0],
      [0, 0, 1, 1, 0, 0, 0, 0],
      [0, 1, 1, 0, 0, 0, 0, 0],
      [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    // 3
    [
      [0, 1, 1, 1, 1, 1, 0, 0],
      [1, 1, 0, 0, 0, 1, 1, 0],
      [0, 0, 0, 0, 0, 1, 1, 0],
      [0, 0, 1, 1, 1, 1, 0, 0],
      [0, 0, 0, 0, 0, 1, 1, 0],
      [0, 0, 0, 0, 0, 1, 1, 0],
      [1, 1, 0, 0, 0, 1, 1, 0],
      [0, 1, 1, 1, 1, 1, 0, 0],
    ],
    // 4
    [
      [0, 0, 0, 0, 1, 1, 0, 0],
      [0, 0, 0, 1, 1, 1, 0, 0],
      [0, 0, 1, 0, 1, 1, 0, 0],
      [0, 1, 0, 0, 1, 1, 0, 0],
      [1, 1, 1, 1, 1, 1, 1, 1],
      [0, 0, 0, 0, 1, 1, 0, 0],
      [0, 0, 0, 0, 1, 1, 0, 0],
      [0, 0, 0, 0, 1, 1, 0, 0],
    ],
  ];

  // Flatten and generate augmented samples with subtle noise
  const samples: TrainingSample[] = [];
  patterns.forEach((pattern, digit) => {
    const flatBase = pattern.flat();
    // Create one-hot target
    const target = [0, 0, 0, 0, 0];
    target[digit] = 1;

    for (let s = 0; s < 12; s++) {
      const noisyInput = flatBase.map((v) => {
        const noise = (Math.random() - 0.5) * 0.15;
        return Math.max(0, Math.min(1, v + noise));
      });
      samples.push({
        input: noisyInput,
        target: [...target],
        label: `数字 ${digit}`,
      });
    }
  });

  return { samples, digitPatterns: patterns };
}

// 6. House Price Continuous Regression Dataset
export function getHousePriceRegressionDataset(): TrainingSample[] {
  // Features: [normalized area (50-250m2), normalized bedrooms (1-5), normalized location score (0-10)]
  // Target: [normalized price in 10k CNY (100-800)]
  const rawData = [
    { area: 60, rooms: 1, loc: 5, price: 150 },
    { area: 75, rooms: 2, loc: 6, price: 210 },
    { area: 90, rooms: 3, loc: 7, price: 320 },
    { area: 110, rooms: 3, loc: 8, price: 430 },
    { area: 130, rooms: 4, loc: 7.5, price: 490 },
    { area: 150, rooms: 4, loc: 9, price: 620 },
    { area: 180, rooms: 5, loc: 8.5, price: 710 },
    { area: 220, rooms: 5, loc: 9.5, price: 850 },
    { area: 55, rooms: 1, loc: 3, price: 110 },
    { area: 85, rooms: 2, loc: 4, price: 190 },
    { area: 100, rooms: 3, loc: 5, price: 260 },
    { area: 140, rooms: 3, loc: 6, price: 380 },
    { area: 160, rooms: 4, loc: 5.5, price: 420 },
    { area: 200, rooms: 4, loc: 8, price: 670 },
    { area: 70, rooms: 2, loc: 9, price: 330 },
    { area: 95, rooms: 3, loc: 8.5, price: 410 },
    { area: 125, rooms: 3, loc: 9.5, price: 580 },
    { area: 175, rooms: 4, loc: 6.5, price: 510 },
  ];

  return rawData.map((d) => ({
    input: [
      (d.area - 50) / 200, // 0~1
      (d.rooms - 1) / 5, // 0~1
      d.loc / 10, // 0~1
    ],
    target: [(d.price - 100) / 800], // 0~1
    label: `${d.price} 万元 (${d.area}m², ${d.rooms}室)`,
  }));
}

// 7. Customer Churn Binary Classification Dataset
export function getCustomerChurnDataset(): TrainingSample[] {
  // Features: [Tenure (0-72 mo), Monthly Charges (20-120$), Support Calls (0-9)]
  // Target: [Churn: 1 or 0]
  const data = [
    { tenure: 2, charge: 85, calls: 5, churn: 1 },
    { tenure: 4, charge: 95, calls: 6, churn: 1 },
    { tenure: 48, charge: 35, calls: 0, churn: 0 },
    { tenure: 60, charge: 45, calls: 1, churn: 0 },
    { tenure: 6, charge: 110, calls: 4, churn: 1 },
    { tenure: 36, charge: 65, calls: 2, churn: 0 },
    { tenure: 12, charge: 75, calls: 3, churn: 1 },
    { tenure: 55, charge: 25, calls: 0, churn: 0 },
    { tenure: 1, charge: 100, calls: 7, churn: 1 },
    { tenure: 70, charge: 50, calls: 1, churn: 0 },
    { tenure: 18, charge: 80, calls: 4, churn: 1 },
    { tenure: 40, charge: 40, calls: 1, churn: 0 },
    { tenure: 8, charge: 90, calls: 5, churn: 1 },
    { tenure: 65, charge: 60, calls: 0, churn: 0 },
    { tenure: 3, charge: 105, calls: 6, churn: 1 },
    { tenure: 50, charge: 30, calls: 1, churn: 0 },
    { tenure: 22, charge: 70, calls: 2, churn: 0 },
    { tenure: 15, charge: 88, calls: 4, churn: 1 },
  ];

  return data.map((d) => ({
    input: [d.tenure / 72, (d.charge - 20) / 100, d.calls / 9],
    target: [d.churn],
    label: d.churn === 1 ? "流失客户" : "留存客户",
  }));
}

// 8. Iris 3-Class Dataset
export function getIrisDataset(): TrainingSample[] {
  // Features: [Sepal Length, Sepal Width, Petal Length, Petal Width] normalized
  // Targets: [Setosa, Versicolor, Virginica] (one-hot)
  const rawIris = [
    // Setosa (Class 0)
    { sl: 5.1, sw: 3.5, pl: 1.4, pw: 0.2, c: 0 },
    { sl: 4.9, sw: 3.0, pl: 1.4, pw: 0.2, c: 0 },
    { sl: 4.7, sw: 3.2, pl: 1.3, pw: 0.2, c: 0 },
    { sl: 5.0, sw: 3.6, pl: 1.4, pw: 0.2, c: 0 },
    { sl: 5.4, sw: 3.9, pl: 1.7, pw: 0.4, c: 0 },
    { sl: 4.6, sw: 3.4, pl: 1.4, pw: 0.3, c: 0 },
    // Versicolor (Class 1)
    { sl: 7.0, sw: 3.2, pl: 4.7, pw: 1.4, c: 1 },
    { sl: 6.4, sw: 3.2, pl: 4.5, pw: 1.5, c: 1 },
    { sl: 6.9, sw: 3.1, pl: 4.9, pw: 1.5, c: 1 },
    { sl: 5.5, sw: 2.3, pl: 4.0, pw: 1.3, c: 1 },
    { sl: 6.5, sw: 2.8, pl: 4.6, pw: 1.5, c: 1 },
    { sl: 5.7, sw: 2.8, pl: 4.5, pw: 1.3, c: 1 },
    // Virginica (Class 2)
    { sl: 6.3, sw: 3.3, pl: 6.0, pw: 2.5, c: 2 },
    { sl: 5.8, sw: 2.7, pl: 5.1, pw: 1.9, c: 2 },
    { sl: 7.1, sw: 3.0, pl: 5.9, pw: 2.1, c: 2 },
    { sl: 6.3, sw: 2.9, pl: 5.6, pw: 1.8, c: 2 },
    { sl: 6.5, sw: 3.0, pl: 5.8, pw: 2.2, c: 2 },
    { sl: 7.6, sw: 3.0, pl: 6.6, pw: 2.1, c: 2 },
  ];

  const speciesNames = ["山鸢尾 (Setosa)", "杂色鸢尾 (Versicolor)", "维吉尼亚鸢尾 (Virginica)"];

  return rawIris.map((d) => {
    const target = [0, 0, 0];
    target[d.c] = 1;
    return {
      input: [
        (d.sl - 4.0) / 4.0,
        (d.sw - 2.0) / 2.5,
        (d.pl - 1.0) / 6.0,
        d.pw / 2.5,
      ],
      target,
      label: speciesNames[d.c],
    };
  });
}

// 9. Nonlinear Function Wave Fitting (Universal Approximation)
export function getNonlinearWaveDataset(points: number = 40): TrainingSample[] {
  const samples: TrainingSample[] = [];
  for (let i = 0; i < points; i++) {
    const x = i / (points - 1); // 0 to 1
    // Complex composite wave: sin(2pi*x)*exp(-0.5x) + 0.3*cos(4pi*x)
    const rawY = Math.sin(2 * Math.PI * x) * Math.exp(-0.4 * x) + 0.3 * Math.cos(4 * Math.PI * x);
    const normY = (rawY + 1.2) / 2.4; // normalize to ~0..1
    samples.push({
      input: [x],
      target: [normY],
      label: `x=${x.toFixed(2)}, y=${normY.toFixed(3)}`,
    });
  }
  return samples;
}

// Master Case Studies Definition (Updated 6 Classic Cases with Case 1: 2-Input Thermal Comfort Regression)
export const ALL_CASE_STUDIES: CaseStudy[] = [
  {
    id: "thermal_comfort",
    title: "1. 智能温湿度体感指数与热舒适度预测",
    subtitle: "环境温度与相对湿度双变量输入、0.8 训练学习与 0.2 预测集体感温度回归",
    category: "regression",
    iconName: "ThermometerSun",
    inputDim: 2,
    outputDim: 1,
    recommendedTopology: [2, 4, 1],
    recommendedActivation: "leaky_relu",
    recommendedOutputActivation: "linear",
    recommendedLr: 0.05,
    recommendedOptimizer: "adam",
    recommendedLoss: "mse",
    description: "针对智能家居空调恒温与气象舒适度监测场景，以【环境温度 T (16~38°C)】与【相对湿度 RH (20~90%)】作为双维度输入特征，拟合人体体感温度 (Apparent Temperature) 非线性回归超曲面。经 0.8 训练集学习后在 0.2 预测集上检验预测精度与 R² 拟合优度，支持在线双特征推断与舒适度防暑预警。",
    realWorldContext: "智能家居空调自适应恒温变频控制、智慧建筑热负荷感知、气象防暑降温指数播报、数据中心温湿度精细化节能调控。",
    featureNames: ["环境温度 T (16~38°C)", "相对湿度 RH (20~90%)"],
    targetNames: ["预测体感温度 T_feel (15~45°C)"],
    samples: generateThermalComfortDataset(60, 0.03),
    boundarySupport: false,
  },
  {
    id: "mnist_digit",
    title: "2. MNIST 手写数字识别",
    subtitle: "8×8 特征矩阵、0.8 训练学习与 0.2 测试集 Softmax 概率分类评估",
    category: "multiclass",
    iconName: "Binary",
    inputDim: 64,
    outputDim: 5,
    recommendedTopology: [64, 16, 5],
    recommendedActivation: "relu",
    recommendedOutputActivation: "softmax",
    recommendedLr: 0.15,
    recommendedOptimizer: "adam",
    recommendedLoss: "cross_entropy",
    description: "经典计算机视觉入门基准，将 8×8 手写像素灰度向量投影至深层隐空间，经 0.8 训练集学习后在 0.2 测试集上检验数字 0~4 的 Softmax 识别准确率与混淆表现。",
    realWorldContext: "邮政编码自动分拣、银行支票手写金额识别、文档光学字符识别 (OCR)。",
    featureNames: Array.from({ length: 64 }, (_, i) => `像素 p[${Math.floor(i / 8)},${i % 8}]`),
    targetNames: ["数字 0", "数字 1", "数字 2", "数字 3", "数字 4"],
    samples: getMNISTDigitSamples().samples,
    boundarySupport: false,
  },
  {
    id: "house_price",
    title: "3. 房价多维连续回归预测",
    subtitle: "面积/居室/地段多特征、0.8 训练学习与 0.2 预测集残差分析",
    category: "regression",
    iconName: "Home",
    inputDim: 3,
    outputDim: 1,
    recommendedTopology: [3, 6, 4, 1],
    recommendedActivation: "leaky_relu",
    recommendedOutputActivation: "linear",
    recommendedLr: 0.08,
    recommendedOptimizer: "adam",
    recommendedLoss: "mse",
    description: "利用房屋建筑面积、居室数与学区地段分 3 大维度，拟合连续的房价回归超曲面，划分 0.8 训练与 0.2 预测，评估模型对未见房源真实总价的估值误差与散点对比。",
    realWorldContext: "二手房在线估价系统、房地产资产配置模型、银行抵押贷款风控估值。",
    featureNames: ["建筑面积 (50~250m²)", "居室数 (1~5室)", "学区地段综合分 (0~10分)"],
    targetNames: ["预测总价 (万元)"],
    samples: getHousePriceRegressionDataset(),
    boundarySupport: false,
  },
  {
    id: "customer_churn",
    title: "4. 客户流失二分类预测",
    subtitle: "在网时长/月资费/投诉量、0.8 训练与 0.2 测试集流失风险预警",
    category: "binary",
    iconName: "Users",
    inputDim: 3,
    outputDim: 1,
    recommendedTopology: [3, 5, 1],
    recommendedActivation: "sigmoid",
    recommendedOutputActivation: "sigmoid",
    recommendedLr: 0.1,
    recommendedOptimizer: "adam",
    recommendedLoss: "binary_cross_entropy",
    description: "根据用户入网时长、每月套餐资费与客服投诉工单频率，经 0.8 训练集学习后在 0.2 测试集上推断流失概率分布，给出高风险客户预警与混淆矩阵。",
    realWorldContext: "SaaS 订阅续费预警、电信运营商精准关怀、电商高净值会员留存。",
    featureNames: ["在网时长 (0~72月)", "每月套餐消费 ($20~$120)", "近半年投诉工单数 (0~9次)"],
    targetNames: ["流失概率 (0=留存, 1=流失)"],
    samples: getCustomerChurnDataset(),
    boundarySupport: false,
  },
  {
    id: "iris_species",
    title: "5. 鸢尾花 (Iris) 三分类",
    subtitle: "4 维花瓣特征、0.8 训练学习与 0.2 测试集 Softmax 判决矩阵",
    category: "multiclass",
    iconName: "Flower",
    inputDim: 4,
    outputDim: 3,
    recommendedTopology: [4, 6, 3],
    recommendedActivation: "tanh",
    recommendedOutputActivation: "softmax",
    recommendedLr: 0.12,
    recommendedOptimizer: "adam",
    recommendedLoss: "cross_entropy",
    description: "Fisher 1936 年经典统计学数据集，利用花萼长宽与花瓣长宽 4 维特征，在 0.8 训练集上收敛后，在 0.2 测试集上验证山鸢尾、杂色鸢尾与维吉尼亚鸢尾的泛化分类能力。",
    realWorldContext: "植物标本自动分类、生物物种形态学分析、医学病理切片分型。",
    featureNames: ["花萼长度 (Sepal L)", "花萼宽度 (Sepal W)", "花瓣长度 (Petal L)", "花瓣宽度 (Petal W)"],
    targetNames: ["山鸢尾 (Setosa)", "杂色鸢尾 (Versicolor)", "维吉尼亚 (Virginica)"],
    samples: getIrisDataset(),
    boundarySupport: false,
  },
  {
    id: "nonlinear_wave",
    title: "6. 非线性函数复合波形逼近",
    subtitle: "通用近似定理 (UAT)、0.8 训练与 0.2 预测集连续曲线拟合评估",
    category: "approximation",
    iconName: "Activity",
    inputDim: 1,
    outputDim: 1,
    recommendedTopology: [1, 10, 8, 1],
    recommendedActivation: "tanh",
    recommendedOutputActivation: "linear",
    recommendedLr: 0.05,
    recommendedOptimizer: "adam",
    recommendedLoss: "mse",
    description: "通过复合阻尼正弦波形 $y = \\sin(2\\pi x) e^{-0.4x} + 0.3\\cos(4\\pi x)$，以 0.8 离散采样点训练网络，并在 0.2 预测测试集上评估连续逼近曲线与真实波形的拟合精度与泛化效果。",
    realWorldContext: "声学音频信号合成、金融时序波动拟合、物理动力学系统辨识。",
    featureNames: ["输入自变量 X (0.0 ~ 1.0)"],
    targetNames: ["目标波形 Y"],
    samples: getNonlinearWaveDataset(50),
    boundarySupport: false,
  },
];

/**
 * Dataset Export Utilities for 6 Classic Cases & Toy Sets
 */

// Helper to escape CSV field
function escapeCsv(val: any): string {
  if (val === undefined || val === null) return "";
  const str = String(val);
  if (str.includes(",") || str.includes('"') || str.includes("\n")) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

// Generate Standard RFC-4180 CSV
export function exportCaseStudyToCSV(
  caseStudy: CaseStudy,
  subset: "all" | "train" | "val" = "all",
  customSamples?: TrainingSample[]
): string {
  const allSamples = customSamples || caseStudy.samples;
  const { train, test: val } = splitTrainTest(allSamples, 0.8);
  const targetSamples = subset === "train" ? train : subset === "val" ? val : allSamples;

  // Build CSV Header
  const headers: string[] = ["sample_id", "split_group"];
  caseStudy.featureNames.forEach((fn, idx) => {
    headers.push(`feature_${idx + 1}_${fn.replace(/[\s\(\)\/]/g, "_")}`);
  });
  caseStudy.targetNames.forEach((tn, idx) => {
    headers.push(`target_${idx + 1}_${tn.replace(/[\s\(\)\/]/g, "_")}`);
  });
  headers.push("sample_label_description");

  const lines: string[] = [headers.map(escapeCsv).join(",")];

  targetSamples.forEach((s, idx) => {
    // Determine if this sample is in train or val
    const isVal = val.includes(s);
    const splitGroup = isVal ? "val_20pct" : "train_80pct";

    const row: any[] = [idx + 1, splitGroup];

    // Add inputs
    s.input.forEach((val) => row.push(typeof val === "number" ? val.toFixed(4) : val));

    // Add targets
    s.target.forEach((val) => row.push(typeof val === "number" ? val.toFixed(4) : val));

    // Add label
    row.push(s.label ?? "");

    lines.push(row.map(escapeCsv).join(","));
  });

  return lines.join("\n");
}

// Generate Structured JSON
export function exportCaseStudyToJSON(
  caseStudy: CaseStudy,
  subset: "all" | "train" | "val" = "all",
  customSamples?: TrainingSample[]
): string {
  const allSamples = customSamples || caseStudy.samples;
  const { train, test: val } = splitTrainTest(allSamples, 0.8);
  const targetSamples = subset === "train" ? train : subset === "val" ? val : allSamples;

  const exportObj = {
    metadata: {
      caseId: caseStudy.id,
      title: caseStudy.title,
      subtitle: caseStudy.subtitle,
      category: caseStudy.category,
      exportedAt: new Date().toISOString(),
      subsetMode: subset,
      trainSplitRatio: 0.8,
      valSplitRatio: 0.2,
      totalSamplesInSubset: targetSamples.length,
      fullDatasetTotal: allSamples.length,
      trainCount: train.length,
      valCount: val.length,
      inputDimension: caseStudy.inputDim,
      outputDimension: caseStudy.outputDim,
      featureNames: caseStudy.featureNames,
      targetNames: caseStudy.targetNames,
      recommendedTopology: caseStudy.recommendedTopology,
      recommendedActivation: caseStudy.recommendedActivation,
      recommendedOptimizer: caseStudy.recommendedOptimizer,
      recommendedLoss: caseStudy.recommendedLoss,
    },
    samples: targetSamples.map((s, idx) => {
      const isVal = val.includes(s);
      return {
        id: idx + 1,
        split: isVal ? "validation_20pct" : "train_80pct",
        features: s.input,
        targets: s.target,
        label: s.label,
      };
    }),
  };

  return JSON.stringify(exportObj, null, 2);
}

// Browser Triggered Direct File Download
export function triggerFileDownload(content: string, filename: string, mimeType: string = "text/csv;charset=utf-8;") {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.setAttribute("download", filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

