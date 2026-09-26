import express, { Request, Response } from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

let aiClient: GoogleGenAI | null = null;
function getAI(): GoogleGenAI | null {
  if (!aiClient && process.env.GEMINI_API_KEY) {
    aiClient = new GoogleGenAI({
      apiKey: process.env.GEMINI_API_KEY,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return aiClient;
}

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json({ limit: "10mb" }));

  // Health check
  app.get("/api/health", (_req: Request, res: Response) => {
    res.json({ status: "ok", hasAiKey: !!process.env.GEMINI_API_KEY });
  });

  // AI Chat Endpoint
  app.post("/api/chat", async (req: Request, res: Response) => {
    try {
      const { message, history, context } = req.body;
      const ai = getAI();

      if (!ai) {
        // High quality offline fallback responses for common neural network questions
        const offlineReply = generateOfflineResponse(message, context);
        return res.json({ reply: offlineReply, mode: "heuristic" });
      }

      const systemPrompt = `你是一位世界顶尖的深度学习与神经网络计算科学家、反向传播（Backpropagation）算法教学专家。
你正在“BP 神经网络与反向传播演化实验室”中协助学习者与研究人员。
你的风格：专业、通俗透彻、数学严密、淡雅严谨，多用结构化要点、公式推导（Markdown LaTeX）与直观几何直觉剖析。

上下文环境信息：
${context ? JSON.stringify(context, null, 2) : "用户正在探索 BP 神经网络基础理论与仿真实验。"}

请针对用户的疑问提供高水准、通俗易懂且带有公式推导与几何直觉的中文简体解答。`;

      const contents = [];
      if (history && Array.isArray(history)) {
        for (const item of history.slice(-6)) {
          contents.push({
            role: item.role === "user" ? "user" : "model",
            parts: [{ text: item.content }],
          });
        }
      }
      contents.push({
        role: "user",
        parts: [{ text: message }],
      });

      const response = await ai.models.generateContent({
        model: "gemini-3.7-flash",
        contents: contents as any,
        config: {
          systemInstruction: systemPrompt,
          temperature: 0.7,
        },
      });

      res.json({ reply: response.text || "未能生成有效回复，请重试。", mode: "gemini" });
    } catch (err: any) {
      console.error("Gemini chat error:", err);
      const fallback = generateOfflineResponse(req.body.message, req.body.context);
      res.json({ reply: fallback, mode: "fallback", error: err?.message });
    }
  });

  // AI Diagnosis Endpoint
  app.post("/api/diagnose", async (req: Request, res: Response) => {
    try {
      const { state } = req.body;
      const ai = getAI();

      if (!ai) {
        return res.json(generateOfflineDiagnosis(state));
      }

      const prompt = `请作为神经网络诊断系统，对以下正在运行的 BP 神经网络实验状态进行深度体检诊断并返回结构化建议：
实验状态：
- 当前轮次 (Epoch): ${state.epoch}
- 当前损失 (Loss): ${state.loss?.toFixed?.(6) ?? state.loss}
- 损失变化趋势: 最近5轮损失 [${state.lossHistory?.slice?.(-5)?.map?.((v: number) => v.toFixed(5))?.join(", ")}]
- 学习率 (Learning Rate): ${state.learningRate}
- 激活函数 (Activation): ${state.activation}
- 优化器 (Optimizer): ${state.optimizer}
- 网络拓扑 (Topology): ${JSON.stringify(state.topology)}
- 权重范数 (Weight Norm): ${state.weightNorm?.toFixed?.(4) ?? "N/A"}
- 平均梯度范数 (Grad Norm): ${state.gradNorm?.toFixed?.(6) ?? "N/A"}
- 任务类型/数据集: ${state.dataset || "自定义"}

请分析：
1. 是否存在梯度消失、梯度爆炸、死神经元、振荡发散、过拟合或鞍点停滞风险？
2. 当前学习率与优化器搭配是否合理？
3. 激活函数选择对深层反向传播的影响。
4. 具体可执行的超参数调优建议。
请用精炼、专业的中文Markdown格式给出诊断报告。`;

      const response = await ai.models.generateContent({
        model: "gemini-3.7-flash",
        contents: prompt,
        config: {
          systemInstruction: "你是一个专业的深度学习模型诊断与调优专家，输出严谨有据的体检与调优报告。",
          temperature: 0.4,
        },
      });

      res.json({
        report: response.text,
        healthScore: calculateHealthScore(state),
        status: detectNetworkStatus(state),
      });
    } catch (err: any) {
      console.error("Diagnosis error:", err);
      res.json(generateOfflineDiagnosis(req.body.state));
    }
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (_req: Request, res: Response) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`BP Neural Network Lab Server running on http://localhost:${PORT}`);
  });
}

function calculateHealthScore(state: any): number {
  if (!state) return 85;
  let score = 95;
  const grad = state.gradNorm ?? 0.1;
  const loss = state.loss ?? 0.5;
  const lr = state.learningRate ?? 0.05;

  if (grad < 1e-5 && state.epoch > 20 && loss > 0.3) score -= 35; // 梯度消失
  if (grad > 50 || isNaN(loss) || !isFinite(loss)) score -= 60; // 梯度爆炸
  if (lr > 2.0) score -= 15;
  if (state.lossHistory && state.lossHistory.length >= 4) {
    const last4 = state.lossHistory.slice(-4);
    const isOscillating = (last4[1] > last4[0] && last4[2] < last4[1] && last4[3] > last4[2]);
    if (isOscillating) score -= 20; // 震荡
  }
  return Math.max(10, Math.min(100, score));
}

function detectNetworkStatus(state: any): string {
  if (!state) return "正常收敛中";
  const grad = state.gradNorm ?? 0.1;
  const loss = state.loss ?? 0.5;
  if (isNaN(loss) || !isFinite(loss) || grad > 100) return "⚠️ 梯度爆炸 / 数值溢出";
  if (grad < 1e-6 && loss > 0.2 && state.epoch > 30) return "⚠️ 梯度消失 / 神经元休眠";
  if (state.epoch > 50 && loss < 0.01) return "✅ 高度收敛 (注意防过拟合)";
  if (loss < 0.1) return "🟢 良好拟合状态";
  return "⚡ 稳定演化训练中";
}

function generateOfflineDiagnosis(state: any) {
  const status = detectNetworkStatus(state);
  const score = calculateHealthScore(state);
  const activation = state?.activation || "Sigmoid";
  const lr = state?.learningRate || 0.1;
  const loss = state?.loss ?? 0.35;

  let advice = `### 🔍 网络状态体检总结
- **健康评分**：${score}/100（${status}）
- **当前损失**：${typeof loss === "number" ? loss.toFixed(5) : loss}
- **激活函数评估**：当前采用 \`${activation}\`。`;

  if (activation === "Sigmoid" && state?.topology?.length > 3) {
    advice += `\n- ⚠️ **警惕梯度消失**：Sigmoid 导数最大值为 0.25，在层数大于 3 时反向传播连乘会导致底层梯度极速衰减至近乎 0。建议切换为 **ReLU** 或 **LeakyReLU**。`;
  } else if (activation === "ReLU") {
    advice += `\n- ✨ **ReLU 优势**：正半区导数恒为 1，有效克服深层梯度消失；但若学习率过大（如 $\\eta > 0.5$），注意防范“神经元坏死（Dying ReLU）”。`;
  }

  if (lr > 1.0) {
    advice += `\n- ⚠️ **学习率偏高**（$\\eta = ${lr}$）：容易在损失曲面峡谷两壁剧烈跳跃震荡，建议调小至 \`0.05 ~ 0.2\` 或开启 **Momentum / Adam** 优化器。`;
  } else {
    advice += `\n- 💡 **调优建议**：当前学习率较为温和。若拟合速度较慢，可尝试适当加大隐藏层神经元数量，或开启动态动量优化。`;
  }

  return {
    report: advice,
    healthScore: score,
    status,
  };
}

function generateOfflineResponse(question: string, context?: any): string {
  const q = (question || "").toLowerCase();
  if (q.includes("梯度消失") || q.includes("sigmoid")) {
    return `### 💡 为什么 Sigmoid 激活函数极易引发“梯度消失”？

1. **数学导数上限限制**：
   Sigmoid 函数定义为 $\\sigma(z) = \\frac{1}{1 + e^{-z}}$，其一阶导数为：
   $$\\sigma'(z) = \\sigma(z)(1 - \\sigma(z))$$
   当 $z=0$ 时，$\\sigma'(z)$ 取得理论最大值 $\\mathbf{0.25}$。当 $|z| > 4$ 时，$\\sigma'(z) \\to 0$（饱和区）。

2. **链式法则的指数级衰减**：
   在 $L$ 层深层网络中，输入层权重梯度需经过多层连乘：
   $$\\frac{\\partial J}{\\partial W^{(1)}} = \\delta^{(L)} \\prod_{l=2}^{L} \\left( W^{(l)T} \\cdot \\text{diag}(\\sigma'(z^{(l-1)})) \\right) \\cdot (a^{(0)})^T$$
   每一层贡献的导数因子 $\\le 0.25$。经过 4 层隐藏层时，梯度缩减因子至多为 $0.25^4 \\approx 0.0039$；经过 10 层时降至 $10^{-6}$，底层参数几乎无法得到任何更新！

3. **破解方案**：
   - 使用 **ReLU**：$f(x) = \\max(0, x)$，正半区导数恒为 1，无梯度饱和区；
   - 权重初始化优化：采用 **He 初始化** 或 **Xavier 初始化**；
   - 引入 **Batch Normalization** 或 **残差连接 (ResNet)**。`;
  }

  if (q.includes("学习率") || q.includes("learning rate") || q.includes("eta")) {
    return `### 🎯 学习率 $\\eta$ 对 BP 训练的决定性影响

1. **$\\eta$ 过大（如 $\\eta > 1.0$）**：
   - 步长跨越谷底，在损失函数两壁剧烈震荡（Overshooting）；
   - 极端情况下损失变为 \`NaN\` 或 \`Infinity\`（数值溢出/梯度爆炸）。

2. **$\\eta$ 过小（如 $\\eta < 0.0001$）**：
   - 权重更新极其缓慢，耗费过多轮次；
   - 容易陷入平坦区（Plateau）或局部极小值/鞍点停滞。

3. **经典选型与自适应策略**：
   - **默认推荐**：纯 SGD 建议 $0.05 \\sim 0.2$；Adam 建议 $0.001 \\sim 0.01$；
   - **学习率衰减（LR Decay）**：$\\eta_t = \\frac{\\eta_0}{1 + k \\cdot t}$，前期大步探索，后期细步收敛。`;
  }

  if (q.includes("链式法则") || q.includes("反向传播") || q.includes("推导")) {
    return `### 📐 反向传播核心公式与链式法则四联公式

设第 $l$ 层的净输入为 $z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$，激活值为 $a^{(l)} = \\sigma(z^{(l)})$。

定义第 $l$ 层的误差项（敏感度）为：
$$\\delta^{(l)} \\triangleq \\frac{\\partial J}{\\partial z^{(l)}}$$

1. **输出层误差**（以 MSE 损失 $J = \\frac{1}{2}\\|y - a^{(L)}\\|^2$ 为例）：
   $$\\delta^{(L)} = (a^{(L)} - y) \\odot \\sigma'(z^{(L)})$$

2. **误差反向逆流传递**（由 $l+1$ 层逆推第 $l$ 层）：
   $$\\delta^{(l)} = \\left( (W^{(l+1)})^T \\delta^{(l+1)} \\right) \\odot \\sigma'(z^{(l)})$$

3. **权重梯度矩阵更新**：
   $$\\frac{\\partial J}{\\partial W^{(l)}} = \\delta^{(l)} (a^{(l-1)})^T, \\quad W^{(l)} \\leftarrow W^{(l)} - \\eta \\frac{\\partial J}{\\partial W^{(l)}}$$

4. **偏置梯度向量更新**：
   $$\\frac{\\partial J}{\\partial b^{(l)}} = \\delta^{(l)}, \\quad b^{(l)} \\leftarrow b^{(l)} - \\eta \\delta^{(l)}$$`;
  }

  return `### 🔬 BP 神经网络智能助手解答
感谢您的提问！BP（Backpropagation）算法的核心精髓在于**“前向计算误差，反向传播梯度”**。

- **前向传播**：将输入向量 $x$ 逐层做线性映射 $W a + b$ 与非线性激活 $\\sigma(z)$，最终在输出端得到预测值 $\\hat{y}$。
- **误差逆流**：利用多元微积分**链式法则**，将损失函数关于预测值的误差 $\\delta^{(L)}$ 逐层向后倒推，计算出所有权重参数的偏导数 $\\frac{\\partial J}{\\partial W}$。
- **参数演化**：沿负梯度方向更新权重 $W \\leftarrow W - \\eta \\nabla_W J$，驱动决策边界逐步扭曲拟合真实目标。

您可以在左侧调整网络层数、激活函数与学习率，并在画布中观察信号脉冲流与 2D 决策边界的动态收敛过程！`;
}

startServer();
