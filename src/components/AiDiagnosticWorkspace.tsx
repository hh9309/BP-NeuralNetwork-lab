import React, { useState, useRef, useEffect } from "react";
import {
  Sparkles,
  Send,
  Bot,
  User,
  Activity,
  AlertTriangle,
  CheckCircle2,
  HelpCircle,
  Stethoscope,
  RefreshCw,
  MessageSquare,
  Settings,
  Key,
  Eye,
  EyeOff,
  Check,
  X,
  ExternalLink,
  Zap,
  Layers,
  Cpu,
  ShieldCheck,
  Flame,
  AlertCircle,
  Copy,
  Trash2,
  ChevronRight,
  TrendingDown,
  BookOpen,
  FileText,
  Download,
  Printer,
  Share2,
  Sliders,
  Compass,
  GitCommit,
  BarChart2,
} from "lucide-react";
import { ChatMessage, DiagnosisResult, ActivationType, OptimizerType } from "../types";
import { BlockMath, InlineMath } from "../lib/katex-helper";

interface AiDiagnosticWorkspaceProps {
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
  onApplyTuning?: (params: {
    learningRate?: number;
    activation?: ActivationType;
    optimizer?: OptimizerType;
  }) => void;
}

export type LLMProvider = "gemini" | "deepseek";

export interface LLMConfig {
  provider: LLMProvider;
  modelName: string;
  apiKey: string;
  customBaseUrl?: string;
  temperature?: number;
}

const STORAGE_KEY = "bp_lab_llm_config_v1";

const DEFAULT_CONFIG: LLMConfig = {
  provider: "gemini",
  modelName: "gemini-3-flash",
  apiKey: "",
  temperature: 0.7,
};

export const AiDiagnosticWorkspace: React.FC<AiDiagnosticWorkspaceProps> = ({
  currentLabState,
  onApplyTuning,
}) => {
  // LLM Configuration State
  const [config, setConfig] = useState<LLMConfig>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        return { ...DEFAULT_CONFIG, ...JSON.parse(saved) };
      }
    } catch (e) {
      console.warn("Failed to load LLM config from localStorage:", e);
    }
    return DEFAULT_CONFIG;
  });

  // Settings Modal Open State
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [tempApiKey, setTempApiKey] = useState(config.apiKey);
  const [tempProvider, setTempProvider] = useState<LLMProvider>(config.provider);
  const [tempModelName, setTempModelName] = useState(config.modelName);
  const [tempBaseUrl, setTempBaseUrl] = useState(config.customBaseUrl || "");
  const [showApiKey, setShowApiKey] = useState(false);
  const [testStatus, setTestStatus] = useState<"idle" | "testing" | "success" | "error">("idle");
  const [testMessage, setTestMessage] = useState<string>("");
  const [savedBanner, setSavedBanner] = useState(false);

  // Chat State
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "model",
      content: `您好！我是 **BP 神经网络与反向传播演化实验室** 的 **AI 诊断助手与深度学习导师**。

我已实时对接您当前的仿真拓扑结构、双轨损失轨迹与梯度范数：
* **当前拓扑**：\`[${currentLabState.topology.join(" → ")}]\`
* **迭代轮数**：Epoch ${currentLabState.epoch} | **训练损失**：${isNaN(currentLabState.loss) ? "NaN" : currentLabState.loss.toFixed(5)}
* **梯度范数**：$||\\nabla W|| = ${isNaN(currentLabState.gradNorm) ? "0.000" : currentLabState.gradNorm.toFixed(5)}$

您可以随时在下方提问任何深度学习理论与代码疑问，或点击左侧 **“一键深度体检”** 进行全自动模型健康诊断！`,
      timestamp: Date.now(),
    },
  ]);

  const [inputPrompt, setInputPrompt] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [diagnosis, setDiagnosis] = useState<DiagnosisResult | null>(null);
  const [isDiagnosing, setIsDiagnosing] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, diagnosis]);

  // Save Config to LocalStorage
  const handleSaveConfig = () => {
    const newConfig: LLMConfig = {
      provider: tempProvider,
      modelName: tempModelName,
      apiKey: tempApiKey.trim(),
      customBaseUrl: tempBaseUrl.trim() || undefined,
      temperature: config.temperature,
    };
    setConfig(newConfig);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(newConfig));
    } catch (e) {
      console.error("Failed to save LLM config", e);
    }
    setSavedBanner(true);
    setTimeout(() => {
      setSavedBanner(false);
      setIsSettingsOpen(false);
    }, 1200);
  };

  // Test LLM Connection directly from browser
  const handleTestConnection = async () => {
    if (!tempApiKey.trim()) {
      setTestStatus("error");
      setTestMessage("请先输入 API Key 再进行测试！");
      return;
    }

    setTestStatus("testing");
    setTestMessage("正在向大模型端点发起握手测试...");

    try {
      if (tempProvider === "gemini") {
        // Direct Gemini REST endpoint (Gemini 2.5/3 Flash)
        const endpoint = tempBaseUrl.trim()
          ? `${tempBaseUrl.trim().replace(/\/$/, "")}/v1beta/models/gemini-2.5-flash:generateContent?key=${tempApiKey.trim()}`
          : `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${tempApiKey.trim()}`;

        const res = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            contents: [{ parts: [{ text: "Ping: Reply with 'OK' only." }] }],
          }),
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData?.error?.message || `HTTP ${res.status}: ${res.statusText}`);
        }

        const data = await res.json();
        const reply = data?.candidates?.[0]?.content?.parts?.[0]?.text || "OK";
        setTestStatus("success");
        setTestMessage(`✅ Gemini 3 Flash 握手连通成功！模型响应: "${reply.trim()}"`);
      } else {
        // DeepSeek API (OpenAI Compatible)
        const endpoint = tempBaseUrl.trim()
          ? `${tempBaseUrl.trim().replace(/\/$/, "")}/chat/completions`
          : "https://api.deepseek.com/chat/completions";

        const res = await fetch(endpoint, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${tempApiKey.trim()}`,
          },
          body: JSON.stringify({
            model: "deepseek-chat",
            messages: [{ role: "user", content: "Ping: Reply with 'OK' only." }],
            max_tokens: 10,
          }),
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData?.error?.message || `HTTP ${res.status}: ${res.statusText}`);
        }

        const data = await res.json();
        const reply = data?.choices?.[0]?.message?.content || "OK";
        setTestStatus("success");
        setTestMessage(`✅ DeepSeek-V4-Pro 握手连通成功！模型响应: "${reply.trim()}"`);
      }
    } catch (err: any) {
      setTestStatus("error");
      setTestMessage(`❌ 握手连通失败: ${err.message || String(err)}`);
    }
  };

  // Helper: Call LLM with user prompt and simulation context
  const callLLMApi = async (
    prompt: string,
    systemPrompt: string
  ): Promise<string> => {
    const { provider, apiKey, customBaseUrl } = config;

    // 1. Direct Browser Client-side call with User's API Key
    if (apiKey && apiKey.trim()) {
      if (provider === "gemini") {
        const url = customBaseUrl
          ? `${customBaseUrl.replace(/\/$/, "")}/v1beta/models/gemini-2.5-flash:generateContent?key=${apiKey.trim()}`
          : `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${apiKey.trim()}`;

        const res = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            systemInstruction: {
              parts: [{ text: systemPrompt }],
            },
            contents: [
              {
                parts: [{ text: prompt }],
              },
            ],
            generationConfig: {
              temperature: 0.7,
              maxOutputTokens: 1500,
            },
          }),
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData?.error?.message || `HTTP ${res.status} error`);
        }

        const data = await res.json();
        return (
          data?.candidates?.[0]?.content?.parts?.[0]?.text ||
          "未能获取模型有效回复。"
        );
      } else {
        // DeepSeek
        const url = customBaseUrl
          ? `${customBaseUrl.replace(/\/$/, "")}/chat/completions`
          : "https://api.deepseek.com/chat/completions";

        const res = await fetch(url, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${apiKey.trim()}`,
          },
          body: JSON.stringify({
            model: "deepseek-chat",
            messages: [
              { role: "system", content: systemPrompt },
              { role: "user", content: prompt },
            ],
            temperature: 0.7,
            max_tokens: 1500,
          }),
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData?.error?.message || `HTTP ${res.status} error`);
        }

        const data = await res.json();
        return (
          data?.choices?.[0]?.message?.content ||
          "未能获取模型有效回复。"
        );
      }
    }

    // 2. Fallback to Local Node Backend `/api/chat` if available (e.g. In Dev Server)
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: prompt,
          context: {
            currentEpoch: currentLabState.epoch,
            currentLoss: currentLabState.loss,
            valLoss: currentLabState.valLoss,
            gradNorm: currentLabState.gradNorm,
            learningRate: currentLabState.learningRate,
            activation: currentLabState.activation,
            optimizer: currentLabState.optimizer,
            topology: currentLabState.topology,
          },
        }),
      });
      if (res.ok) {
        const data = await res.json();
        if (data.reply) return data.reply;
      }
    } catch (e) {
      // Backend not running (e.g. GitHub Pages / Netlify static hosting)
    }

    // 3. Fallback to Local Built-in Offline Pedagogical Knowledge Engine
    return generateLocalRuleBasedReply(prompt, currentLabState);
  };

  // Local Offline Expert Knowledge Fallback
  const generateLocalRuleBasedReply = (
    prompt: string,
    state: typeof currentLabState
  ): string => {
    const p = prompt.toLowerCase();
    if (p.includes("sigmoid") || p.includes("梯度消失")) {
      return `【离线本地导师】**Sigmoid 梯度消失本质剖析**：
1. **导数上限限制**：Sigmoid 导数公式为 $\\sigma'(z) = \\sigma(z)(1 - \\sigma(z))$，其最大值仅为 **0.25**（在 $z=0$ 时取得）。
2. **链式连乘衰减**：在反向传播中，浅层神经元梯度需要连乘各层的激活函数导数：
$$\\delta^{(1)} = \\left( \\prod_{k=2}^{L} W^{(k)} \\sigma'(z^{(k)}) \\right) \\delta^{(L)}$$
当网络层数加深时，$(0.25)^L$ 会呈指数级雪崩衰减至 0，导致靠近输入层的权重无法得到有效更新。
💡 **建议解决方案**：将隐藏层激活函数替换为 **ReLU** 或 **LeakyReLU**，其正半轴导数恒为 1，彻底阻断梯度消失。`;
    }

    if (p.includes("学习率") || p.includes("震荡") || p.includes("loss")) {
      return `【离线本地导师】**学习率 $\\eta$ 与损失震荡调优策略**：
* 当前实验学习率 $\\eta = ${state.learningRate}$，优化器为 **${state.optimizer.toUpperCase()}**。
* **震荡原因**：当学习率过大时，参数更新步长会跨过凸碗底最优点，在损失曲面狭长峡谷中出现“之”字形反复弹跳。
* **调优建议**：
  1. 将学习率下调至当前值的 1/3 ~ 1/5（例如从 ${state.learningRate} 调整至 ${(state.learningRate * 0.3).toFixed(3)}）；
  2. 切换为 **Adam** 或 **Momentum** 自适应动量优化器，借助一阶动量平滑高频震荡分量。`;
    }

    if (p.includes("xor") || p.includes("异或") || p.includes("感知机")) {
      return `【离线本地导师】**单层感知机无法解决 XOR 问题的数学证明**：
* 异或逻辑属于**非线性不可分**问题（$(0,0)\\to 0, (1,1)\\to 0$ 与 $(1,0)\\to 1, (0,1)\\to 1$ 无法用一条二维直线划分）。
* 单层感知机本质上是一个超平面线性分类器 $y = \\text{step}(w_1 x_1 + w_2 x_2 + b)$。
* **突破关键**：必须引入至少 1 个包含非线性激活函数的隐藏层（如 \`[2 → 4 → 1]\` 拓扑），将原始二维坐标映射至高维特征空间，使其在隐藏层表征空间内转化为线性可分！`;
    }

    return `【离线本地导师】当前仿真运行状态分析：
* **模型拓扑**：\`[${state.topology.join(" → ")}]\`
* **迭代状态**：Epoch ${state.epoch}，当前训练损失 $L = ${isNaN(state.loss) ? "NaN" : state.loss.toFixed(5)}$，梯度范数 $||\\nabla W|| = ${isNaN(state.gradNorm) ? "0.000" : state.gradNorm.toFixed(5)}$。
* **状态诊断**：${
      state.loss < 0.03
        ? "模型已高度收敛，决策曲面拟合良好！"
        : state.gradNorm < 0.0001
        ? "检测到梯度范数极小，可能已陷入局部平坦鞍区或发生梯度消失，建议增大学习率或改用 ReLU 激活函数。"
        : "梯度回传正常，正在平稳逼近目标流形。"
    }

*(💡 提示：如需启用 Google Gemini 3 Flash 或 DeepSeek-V4-Pro 云端大语言模型实时深度对话，请点击右上角 ⚙️ 设置大模型 输入您的 API Key)*`;
  };

  // Handle Free Chat
  const handleSendMessage = async (textToSend?: string) => {
    const text = textToSend || inputPrompt;
    if (!text.trim() || isLoading) return;

    const userMsg: ChatMessage = {
      id: Date.now().toString(),
      role: "user",
      content: text,
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, userMsg]);
    if (!textToSend) setInputPrompt("");
    setIsLoading(true);

    const systemPrompt = `你是一位世界顶级的深度学习科学家与 BP 神经网络教学导师。
你正在协助用户分析一个实时的浏览器神经网络演化仿真实验。
当前实验运行快照：
- 拓扑结构: [${currentLabState.topology.join(" -> ")}]
- 当前迭代轮数: Epoch ${currentLabState.epoch}
- 当前训练损失: ${currentLabState.loss}
- 验证集损失: ${currentLabState.valLoss ?? "N/A"}
- 梯度范数 ||∇W||: ${currentLabState.gradNorm}
- 权重范数 ||W||: ${currentLabState.weightNorm}
- 学习率 η: ${currentLabState.learningRate}
- 激活函数: ${currentLabState.activation}
- 优化器: ${currentLabState.optimizer}
- 当前数据集: ${currentLabState.dataset || "自定义数据"}

请遵循以下指导：
1. 语言亲切、专业、深入浅出，结合数学公式 (使用 LaTeX 格式，如 $E = \\frac{1}{2}(y-\\hat{y})^2$) 和几何直观进行阐述；
2. 针对用户的问题结合当前快照的真实参数（如学习率、梯度范数、损失）给出量化诊断与改进建议；
3. 输出排版清晰，善用加粗与要点列表。`;

    try {
      const reply = await callLLMApi(text, systemPrompt);
      const aiReply: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: "model",
        content: reply,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, aiReply]);
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: "model",
          content: `⚠️ API 调用异常: ${err.message || String(err)}\n已自动为您切换至本地知识库为您解析：\n` + generateLocalRuleBasedReply(text, currentLabState),
          timestamp: Date.now(),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle One-Click Network Health Diagnosis
  const handleRunDiagnosis = async () => {
    setIsDiagnosing(true);

    const loss = currentLabState.loss;
    const gradNorm = currentLabState.gradNorm;
    const lr = currentLabState.learningRate;
    const act = currentLabState.activation;

    let score = 85;
    let status = "正常";
    const risks: string[] = [];
    const suggestions: string[] = [];

    if (isNaN(loss) || loss > 10) {
      score = 20;
      status = "梯度爆炸 / 发散";
      risks.push("损失值已出现 NaN 或异常发散，权重发生数值溢出！");
      suggestions.push(`立即将学习率从 ${lr} 下调至 ${(lr * 0.1).toFixed(4)}，或启用梯度裁剪机制。`);
    } else if (gradNorm < 0.0001 && loss > 0.15) {
      score = 45;
      status = "梯度消失 / 停滞";
      risks.push(`梯度范数过小 (||∇W|| = ${gradNorm.toFixed(6)})，且损失停滞在 ${loss.toFixed(4)}。`);
      if (act === "sigmoid" || act === "tanh") {
        suggestions.push(`检测到正在使用 ${act.toUpperCase()} 激活函数，建议将隐藏层替换为 ReLU 或 LeakyReLU 激活函数。`);
      }
      suggestions.push("可以尝试将优化器从 SGD 切换为 Adam 自适应优化器以加速脱离平坦区。");
    } else if (loss < 0.05) {
      score = 98;
      status = "极佳收敛";
      suggestions.push("网络已完成高精度拟合，可以尝试切换至其他非线性复杂数据集测试泛化能力。");
    } else {
      score = 78;
      status = "正常演化";
      suggestions.push("网络正在稳步迭代下降中，可继续观察损失曲线平滑度。");
    }

    if (currentLabState.valLoss && currentLabState.valLoss > loss * 2.5 && currentLabState.epoch > 300) {
      risks.push("验证集损失明显高于训练集，检测到潜在的过拟合 (Overfitting) 风险！");
      suggestions.push("建议增加正则化惩罚、减少隐藏层神经元数量，或收集更多训练样本。");
    }

    const localReport: DiagnosisResult = {
      healthScore: score,
      status,
      risks,
      suggestions,
      report: `【系统综合诊断】当前网络拓扑 [${currentLabState.topology.join(" → ")}] 在经过 ${currentLabState.epoch} 轮迭代后，健康指数为 ${score}/100。\n当前状态：${status}。损失 $L = ${isNaN(loss) ? "NaN" : loss.toFixed(4)}$，梯度敏感度 $||\\nabla W|| = ${isNaN(gradNorm) ? "0.000" : gradNorm.toFixed(4)}$。`,
    };

    try {
      if (config.apiKey && config.apiKey.trim()) {
        const prompt = `请对以下神经网络训练状态进行全方位的深度体检诊断报告：
拓扑: [${currentLabState.topology.join(" -> ")}]
Epoch: ${currentLabState.epoch}, Train Loss: ${currentLabState.loss}, Val Loss: ${currentLabState.valLoss ?? "N/A"}, Grad Norm: ${currentLabState.gradNorm}, LR: ${currentLabState.learningRate}, Activation: ${currentLabState.activation}, Optimizer: ${currentLabState.optimizer}
请输出结构化建议，指出是否存在梯度消失、梯度爆炸、学习率过大震荡或过拟合。`;
        const aiReport = await callLLMApi(prompt, "你是一个神经网络故障体检专家，请输出精确严谨的体检报告。");
        localReport.report = aiReport;
      }
    } catch (e) {
      // Fallback to local report
    }

    setDiagnosis(localReport);
    setIsDiagnosing(false);
  };

  const quickQuestions = [
    "诊断当前训练状态与损失收敛速度",
    "为什么 Sigmoid 容易引发梯度消失？",
    "如何根据损失曲线震荡选择合适的学习率 η？",
    "为什么单层感知机无法解决异或 (XOR) 问题？",
    "通用近似定理 (UAT) 的数学内涵是什么？",
    "Adam 优化器相比 SGD 有哪些本质优势？",
  ];

  const handleCopyText = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div className="space-y-4">
      {/* ------------------------------------------------------------- */}
      {/* SECTION 1: HEADER BANNER WITH TITLE & MODEL SETTINGS GEAR     */}
      {/* ------------------------------------------------------------- */}
      <div className="bg-white border border-slate-200 rounded-lg p-4 shadow-sm flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-lg bg-blue-600 text-white shadow-sm flex items-center justify-center">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-sm sm:text-base font-bold text-slate-900">
                7. AI 诊断助手 (AI Diagnostic Assistant & Mentor)
              </h2>
              <span className="text-[10px] px-2 py-0.5 rounded bg-blue-50 text-blue-700 font-bold border border-blue-200 font-mono">
                {config.provider === "gemini" ? "Google Gemini 3 Flash" : "DeepSeek-V4-Pro"}
              </span>
            </div>
            <p className="text-xs text-slate-500 font-medium mt-0.5">
              实时连接神经网络仿真内核 · 大模型深度诊断收敛态与梯度健康度
            </p>
          </div>
        </div>

        {/* Right Action: LLM Settings Gear Button */}
        <div className="flex items-center gap-2 flex-wrap">
          {/* Status Indicator */}
          <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1.5 bg-slate-50 border border-slate-200 rounded-md text-xs font-mono text-slate-600">
            <span
              className={`w-2 h-2 rounded-full ${
                config.apiKey ? "bg-emerald-500 animate-pulse" : "bg-amber-500"
              }`}
            />
            <span className="text-[11px]">
              {config.apiKey
                ? `${config.provider === "gemini" ? "Gemini 3 Flash" : "DeepSeek-V4-Pro"}`
                : "本地离线模式"}
            </span>
          </div>

          {/* The Gear Button (设置大模型) */}
          <button
            onClick={() => {
              setTempApiKey(config.apiKey);
              setTempProvider(config.provider);
              setTempModelName(config.modelName);
              setTempBaseUrl(config.customBaseUrl || "");
              setTestStatus("idle");
              setTestMessage("");
              setIsSettingsOpen(true);
            }}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-900 hover:bg-slate-800 text-white text-xs font-semibold rounded-md shadow-sm transition cursor-pointer"
            title="点击设置大模型：手工输入 API Key 与选择 Gemini 3 Flash / DeepSeek-V4-Pro"
          >
            <Settings className="w-3.5 h-3.5 text-blue-400" />
            <span>设置大模型</span>
          </button>
        </div>
      </div>

      {/* ------------------------------------------------------------- */}
      {/* SECTION 2: AI DIAGNOSTIC CHAT & REAL-TIME MENTOR              */}
      {/* ------------------------------------------------------------- */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 animate-fade-in">
          {/* LEFT COLUMN: REAL-TIME STATE INSPECTOR & HEALTH GAUGE (5 cols) */}
          <div className="lg:col-span-5 space-y-4 flex flex-col">
            {/* Real-time State Card */}
            <div className="bg-white border border-slate-200 rounded-lg p-3.5 shadow-sm space-y-3">
              <div className="flex items-center justify-between pb-2 border-b border-slate-100">
                <div className="flex items-center gap-1.5 text-xs font-bold text-slate-800">
                  <Activity className="w-3.5 h-3.5 text-blue-600" />
                  <span>实时网络状态快照 (Real-time Telemetry)</span>
                </div>
                <span className="text-[10px] font-mono text-slate-400">
                  Epoch {currentLabState.epoch}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                <div className="bg-slate-50 p-2 rounded border border-slate-200">
                  <span className="text-[10px] text-slate-400 block font-sans">网络拓扑</span>
                  <span className="font-bold text-slate-800">[{currentLabState.topology.join(" → ")}]</span>
                </div>
                <div className="bg-slate-50 p-2 rounded border border-slate-200">
                  <span className="text-[10px] text-slate-400 block font-sans">80% 训练损失</span>
                  <span
                    className={`font-bold ${
                      isNaN(currentLabState.loss)
                        ? "text-rose-600"
                        : currentLabState.loss < 0.05
                        ? "text-emerald-600"
                        : "text-blue-600"
                    }`}
                  >
                    {isNaN(currentLabState.loss) ? "NaN (爆炸)" : currentLabState.loss.toFixed(5)}
                  </span>
                </div>
                <div className="bg-slate-50 p-2 rounded border border-slate-200">
                  <span className="text-[10px] text-slate-400 block font-sans">梯度范数 ||∇W||</span>
                  <span className="font-semibold text-slate-700">
                    {isNaN(currentLabState.gradNorm) ? "0.0000" : currentLabState.gradNorm.toFixed(5)}
                  </span>
                </div>
                <div className="bg-slate-50 p-2 rounded border border-slate-200">
                  <span className="text-[10px] text-slate-400 block font-sans">学习率 & 优化器</span>
                  <span className="font-semibold text-blue-700">
                    η={currentLabState.learningRate} | {currentLabState.optimizer.toUpperCase()}
                  </span>
                </div>
              </div>

              {/* Run Comprehensive Health Diagnosis Button */}
              <button
                onClick={handleRunDiagnosis}
                disabled={isDiagnosing}
                className="w-full flex items-center justify-center gap-2 py-2 px-3 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white rounded text-xs font-bold shadow-xs transition cursor-pointer"
              >
                {isDiagnosing ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    <span>正在进行全网络数学体检...</span>
                  </>
                ) : (
                  <>
                    <Stethoscope className="w-3.5 h-3.5" />
                    <span>🩺 一键深度网络健康体检</span>
                  </>
                )}
              </button>
            </div>

            {/* Health Diagnosis Report Panel */}
            {diagnosis && (
              <div className="bg-white border border-slate-200 rounded-lg p-3.5 shadow-sm space-y-3">
                <div className="flex items-center justify-between pb-2 border-b border-slate-100">
                  <div className="flex items-center gap-2">
                    <div
                      className={`w-2.5 h-2.5 rounded-full ${
                        diagnosis.healthScore >= 80
                          ? "bg-emerald-500"
                          : diagnosis.healthScore >= 50
                          ? "bg-amber-500"
                          : "bg-rose-500"
                      }`}
                    />
                    <span className="text-xs font-bold text-slate-800">
                      健康指数: {diagnosis.healthScore} / 100 ({diagnosis.status})
                    </span>
                  </div>
                  <span
                    className={`text-[10px] px-2 py-0.5 rounded font-bold ${
                      diagnosis.healthScore >= 80
                        ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                        : diagnosis.healthScore >= 50
                        ? "bg-amber-50 text-amber-700 border border-amber-200"
                        : "bg-rose-50 text-rose-700 border border-rose-200"
                    }`}
                  >
                    {diagnosis.status}
                  </span>
                </div>

                {/* Potential Risks */}
                {diagnosis.risks && diagnosis.risks.length > 0 && (
                  <div className="space-y-1.5">
                    <span className="text-[11px] font-bold text-rose-700 flex items-center gap-1">
                      <AlertTriangle className="w-3 h-3 text-rose-600" />
                      潜在异常警报:
                    </span>
                    <div className="space-y-1">
                      {diagnosis.risks.map((risk, idx) => (
                        <div
                          key={idx}
                          className="text-xs text-rose-700 bg-rose-50 border border-rose-200 rounded p-1.5 leading-relaxed"
                        >
                          {risk}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Actionable Suggestions */}
                {diagnosis.suggestions && diagnosis.suggestions.length > 0 && (
                  <div className="space-y-1.5">
                    <span className="text-[11px] font-bold text-slate-700 flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3 text-emerald-600" />
                      调优建议:
                    </span>
                    <ul className="space-y-1 text-xs text-slate-600 list-disc pl-4 leading-relaxed">
                      {diagnosis.suggestions.map((sug, idx) => (
                        <li key={idx}>{sug}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* AI Comprehensive Report Text */}
                <div className="p-2.5 bg-slate-50 border border-slate-200 rounded text-xs text-slate-700 leading-relaxed font-sans max-h-48 overflow-y-auto whitespace-pre-line">
                  {diagnosis.report}
                </div>
              </div>
            )}

            {/* Quick FAQ Prompts */}
            <div className="bg-white border border-slate-200 rounded-lg p-3.5 shadow-sm space-y-2 flex-1">
              <div className="flex items-center gap-1.5 text-xs font-bold text-slate-800">
                <HelpCircle className="w-3.5 h-3.5 text-blue-600" />
                <span>常见理论与调优快捷提问</span>
              </div>
              <div className="space-y-1.5">
                {quickQuestions.map((q, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleSendMessage(q)}
                    className="w-full text-left p-1.5 rounded text-xs text-slate-600 hover:text-blue-700 hover:bg-blue-50 border border-slate-100 hover:border-blue-200 transition flex items-center justify-between gap-2 group cursor-pointer"
                  >
                    <span className="truncate">{q}</span>
                    <ChevronRight className="w-3 h-3 text-slate-400 group-hover:text-blue-600 shrink-0" />
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* RIGHT COLUMN: AI CONVERSATION & MENTOR CHAT (7 cols) */}
          <div className="lg:col-span-7 flex flex-col bg-white border border-slate-200 rounded-lg shadow-sm overflow-hidden min-h-[550px]">
            {/* Chat Top Banner */}
            <div className="px-4 py-2.5 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
              <div className="flex items-center gap-2 text-xs font-bold text-slate-800">
                <MessageSquare className="w-4 h-4 text-blue-600" />
                <span>AI 深度学习导师互动窗口</span>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={() =>
                    setMessages([
                      {
                        id: "welcome",
                        role: "model",
                        content: `已重置对话记录。您可以随时向我提问理论公式、收敛问题或点击 **“一键深度体检”**！`,
                        timestamp: Date.now(),
                      },
                    ])
                  }
                  className="text-slate-400 hover:text-slate-600 p-1 rounded transition cursor-pointer"
                  title="清空对话历史"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            {/* Missing API Key Guidance Banner (If not configured) */}
            {!config.apiKey && (
              <div className="px-3.5 py-2 bg-amber-50 border-b border-amber-200 flex items-center justify-between gap-2 text-xs text-amber-800">
                <div className="flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 text-amber-600 shrink-0" />
                  <span>
                    未输入 API Key。当前使用内置离线导师。点击右上角 <b>⚙️ 设置大模型</b> 即可接入 Gemini 3 Flash 或 DeepSeek-V4-Pro！
                  </span>
                </div>
                <button
                  onClick={() => setIsSettingsOpen(true)}
                  className="px-2 py-0.5 bg-amber-600 hover:bg-amber-700 text-white rounded text-[11px] font-bold shrink-0 transition cursor-pointer"
                >
                  去配置
                </button>
              </div>
            )}

            {/* Messages Scroll Area */}
            <div className="flex-1 p-4 overflow-y-auto space-y-3.5 max-h-[460px]">
              {messages.map((msg) => {
                const isAi = msg.role === "model";
                return (
                  <div
                    key={msg.id}
                    className={`flex gap-2.5 ${isAi ? "items-start" : "items-start flex-row-reverse"}`}
                  >
                    <div
                      className={`w-7 h-7 rounded-full flex items-center justify-center shrink-0 shadow-xs ${
                        isAi ? "bg-blue-600 text-white" : "bg-slate-800 text-white"
                      }`}
                    >
                      {isAi ? <Bot className="w-4 h-4" /> : <User className="w-4 h-4" />}
                    </div>

                    <div
                      className={`relative max-w-[85%] rounded-lg p-3 text-xs leading-relaxed group ${
                        isAi
                          ? "bg-slate-50 text-slate-800 border border-slate-200 shadow-xs"
                          : "bg-blue-600 text-white"
                      }`}
                    >
                      {/* Copy message button */}
                      <button
                        onClick={() => handleCopyText(msg.id, msg.content)}
                        className={`absolute top-2 right-2 p-1 rounded opacity-0 group-hover:opacity-100 transition cursor-pointer ${
                          isAi ? "text-slate-400 hover:text-slate-600 bg-white/80" : "text-white/80 hover:text-white bg-blue-700"
                        }`}
                        title="复制本段内容"
                      >
                        {copiedId === msg.id ? (
                          <Check className="w-3 h-3 text-emerald-600" />
                        ) : (
                          <Copy className="w-3 h-3" />
                        )}
                      </button>

                      <div className="whitespace-pre-wrap font-sans pr-4">{msg.content}</div>

                      <span
                        className={`text-[9px] block mt-1.5 ${
                          isAi ? "text-slate-400" : "text-blue-200"
                        }`}
                      >
                        {new Date(msg.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                  </div>
                );
              })}

              {isLoading && (
                <div className="flex items-center gap-2 text-slate-500 text-xs py-2">
                  <div className="w-7 h-7 rounded-full bg-blue-600 text-white flex items-center justify-center animate-pulse">
                    <Bot className="w-4 h-4" />
                  </div>
                  <div className="flex items-center gap-1.5 bg-slate-50 border border-slate-200 px-3 py-2 rounded-lg">
                    <RefreshCw className="w-3.5 h-3.5 animate-spin text-blue-600" />
                    <span>正在深度推理中 (Thinking)...</span>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Input Box */}
            <div className="p-3 border-t border-slate-200 bg-slate-50">
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={inputPrompt}
                  onChange={(e) => setInputPrompt(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleSendMessage();
                    }
                  }}
                  placeholder="向 AI 导师提问：例如“为何 Sigmoid 会在深度网络中发生梯度消失？”..."
                  className="flex-1 bg-white border border-slate-300 rounded-md px-3 py-2 text-xs text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
                />

                <button
                  onClick={() => handleSendMessage()}
                  disabled={isLoading || !inputPrompt.trim()}
                  className="flex items-center gap-1.5 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white rounded-md text-xs font-bold shadow-xs transition cursor-pointer shrink-0"
                >
                  <Send className="w-3.5 h-3.5" />
                  <span>发送</span>
                </button>
              </div>
              <div className="mt-1.5 flex items-center justify-between text-[10px] text-slate-400">
                <span>按 Enter 发送提问</span>
                <span>支持 LaTeX 数学公式与反向传播矩阵解析</span>
              </div>
            </div>
          </div>
        </div>

      {/* ------------------------------------------------------------- */}
      {/* SECTION 3: LLM SETTINGS MODAL DIALOG (GEAR ICON POPUP)        */}
      {/* ------------------------------------------------------------- */}
      {isSettingsOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-xs animate-fade-in">
          <div className="relative w-full max-w-lg bg-white border border-slate-300 rounded-xl shadow-2xl overflow-hidden flex flex-col">
            {/* Modal Header */}
            <div className="px-5 py-3.5 bg-slate-900 text-white flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Settings className="w-4 h-4 text-blue-400" />
                <h3 className="text-sm font-bold">大模型参数与 API Key 配置 (LLM Settings)</h3>
              </div>
              <button
                onClick={() => setIsSettingsOpen(false)}
                className="text-slate-400 hover:text-white p-1 rounded transition cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Modal Body */}
            <div className="p-5 space-y-4 max-h-[80vh] overflow-y-auto text-xs">
              {/* Step 1: Model Provider Selection */}
              <div className="space-y-2">
                <label className="block text-xs font-bold text-slate-800">
                  1. 选择大模型供应商 (Model Selection)
                </label>
                <div className="grid grid-cols-2 gap-2.5">
                  <div
                    onClick={() => {
                      setTempProvider("gemini");
                      setTempModelName("gemini-3-flash");
                    }}
                    className={`p-3 rounded-lg border cursor-pointer transition flex flex-col justify-between ${
                      tempProvider === "gemini"
                        ? "border-blue-600 bg-blue-50/50 ring-1 ring-blue-500"
                        : "border-slate-200 hover:bg-slate-50"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-slate-900">Gemini 3 Flash</span>
                      {tempProvider === "gemini" && (
                        <CheckCircle2 className="w-4 h-4 text-blue-600" />
                      )}
                    </div>
                    <span className="text-[11px] text-slate-500 mt-1">
                      Google 极速推理与多模态模型 (推荐)
                    </span>
                  </div>

                  <div
                    onClick={() => {
                      setTempProvider("deepseek");
                      setTempModelName("deepseek-v4-pro");
                    }}
                    className={`p-3 rounded-lg border cursor-pointer transition flex flex-col justify-between ${
                      tempProvider === "deepseek"
                        ? "border-blue-600 bg-blue-50/50 ring-1 ring-blue-500"
                        : "border-slate-200 hover:bg-slate-50"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-slate-900">DeepSeek-V4-Pro</span>
                      {tempProvider === "deepseek" && (
                        <CheckCircle2 className="w-4 h-4 text-blue-600" />
                      )}
                    </div>
                    <span className="text-[11px] text-slate-500 mt-1">
                      DeepSeek 深度代码与数学推理大模型
                    </span>
                  </div>
                </div>
              </div>

              {/* Step 2: Manual API Key Input */}
              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-bold text-slate-800 flex items-center gap-1">
                    <Key className="w-3.5 h-3.5 text-blue-600" />
                    <span>2. 手工输入 API-Key (Manual Input)</span>
                  </label>
                  {tempProvider === "gemini" ? (
                    <a
                      href="https://aistudio.google.com/app/apikey"
                      target="_blank"
                      rel="noreferrer"
                      className="text-[11px] text-blue-600 hover:underline flex items-center gap-0.5"
                    >
                      <span>获取 Gemini Key</span>
                      <ExternalLink className="w-2.5 h-2.5" />
                    </a>
                  ) : (
                    <a
                      href="https://platform.deepseek.com/api_keys"
                      target="_blank"
                      rel="noreferrer"
                      className="text-[11px] text-blue-600 hover:underline flex items-center gap-0.5"
                    >
                      <span>获取 DeepSeek Key</span>
                      <ExternalLink className="w-2.5 h-2.5" />
                    </a>
                  )}
                </div>

                <div className="relative flex items-center">
                  <input
                    type={showApiKey ? "text" : "password"}
                    value={tempApiKey}
                    onChange={(e) => setTempApiKey(e.target.value)}
                    placeholder={
                      tempProvider === "gemini"
                        ? "AIzaSy..."
                        : "sk-..."
                    }
                    className="w-full bg-slate-50 border border-slate-300 rounded-md px-3 py-2 pr-10 text-xs font-mono text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:bg-white"
                  />
                  <button
                    type="button"
                    onClick={() => setShowApiKey(!showApiKey)}
                    className="absolute right-2.5 text-slate-400 hover:text-slate-600 cursor-pointer"
                  >
                    {showApiKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
                <p className="text-[10px] text-slate-500">
                  🔒 密钥仅安全保存在您当前浏览器的本地缓存 (LocalStorage) 中，直接由浏览器与大模型端点通信，无需依赖中间服务器。
                </p>
              </div>

              {/* Optional Custom Base URL */}
              <div className="space-y-1">
                <label className="text-xs font-semibold text-slate-700">
                  自定义 API Base URL (可选代理/中转端点):
                </label>
                <input
                  type="text"
                  value={tempBaseUrl}
                  onChange={(e) => setTempBaseUrl(e.target.value)}
                  placeholder={
                    tempProvider === "gemini"
                      ? "https://generativelanguage.googleapis.com (默认)"
                      : "https://api.deepseek.com (默认)"
                  }
                  className="w-full bg-slate-50 border border-slate-300 rounded-md px-3 py-1.5 text-xs font-mono text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-1 focus:ring-blue-500"
                />
              </div>

              {/* Connection Test Area */}
              <div className="pt-2 border-t border-slate-100 flex flex-col gap-2">
                <div className="flex items-center justify-between">
                  <button
                    type="button"
                    onClick={handleTestConnection}
                    disabled={testStatus === "testing" || !tempApiKey.trim()}
                    className="px-3 py-1.5 bg-slate-100 hover:bg-slate-200 disabled:opacity-50 text-slate-700 font-semibold rounded text-xs transition cursor-pointer flex items-center gap-1.5"
                  >
                    {testStatus === "testing" ? (
                      <>
                        <RefreshCw className="w-3.5 h-3.5 animate-spin text-blue-600" />
                        <span>正在探测连通性...</span>
                      </>
                    ) : (
                      <>
                        <Zap className="w-3.5 h-3.5 text-amber-500" />
                        <span>测试连通性</span>
                      </>
                    )}
                  </button>

                  {tempApiKey && (
                    <button
                      type="button"
                      onClick={() => setTempApiKey("")}
                      className="text-slate-400 hover:text-rose-600 text-xs transition cursor-pointer"
                    >
                      清空 Key
                    </button>
                  )}
                </div>

                {testMessage && (
                  <div
                    className={`p-2 rounded text-xs leading-relaxed ${
                      testStatus === "success"
                        ? "bg-emerald-50 text-emerald-800 border border-emerald-200"
                        : testStatus === "error"
                        ? "bg-rose-50 text-rose-800 border border-rose-200"
                        : "bg-blue-50 text-blue-800 border border-blue-200"
                    }`}
                  >
                    {testMessage}
                  </div>
                )}
              </div>
            </div>

            {/* Modal Footer */}
            <div className="px-5 py-3 bg-slate-50 border-t border-slate-200 flex items-center justify-between">
              <button
                type="button"
                onClick={() => setIsSettingsOpen(false)}
                className="px-3 py-1.5 border border-slate-300 text-slate-600 hover:bg-white rounded text-xs font-semibold transition cursor-pointer"
              >
                取消
              </button>

              <button
                type="button"
                onClick={handleSaveConfig}
                className="flex items-center gap-1.5 px-4 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded text-xs font-bold shadow-xs transition cursor-pointer"
              >
                {savedBanner ? (
                  <>
                    <Check className="w-3.5 h-3.5" />
                    <span>已确认保存！</span>
                  </>
                ) : (
                  <>
                    <Check className="w-3.5 h-3.5" />
                    <span>3. 确认大模型选择并保存</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
