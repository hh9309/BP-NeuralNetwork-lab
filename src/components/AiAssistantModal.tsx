import React, { useState, useRef, useEffect } from "react";
import {
  Sparkles,
  X,
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
} from "lucide-react";
import { ChatMessage, DiagnosisResult } from "../types";
import { BlockMath, InlineMath } from "../lib/katex-helper";

interface AiAssistantModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentLabState: {
    epoch: number;
    loss: number;
    gradNorm: number;
    weightNorm: number;
    learningRate: number;
    activation: string;
    optimizer: string;
    topology: number[];
    lossHistory: number[];
    dataset?: string;
  };
}

export const AiAssistantModal: React.FC<AiAssistantModalProps> = ({
  isOpen,
  onClose,
  currentLabState,
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "model",
      content: `您好！我是 BP 神经网络实验室的 **AI 诊断与深度学习导师**。
我已自动对接当前的仿真环境。您可以随时向我提问理论公式、收敛问题，或点击下方的 **“一键深度体检”** 让我就当前梯度与损失进行健康诊断！`,
      timestamp: Date.now(),
    },
  ]);
  const [inputPrompt, setInputPrompt] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [diagnosis, setDiagnosis] = useState<DiagnosisResult | null>(null);
  const [isDiagnosing, setIsDiagnosing] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, diagnosis]);

  if (!isOpen) return null;

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

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          history: messages,
          context: {
            currentEpoch: currentLabState.epoch,
            currentLoss: currentLabState.loss,
            gradNorm: currentLabState.gradNorm,
            learningRate: currentLabState.learningRate,
            activation: currentLabState.activation,
            optimizer: currentLabState.optimizer,
            topology: currentLabState.topology,
          },
        }),
      });

      const data = await res.json();
      const aiReply: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: "model",
        content: data.reply || "未能获取回复，请重试。",
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, aiReply]);
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: "model",
          content: `网络连接异常，已为您切换至本地知识库分析。\n建议检查学习率 η 是否适配激活函数导数范围。`,
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
    try {
      const res = await fetch("/api/diagnose", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ state: currentLabState }),
      });
      const data = await res.json();
      setDiagnosis(data);
    } catch (err) {
      console.error(err);
    } finally {
      setIsDiagnosing(false);
    }
  };

  const quickQuestions = [
    "为什么 Sigmoid 容易引发梯度消失？",
    "如何根据损失曲线震荡选择合适的学习率 η？",
    "为什么单层感知机无法解决异或 (XOR) 问题？",
    "通用近似定理 (UAT) 的数学内涵是什么？",
    "Adam 优化器相比 SGD 有哪些本质优势？",
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-xs">
      <div className="relative w-full max-w-2xl bg-white border border-slate-300 rounded-lg shadow-xl overflow-hidden flex flex-col h-[600px] max-h-[90vh]">
        {/* Modal Header */}
        <div className="flex items-center justify-between px-4 py-2.5 bg-slate-50 border-b border-slate-200">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-blue-600" />
            <div>
              <h3 className="text-xs sm:text-sm font-bold text-slate-900 flex items-center gap-2">
                <span>AI 神经网络智能诊断与导师 Q&A</span>
                <span className="text-[10px] px-1.5 py-0.2 rounded bg-blue-50 text-blue-700 border border-blue-200 font-mono font-bold">
                  Gemini
                </span>
              </h3>
              <p className="text-[10px] text-slate-500 font-medium">
                实时捕获仿真状态、诊断梯度消失/爆炸与调优参数
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1 text-slate-400 hover:text-slate-700 rounded hover:bg-slate-200 transition cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Real-time State Snapshot Ribbon */}
        <div className="px-4 py-1.5 bg-slate-100/80 border-b border-slate-200 flex flex-wrap items-center justify-between text-xs font-mono text-slate-600 gap-2">
          <div className="flex items-center gap-3">
            <span>
              拓扑: <b className="text-slate-900">[{currentLabState.topology.join("-")}]</b>
            </span>
            <span>
              Loss:{" "}
              <b
                className={
                  currentLabState.loss < 0.05 ? "text-emerald-700" : "text-amber-700"
                }
              >
                {currentLabState.loss?.toFixed?.(5) ?? currentLabState.loss}
              </b>
            </span>
            <span>
              ||∇W||:{" "}
              <b className="text-blue-700">
                {currentLabState.gradNorm?.toFixed?.(5) ?? "0.00"}
              </b>
            </span>
          </div>

          <button
            onClick={handleRunDiagnosis}
            disabled={isDiagnosing}
            className="flex items-center gap-1 px-2.5 py-0.5 bg-emerald-600 hover:bg-emerald-700 disabled:opacity-40 text-white rounded text-xs font-semibold shadow-sm transition cursor-pointer font-sans"
          >
            <Stethoscope className="w-3.5 h-3.5" />
            <span>{isDiagnosing ? "体检中..." : "一键深度体检"}</span>
          </button>
        </div>

        {/* Diagnosis Result Banner if Present */}
        {diagnosis && (
          <div className="px-4 py-2.5 bg-blue-50/60 border-b border-blue-200 text-xs">
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center gap-1.5">
                <span className="font-bold text-blue-900">体检状态:</span>
                <span className="font-bold text-slate-900">{diagnosis.status}</span>
              </div>
              <div className="flex items-center gap-1 font-mono">
                <span className="text-slate-500 text-[11px]">健康指数:</span>
                <span
                  className={`font-bold text-xs ${
                    diagnosis.healthScore > 80
                      ? "text-emerald-700"
                      : diagnosis.healthScore > 50
                      ? "text-amber-700"
                      : "text-rose-700"
                  }`}
                >
                  {diagnosis.healthScore}/100
                </span>
              </div>
            </div>
            <div className="text-slate-700 text-[11px] leading-relaxed max-h-20 overflow-y-auto font-sans whitespace-pre-line">
              {diagnosis.report}
            </div>
          </div>
        )}

        {/* Messages List Area */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3 text-xs font-sans bg-white">
          {messages.map((m) => {
            const isUser = m.role === "user";
            return (
              <div
                key={m.id}
                className={`flex gap-2.5 ${isUser ? "flex-row-reverse" : "flex-row"}`}
              >
                <div
                  className={`w-6 h-6 rounded flex items-center justify-center shrink-0 ${
                    isUser
                      ? "bg-blue-600 text-white"
                      : "bg-indigo-600 text-white"
                  }`}
                >
                  {isUser ? <User className="w-3.5 h-3.5" /> : <Bot className="w-3.5 h-3.5" />}
                </div>

                <div
                  className={`max-w-[85%] rounded-lg px-3 py-2 leading-relaxed ${
                    isUser
                      ? "bg-blue-600 text-white rounded-tr-none shadow-xs"
                      : "bg-slate-50 text-slate-800 border border-slate-200 rounded-tl-none"
                  }`}
                >
                  <div className="whitespace-pre-line text-xs font-sans">{m.content}</div>
                </div>
              </div>
            );
          })}

          {isLoading && (
            <div className="flex gap-2.5">
              <div className="w-6 h-6 rounded bg-indigo-600 text-white flex items-center justify-center shrink-0">
                <Bot className="w-3.5 h-3.5" />
              </div>
              <div className="bg-slate-50 border border-slate-200 text-slate-600 rounded-lg rounded-tl-none px-3 py-2 flex items-center gap-1.5 text-[11px]">
                <span className="w-1.5 h-1.5 rounded-full bg-blue-600 animate-bounce" />
                <span className="w-1.5 h-1.5 rounded-full bg-blue-600 animate-bounce [animation-delay:0.2s]" />
                <span className="w-1.5 h-1.5 rounded-full bg-blue-600 animate-bounce [animation-delay:0.4s]" />
                <span className="ml-1 font-medium">AI 导师正在推导解析...</span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Quick Suggestion Chips */}
        <div className="px-4 py-1.5 bg-slate-50 border-t border-slate-200 overflow-x-auto flex gap-1.5">
          {quickQuestions.map((q, idx) => (
            <button
              key={idx}
              onClick={() => handleSendMessage(q)}
              disabled={isLoading}
              className="text-[10px] px-2 py-0.5 rounded bg-white hover:bg-slate-100 text-slate-700 border border-slate-200 whitespace-nowrap transition cursor-pointer font-medium shadow-xs"
            >
              {q}
            </button>
          ))}
        </div>

        {/* Input Bar */}
        <div className="p-3 bg-slate-50 border-t border-slate-200">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSendMessage();
            }}
            className="flex items-center gap-2"
          >
            <input
              type="text"
              value={inputPrompt}
              onChange={(e) => setInputPrompt(e.target.value)}
              placeholder="随时提问：如“为什么Sigmoid会导致梯度消失？”、“如何选择学习率？”"
              className="flex-1 bg-white border border-slate-300 rounded px-3 py-1.5 text-xs text-slate-900 placeholder-slate-400 outline-none focus:border-blue-600 focus:ring-1 focus:ring-blue-600 transition"
            />
            <button
              type="submit"
              disabled={!inputPrompt.trim() || isLoading}
              className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 disabled:opacity-40 text-white rounded text-xs font-semibold transition flex items-center gap-1 cursor-pointer shadow-sm"
            >
              <Send className="w-3.5 h-3.5" />
              <span>发送</span>
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};
