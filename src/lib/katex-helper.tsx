import React, { useEffect, useRef } from "react";
import katex from "katex";

interface MathViewProps {
  math: string;
  block?: boolean;
  className?: string;
}

export const MathView: React.FC<MathViewProps> = ({ math, block = false, className = "" }) => {
  const containerRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    try {
      katex.render(math, containerRef.current, {
        displayMode: block,
        throwOnError: false,
        output: "htmlAndMathml",
      });
    } catch (e) {
      if (containerRef.current) {
        containerRef.current.textContent = math;
      }
    }
  }, [math, block]);

  return <span ref={containerRef} className={`inline-block ${className}`} />;
};

export const InlineMath: React.FC<{ math: string; className?: string }> = ({ math, className }) => (
  <MathView math={math} block={false} className={className} />
);

export const BlockMath: React.FC<{ math: string; className?: string }> = ({ math, className }) => (
  <div className={`overflow-x-auto py-1 text-center ${className || ""}`}>
    <MathView math={math} block={true} />
  </div>
);
