import React, { useEffect, useRef } from 'react';
import { Layers } from 'lucide-react';

function AnimatedBar({ pct, gradient, glow, delay = 0 }) {
  const barRef = useRef(null);

  useEffect(() => {
    if (!barRef.current) return;
    barRef.current.style.width = '0%';
    const timer = setTimeout(() => {
      if (barRef.current) {
        barRef.current.style.transition = `width 1.2s cubic-bezier(0.4,0,0.2,1) ${delay}s`;
        barRef.current.style.width = `${Math.min(100, Math.max(0, pct))}%`;
      }
    }, 50);
    return () => clearTimeout(timer);
  }, [pct, delay]);

  return (
    <div
      style={{
        height: '11px',
        background: 'rgba(22,137,245,0.10)',
        borderRadius: '999px',
        overflow: 'hidden',
        flex: 1,
      }}
    >
      <div
        ref={barRef}
        style={{
          height: '100%',
          background: gradient,
          borderRadius: '999px',
          boxShadow: `0 2px 8px ${glow}`,
          width: '0%',
        }}
      />
    </div>
  );
}

export default function OverallAnalysisCard({
  visualScore = null,
  nlpScore = null,
  combinedScore = null,
  visualFeatureScores = {},
}) {
  const bars = [];

  if (visualFeatureScores && Object.keys(visualFeatureScores).length > 0) {
    if (visualFeatureScores.stroke_similarity !== undefined) {
      const p = Number((visualFeatureScores.stroke_similarity * 100).toFixed(1));
      bars.push({
        label: 'Stroke Width & Ink Match',
        pct: p,
        display: `${p}%`,
        gradient: 'linear-gradient(90deg, #1689F5 0%, #3BABF5 100%)',
        glow: 'rgba(22,137,245,0.35)',
      });
    }

    if (visualFeatureScores.slant_similarity !== undefined) {
      const p = Number((visualFeatureScores.slant_similarity * 100).toFixed(1));
      bars.push({
        label: 'Slant & Angle Consistency',
        pct: p,
        display: `${p}%`,
        gradient: 'linear-gradient(90deg, #08C5E8 0%, #3BEAFF 100%)',
        glow: 'rgba(8,197,232,0.35)',
      });
    }

    if (visualFeatureScores.letter_proportion_similarity !== undefined) {
      const p = Number((visualFeatureScores.letter_proportion_similarity * 100).toFixed(1));
      bars.push({
        label: 'Letter Proportions & Spacing',
        pct: p,
        display: `${p}%`,
        gradient: 'linear-gradient(90deg, #17C98B 0%, #38EFA9 100%)',
        glow: 'rgba(23,201,139,0.35)',
      });
    }
  }

  // If NLP is present (Phase 2), show NLP bar
  if (nlpScore !== null && nlpScore !== undefined) {
    const p = Number((nlpScore * 100).toFixed(1));
    bars.push({
      label: 'Linguistic (NLP) Similarity',
      pct: p,
      display: `${p}%`,
      gradient: 'linear-gradient(90deg, #9333EA 0%, #C084FC 100%)',
      glow: 'rgba(147,51,234,0.35)',
    });
  }

  // Overall visual / combined score bar
  const totalScore = combinedScore ?? visualScore;
  if (totalScore !== null && totalScore !== undefined) {
    const p = Number((totalScore * 100).toFixed(1));
    bars.push({
      label: nlpScore != null ? 'Combined Fusion Score' : 'Overall Visual Match',
      pct: p,
      display: `${p}%`,
      gradient: 'linear-gradient(90deg, #FFD629 0%, #FFBE00 100%)',
      glow: 'rgba(255,214,41,0.40)',
    });
  }

  return (
    <div className="glass-card flex flex-col fade-in" style={{ padding: '24px 22px', flex: 1 }}>
      {/* Header */}
      <div className="flex items-center gap-2" style={{ marginBottom: '24px' }}>
        <Layers size={20} strokeWidth={2} style={{ color: '#1689F5' }} />
        <span style={{ fontWeight: 700, fontSize: '1rem', color: '#071C54' }}>Overall Analysis</span>
      </div>

      {/* Bars */}
      <div className="flex flex-col" style={{ gap: '18px' }}>
        {bars.map((bar, i) => (
          <div key={bar.label} className="flex flex-col" style={{ gap: '7px' }}>
            <div className="flex items-center gap-3">
              <span style={{ fontSize: '0.82rem', color: '#071C54', fontWeight: 500, flex: 1, whiteSpace: 'nowrap' }}>
                {bar.label}
              </span>
              <span style={{ fontSize: '0.82rem', fontWeight: 700, color: '#071C54', minWidth: '46px', textAlign: 'right' }}>
                {bar.display}
              </span>
            </div>
            <AnimatedBar pct={bar.pct} gradient={bar.gradient} glow={bar.glow} delay={i * 0.15} />
          </div>
        ))}
      </div>
    </div>
  );
}
