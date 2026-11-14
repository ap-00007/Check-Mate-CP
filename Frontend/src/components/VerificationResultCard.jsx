import React, { useEffect, useRef } from 'react';
import { ShieldCheck, CheckCircle, XCircle, AlertTriangle } from 'lucide-react';

export default function VerificationResultCard({
  percentage = null,
  status = 'compatible',
  message = '',
}) {
  const circleRef = useRef(null);

  const displayPct = percentage !== null && percentage !== undefined ? Number(percentage) : 0;
  const isAvailable = percentage !== null && percentage !== undefined;

  const CX = 100;
  const CY = 100;
  const R = 82;
  const SW = 12;
  const circumference = 2 * Math.PI * R;
  const dashOffset = isAvailable
    ? circumference - (Math.min(100, Math.max(0, displayPct)) / 100) * circumference
    : circumference;

  useEffect(() => {
    const el = circleRef.current;
    if (!el) return;

    el.style.transition = 'none';
    el.style.strokeDashoffset = String(circumference);

    const raf = requestAnimationFrame(() => {
      el.style.transition = 'stroke-dashoffset 1.4s cubic-bezier(0.4, 0, 0.2, 1)';
      el.style.strokeDashoffset = String(dashOffset);
    });
    return () => cancelAnimationFrame(raf);
  }, [circumference, dashOffset]);

  const getStatusBadge = () => {
    if (status === 'compatible') {
      return {
        icon: <CheckCircle size={16} strokeWidth={2.5} style={{ color: '#17C98B' }} />,
        text: 'Same Author',
        color: '#17C98B',
        bg: 'rgba(23,201,139,0.14)',
        border: 'rgba(23,201,139,0.35)',
      };
    }
    if (status === 'not_compatible') {
      return {
        icon: <XCircle size={16} strokeWidth={2.5} style={{ color: '#EF4444' }} />,
        text: 'Different Author',
        color: '#EF4444',
        bg: 'rgba(239,68,68,0.12)',
        border: 'rgba(239,68,68,0.35)',
      };
    }
    return {
      icon: <AlertTriangle size={16} strokeWidth={2.5} style={{ color: '#F59E0B' }} />,
      text: 'Insufficient Evidence',
      color: '#F59E0B',
      bg: 'rgba(245,158,11,0.14)',
      border: 'rgba(245,158,11,0.35)',
    };
  };

  const badge = getStatusBadge();

  return (
    <div
      className="glass-card flex flex-col fade-in"
      style={{
        padding: '22px 20px 20px',
        minHeight: '310px',
        alignItems: 'flex-start',
        justifyContent: 'space-between',
      }}
    >
      {/* Header */}
      <div className="flex items-center gap-2">
        <ShieldCheck size={22} strokeWidth={2} style={{ color: '#1689F5' }} />
        <span style={{ fontWeight: 700, fontSize: '1rem', color: '#071C54' }}>
          Verification Result
        </span>
      </div>

      {/* Circular progress */}
      <div
        className="flex items-center justify-center w-full"
        style={{ flex: 1, paddingTop: '10px', paddingBottom: '10px' }}
      >
        <div style={{ position: 'relative', width: 200, height: 200 }}>
          <svg width="200" height="200" viewBox="0 0 200 200" style={{ display: 'block' }}>
            <defs>
              <linearGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#08C5E8" />
                <stop offset="100%" stopColor="#1689F5" />
              </linearGradient>
            </defs>

            {/* Track */}
            <circle
              cx={CX}
              cy={CY}
              r={R}
              fill="none"
              stroke="rgba(22,137,245,0.13)"
              strokeWidth={SW}
            />

            {/* Animated progress arc */}
            <circle
              ref={circleRef}
              cx={CX}
              cy={CY}
              r={R}
              fill="none"
              stroke="url(#ringGrad)"
              strokeWidth={SW}
              strokeLinecap="round"
              strokeDasharray={circumference}
              strokeDashoffset={circumference}
              transform={`rotate(-90 ${CX} ${CY})`}
              style={{ filter: 'drop-shadow(0 0 7px rgba(8,197,232,0.55))' }}
            />
          </svg>

          {/* Text overlay */}
          <div
            style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              pointerEvents: 'none',
            }}
          >
            <span
              style={{
                fontWeight: 800,
                fontSize: isAvailable ? '1.95rem' : '1.5rem',
                lineHeight: 1,
                color: '#071C54',
                letterSpacing: '-0.02em',
              }}
            >
              {isAvailable ? `${displayPct.toFixed(1)}%` : 'N/A'}
            </span>
            <span
              style={{
                fontSize: '0.80rem',
                color: '#52709C',
                fontWeight: 500,
                marginTop: '5px',
              }}
            >
              Visual Similarity
            </span>
          </div>
        </div>
      </div>

      {/* Status badge */}
      <div className="flex items-center justify-center w-full">
        <div
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '7px',
            background: badge.bg,
            border: `1.5px solid ${badge.border}`,
            borderRadius: '999px',
            padding: '7px 22px',
          }}
        >
          {badge.icon}
          <span style={{ fontWeight: 700, fontSize: '0.88rem', color: badge.color }}>
            {badge.text}
          </span>
        </div>
      </div>
    </div>
  );
}
