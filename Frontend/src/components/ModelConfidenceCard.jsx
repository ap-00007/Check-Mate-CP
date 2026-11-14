import React from 'react';
import { Gauge } from 'lucide-react';

export default function ModelConfidenceCard({ value = null }) {
  const isAvailable = value !== null && value !== undefined;
  const numVal = isAvailable ? Number(value) : 0;

  const getBadge = () => {
    if (!isAvailable) {
      return { text: 'No Data', color: '#52709C', bg: 'rgba(82,112,156,0.14)', border: 'rgba(82,112,156,0.3)' };
    }
    if (numVal >= 0.80 || numVal <= 0.20) {
      return { text: 'High Confidence', color: '#17C98B', bg: 'rgba(23, 201, 139, 0.14)', border: 'rgba(23,201,139,0.35)' };
    }
    if (numVal >= 0.65 || numVal <= 0.35) {
      return { text: 'Moderate Confidence', color: '#1689F5', bg: 'rgba(22, 137, 245, 0.14)', border: 'rgba(22,137,245,0.35)' };
    }
    return { text: 'Borderline Match', color: '#F59E0B', bg: 'rgba(245, 158, 11, 0.14)', border: 'rgba(245,158,11,0.35)' };
  };

  const badge = getBadge();

  return (
    <div
      className="glass-card flex flex-col fade-in"
      style={{ padding: '24px 20px', minHeight: '310px', alignItems: 'center', justifyContent: 'center' }}
    >
      {/* Header */}
      <div className="flex items-center gap-2 w-full" style={{ marginBottom: '24px' }}>
        <Gauge size={22} strokeWidth={2} style={{ color: '#1689F5' }} />
        <span style={{ fontWeight: 700, fontSize: '1rem', color: '#071C54' }}>Model Confidence</span>
      </div>

      {/* Value */}
      <div
        style={{
          fontWeight: 800,
          fontSize: isAvailable ? '3rem' : '2.2rem',
          letterSpacing: '-0.02em',
          background: 'linear-gradient(135deg, #071C54 0%, #1689F5 100%)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          backgroundClip: 'text',
          lineHeight: 1.1,
          textAlign: 'center',
          marginBottom: '20px',
        }}
      >
        {isAvailable ? numVal.toFixed(3) : '—'}
      </div>

      {/* Badge */}
      <div
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '7px',
          background: badge.bg,
          border: `1.5px solid ${badge.border}`,
          borderRadius: '999px',
          padding: '7px 20px',
        }}
      >
        <span style={{ fontWeight: 700, fontSize: '0.88rem', color: badge.color }}>
          {badge.text}
        </span>
      </div>
    </div>
  );
}
