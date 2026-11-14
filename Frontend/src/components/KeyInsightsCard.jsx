import React from 'react';
import { Lightbulb, CheckCircle2, AlertCircle } from 'lucide-react';

export default function KeyInsightsCard({
  insights = [],
  status = 'compatible',
  visualFeatureScores = {},
  warnings = [],
}) {
  let displayInsights = [];

  if (insights && insights.length > 0) {
    displayInsights = insights.map((text) => ({ text, isWarning: false }));
  } else {
    // Generate dynamic insights from actual visual scores
    const stroke = visualFeatureScores?.stroke_similarity;
    const slant = visualFeatureScores?.slant_similarity;
    const prop = visualFeatureScores?.letter_proportion_similarity;

    if (status === 'compatible') {
      displayInsights.push({
        text: 'Handwriting physical profiles show high stylistic consistency across both samples.',
        isWarning: false,
      });
      if (stroke !== undefined && stroke >= 0.75) {
        displayInsights.push({
          text: `Matching pen stroke thickness and ink pressure distribution (${(stroke * 100).toFixed(1)}% match).`,
          isWarning: false,
        });
      }
      if (slant !== undefined && slant >= 0.75) {
        displayInsights.push({
          text: `Consistent character slant angle and stroke orientation (${(slant * 100).toFixed(1)}% match).`,
          isWarning: false,
        });
      }
      if (prop !== undefined && prop >= 0.75) {
        displayInsights.push({
          text: `Proportional letter height-to-width ratio and baseline geometry match (${(prop * 100).toFixed(1)}% match).`,
          isWarning: false,
        });
      }
    } else if (status === 'not_compatible') {
      displayInsights.push({
        text: 'Handwriting profiles show significant physical divergence between the two samples.',
        isWarning: false,
      });
      if (stroke !== undefined && stroke < 0.70) {
        displayInsights.push({
          text: 'Noticeable divergence in stroke thickness or ink deposit patterns.',
          isWarning: false,
        });
      }
      if (slant !== undefined && slant < 0.70) {
        displayInsights.push({
          text: 'Discrepancy detected in handwriting slant angle and stroke orientation.',
          isWarning: false,
        });
      }
      if (prop !== undefined && prop < 0.70) {
        displayInsights.push({
          text: 'Distinct connected-component letter height and aspect ratio patterns.',
          isWarning: false,
        });
      }
    } else {
      displayInsights.push({
        text: 'Insufficient ink or stroke detail detected to compute a definitive stylistic match.',
        isWarning: true,
      });
    }

    if (warnings && warnings.length > 0) {
      warnings.forEach((w) => {
        displayInsights.push({ text: w, isWarning: true });
      });
    }
  }

  return (
    <div className="glass-card flex flex-col fade-in" style={{ padding: '24px 22px' }}>
      {/* Header */}
      <div className="flex items-center gap-2" style={{ marginBottom: '20px' }}>
        <Lightbulb size={20} strokeWidth={2} style={{ color: '#1689F5' }} />
        <span style={{ fontWeight: 700, fontSize: '1rem', color: '#071C54' }}>Key Insights</span>
      </div>

      {/* Insight rows */}
      <div className="flex flex-col" style={{ gap: '12px' }}>
        {displayInsights.map((item, i) => (
          <div key={i} className="flex items-start gap-3">
            <div
              style={{
                flexShrink: 0,
                width: '20px',
                height: '20px',
                borderRadius: '50%',
                background: item.isWarning ? 'rgba(245,158,11,0.15)' : 'rgba(23,201,139,0.15)',
                border: `1.5px solid ${item.isWarning ? 'rgba(245,158,11,0.4)' : 'rgba(23,201,139,0.4)'}`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginTop: '1px',
              }}
            >
              {item.isWarning ? (
                <AlertCircle size={13} strokeWidth={2.5} style={{ color: '#F59E0B' }} />
              ) : (
                <CheckCircle2 size={13} strokeWidth={2.5} style={{ color: '#17C98B' }} />
              )}
            </div>
            <span style={{ fontSize: '0.83rem', color: '#071C54', lineHeight: 1.5, fontWeight: 500 }}>
              {item.text}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
