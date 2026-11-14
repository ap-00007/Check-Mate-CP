import React from 'react';
import { BarChart2 } from 'lucide-react';

export default function FeatureComparisonCard({ rows = [] }) {
  const displayRows = rows && rows.length > 0 ? rows : [];

  return (
    <div className="glass-card flex flex-col fade-in" style={{ padding: '24px 22px' }}>
      {/* Header */}
      <div className="flex items-center gap-2" style={{ marginBottom: '16px' }}>
        <BarChart2 size={20} strokeWidth={2} style={{ color: '#1689F5' }} />
        <span style={{ fontWeight: 700, fontSize: '1rem', color: '#071C54' }}>Feature Comparison</span>
      </div>

      {/* Table wrapper */}
      <div style={{ overflowX: 'auto', borderRadius: '12px', border: '1px solid rgba(22,137,245,0.14)' }}>
        <table className="feature-table">
          <thead>
            <tr>
              <th style={{ textAlign: 'left' }}>Feature</th>
              <th style={{ textAlign: 'right' }}>Reference</th>
              <th style={{ textAlign: 'right' }}>Comparison</th>
              <th style={{ textAlign: 'right' }}>Δ (Difference)</th>
            </tr>
          </thead>
          <tbody>
            {displayRows.length > 0 ? (
              displayRows.map((row, i) => (
                <tr key={i}>
                  <td style={{ fontWeight: 600 }}>{row.feature}</td>
                  <td style={{ textAlign: 'right', color: '#1689F5', fontWeight: 600 }}>
                    {typeof row.reference === 'number' ? row.reference.toFixed(3) : row.reference}
                  </td>
                  <td style={{ textAlign: 'right', color: '#08C5E8', fontWeight: 600 }}>
                    {typeof row.comparison === 'number' ? row.comparison.toFixed(3) : row.comparison}
                  </td>
                  <td style={{ textAlign: 'right', color: '#52709C', fontWeight: 500 }}>
                    {typeof (row.absolute_difference ?? row.delta) === 'number'
                      ? (row.absolute_difference ?? row.delta).toFixed(3)
                      : (row.absolute_difference ?? row.delta)}
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={4} style={{ textAlign: 'center', padding: '16px', color: '#52709C' }}>
                  No feature comparison data available.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
