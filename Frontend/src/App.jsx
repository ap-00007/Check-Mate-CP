import React, { useState } from 'react';
import { Fingerprint, ChevronRight, ChevronDown, Loader2, AlertCircle } from 'lucide-react';
import UploadCard from './components/UploadCard';
import VerificationResultCard from './components/VerificationResultCard';
import OverallAnalysisCard from './components/OverallAnalysisCard';
import ModelConfidenceCard from './components/ModelConfidenceCard';
import KeyInsightsCard from './components/KeyInsightsCard';
import FeatureComparisonCard from './components/FeatureComparisonCard';

export default function App() {
  const [refFile, setRefFile] = useState(null);
  const [cmpFile, setCmpFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [resultData, setResultData] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async () => {
    if (loading) return;

    if (!refFile || !cmpFile) {
      setError('Please upload both a reference and a comparison handwriting image before submitting.');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('reference_image', refFile);
    formData.append('comparison_image', cmpFile);

    try {
    // Try direct backend first (127.0.0.1:8000), then localhost:8000, then proxy
    const endpoints = [
      'http://127.0.0.1:8000/api/v1/verify',
      'http://localhost:8000/api/v1/verify',
      '/api/v1/verify',
    ];

    let lastError = null;
    let data = null;

    for (const endpoint of endpoints) {
      try {
        const response = await fetch(endpoint, {
          method: 'POST',
          body: formData,
        });

        const contentType = response.headers.get('content-type') || '';
        if (response.ok && contentType.includes('application/json')) {
          data = await response.json();
          break;
        } else if (!response.ok) {
          const errJson = await response.json().catch(() => null);
          const detail = errJson?.detail || `Server returned ${response.status} from ${endpoint}`;
          lastError = new Error(detail);
        }
      } catch (err) {
        lastError = err;
      }
    }

    if (data) {
      setResultData(data);
    } else {
      console.error('Verification failed across all endpoints:', lastError);
      setError(
        lastError?.message ||
        'Unable to connect to the backend server. Please verify that FastAPI is running on port 8000.'
      );
    }
    } catch (err) {
      console.error('Verification request failed:', err);
      setError(
        err.message || 'Unable to connect to the backend server. Please verify that FastAPI is running on port 8000.'
      );
    } finally {
      setLoading(false);
    }
  };

  const showResults = Boolean(resultData);
  const primaryScore = resultData?.combined_score ?? resultData?.visual_score ?? null;
  const percentage = primaryScore !== null ? primaryScore * 100 : null;

  return (
    <div
      style={{
        minHeight: '100vh',
        background: 'linear-gradient(160deg, #DCEEFF 0%, #C8E5FF 50%, #D4EAFF 100%)',
        paddingTop: '48px',
        paddingBottom: '64px',
        paddingLeft: '16px',
        paddingRight: '16px',
      }}
    >
      {/* ───── MAX-WIDTH WRAPPER ───── */}
      <div style={{ maxWidth: '1200px', margin: '0 auto' }}>

        {/* ════════════════════════════════
            SECTION A: HEADER + UPLOAD
            ════════════════════════════════ */}
        <div className="flex flex-col items-center" style={{ marginBottom: '32px' }}>

          {/* Title */}
          <h1
            style={{
              fontWeight: 800,
              fontSize: 'clamp(2.4rem, 5vw, 3.4rem)',
              letterSpacing: '-0.02em',
              lineHeight: 1.1,
              marginBottom: '12px',
              textAlign: 'center',
            }}
          >
            <span style={{ color: '#071C54' }}>Check </span>
            <span
              style={{
                background: 'linear-gradient(90deg, #08C5E8 0%, #1689F5 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
              }}
            >
              Mate
            </span>
          </h1>

          <p
            style={{
              color: '#52709C',
              fontSize: '1rem',
              fontWeight: 500,
              marginBottom: '28px',
              textAlign: 'center',
            }}
          >
            Handwriting Author Verification — Computer Vision &amp; Stylometric Analysis
          </p>

          {/* Upload container */}
          <div
            className="glass-outer w-full"
            style={{ padding: '24px', marginBottom: '24px' }}
          >
            <div
              className="upload-row-flex flex gap-5"
              style={{ flexWrap: 'wrap' }}
            >
              <UploadCard
                title="Reference Handwriting"
                subtitle="Upload the known handwriting sample"
                side="left"
                onFileChange={(f) => {
                  setRefFile(f);
                  setError(null);
                }}
              />
              <UploadCard
                title="Comparison Handwriting"
                subtitle="Upload the handwriting sample to be verified"
                side="right"
                onFileChange={(f) => {
                  setCmpFile(f);
                  setError(null);
                }}
              />
            </div>
          </div>

          {/* Error Banner */}
          {error && (
            <div
              className="fade-in flex items-center gap-2"
              style={{
                background: 'rgba(239, 68, 68, 0.12)',
                border: '1.5px solid rgba(239, 68, 68, 0.35)',
                color: '#DC2626',
                borderRadius: '12px',
                padding: '12px 20px',
                marginBottom: '20px',
                fontSize: '0.88rem',
                fontWeight: 600,
                maxWidth: '600px',
              }}
            >
              <AlertCircle size={18} style={{ flexShrink: 0 }} />
              <span>{error}</span>
            </div>
          )}

          {/* Submit button */}
          <button
            className="btn-primary"
            onClick={handleSubmit}
            disabled={loading}
            style={loading ? { opacity: 0.75, cursor: 'not-allowed' } : {}}
          >
            {loading ? (
              <Loader2 size={19} strokeWidth={2.2} style={{ animation: 'spin 1s linear infinite' }} />
            ) : (
              <Fingerprint size={19} strokeWidth={2} />
            )}
            {loading ? 'Verifying Styles…' : 'Submit for Verification'}
            {!loading && <ChevronRight size={18} strokeWidth={2.5} />}
          </button>
        </div>

        {/* ════════════════════════════════
            DECORATIVE CHEVRONS
            ════════════════════════════════ */}
        {showResults && (
          <div
            className="flex flex-col items-center chevron-deco"
            style={{ marginBottom: '24px', gap: '-6px' }}
          >
            <ChevronDown size={26} strokeWidth={2.5} style={{ marginBottom: '-10px' }} />
            <ChevronDown size={26} strokeWidth={2.5} />
          </div>
        )}

        {/* ════════════════════════════════
            SECTION B: RESULTS DASHBOARD
            ════════════════════════════════ */}
        {showResults && resultData && (
          <div
            className="glass-outer fade-in"
            style={{ padding: '24px' }}
          >
            {/* ── TOP ROW: 3 columns ── */}
            <div
              className="results-top-grid grid gap-5"
              style={{
                gridTemplateColumns: '1fr 2fr 1fr',
                marginBottom: '20px',
              }}
            >
              <VerificationResultCard
                percentage={percentage}
                status={resultData.status}
                message={resultData.message}
              />
              <OverallAnalysisCard
                visualScore={resultData.visual_score}
                nlpScore={resultData.nlp_score}
                combinedScore={resultData.combined_score}
                visualFeatureScores={resultData.visual_feature_scores}
              />
              <ModelConfidenceCard
                value={resultData.combined_score ?? resultData.visual_score}
              />
            </div>

            {/* ── BOTTOM ROW: 2 columns ── */}
            <div
              className="results-bottom-grid grid gap-5"
              style={{ gridTemplateColumns: '1fr 1.4fr' }}
            >
              <KeyInsightsCard
                status={resultData.status}
                visualFeatureScores={resultData.visual_feature_scores}
                warnings={resultData.warnings}
              />
              <FeatureComparisonCard
                rows={resultData.feature_comparison}
              />
            </div>
          </div>
        )}
      </div>

      {/* Spin keyframe injected inline for the loader icon */}
      <style>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        @media (max-width: 900px) {
          .results-top-grid { grid-template-columns: 1fr 1fr !important; }
          .results-bottom-grid { grid-template-columns: 1fr !important; }
        }
        @media (max-width: 640px) {
          .results-top-grid { grid-template-columns: 1fr !important; }
          .upload-row { flex-direction: column !important; }
        }
      `}</style>
    </div>
  );
}
