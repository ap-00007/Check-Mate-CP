import React, { useCallback, useRef, useState } from 'react';
import { FileText, ImageUp, X } from 'lucide-react';

export default function UploadCard({ title, subtitle, side, onFileChange }) {
  const inputRef = useRef(null);
  const [preview, setPreview] = useState(null);
  const [dragging, setDragging] = useState(false);

  const accept = 'image/jpeg,image/jpg,image/png,image/webp';

  const handleFile = useCallback((file) => {
    if (!file) return;
    if (!['image/jpeg', 'image/jpg', 'image/png', 'image/webp'].includes(file.type)) return;
    const url = URL.createObjectURL(file);
    setPreview(url);
    onFileChange && onFileChange(file);
  }, [onFileChange]);

  const handleChange = (e) => handleFile(e.target.files[0]);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const handleRemove = (e) => {
    e.stopPropagation();
    setPreview(null);
    if (inputRef.current) inputRef.current.value = '';
    onFileChange && onFileChange(null);
  };

  return (
    <div className="glass-card flex-1 p-5 flex flex-col gap-4 min-w-0">
      {/* Card header */}
      <div className="flex items-center gap-3">
        <div
          style={{
            background: 'linear-gradient(135deg, rgba(8,197,232,0.18) 0%, rgba(22,137,245,0.18) 100%)',
            borderRadius: '10px',
            padding: '8px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <FileText size={22} strokeWidth={1.8} style={{ color: '#1689F5' }} />
        </div>
        <div>
          <div style={{ fontWeight: 700, fontSize: '1rem', color: '#071C54', lineHeight: 1.2 }}>
            {title}
          </div>
          <div style={{ fontSize: '0.78rem', color: '#52709C', marginTop: '2px' }}>
            {subtitle}
          </div>
        </div>
      </div>

      {/* Upload zone */}
      <div
        className={`upload-zone flex flex-col items-center justify-center relative${dragging ? ' drag-over' : ''}`}
        style={{ minHeight: '160px', padding: '20px' }}
        onClick={() => !preview && inputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          className="hidden"
          onChange={handleChange}
        />

        {preview ? (
          <div className="relative w-full h-full flex items-center justify-center">
            <img
              src={preview}
              alt="Preview"
              style={{
                maxHeight: '130px',
                maxWidth: '100%',
                borderRadius: '10px',
                objectFit: 'contain',
                boxShadow: '0 4px 16px rgba(22,137,245,0.18)',
              }}
            />
            <button
              onClick={handleRemove}
              style={{
                position: 'absolute',
                top: '-8px',
                right: '-8px',
                background: 'linear-gradient(135deg,#08C5E8,#1689F5)',
                border: 'none',
                borderRadius: '50%',
                width: '24px',
                height: '24px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                boxShadow: '0 2px 8px rgba(22,137,245,0.35)',
              }}
            >
              <X size={13} color="#fff" />
            </button>
            <button
              onClick={(e) => { e.stopPropagation(); inputRef.current?.click(); }}
              style={{
                position: 'absolute',
                bottom: '-8px',
                right: '-8px',
                background: 'linear-gradient(135deg,#08C5E8,#1689F5)',
                border: 'none',
                borderRadius: '50%',
                width: '24px',
                height: '24px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                boxShadow: '0 2px 8px rgba(22,137,245,0.35)',
              }}
            >
              <ImageUp size={13} color="#fff" />
            </button>
          </div>
        ) : (
          <>
            {/* Upload icon */}
            <div style={{ marginBottom: '10px', position: 'relative' }}>
              <svg width="52" height="52" viewBox="0 0 52 52" fill="none">
                <rect width="52" height="52" rx="12" fill="rgba(22,137,245,0.07)" />
                <rect x="12" y="14" width="28" height="24" rx="4" fill="none" stroke="#3BABF5" strokeWidth="1.8" />
                <path d="M20 26 L26 20 L32 26" stroke="#1689F5" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" fill="none" />
                <path d="M26 20 L26 32" stroke="#1689F5" strokeWidth="1.8" strokeLinecap="round" fill="none" />
                <circle cx="38" cy="14" r="6" fill="white" stroke="none" />
                <line x1="38" y1="11" x2="38" y2="17" stroke="#08C5E8" strokeWidth="1.8" strokeLinecap="round" />
                <line x1="35" y1="14" x2="41" y2="14" stroke="#08C5E8" strokeWidth="1.8" strokeLinecap="round" />
              </svg>
            </div>
            <div style={{ fontWeight: 700, fontSize: '0.88rem', color: '#071C54', textAlign: 'center' }}>
              Drag &amp; drop image here
            </div>
            <div style={{ fontSize: '0.78rem', color: '#52709C', marginTop: '4px', textAlign: 'center' }}>
              or click to browse
            </div>
          </>
        )}
      </div>
    </div>
  );
}
