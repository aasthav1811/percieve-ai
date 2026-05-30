// src/components/InputPanel.jsx
import React, { useRef, useState } from 'react'

export default function InputPanel({ text, onTextChange, onImageChange, onAnalyze, loading, imagePreview }) {
  const fileRef = useRef(null)
  const [dragOver, setDragOver] = useState(false)

  function handleDrop(e) {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) onImageChange(file)
  }

  function handleFile(e) {
    const file = e.target.files[0]
    if (file) onImageChange(file)
  }

  return (
    <div className="card overflow-hidden">
      <div className="panel-header">
        <span className="w-[7px] h-[7px] rounded-full bg-[#cfe2f7] flex-shrink-0" />
        <span>Input Stream</span>
        <span className="ml-auto font-mono text-[9px] text-muted">Vision + NLP</span>
      </div>

      <div className="p-4 flex flex-col gap-4">
        {/* Image drop zone */}
        <div
          className={`relative border rounded-lg transition-all duration-200 cursor-pointer overflow-hidden
            ${dragOver
              ? 'border-teal bg-[#0ec9a815]'
              : 'border-[#1a3a6e50] hover:border-[#1a3a6e] bg-[#081527]'}`}
          style={{ minHeight: '120px' }}
          onClick={() => fileRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
        >
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleFile}
          />
          {imagePreview ? (
            <img
              src={imagePreview}
              alt="Face preview"
              className="w-full h-full object-cover"
              style={{ maxHeight: '200px' }}
            />
          ) : (
            <div className="flex flex-col items-center justify-center gap-2 py-8 text-center">
              <div className="font-mono text-3xl opacity-30">⬡</div>
              <p className="font-mono text-[11px] text-muted">Drop face image or click to upload</p>
              <p className="font-mono text-[9px] text-[#3a5a7a]">JPEG · PNG · WEBP</p>
            </div>
          )}
          {/* scanline overlay while no image */}
          {!imagePreview && (
            <div className="absolute inset-0 pointer-events-none overflow-hidden">
              <div className="scanline-animate absolute left-0 right-0 h-px bg-teal opacity-20" />
            </div>
          )}
        </div>

        {/* Text area */}
        <div className="flex flex-col gap-1">
          <label className="font-mono text-[10px] text-muted tracking-[0.1em] uppercase">
            Transcript / Message
          </label>
          <textarea
            value={text}
            onChange={e => onTextChange(e.target.value)}
            placeholder="Enter customer speech or chat message…"
            rows={4}
            className="
              w-full bg-[#081527] border border-[#1a3a6e40] rounded-lg p-3
              font-mono text-[12px] text-[#cfe2f7] leading-[1.8] resize-none
              placeholder:text-[#2a4a6a] outline-none
              focus:border-teal focus:bg-[#08182e]
              transition-all duration-200
            "
          />
        </div>

        {/* Analyze button */}
        <button
          onClick={onAnalyze}
          disabled={loading || (!text.trim())}
          className="
            relative overflow-hidden w-full py-3 rounded-md
            font-display font-semibold text-sm
            transition-all duration-200
            disabled:opacity-40 disabled:cursor-not-allowed
            enabled:cursor-pointer
          "
          style={{
            background: loading
              ? 'linear-gradient(135deg, #0a6b5b, #0a4b7a)'
              : 'linear-gradient(135deg, #0ec9a8, #0a8aff)',
            color: '#050d1b',
          }}
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <span className="inline-block w-3 h-3 border-2 border-[#050d1b] border-t-transparent rounded-full animate-spin" />
              Analyzing…
            </span>
          ) : (
            '▶  Analyze Emotion'
          )}
          {/* shimmer on idle */}
          {!loading && (
            <div className="absolute inset-0 bg-white opacity-0 hover:opacity-10 transition-opacity duration-200" />
          )}
        </button>

        {/* API status hint */}
        <p className="font-mono text-[9px] text-[#2a4a6a] text-center">
          Demo mode active — connects to FastAPI on :8000 when available
        </p>
      </div>
    </div>
  )
}
