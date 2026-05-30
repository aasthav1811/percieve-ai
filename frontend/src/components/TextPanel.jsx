// src/components/TextPanel.jsx
import React from 'react'
import { EMOTION_COLORS } from '../lib/api'

const EMOTION_ORDER = ['Happy', 'Angry', 'Fear', 'Sad', 'Neutral', 'Surprised', 'Disgust']

export default function TextPanel({ result, inputText }) {
  const probs     = result?.text_probs     || {}
  const sentiment = result?.sentiment_score ?? null
  const keywords  = result?.keywords       || []
  const emotion   = result?.text_emotion   || '—'

  const sentimentLabel =
    sentiment === null ? '—'
    : sentiment > 0.6  ? 'Positive'
    : sentiment < 0.4  ? 'Negative'
    : 'Neutral'

  const sentimentColor =
    sentiment === null ? '#5e82aa'
    : sentiment > 0.6  ? '#30d988'
    : sentiment < 0.4  ? '#ff4d6a'
    : '#5e82aa'

  return (
    <div className="card overflow-hidden">
      <div className="panel-header">
        <span className="w-[7px] h-[7px] rounded-full bg-amber flex-shrink-0" />
        <span>NLP Sentiment Analysis</span>
        <span className="ml-auto text-amber font-mono text-[9px]">RoBERTa-base</span>
      </div>

      <div className="p-4 flex flex-col gap-4">
        {/* Input text preview with keyword highlights */}
        <div className="bg-[#081527] border border-[#1a3a6e30] rounded-lg p-3 min-h-[72px]">
          <p className="font-mono text-[11px] text-[#8bb0d0] leading-[1.8]">
            {inputText ? renderKeywords(inputText, keywords) : (
              <span className="text-muted italic">Awaiting text input…</span>
            )}
          </p>
        </div>

        {/* Sentiment score */}
        <div className="flex items-center gap-3">
          <span className="font-mono text-[10px] text-muted">Sentiment</span>
          <div className="flex-1 h-1.5 bg-[#1a3a6e40] rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-700"
              style={{
                width: `${(sentiment ?? 0.5) * 100}%`,
                background: `linear-gradient(90deg, #ff4d6a, #f5a623, #30d988)`,
              }}
            />
          </div>
          <span className="font-mono text-[10px]" style={{ color: sentimentColor }}>
            {sentiment !== null ? `${(sentiment * 100).toFixed(0)}% ${sentimentLabel}` : '—'}
          </span>
        </div>

        {/* Emotion probability bars */}
        <div className="flex flex-col gap-[6px]">
          {EMOTION_ORDER.map(name => {
            const val   = probs[name] ?? 0
            const color = EMOTION_COLORS[name]
            const isTop = name === emotion
            return (
              <div key={name} className="flex items-center gap-2">
                <span
                  className="font-mono text-[10px] w-[62px] flex-shrink-0 transition-colors duration-300"
                  style={{ color: isTop ? color : '#5e82aa' }}
                >
                  {name}
                </span>
                <div className="flex-1 h-1 bg-[#1a3a6e40] rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-700 ease-out"
                    style={{ width: `${val * 100}%`, background: color, opacity: isTop ? 1 : 0.5 }}
                  />
                </div>
                <span className="font-mono text-[10px] text-muted w-8 text-right">
                  {(val * 100).toFixed(0)}%
                </span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

function renderKeywords(text, keywords) {
  if (!keywords || keywords.length === 0) return text

  const SENTIMENT_COLORS = { pos: '#30d988', neg: '#ff6b77', neu: '#5e82aa' }
  const parts = []
  let remaining = text

  keywords.forEach(([kw, sentiment]) => {
    const idx = remaining.toLowerCase().indexOf(kw.toLowerCase())
    if (idx === -1) return
    if (idx > 0) parts.push(remaining.slice(0, idx))
    parts.push(
      <mark
        key={kw}
        style={{
          background: `${SENTIMENT_COLORS[sentiment]}18`,
          color: SENTIMENT_COLORS[sentiment],
          borderBottom: `1px solid ${SENTIMENT_COLORS[sentiment]}60`,
          borderRadius: '2px',
          padding: '0 2px',
        }}
      >
        {remaining.slice(idx, idx + kw.length)}
      </mark>
    )
    remaining = remaining.slice(idx + kw.length)
  })

  if (remaining) parts.push(remaining)
  return parts
}
