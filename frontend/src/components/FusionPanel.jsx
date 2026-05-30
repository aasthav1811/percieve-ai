// src/components/FusionPanel.jsx
import React from 'react'
import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer } from 'recharts'
import { EMOTION_COLORS } from '../lib/api'

const EMOTION_ORDER = ['Happy', 'Angry', 'Fear', 'Sad', 'Neutral', 'Surprised', 'Disgust']

const RISK_CONFIG = {
  low:    { label: 'LOW RISK',    color: '#30d988', bg: 'rgba(48,217,136,0.08)',  bar: '#30d988' },
  medium: { label: 'MEDIUM RISK', color: '#f5a623', bg: 'rgba(245,166,35,0.08)', bar: '#f5a623' },
  high:   { label: 'HIGH RISK',   color: '#ff4d6a', bg: 'rgba(255,77,106,0.08)', bar: '#ff4d6a' },
}

export default function FusionPanel({ result }) {
  const emotion   = result?.fused_emotion  || '—'
  const probs     = result?.fused_probs    || {}
  const conf      = result?.confidence     ?? null
  const valence   = result?.valence        ?? null
  const arousal   = result?.arousal        ?? null
  const risk      = result?.risk_level     || 'low'
  const weights   = result?.modal_weights  || { visual: 0.5, text: 0.5 }
  const rec       = result?.recommendation || ''
  const ms        = result?.inference_ms   ?? null

  const riskCfg = RISK_CONFIG[risk]

  const radarData = EMOTION_ORDER.map(name => ({
    name,
    value: Math.round((probs[name] ?? 0) * 100),
  }))

  const emoColor = EMOTION_COLORS[emotion] || '#5e82aa'

  return (
    <div className="card overflow-hidden glow-purple" style={{ borderColor: `${emoColor}30` }}>
      <div className="panel-header">
        <span className="w-[7px] h-[7px] rounded-full bg-purple flex-shrink-0" />
        <span>Cross-Modal Fusion Output</span>
        <span className="ml-auto text-purple font-mono text-[9px]">Attention Gate</span>
      </div>

      <div className="p-4 flex flex-col gap-4">
        {/* Primary emotion + confidence */}
        <div className="text-center py-3 rounded-lg border"
          style={{ background: `${emoColor}10`, borderColor: `${emoColor}30` }}>
          <div className="font-mono text-[10px] text-muted mb-1 tracking-[0.15em] uppercase">Fused Emotion</div>
          <div className="font-display text-3xl font-semibold transition-all duration-500"
            style={{ color: emoColor }}>
            {emotion}
          </div>
          {conf !== null && (
            <div className="font-mono text-[10px] mt-1" style={{ color: emoColor }}>
              {(conf * 100).toFixed(1)}% confidence
            </div>
          )}
        </div>

        {/* Radar chart */}
        <div className="h-[160px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart data={radarData} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
              <PolarGrid stroke="rgba(30,74,138,0.3)" />
              <PolarAngleAxis
                dataKey="name"
                tick={{ fill: '#5e82aa', fontSize: 9, fontFamily: 'IBM Plex Mono' }}
              />
              <Radar
                name="Emotion"
                dataKey="value"
                stroke={emoColor}
                fill={emoColor}
                fillOpacity={0.15}
                strokeWidth={1.5}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Valence / Arousal */}
        <div className="grid grid-cols-2 gap-2">
          <ScalarMetric
            label="Valence"
            value={valence}
            min={-1} max={1}
            format={v => `${v >= 0 ? '+' : ''}${v.toFixed(2)}`}
            colorFn={v => v > 0.2 ? '#30d988' : v < -0.2 ? '#ff4d6a' : '#5e82aa'}
          />
          <ScalarMetric
            label="Arousal"
            value={arousal}
            min={0} max={1}
            format={v => v.toFixed(2)}
            colorFn={v => v > 0.6 ? '#ff4d6a' : v < 0.3 ? '#9b74f7' : '#f5a623'}
          />
        </div>

        {/* Modal weights */}
        <div>
          <div className="font-mono text-[10px] text-muted mb-2 tracking-[0.12em] uppercase">
            Attention Gate Weights
          </div>
          <div className="flex h-2 rounded-full overflow-hidden gap-[2px]">
            <div
              className="h-full rounded-l-full transition-all duration-700 relative group"
              style={{ width: `${weights.visual * 100}%`, background: '#0ec9a8' }}
              title={`Visual: ${(weights.visual * 100).toFixed(0)}%`}
            />
            <div
              className="h-full rounded-r-full transition-all duration-700"
              style={{ width: `${weights.text * 100}%`, background: '#f5a623' }}
              title={`Text: ${(weights.text * 100).toFixed(0)}%`}
            />
          </div>
          <div className="flex justify-between mt-1">
            <span className="font-mono text-[9px] text-teal">Vision {(weights.visual * 100).toFixed(0)}%</span>
            <span className="font-mono text-[9px] text-amber">Text {(weights.text * 100).toFixed(0)}%</span>
          </div>
        </div>

        {/* Risk level */}
        <div className="rounded-lg border p-3 flex flex-col gap-1"
          style={{ background: riskCfg.bg, borderColor: `${riskCfg.color}30` }}>
          <div className="flex items-center justify-between">
            <span className="font-mono text-[10px] font-semibold tracking-[0.15em]"
              style={{ color: riskCfg.color }}>
              ● {riskCfg.label}
            </span>
            {ms !== null && (
              <span className="font-mono text-[9px] text-muted">{ms.toFixed(0)}ms</span>
            )}
          </div>
          {rec && (
            <p className="font-mono text-[10px] text-[#8bb0d0] leading-[1.7] mt-1">{rec}</p>
          )}
        </div>
      </div>
    </div>
  )
}

function ScalarMetric({ label, value, min, max, format, colorFn }) {
  const pct = value !== null ? ((value - min) / (max - min)) * 100 : 50
  const color = value !== null ? colorFn(value) : '#5e82aa'

  return (
    <div className="bg-[#081527] border border-[#1a3a6e30] rounded-lg p-3">
      <div className="font-mono text-[9px] text-muted mb-2 tracking-[0.1em] uppercase">{label}</div>
      <div className="font-display text-xl font-semibold mb-2 transition-colors duration-500"
        style={{ color }}>
        {value !== null ? format(value) : '—'}
      </div>
      <div className="h-1 bg-[#1a3a6e40] rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  )
}
