// src/components/FacePanel.jsx
import React from 'react'
import { EMOTION_COLORS } from '../lib/api'

const EMOTION_ORDER = ['Happy', 'Angry', 'Fear', 'Sad', 'Neutral', 'Surprised', 'Disgust']

const MOUTH_PATHS = {
  Happy:    'M46,84 Q65,96 84,84',
  Angry:    'M46,92 Q65,80 84,92',
  Fear:     'M48,88 Q65,86 82,88',
  Sad:      'M46,92 Q65,80 84,92',
  Neutral:  'M48,87 Q65,87 82,87',
  Surprised:'M55,82 Q65,94 75,82',
  Disgust:  'M47,90 Q65,82 83,90',
}

const BROW_L = {
  Happy:    'M33,35 Q45,29 57,34',
  Angry:    'M33,32 Q45,26 57,31',
  Fear:     'M33,33 Q45,28 57,33',
  Sad:      'M33,38 Q45,34 57,38',
  Neutral:  'M33,35 Q45,32 57,35',
  Surprised:'M33,30 Q45,24 57,30',
  Disgust:  'M33,33 Q45,30 57,35',
}

const BROW_R = {
  Happy:    'M73,34 Q85,29 97,35',
  Angry:    'M73,31 Q85,26 97,32',
  Fear:     'M73,33 Q85,28 97,33',
  Sad:      'M73,38 Q85,34 97,38',
  Neutral:  'M73,35 Q85,32 97,35',
  Surprised:'M73,30 Q85,24 97,30',
  Disgust:  'M73,35 Q85,30 97,33',
}

const GLOW_COLORS = {
  Happy:    'rgba(48,217,136,0.12)',
  Angry:    'rgba(255,77,106,0.12)',
  Fear:     'rgba(245,166,35,0.10)',
  Sad:      'rgba(155,116,247,0.10)',
  Neutral:  'rgba(94,130,170,0.08)',
  Surprised:'rgba(14,201,168,0.10)',
  Disgust:  'rgba(255,107,119,0.10)',
}

export default function FacePanel({ result }) {
  const emotion = result?.facial_emotion || 'Neutral'
  const probs   = result?.facial_probs   || {}

  return (
    <div className="card processing-ring overflow-hidden glow-teal">
      <div className="panel-header">
        <span className="w-[7px] h-[7px] rounded-full bg-teal flex-shrink-0" />
        <span>Facial Expression Analysis</span>
        <span className="ml-auto text-teal font-mono text-[9px]">CNN / ViT</span>
      </div>

      <div className="p-4">
        {/* SVG Face */}
        <div className="flex justify-center mb-4 relative">
          <svg viewBox="0 0 130 130" className="w-[130px] h-[130px]">
            <ellipse cx="65" cy="65" rx="48" ry="52" fill="#0d2040" stroke="#1a3a6e" strokeWidth="1.5"/>
            <ellipse cx="45" cy="48" rx="12" ry="9"  fill="rgba(14,201,168,0.15)" stroke="rgba(14,201,168,0.3)" strokeWidth="0.5"/>
            <ellipse cx="85" cy="48" rx="12" ry="9"  fill="rgba(14,201,168,0.15)" stroke="rgba(14,201,168,0.3)" strokeWidth="0.5"/>
            <circle  cx="45" cy="48" r="4" fill="#0ec9a8" opacity="0.7"/>
            <circle  cx="85" cy="48" r="4" fill="#0ec9a8" opacity="0.7"/>
            <path d={BROW_L[emotion]} fill="none" stroke="rgba(14,201,168,0.5)" strokeWidth="2" strokeLinecap="round"
                  style={{ transition: 'd 0.6s ease' }}/>
            <path d={BROW_R[emotion]} fill="none" stroke="rgba(14,201,168,0.5)" strokeWidth="2" strokeLinecap="round"
                  style={{ transition: 'd 0.6s ease' }}/>
            <path d={MOUTH_PATHS[emotion]} fill="none" stroke="rgba(14,201,168,0.6)" strokeWidth="2.5" strokeLinecap="round"
                  style={{ transition: 'd 0.6s ease' }}/>
            <ellipse cx="65" cy="65" rx="48" ry="52" fill={GLOW_COLORS[emotion]} stroke="none"
                     style={{ transition: 'fill 0.6s ease' }}/>
            {/* scan grid */}
            <line x1="20" y1="65" x2="110" y2="65" stroke="rgba(14,201,168,0.06)" strokeWidth="0.5"/>
            <line x1="65" y1="13" x2="65"  y2="117" stroke="rgba(14,201,168,0.06)" strokeWidth="0.5"/>
            {/* corner brackets */}
            <path d="M17,22 L17,13 L26,13" fill="none" stroke="#0ec9a8" strokeWidth="1" opacity="0.5"/>
            <path d="M113,22 L113,13 L104,13" fill="none" stroke="#0ec9a8" strokeWidth="1" opacity="0.5"/>
            <path d="M17,108 L17,117 L26,117" fill="none" stroke="#0ec9a8" strokeWidth="1" opacity="0.5"/>
            <path d="M113,108 L113,117 L104,117" fill="none" stroke="#0ec9a8" strokeWidth="1" opacity="0.5"/>
          </svg>
          {/* scanline */}
          <div className="absolute left-0 right-0 h-px bg-teal opacity-60 scanline-animate pointer-events-none" />
        </div>

        {/* Emotion bars */}
        <div className="flex flex-col gap-[6px]">
          {EMOTION_ORDER.map(name => {
            const val = probs[name] ?? 0
            const color = EMOTION_COLORS[name]
            return (
              <div key={name} className="flex items-center gap-2">
                <span className="font-mono text-[10px] text-muted w-[62px] flex-shrink-0">{name}</span>
                <div className="flex-1 h-1 bg-[#1a3a6e40] rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-700 ease-out"
                    style={{ width: `${val * 100}%`, background: color }}
                  />
                </div>
                <span className="font-mono text-[10px] text-muted w-8 text-right">{(val*100).toFixed(0)}%</span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
