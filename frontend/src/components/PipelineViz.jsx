// src/components/PipelineViz.jsx
import React from 'react'

const STAGES = [
  { id: 0, label: 'Face Detection',     sub: 'MTCNN',          color: '#0ec9a8' },
  { id: 1, label: 'Visual Encoding',    sub: 'ResNet-50',       color: '#0ec9a8' },
  { id: 2, label: 'Text Encoding',      sub: 'RoBERTa-base',   color: '#f5a623' },
  { id: 3, label: 'Cross-Modal Fusion', sub: 'Attention Gate', color: '#9b74f7' },
  { id: 4, label: 'Emotion Output',     sub: '7-class + Risk', color: '#30d988' },
]

export default function PipelineViz({ activeStage }) {
  return (
    <div className="card p-4">
      <div className="section-label">5-Stage Inference Pipeline</div>

      <div className="flex items-center gap-0 overflow-x-auto pb-1">
        {STAGES.map((stage, i) => (
          <React.Fragment key={stage.id}>
            {/* Stage box */}
            <div
              className="flex flex-col items-center text-center flex-shrink-0 transition-all duration-500"
              style={{ minWidth: '100px' }}
            >
              {/* Icon / number */}
              <div
                className="w-8 h-8 rounded-full flex items-center justify-center mb-2 font-mono text-[11px] font-semibold border transition-all duration-500"
                style={{
                  background: activeStage >= stage.id ? `${stage.color}20` : 'transparent',
                  borderColor: activeStage >= stage.id ? stage.color : '#1a3a6e40',
                  color:       activeStage >= stage.id ? stage.color : '#3a5a7a',
                }}
              >
                {activeStage > stage.id ? '✓' : stage.id + 1}
              </div>
              <span
                className="font-mono text-[9px] leading-[1.4] transition-colors duration-500"
                style={{ color: activeStage >= stage.id ? '#cfe2f7' : '#3a5a7a' }}
              >
                {stage.label}
              </span>
              <span
                className="font-mono text-[8px] mt-0.5 transition-colors duration-500"
                style={{ color: activeStage >= stage.id ? stage.color : '#2a4a6a' }}
              >
                {stage.sub}
              </span>
            </div>

            {/* Connector arrow */}
            {i < STAGES.length - 1 && (
              <div className="flex-1 flex items-center justify-center px-1" style={{ minWidth: '24px' }}>
                <svg viewBox="0 0 24 8" className="w-6" style={{ opacity: activeStage > i ? 0.8 : 0.2 }}>
                  <line x1="0" y1="4" x2="18" y2="4"
                    stroke={activeStage > i ? STAGES[i].color : '#1a3a6e'}
                    strokeWidth="1.5"
                    style={{ transition: 'stroke 0.5s' }}
                  />
                  <polygon points="18,1 24,4 18,7"
                    fill={activeStage > i ? STAGES[i].color : '#1a3a6e'}
                    style={{ transition: 'fill 0.5s' }}
                  />
                </svg>
              </div>
            )}
          </React.Fragment>
        ))}
      </div>
    </div>
  )
}
