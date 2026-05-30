// src/components/ScenarioSelector.jsx
import React from 'react'
import { DEMO_SCENARIOS } from '../lib/api'

export default function ScenarioSelector({ activeId, onSelect }) {
  return (
    <div className="flex items-center gap-2 flex-wrap">
      <span className="font-mono text-[10px] text-muted tracking-[0.12em] uppercase mr-1">
        Demo Scenarios
      </span>
      {DEMO_SCENARIOS.map(s => (
        <button
          key={s.id}
          onClick={() => onSelect(s.id)}
          className={`
            flex items-center gap-1.5 px-3 py-1.5 rounded-md border
            font-mono text-[11px] transition-all duration-200 cursor-pointer
            ${activeId === s.id
              ? 'bg-[#0ec9a815] border-teal text-teal'
              : 'bg-transparent border-[#1a3a6e50] text-muted hover:border-[#1a3a6e] hover:text-[#cfe2f7]'}
          `}
        >
          <span>{s.emoji}</span>
          <span>{s.name}</span>
        </button>
      ))}
    </div>
  )
}
