// src/components/Header.jsx
import React, { useEffect, useState } from 'react'
import { healthCheck } from '../lib/api'

const NAV_LINKS = [
  { label: 'Dashboard', action: () => window.scrollTo({ top: 0, behavior: 'smooth' }) },
  { label: 'Pipeline',  action: () => document.getElementById('pipeline-section')?.scrollIntoView({ behavior: 'smooth' }) },
  { label: 'Docs',      action: () => window.open('http://localhost:8000/docs', '_blank') },
  { label: 'GitHub',    action: () => window.open('https://github.com/', '_blank') },
]

export default function Header() {
  const [apiStatus, setApiStatus] = useState('checking')
  const [activeNav, setActiveNav] = useState('Dashboard')

  useEffect(() => {
    healthCheck()
      .then(() => setApiStatus('online'))
      .catch(() => setApiStatus('offline'))
  }, [])

  const statusDot = {
    checking: 'bg-amber animate-pulse',
    online:   'bg-green',
    offline:  'bg-red',
  }[apiStatus]

  const statusLabel = {
    checking: 'Connecting…',
    online:   'API Online',
    offline:  'Demo Mode',
  }[apiStatus]

  return (
    <header className="flex items-center justify-between px-6 py-4 border-b border-[#1a3a6e40] bg-[#050d1b] sticky top-0 z-50">
      {/* Logo */}
      <div
        className="flex items-center gap-3 cursor-pointer"
        onClick={() => { setActiveNav('Dashboard'); window.scrollTo({ top: 0, behavior: 'smooth' }) }}
      >
        <div className="w-8 h-8 rounded-md bg-teal flex items-center justify-center">
          <span className="font-mono font-bold text-bg text-[11px]">EM</span>
        </div>
        <div>
          <div className="font-display font-semibold text-[15px] gradient-text">perceive_ai</div>
          <div className="font-mono text-[9px] text-muted tracking-[0.1em]">MULTIMODAL EMOTIONAL INTELLIGENCE</div>
        </div>
      </div>

      {/* Center nav */}
      <nav className="hidden md:flex items-center gap-1">
        {NAV_LINKS.map(({ label, action }) => {
          const isActive = activeNav === label
          return (
            <button
              key={label}
              onClick={() => { setActiveNav(label); action() }}
              className={`
                font-mono text-[11px] px-4 py-2 rounded-md border transition-all duration-150 cursor-pointer
                ${isActive
                  ? 'text-teal bg-[#0ec9a815] border-[#0ec9a830]'
                  : 'text-muted bg-transparent border-transparent hover:text-[#cfe2f7] hover:bg-[#0c1e38]'}
              `}
            >
              {label}
            </button>
          )
        })}
      </nav>

      {/* Status */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5 bg-[#081527] border border-[#1a3a6e40] rounded-full px-3 py-1.5">
          <span className={`w-[6px] h-[6px] rounded-full flex-shrink-0 ${statusDot}`} />
          <span className="font-mono text-[10px] text-muted">{statusLabel}</span>
        </div>
        <button
          onClick={() => window.open('http://localhost:8000/docs', '_blank')}
          className="hidden sm:flex items-center gap-1 bg-[#081527] border border-[#1a3a6e40] rounded-full px-3 py-1.5 cursor-pointer hover:border-teal transition-colors duration-150"
        >
          <span className="font-mono text-[10px] text-muted">FastAPI :8000</span>
        </button>
      </div>
    </header>
  )
}
