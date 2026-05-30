// src/hooks/useTypewriter.js
import { useState, useEffect, useRef } from 'react'

export function useTypewriter(text, speed = 22) {
  const [displayed, setDisplayed] = useState('')
  const timerRef = useRef(null)

  useEffect(() => {
    setDisplayed('')
    if (!text) return
    let i = 0
    timerRef.current = setInterval(() => {
      if (i < text.length) {
        setDisplayed(text.slice(0, ++i))
      } else {
        clearInterval(timerRef.current)
      }
    }, speed)
    return () => clearInterval(timerRef.current)
  }, [text, speed])

  return displayed
}
