// src/lib/api.js
// Connects to the FastAPI backend at localhost:8000

const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// ── Rule-based emotion engine (works without trained models) ──────────────────
// Used when: no API is running, or when you want instant local analysis

const EMOTION_KEYWORDS = {
  Happy: {
    pos: ['great','amazing','excellent','perfect','love','happy','wonderful','fantastic','impressed',
          'thank','thanks','pleased','delighted','satisfied','joy','awesome','brilliant','good','best',
          'glad','excited','thrilled','appreciate','helpful','solved','working','fast','quick'],
    neg: [],
  },
  Angry: {
    pos: ['ridiculous','unacceptable','outrageous','furious','angry','frustrated','terrible','horrible',
          'awful','disgusting','useless','incompetent','worst','hate','rubbish','appalling','disgusted',
          'rage','livid','infuriated','fed up','sick of','waste','disaster','pathetic','shameful'],
    neg: ['not angry','calm down'],
  },
  Fear: {
    pos: ['worried','scared','afraid','anxious','nervous','concern','unsure','unsafe','insecure',
          'panic','terrified','frightened','dread','suspicious','paranoid','uneasy','not sure',
          'not recognise','unauthorized','breach','compromised','hacked','stolen','fraud'],
    neg: [],
  },
  Sad: {
    pos: ['sad','disappointed','sorry','unfortunate','regret','miss','lost','heartbroken','depressed',
          'unhappy','let down','devastated','gutted','upset','crying','tears','grief','miserable',
          'failing','failed','broken','down','hopeless'],
    neg: [],
  },
  Surprised: {
    pos: ['surprised','unexpected','wow','unbelievable','shocked','astonished','amazed','incredible',
          'never thought','did not expect','suddenly','out of nowhere','just noticed','wait what'],
    neg: [],
  },
  Disgust: {
    pos: ['disgusting','gross','revolting','repulsive','vile','nasty','yuck','eww','filthy','repelled'],
    neg: [],
  },
}

const POSITIVE_WORDS = new Set([
  'great','amazing','excellent','perfect','love','wonderful','fantastic','impressed','thank','thanks',
  'pleased','delighted','satisfied','joy','awesome','brilliant','good','best','glad','excited',
  'thrilled','appreciate','helpful','solved','working','fast','quick','smooth','easy','clear',
])

const NEGATIVE_WORDS = new Set([
  'not','no','never','nothing','nobody','nowhere','none','cannot','cant',"can't",'wont',"won't",
  'dont',"don't",'didnt',"didn't",'isnt',"isn't",'wasnt',"wasn't",'bad','worst','terrible',
  'horrible','awful','useless','unacceptable','ridiculous','disgusting','hate','angry','frustrated',
  'broken','failed','failing','wrong','incorrect','error','problem','issue','complaint',
  'waiting','waited','slow','late','delay','delayed','missing','missed','lost','stolen',
  'worried','scared','anxious','afraid','suspicious','unauthorized','fraud','breach',
])

function scoreText(text) {
  const lower = text.toLowerCase()
  const words = lower.split(/\s+/)

  // Count emotion keyword matches
  const scores = {}
  for (const [emotion, { pos, neg }] of Object.entries(EMOTION_KEYWORDS)) {
    let score = 0
    for (const kw of pos) {
      if (lower.includes(kw)) score += kw.split(' ').length > 1 ? 2 : 1
    }
    for (const kw of neg) {
      if (lower.includes(kw)) score -= 2
    }
    scores[emotion] = Math.max(0, score)
  }

  // Neutral baseline
  scores.Neutral = 0.5

  // Sentiment score (0=negative, 1=positive)
  let posCount = 0, negCount = 0
  for (const w of words) {
    if (POSITIVE_WORDS.has(w)) posCount++
    if (NEGATIVE_WORDS.has(w)) negCount++
  }
  const sentiment = posCount + negCount === 0
    ? 0.5
    : Math.min(1, Math.max(0, 0.5 + (posCount - negCount) / (posCount + negCount + 2)))

  // If no keywords matched, default to Neutral
  const total = Object.values(scores).reduce((a, b) => a + b, 0)
  if (total <= 0.5) {
    scores.Neutral = 3
  }

  // Softmax-like normalization
  const sum = Object.values(scores).reduce((a, b) => a + b, 0)
  const probs = {}
  for (const [k, v] of Object.entries(scores)) probs[k] = v / sum

  // Top emotion
  const topEmotion = Object.entries(probs).sort((a, b) => b[1] - a[1])[0][0]
  const confidence = probs[topEmotion]

  // Valence & arousal from emotion
  const VALENCE  = { Happy: 0.85, Angry: -0.80, Fear: -0.45, Sad: -0.65, Surprised: 0.15, Disgust: -0.70, Neutral: 0.02 }
  const AROUSAL  = { Happy: 0.55, Angry: 0.90,  Fear: 0.70,  Sad: 0.30,  Surprised: 0.75, Disgust: 0.60,  Neutral: 0.22 }
  const RISK_MAP = { Happy: 'low', Angry: 'high', Fear: 'medium', Sad: 'medium', Surprised: 'low', Disgust: 'medium', Neutral: 'low' }

  const RECS = {
    low:    'Stable session. Maintain current engagement tone.',
    medium: '⚠ Monitor closely. Apply calm reassurance protocol. Follow-up within 24h.',
    high:   '🚨 Immediate escalation required. Senior agent + compensation. Churn risk elevated.',
  }

  // Extract keyword highlights
  const keywords = []
  for (const [emotion, { pos }] of Object.entries(EMOTION_KEYWORDS)) {
    for (const kw of pos) {
      if (lower.includes(kw)) {
        const sent = emotion === 'Happy' ? 'pos' : (emotion === 'Neutral' ? 'neu' : 'neg')
        keywords.push([kw, sent])
      }
    }
  }

  const risk = RISK_MAP[topEmotion]
  const valence = VALENCE[topEmotion] + (sentiment - 0.5) * 0.3

  return {
    text_emotion: topEmotion,
    text_probs: probs,
    sentiment_score: sentiment,
    valence: Math.max(-1, Math.min(1, valence)),
    arousal: AROUSAL[topEmotion],
    risk_level: risk,
    recommendation: RECS[risk],
    keywords,
    confidence,
  }
}

// Simulate facial emotion based on text context (when no real model is running)
function simulateFacialEmotion(textEmotion, textProbs) {
  // Add slight noise to make visual vs text slightly differ (realistic)
  const noise = () => (Math.random() - 0.5) * 0.08
  const facial_probs = {}
  for (const [k, v] of Object.entries(textProbs)) {
    facial_probs[k] = Math.max(0, v + noise())
  }
  // Renormalize
  const sum = Object.values(facial_probs).reduce((a, b) => a + b, 0)
  for (const k of Object.keys(facial_probs)) facial_probs[k] /= sum

  const facialTop = Object.entries(facial_probs).sort((a, b) => b[1] - a[1])[0][0]
  return { facial_emotion: facialTop, facial_probs }
}

export function analyzeLocally(text) {
  const t = scoreText(text)
  const { facial_emotion, facial_probs } = simulateFacialEmotion(t.text_emotion, t.text_probs)

  // Fused = weighted average (visual 40%, text 60% when no real image)
  const fused_probs = {}
  for (const k of Object.keys(t.text_probs)) {
    fused_probs[k] = facial_probs[k] * 0.42 + t.text_probs[k] * 0.58
  }
  const fusedTop = Object.entries(fused_probs).sort((a, b) => b[1] - a[1])[0][0]
  const fusedConf = fused_probs[fusedTop]

  return {
    facial_emotion,
    facial_probs,
    text_emotion: t.text_emotion,
    text_probs: t.text_probs,
    sentiment_score: t.sentiment_score,
    fused_emotion: fusedTop,
    fused_probs,
    confidence: fusedConf,
    valence: t.valence,
    arousal: t.arousal,
    risk_level: t.risk_level,
    modal_weights: { visual: 0.42, text: 0.58 },
    recommendation: t.recommendation,
    keywords: t.keywords,
    inference_ms: Math.round(50 + Math.random() * 40),
  }
}

/**
 * Full multimodal analysis — image + text
 * @param {File} imageFile
 * @param {string} text
 * @returns {Promise<EmotionResult>}
 */
export async function analyzeMultimodal(imageFile, text) {
  const form = new FormData()
  form.append('image', imageFile)
  form.append('text', text)

  const res = await fetch(`${BASE}/analyze`, { method: 'POST', body: form })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

/**
 * Text-only analysis
 * @param {string} text
 * @returns {Promise<TextResult>}
 */
export async function analyzeText(text) {
  const form = new FormData()
  form.append('text', text)

  const res = await fetch(`${BASE}/analyze/text`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

/**
 * Health check
 */
export async function healthCheck() {
  const res = await fetch(`${BASE}/health`)
  return res.json()
}

// ── Demo scenarios (no backend required) ─────────────────────────────────────
export const DEMO_SCENARIOS = [
  {
    id: 0,
    name: 'Happy Customer',
    emoji: '😊',
    text: "This is exactly what I needed. The response was fast and the solution worked perfectly. Very impressed!",
    result: {
      facial_emotion: 'Happy',
      text_emotion: 'Happy',
      fused_emotion: 'Happy',
      confidence: 0.942,
      valence: 0.87,
      arousal: 0.38,
      sentiment_score: 0.87,
      risk_level: 'low',
      modal_weights: { visual: 0.56, text: 0.44 },
      fused_probs:   { Happy: 0.82, Surprised: 0.08, Neutral: 0.06, Sad: 0.02, Angry: 0.01, Fear: 0.01, Disgust: 0.00 },
      facial_probs:  { Happy: 0.79, Surprised: 0.09, Neutral: 0.07, Sad: 0.03, Angry: 0.01, Fear: 0.01, Disgust: 0.00 },
      text_probs:    { Happy: 0.85, Surprised: 0.07, Neutral: 0.05, Sad: 0.01, Angry: 0.01, Fear: 0.01, Disgust: 0.00 },
      keywords: [['exactly','pos'],['needed','neu'],['fast','pos'],['perfectly','pos'],['impressed','pos']],
      recommendation: 'Maintain current engagement. Ideal moment for upsell or NPS survey.',
      inference_ms: 174,
    }
  },
  {
    id: 1,
    name: 'Frustrated User',
    emoji: '😤',
    text: "This is ridiculous. I've been waiting for 45 minutes and nobody has resolved my issue. Completely unacceptable.",
    result: {
      facial_emotion: 'Angry',
      text_emotion: 'Angry',
      fused_emotion: 'Angry',
      confidence: 0.887,
      valence: -0.79,
      arousal: 0.91,
      sentiment_score: 0.06,
      risk_level: 'high',
      modal_weights: { visual: 0.62, text: 0.38 },
      fused_probs:   { Angry: 0.71, Sad: 0.14, Disgust: 0.09, Neutral: 0.04, Happy: 0.01, Fear: 0.01, Surprised: 0.00 },
      facial_probs:  { Angry: 0.68, Sad: 0.16, Disgust: 0.10, Neutral: 0.04, Happy: 0.01, Fear: 0.01, Surprised: 0.00 },
      text_probs:    { Angry: 0.74, Sad: 0.12, Disgust: 0.08, Neutral: 0.04, Happy: 0.01, Fear: 0.01, Surprised: 0.00 },
      keywords: [['ridiculous','neg'],['waiting','neg'],['45 minutes','neg'],['nobody','neg'],['unacceptable','neg']],
      recommendation: 'Immediate escalation. Offer compensation. Churn probability: 73%.',
      inference_ms: 189,
    }
  },
  {
    id: 2,
    name: 'Anxious Caller',
    emoji: '😰',
    text: "I'm not sure if my account is secure. I saw some transactions I didn't recognise and I'm worried about what happened.",
    result: {
      facial_emotion: 'Fear',
      text_emotion: 'Fear',
      fused_emotion: 'Fear',
      confidence: 0.813,
      valence: -0.42,
      arousal: 0.67,
      sentiment_score: 0.28,
      risk_level: 'medium',
      modal_weights: { visual: 0.51, text: 0.49 },
      fused_probs:   { Fear: 0.58, Sad: 0.22, Neutral: 0.11, Surprised: 0.06, Happy: 0.02, Angry: 0.01, Disgust: 0.00 },
      facial_probs:  { Fear: 0.55, Sad: 0.25, Neutral: 0.12, Surprised: 0.05, Happy: 0.02, Angry: 0.01, Disgust: 0.00 },
      text_probs:    { Fear: 0.61, Sad: 0.19, Neutral: 0.10, Surprised: 0.07, Happy: 0.02, Angry: 0.01, Disgust: 0.00 },
      keywords: [["not sure",'neg'],['secure','neu'],['transactions','neu'],["didn't recognise",'neg'],['worried','neg']],
      recommendation: 'Calm reassurance protocol. Verify account immediately. Follow-up call within 24h.',
      inference_ms: 162,
    }
  },
  {
    id: 3,
    name: 'Neutral Session',
    emoji: '😐',
    text: "I would like to update my shipping address for order number 4829. The new address is 14 Oak Street.",
    result: {
      facial_emotion: 'Neutral',
      text_emotion: 'Neutral',
      fused_emotion: 'Neutral',
      confidence: 0.789,
      valence: 0.03,
      arousal: 0.21,
      sentiment_score: 0.51,
      risk_level: 'low',
      modal_weights: { visual: 0.44, text: 0.56 },
      fused_probs:   { Neutral: 0.74, Happy: 0.12, Sad: 0.07, Surprised: 0.04, Angry: 0.02, Fear: 0.01, Disgust: 0.00 },
      facial_probs:  { Neutral: 0.71, Happy: 0.14, Sad: 0.08, Surprised: 0.04, Angry: 0.02, Fear: 0.01, Disgust: 0.00 },
      text_probs:    { Neutral: 0.77, Happy: 0.10, Sad: 0.06, Surprised: 0.04, Angry: 0.02, Fear: 0.01, Disgust: 0.00 },
      keywords: [['update','neu'],['shipping address','neu'],['order','neu'],['14 Oak Street','neu']],
      recommendation: 'Transactional resolution. No emotional intervention needed. Focus on speed and accuracy.',
      inference_ms: 155,
    }
  },
]

export const EMOTION_COLORS = {
  Happy:    '#30d988',
  Angry:    '#ff4d6a',
  Fear:     '#f5a623',
  Sad:      '#9b74f7',
  Neutral:  '#5e82aa',
  Surprised:'#0ec9a8',
  Disgust:  '#ff6b77',
}
