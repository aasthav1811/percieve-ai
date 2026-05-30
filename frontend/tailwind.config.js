/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        display: ['"Sora"', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'monospace'],
      },
      colors: {
        bg:      '#050d1b',
        surface: '#081527',
        card:    '#0c1e38',
        card2:   '#0f2548',
        teal:    '#0ec9a8',
        amber:   '#f5a623',
        purple:  '#9b74f7',
        coral:   '#ff6b77',
        muted:   '#5e82aa',
        green:   '#30d988',
        red:     '#ff4d6a',
      },
      animation: {
        'pulse-slow': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'scan': 'scan 2.5s ease-in-out infinite',
        'float': 'float 3s ease-in-out infinite',
      },
      keyframes: {
        scan: {
          '0%':   { top: '0%', opacity: '0' },
          '10%':  { opacity: '0.7' },
          '90%':  { opacity: '0.7' },
          '100%': { top: '100%', opacity: '0' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%':      { transform: 'translateY(-6px)' },
        },
      },
    },
  },
  plugins: [],
}
