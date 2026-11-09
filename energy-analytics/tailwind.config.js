/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        "./src/**/*.{js,jsx,ts,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                paper: {
                    DEFAULT: '#F2F0E9',
                    50: '#FAF9F5',
                    100: '#F6F4EE',
                    200: '#EDEAE1',
                    300: '#E2DED2',
                    400: '#D2CDBE',
                },
                ink: {
                    DEFAULT: '#111813',
                    50: '#F4F5F3',
                    100: '#E6E8E4',
                    200: '#C9CEC7',
                    300: '#9AA398',
                    400: '#6C776C',
                    500: '#4B564C',
                    600: '#334036',
                    700: '#24302A',
                    800: '#18231D',
                    900: '#0E1712',
                    950: '#080F0B',
                },
                signal: {
                    DEFAULT: '#D8F24D',
                    50: '#F8FCE4',
                    100: '#EFF9BC',
                    200: '#E3F483',
                    300: '#D8F24D',
                    400: '#C4DF2F',
                    500: '#A5C01D',
                    600: '#819717',
                },
                moss: {
                    DEFAULT: '#0F3D2E',
                    light: '#1B5A45',
                    dark: '#0A2A20',
                },
                clay: {
                    DEFAULT: '#C75B32',
                    light: '#E17A52',
                },
                sky: {
                    DEFAULT: '#3A6B8A',
                    light: '#5E92B3',
                },
            },
            fontFamily: {
                sans: ['Archivo', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'sans-serif'],
                display: ['"Instrument Serif"', 'Georgia', 'serif'],
                mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
            },
            letterSpacing: {
                tightest: '-0.055em',
            },
            boxShadow: {
                lift: '0 22px 48px -28px rgba(17, 24, 19, 0.35)',
                panel: '0 1px 0 rgba(17, 24, 19, 0.04), 0 18px 40px -32px rgba(17, 24, 19, 0.45)',
                insetline: 'inset 0 0 0 1px rgba(17, 24, 19, 0.08)',
            },
            keyframes: {
                marquee: {
                    '0%': { transform: 'translateX(0)' },
                    '100%': { transform: 'translateX(-50%)' },
                },
                shimmer: {
                    '0%': { backgroundPosition: '200% 0' },
                    '100%': { backgroundPosition: '-200% 0' },
                },
                drawLine: {
                    '0%': { strokeDashoffset: '1000' },
                    '100%': { strokeDashoffset: '0' },
                },
                floatSlow: {
                    '0%, 100%': { transform: 'translateY(0)' },
                    '50%': { transform: 'translateY(-8px)' },
                },
                blink: {
                    '0%, 100%': { opacity: '1' },
                    '50%': { opacity: '0.25' },
                },
                scan: {
                    '0%': { transform: 'translateY(-100%)' },
                    '100%': { transform: 'translateY(400%)' },
                },
            },
            animation: {
                marquee: 'marquee 38s linear infinite',
                shimmer: 'shimmer 1.8s ease-in-out infinite',
                'draw-line': 'drawLine 1.8s cubic-bezier(0.16, 1, 0.3, 1) forwards',
                'float-slow': 'floatSlow 7s ease-in-out infinite',
                blink: 'blink 2.4s ease-in-out infinite',
                scan: 'scan 5s linear infinite',
            },
        },
    },
    plugins: [],
}
