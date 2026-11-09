import React, { useEffect, useId, useRef, useState } from 'react';
import { motion, useInView, animate, useReducedMotion } from 'framer-motion';
import logoMark from '../../assets/loadiq-mark.png';

export const cn = (...classes) => classes.filter(Boolean).join(' ');

export const EASE = [0.16, 1, 0.3, 1];

export const Logo = ({ theme = 'dark', compact = false, className = '' }) => {
  const isLight = theme === 'light';

  return (
    <div className={cn('flex items-center gap-3', className)}>
      <div className="relative flex h-9 w-9 shrink-0 items-center justify-center overflow-hidden rounded-[10px]">
        <img
          src={logoMark}
          alt=""
          aria-hidden="true"
          draggable="false"
          className="h-full w-full object-cover"
        />
      </div>
      {!compact && (
        <div className="leading-none">
          <p className={cn('text-[1.05rem] font-semibold tracking-[-0.05em]', isLight ? 'text-paper' : 'text-ink')}>
            Load<span className="font-normal">IQ</span>
          </p>
          <p className={cn('label-sm mt-1.5', isLight ? 'text-paper/45' : 'text-ink/45')}>
            Grid Intelligence
          </p>
        </div>
      )}
    </div>
  );
};

export const buttonClass = (variant = 'primary', size = 'md', extra = '') => {
  const base =
    'group inline-flex items-center justify-center gap-2 rounded-full font-semibold transition-all duration-300 disabled:cursor-not-allowed disabled:opacity-50';
  const sizes = {
    sm: 'h-9 px-4 text-[0.75rem]',
    md: 'h-11 px-5 text-[0.8125rem]',
    lg: 'h-[3.25rem] px-7 text-sm',
  };
  const variants = {
    primary: 'bg-ink text-paper hover:bg-ink-800 shadow-[0_10px_24px_-16px_rgba(17,24,19,0.8)] active:scale-[0.985]',
    signal: 'bg-signal text-ink hover:bg-signal-400 active:scale-[0.985]',
    outline: 'border border-ink/20 bg-transparent text-ink hover:border-ink/45 hover:bg-ink/[0.04] active:scale-[0.985]',
    soft: 'bg-ink/[0.06] text-ink hover:bg-ink/[0.1] active:scale-[0.985]',
    ghost: 'text-ink hover:bg-ink/[0.06]',
    light: 'bg-paper text-ink hover:bg-white active:scale-[0.985]',
    darkOutline: 'border border-paper/20 bg-transparent text-paper hover:border-paper/40 hover:bg-paper/10 active:scale-[0.985]',
    danger: 'bg-clay text-white hover:bg-clay-light active:scale-[0.985]',
  };

  return cn(base, sizes[size] || sizes.md, variants[variant] || variants.primary, extra);
};

export const Button = ({ variant, size, className = '', children, ...props }) => (
  <button className={buttonClass(variant, size, className)} {...props}>
    {children}
  </button>
);

export const Badge = ({ children, tone = 'neutral', className = '' }) => {
  const tones = {
    neutral: 'border-ink/15 bg-transparent text-ink/60',
    solid: 'border-ink bg-ink text-paper',
    signal: 'border-signal-400 bg-signal/35 text-ink',
    moss: 'border-moss/20 bg-moss/[0.07] text-moss',
    clay: 'border-clay/25 bg-clay/[0.08] text-clay',
    danger: 'border-clay/25 bg-clay/[0.08] text-clay',
    light: 'border-paper/20 bg-paper/10 text-paper/80',
  };

  return (
    <span
      className={cn(
        'label inline-flex items-center gap-2 rounded-full border px-3 py-1.5',
        tones[tone] || tones.neutral,
        className
      )}
    >
      {children}
    </span>
  );
};

export const StatusDot = ({ tone = 'green', pulse = true, className = '' }) => {
  const tones = {
    green: 'bg-emerald-600 text-emerald-600',
    lime: 'bg-signal-400 text-signal-400',
    amber: 'bg-amber-500 text-amber-500',
    red: 'bg-clay text-clay',
    muted: 'bg-ink/25 text-ink/25',
  };

  return (
    <span className={cn('status-dot', tones[tone] || tones.green, className)}>
      {pulse && <span className="absolute inset-0 animate-ping rounded-full bg-current opacity-30" />}
    </span>
  );
};

export const Reveal = ({
  children,
  delay = 0,
  y = 24,
  className = '',
  once = true,
  amount = 0.2,
  as = 'div',
}) => {
  const reduceMotion = useReducedMotion();
  const MotionTag = motion[as] || motion.div;

  return (
    <MotionTag
      className={className}
      initial={reduceMotion ? false : { opacity: 0, y }}
      whileInView={reduceMotion ? undefined : { opacity: 1, y: 0 }}
      viewport={{ once, amount }}
      transition={{ duration: 0.8, delay, ease: EASE }}
    >
      {children}
    </MotionTag>
  );
};

export const Stagger = ({ children, className = '', delayChildren = 0.06, staggerChildren = 0.08 }) => (
  <motion.div
    className={className}
    initial="hidden"
    whileInView="visible"
    viewport={{ once: true, amount: 0.15 }}
    variants={{
      hidden: {},
      visible: { transition: { delayChildren, staggerChildren } },
    }}
  >
    {children}
  </motion.div>
);

export const StaggerItem = ({ children, className = '', y = 22 }) => (
  <motion.div
    className={className}
    variants={{
      hidden: { opacity: 0, y },
      visible: { opacity: 1, y: 0, transition: { duration: 0.72, ease: EASE } },
    }}
  >
    {children}
  </motion.div>
);

export const CountUp = ({ value, decimals = 0, duration = 1.6, prefix = '', suffix = '', className = '' }) => {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: '-40px' });
  const reduceMotion = useReducedMotion();
  const [display, setDisplay] = useState(reduceMotion ? value : 0);

  useEffect(() => {
    if (!inView) return undefined;

    if (reduceMotion) {
      setDisplay(value);
      return undefined;
    }

    const controls = animate(0, Number(value) || 0, {
      duration,
      ease: EASE,
      onUpdate: (latest) => setDisplay(latest),
    });

    return () => controls.stop();
  }, [inView, value, duration, reduceMotion]);

  const formatted = Number(display).toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });

  return (
    <span ref={ref} className={className}>
      {prefix}{formatted}{suffix}
    </span>
  );
};

export const Sparkline = ({
  data = [],
  width = 120,
  height = 36,
  stroke = '#111813',
  fill = 'rgba(216, 242, 77, 0.35)',
  strokeWidth = 1.8,
  className = '',
}) => {
  const id = useId();
  const safeData = data.length > 1 ? data : [0, 0];
  const min = Math.min(...safeData);
  const max = Math.max(...safeData);
  const range = max - min || 1;
  const step = width / (safeData.length - 1);
  const points = safeData.map((value, index) => {
    const x = index * step;
    const y = height - 3 - ((value - min) / range) * (height - 6);
    return [x, y];
  });
  const path = points.map(([x, y], index) => `${index === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`).join(' ');
  const area = `${path} L${width},${height} L0,${height} Z`;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className={className} aria-hidden="true" preserveAspectRatio="none">
      <defs>
        <linearGradient id={`spark-${id}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={fill} />
          <stop offset="100%" stopColor={fill} stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#spark-${id})`} />
      <path d={path} fill="none" stroke={stroke} strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
    </svg>
  );
};

export const SectionIntro = ({ eyebrow, title, description, className = '', titleClassName = '', align = 'left' }) => (
  <div className={cn(align === 'center' && 'text-center', className)}>
    {eyebrow && (
      <div className={cn('mb-5 flex items-center gap-3', align === 'center' && 'justify-center')}>
        <span className="h-1.5 w-1.5 rounded-full bg-signal-500" />
        <span className="label text-ink/50">{eyebrow}</span>
      </div>
    )}
    <h2 className={cn('max-w-4xl text-[2.25rem] leading-[1.02] tracking-[-0.045em] sm:text-5xl lg:text-[3.5rem]', titleClassName)}>
      {title}
    </h2>
    {description && (
      <p className={cn('mt-6 max-w-2xl text-[0.95rem] leading-relaxed text-ink/60 sm:text-base', align === 'center' && 'mx-auto')}>
        {description}
      </p>
    )}
  </div>
);

export const Rule = ({ className = '' }) => <div className={cn('h-px w-full bg-ink/10', className)} />;

export const ArrowUpRight = ({ size = 15, className = '' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className={className} aria-hidden="true">
    <path d="M7 17 17 7M8 7h9v9" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

export const Ticker = ({ items = [], className = '' }) => {
  const doubled = [...items, ...items];

  return (
    <div className={cn('overflow-hidden', className)}>
      <div className="marquee">
        {doubled.map((item, index) => (
          <span key={`${item}-${index}`} className="label flex items-center gap-8 whitespace-nowrap px-5 py-4 text-paper/55">
            {item}
            <span className="h-1 w-1 rounded-full bg-signal-400" />
          </span>
        ))}
      </div>
    </div>
  );
};
