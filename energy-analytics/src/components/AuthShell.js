import React from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowLeft, ArrowUpRight, Check } from 'lucide-react';
import { EASE, Logo, StatusDot } from './ui';

const AuthShell = ({ children, eyebrow, title, description, image, imageAlt, quote, imageCaption }) => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-paper lg:grid lg:grid-cols-[0.92fr_1.08fr]">
      {/* Editorial panel */}
      <aside className="relative hidden overflow-hidden bg-ink text-paper lg:flex lg:flex-col lg:justify-between">
        <img
          src={image}
          alt={imageAlt}
          className="absolute inset-0 h-full w-full object-cover opacity-45"
          loading="eager"
        />
        <div className="absolute inset-0 bg-gradient-to-b from-ink/80 via-ink/55 to-ink" />
        <div className="absolute inset-0 grain" />
        <div className="pointer-events-none absolute inset-0 dot-grid-light opacity-25" />

        <div className="relative flex items-center justify-between p-10 xl:p-12">
          <button onClick={() => navigate('/')} aria-label="LoadIQ home">
            <Logo theme="light" />
          </button>
          <span className="label-sm flex items-center gap-2 text-paper/45">
            <StatusDot tone="lime" /> secure session
          </span>
        </div>

        <div className="relative px-10 pb-12 xl:px-12">
          <motion.div
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.9, ease: EASE }}
            className="max-w-lg"
          >
            <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/[0.04] px-3 py-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-signal" />
              <span className="label-sm text-paper/60">{imageCaption}</span>
            </div>
            <p className="display-serif text-[2.6rem] leading-[1.08] tracking-[-0.02em] text-paper xl:text-[3.2rem]">
              &ldquo;{quote}&rdquo;
            </p>
            <div className="mt-8 flex flex-wrap gap-x-8 gap-y-3">
              {['Render · Railway fallback', 'Firebase auth', 'Local-first history'].map((item) => (
                <span key={item} className="label-sm flex items-center gap-2 text-paper/40">
                  <Check size={12} className="text-signal" /> {item}
                </span>
              ))}
            </div>
          </motion.div>

          <div className="mt-12 flex items-end justify-between border-t border-white/10 pt-6">
            <div>
              <p className="label-sm text-paper/30">Grid intelligence platform</p>
              <p className="mt-2 text-sm text-paper/60">UK · USA · Germany · India</p>
            </div>
            <span className="mono-num text-xs text-paper/30">v4.2.1</span>
          </div>
        </div>
      </aside>

      {/* Form panel */}
      <main className="relative flex min-h-screen flex-col">
        <div className="flex items-center justify-between px-5 py-5 sm:px-8 lg:px-12">
          <button
            onClick={() => navigate('/')}
            className="group inline-flex items-center gap-2 text-[0.75rem] font-medium text-ink/50 transition-colors hover:text-ink"
          >
            <span className="flex h-8 w-8 items-center justify-center rounded-full border border-ink/15 transition-colors group-hover:border-ink/30">
              <ArrowLeft size={14} />
            </span>
            Back to home
          </button>
          <div className="lg:hidden">
            <Logo />
          </div>
          <span className="hidden text-[0.7rem] text-ink/35 sm:block">
            Need help? <span className="text-ink/60">support@loadiq.ai</span>
          </span>
        </div>

        <div className="flex flex-1 items-center justify-center px-5 pb-14 pt-6 sm:px-8 lg:px-12">
          <motion.div
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, ease: EASE }}
            className="w-full max-w-[27rem]"
          >
            <div className="mb-9">
              <div className="mb-4 flex items-center gap-2">
                <span className="h-1.5 w-1.5 rounded-full bg-moss" />
                <span className="label text-ink/40">{eyebrow}</span>
              </div>
              <h1 className="text-[2.1rem] leading-[1.04] tracking-[-0.05em] sm:text-[2.5rem]">{title}</h1>
              <p className="mt-3 text-sm leading-relaxed text-ink/50">{description}</p>
            </div>

            {children}

            <div className="mt-8 flex items-center justify-between border-t border-ink/10 pt-5 text-[0.68rem] text-ink/35">
              <span>Encrypted by Firebase Auth</span>
              <span className="flex items-center gap-1.5">
                Privacy <ArrowUpRight size={11} />
              </span>
            </div>
          </motion.div>
        </div>
      </main>
    </div>
  );
};

export default AuthShell;
