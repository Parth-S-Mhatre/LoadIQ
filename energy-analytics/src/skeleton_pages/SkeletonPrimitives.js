import React from 'react';

const pulse = 'animate-pulse bg-white/5';

export const SkeletonBlock = ({ className = '' }) => (
    <div className={`${pulse} ${className}`.trim()} />
);

export const SkeletonLine = ({ className = '' }) => (
    <SkeletonBlock className={`h-3 rounded-full ${className}`.trim()} />
);

export const SkeletonCard = ({ className = '' }) => (
    <div className={`rounded-3xl border border-white/5 bg-slate-900/60 ${className}`.trim()} />
);

export const SkeletonReason = ({ title, subtitle }) => (
    <div className="mb-8 flex flex-col gap-3">
        <div className="inline-flex w-fit items-center gap-2 rounded-full border border-indigo-400/20 bg-indigo-500/10 px-4 py-2 text-[11px] font-bold uppercase tracking-[0.24em] text-indigo-200">
            <div className="h-2 w-2 rounded-full bg-indigo-400 animate-pulse" />
            Loading Optimized View
        </div>
        <div>
            <h2 className="text-2xl font-black text-white">{title}</h2>
            <p className="mt-2 max-w-2xl text-sm text-slate-400">{subtitle}</p>
        </div>
    </div>
);
