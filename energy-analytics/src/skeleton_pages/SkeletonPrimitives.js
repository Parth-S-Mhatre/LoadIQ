import React from 'react';

const pulse = 'skeleton';

export const SkeletonBlock = ({ className = '' }) => (
    <div className={`${pulse} ${className}`.trim()} />
);

export const SkeletonLine = ({ className = '' }) => (
    <SkeletonBlock className={`h-3 rounded-full ${className}`.trim()} />
);

export const SkeletonCard = ({ className = '' }) => (
    <div className={`rounded-2xl border border-ink/10 bg-paper-50 ${className}`.trim()} />
);

export const SkeletonReason = ({ title, subtitle }) => (
    <div className="mb-8 flex flex-col gap-3">
        <div className="inline-flex w-fit items-center gap-2 rounded-full border border-ink/10 bg-paper-50 px-3.5 py-1.5">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-moss" />
            <span className="label-sm text-ink/45">Loading optimised view</span>
        </div>
        <div>
            <h2 className="text-xl font-medium tracking-[-0.03em] text-ink">{title}</h2>
            <p className="mt-2 max-w-2xl text-xs leading-relaxed text-ink/45">{subtitle}</p>
        </div>
    </div>
);
