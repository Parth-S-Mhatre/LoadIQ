import React from 'react';
import { SkeletonBlock, SkeletonCard, SkeletonLine, SkeletonReason } from './SkeletonPrimitives';

const LandingSkeleton = () => {
    return (
        <div className="min-h-screen overflow-hidden bg-paper px-5 text-ink sm:px-8">
            <div className="mx-auto flex h-[4.5rem] max-w-7xl items-center justify-between border-b border-ink/10">
                <div className="flex items-center gap-3">
                    <SkeletonBlock className="h-9 w-9 rounded-[10px]" />
                    <div className="space-y-2">
                        <SkeletonLine className="w-24" />
                        <SkeletonLine className="w-16" />
                    </div>
                </div>
                <div className="hidden gap-6 md:flex">
                    <SkeletonLine className="w-20" />
                    <SkeletonLine className="w-24" />
                    <SkeletonLine className="w-20" />
                </div>
                <SkeletonBlock className="h-11 w-36 rounded-full" />
            </div>

            <div className="mx-auto flex min-h-[calc(100vh-5rem)] max-w-7xl flex-col justify-center py-16">
                <SkeletonReason
                    title="Preparing the landing experience"
                    subtitle="We are shaping the lightweight hero structure first so the full page can render smoothly on slower connections."
                />

                <div className="grid items-center gap-12 lg:grid-cols-[1.2fr_0.8fr]">
                    <div className="space-y-6">
                        <SkeletonBlock className="h-8 w-52 rounded-full" />
                        <SkeletonBlock className="h-14 w-full max-w-2xl rounded-xl" />
                        <SkeletonBlock className="h-14 w-4/5 rounded-xl" />
                        <div className="space-y-3">
                            <SkeletonLine className="w-full max-w-2xl" />
                            <SkeletonLine className="w-5/6 max-w-xl" />
                        </div>
                        <div className="flex flex-col gap-4 sm:flex-row">
                            <SkeletonBlock className="h-12 w-48 rounded-full" />
                            <SkeletonBlock className="h-12 w-44 rounded-full" />
                        </div>
                    </div>

                    <SkeletonCard className="overflow-hidden p-0">
                        <SkeletonBlock className="h-[26rem] w-full" />
                    </SkeletonCard>
                </div>

                <div className="mt-16 grid gap-3 md:grid-cols-3">
                    {[...Array(3)].map((_, index) => (
                        <SkeletonCard key={index} className="p-7">
                            <SkeletonBlock className="mb-6 h-11 w-11 rounded-xl" />
                            <SkeletonLine className="mb-3 w-40" />
                            <SkeletonLine className="mb-2 w-full" />
                            <SkeletonLine className="w-5/6" />
                        </SkeletonCard>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default LandingSkeleton;
