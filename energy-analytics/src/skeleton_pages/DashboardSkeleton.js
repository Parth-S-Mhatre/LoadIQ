import React from 'react';
import { SkeletonBlock, SkeletonCard, SkeletonLine, SkeletonReason } from './SkeletonPrimitives';

const DashboardSkeleton = ({ backendMode = false }) => {
    return (
        <div className="min-h-screen bg-[#07101d] text-white">
            <aside className="fixed bottom-0 left-0 right-0 z-20 flex h-20 items-center justify-around border-t border-white/5 bg-[#0B1121] px-6 md:top-0 md:h-screen md:w-24 md:flex-col md:justify-start md:gap-4 md:border-r md:border-t-0 md:px-0 md:py-8">
                <SkeletonBlock className="hidden h-12 w-12 rounded-[14px] md:block" />
                {[...Array(6)].map((_, index) => (
                    <SkeletonBlock key={index} className="h-12 w-12 rounded-2xl" />
                ))}
            </aside>

            <main className="min-h-screen p-6 pb-32 md:ml-24 md:p-8 md:pb-12 lg:p-10">
                <SkeletonReason
                    title={backendMode ? 'Waiting for backend readiness' : 'Preparing dashboard layout'}
                    subtitle={backendMode
                        ? 'This route depends on the Render backend. The skeleton stays in place until the service responds, while keeping the page structure stable.'
                        : 'The dashboard shell renders first so users on slower networks get a stable layout before the heavier visuals arrive.'}
                />

                <div className="mb-8 flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
                    <div className="space-y-3">
                        <SkeletonLine className="w-44" />
                        <SkeletonLine className="w-80 max-w-full" />
                    </div>
                    <div className="flex flex-wrap gap-4">
                        <SkeletonBlock className="h-11 w-36 rounded-xl" />
                        <SkeletonBlock className="h-11 w-36 rounded-xl" />
                    </div>
                </div>

                <div className="grid gap-4 md:grid-cols-4">
                    {[...Array(4)].map((_, index) => (
                        <SkeletonCard key={index} className="p-6">
                            <SkeletonLine className="mb-4 w-24" />
                            <SkeletonBlock className="h-10 w-28 rounded-2xl" />
                        </SkeletonCard>
                    ))}
                </div>

                <div className="mt-6 grid gap-6 lg:grid-cols-[1.35fr_0.65fr]">
                    <SkeletonCard className="p-8">
                        <SkeletonLine className="mb-4 w-40" />
                        <SkeletonBlock className="h-[340px] w-full rounded-3xl" />
                    </SkeletonCard>
                    <SkeletonCard className="p-8">
                        <SkeletonLine className="mb-6 w-36" />
                        <div className="space-y-5">
                            {[...Array(5)].map((_, index) => (
                                <div key={index} className="space-y-3">
                                    <SkeletonLine className="w-28" />
                                    <SkeletonBlock className="h-3 w-full rounded-full" />
                                </div>
                            ))}
                        </div>
                    </SkeletonCard>
                </div>

                <div className="mt-6 grid gap-6 lg:grid-cols-[0.72fr_1.28fr]">
                    <SkeletonCard className="p-6">
                        <SkeletonLine className="mb-4 w-32" />
                        <div className="space-y-4">
                            {[...Array(8)].map((_, index) => (
                                <SkeletonBlock key={index} className="h-10 w-full rounded-xl" />
                            ))}
                        </div>
                    </SkeletonCard>
                    <SkeletonCard className="p-6">
                        <SkeletonLine className="mb-4 w-44" />
                        <SkeletonBlock className="h-[420px] w-full rounded-3xl" />
                    </SkeletonCard>
                </div>
            </main>
        </div>
    );
};

export default DashboardSkeleton;
