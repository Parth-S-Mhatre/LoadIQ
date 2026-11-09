import React from 'react';
import { SkeletonBlock, SkeletonCard, SkeletonLine, SkeletonReason } from './SkeletonPrimitives';

const DashboardSkeleton = ({ backendMode = false }) => {
    return (
        <div className="min-h-screen bg-paper text-ink">
            <aside className="fixed bottom-0 left-0 right-0 z-20 flex h-20 items-center justify-around border-t border-white/10 bg-ink px-6 lg:top-0 lg:h-screen lg:w-[16rem] lg:flex-col lg:items-stretch lg:justify-start lg:gap-3 lg:border-r lg:border-t-0 lg:px-4 lg:py-6">
                <SkeletonBlock className="skeleton-dark hidden h-9 w-32 rounded-lg lg:block" />
                {[...Array(6)].map((_, index) => (
                    <SkeletonBlock key={index} className="skeleton-dark h-11 w-11 rounded-xl lg:w-full" />
                ))}
            </aside>

            <main className="min-h-screen p-4 pb-32 sm:p-6 lg:ml-[16rem] lg:p-8">
                <SkeletonReason
                    title={backendMode ? 'Waiting for backend readiness' : 'Preparing dashboard layout'}
                    subtitle={backendMode
                        ? 'This route depends on the Render backend. The skeleton holds the structure until the service responds.'
                        : 'The dashboard shell renders first so slower networks get a stable layout before the heavier visuals arrive.'}
                />

                <div className="mb-6 flex flex-col gap-5 sm:flex-row sm:items-center sm:justify-between">
                    <div className="space-y-3">
                        <SkeletonLine className="w-36" />
                        <SkeletonLine className="w-72 max-w-full" />
                    </div>
                    <div className="flex flex-wrap gap-3">
                        <SkeletonBlock className="h-10 w-32 rounded-full" />
                        <SkeletonBlock className="h-10 w-32 rounded-full" />
                    </div>
                </div>

                <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                    {[...Array(4)].map((_, index) => (
                        <SkeletonCard key={index} className="p-5">
                            <SkeletonLine className="mb-5 w-24" />
                            <SkeletonBlock className="h-9 w-28 rounded-lg" />
                        </SkeletonCard>
                    ))}
                </div>

                <div className="mt-3 grid gap-3 xl:grid-cols-[1.6fr_1fr]">
                    <SkeletonCard className="p-6">
                        <SkeletonLine className="mb-4 w-40" />
                        <SkeletonBlock className="h-[20rem] w-full rounded-xl" />
                    </SkeletonCard>
                    <SkeletonCard className="p-6">
                        <SkeletonLine className="mb-6 w-36" />
                        <div className="space-y-5">
                            {[...Array(5)].map((_, index) => (
                                <div key={index} className="space-y-3">
                                    <SkeletonLine className="w-28" />
                                    <SkeletonBlock className="h-1.5 w-full rounded-full" />
                                </div>
                            ))}
                        </div>
                    </SkeletonCard>
                </div>

                <div className="mt-3 grid gap-3 lg:grid-cols-[0.72fr_1.28fr]">
                    <SkeletonCard className="p-5">
                        <SkeletonLine className="mb-4 w-32" />
                        <div className="space-y-3">
                            {[...Array(8)].map((_, index) => (
                                <SkeletonBlock key={index} className="h-9 w-full rounded-lg" />
                            ))}
                        </div>
                    </SkeletonCard>
                    <SkeletonCard className="p-5">
                        <SkeletonLine className="mb-4 w-44" />
                        <SkeletonBlock className="h-[22rem] w-full rounded-xl" />
                    </SkeletonCard>
                </div>
            </main>
        </div>
    );
};

export default DashboardSkeleton;
