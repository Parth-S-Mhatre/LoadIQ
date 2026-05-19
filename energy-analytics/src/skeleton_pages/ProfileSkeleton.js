import React from 'react';
import { SkeletonBlock, SkeletonCard, SkeletonLine, SkeletonReason } from './SkeletonPrimitives';

const ProfileSkeleton = () => {
    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 px-6 py-8 text-white">
            <div className="mx-auto max-w-6xl">
                <SkeletonReason
                    title="Preparing profile workspace"
                    subtitle="This page does not need the backend, so the skeleton appears only when the network is constrained and the route chunk needs more time to arrive."
                />

                <SkeletonCard className="mb-8 p-6">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                            <SkeletonBlock className="h-12 w-12 rounded-2xl" />
                            <div className="space-y-2">
                                <SkeletonLine className="w-40" />
                                <SkeletonLine className="w-56" />
                            </div>
                        </div>
                        <SkeletonBlock className="h-11 w-40 rounded-xl" />
                    </div>
                </SkeletonCard>

                <div className="mb-8 flex flex-wrap gap-3">
                    {[...Array(4)].map((_, index) => (
                        <SkeletonBlock key={index} className="h-12 w-36 rounded-xl" />
                    ))}
                </div>

                <div className="grid gap-6 md:grid-cols-2">
                    {[...Array(2)].map((_, columnIndex) => (
                        <SkeletonCard key={columnIndex} className="p-6">
                            <div className="space-y-5">
                                {[...Array(4)].map((_, rowIndex) => (
                                    <div key={rowIndex} className="space-y-2">
                                        <SkeletonLine className="w-28" />
                                        <SkeletonBlock className="h-12 w-full rounded-xl" />
                                    </div>
                                ))}
                            </div>
                        </SkeletonCard>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default ProfileSkeleton;
