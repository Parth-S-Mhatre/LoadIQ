import React from 'react';
import { SkeletonBlock, SkeletonCard, SkeletonLine, SkeletonReason } from './SkeletonPrimitives';

const ProfileSkeleton = () => {
    return (
        <div className="min-h-screen bg-paper px-5 py-8 text-ink sm:px-8">
            <div className="mx-auto max-w-6xl">
                <SkeletonReason
                    title="Preparing profile workspace"
                    subtitle="This page does not need the backend, so the skeleton appears only when the network is constrained and the route chunk needs more time to arrive."
                />

                <SkeletonCard className="mb-6 p-5">
                    <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                        <div className="flex items-center gap-4">
                            <SkeletonBlock className="h-12 w-12 rounded-xl" />
                            <div className="space-y-2">
                                <SkeletonLine className="w-40" />
                                <SkeletonLine className="w-56 max-w-full" />
                            </div>
                        </div>
                        <SkeletonBlock className="h-11 w-40 rounded-full" />
                    </div>
                </SkeletonCard>

                <div className="mb-6 flex flex-wrap gap-2">
                    {[...Array(5)].map((_, index) => (
                        <SkeletonBlock key={index} className="h-10 w-28 rounded-full" />
                    ))}
                </div>

                <div className="grid gap-4 md:grid-cols-2">
                    {[...Array(2)].map((_, columnIndex) => (
                        <SkeletonCard key={columnIndex} className="p-5">
                            <div className="space-y-5">
                                {[...Array(4)].map((_, rowIndex) => (
                                    <div key={rowIndex} className="space-y-2">
                                        <SkeletonLine className="w-28" />
                                        <SkeletonBlock className="h-11 w-full rounded-lg" />
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
