import React from 'react';
import { SkeletonBlock, SkeletonCard, SkeletonLine, SkeletonReason } from './SkeletonPrimitives';

const AuthSkeleton = ({ mode = 'login' }) => {
    const isRegister = mode === 'register';

    return (
        <div className="flex min-h-screen items-center justify-center overflow-hidden bg-[#080B14] p-6 text-white">
            <div className="absolute left-8 top-8 flex items-center gap-3">
                <SkeletonBlock className="h-10 w-10 rounded-2xl" />
                <SkeletonLine className="w-24" />
            </div>

            <div className="w-full max-w-[460px]">
                <SkeletonReason
                    title={isRegister ? 'Preparing registration workspace' : 'Preparing sign-in workspace'}
                    subtitle="The skeleton is shown first when the connection is constrained so the form layout appears immediately and the heavier page assets can stream in smoothly."
                />

                <SkeletonCard className="rounded-[2rem] p-10">
                    <div className="mb-8 flex flex-col items-center">
                        <SkeletonBlock className="mb-6 h-16 w-16 rounded-2xl" />
                        <SkeletonLine className="mb-3 w-40" />
                        <SkeletonLine className="w-28" />
                    </div>

                    <div className="space-y-5">
                        {[...Array(isRegister ? 3 : 2)].map((_, index) => (
                            <div key={index} className="space-y-2">
                                <SkeletonLine className="w-24" />
                                <SkeletonBlock className="h-12 w-full rounded-xl" />
                            </div>
                        ))}

                        {isRegister && (
                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <SkeletonLine className="w-24" />
                                    <SkeletonBlock className="h-12 rounded-xl" />
                                </div>
                                <div className="space-y-2">
                                    <SkeletonLine className="w-24" />
                                    <SkeletonBlock className="h-12 rounded-xl" />
                                </div>
                            </div>
                        )}

                        <SkeletonBlock className="mt-6 h-12 w-full rounded-xl" />
                        <div className="my-6 flex items-center gap-4">
                            <div className="h-px flex-1 bg-white/10" />
                            <SkeletonLine className="w-8" />
                            <div className="h-px flex-1 bg-white/10" />
                        </div>
                        <SkeletonBlock className="h-12 w-full rounded-xl" />
                    </div>
                </SkeletonCard>
            </div>
        </div>
    );
};

export default AuthSkeleton;
