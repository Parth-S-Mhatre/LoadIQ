import React from 'react';
import { SkeletonBlock, SkeletonCard, SkeletonLine, SkeletonReason } from './SkeletonPrimitives';

const AuthSkeleton = ({ mode = 'login' }) => {
    const isRegister = mode === 'register';

    return (
        <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-paper p-6 text-ink">
            <div className="absolute left-6 top-6 flex items-center gap-3">
                <SkeletonBlock className="h-9 w-9 rounded-[10px]" />
                <SkeletonLine className="w-24" />
            </div>

            <div className="w-full max-w-[27rem]">
                <SkeletonReason
                    title={isRegister ? 'Preparing registration workspace' : 'Preparing sign-in workspace'}
                    subtitle="The skeleton appears first on constrained connections so the form layout is stable before the heavier assets arrive."
                />

                <SkeletonCard className="p-8">
                    <div className="mb-8">
                        <SkeletonLine className="mb-3 w-28" />
                        <SkeletonBlock className="h-8 w-56 rounded-lg" />
                        <SkeletonLine className="mt-3 w-64 max-w-full" />
                    </div>

                    <div className="space-y-5">
                        {[...Array(isRegister ? 3 : 2)].map((_, index) => (
                            <div key={index} className="space-y-2">
                                <SkeletonLine className="w-24" />
                                <SkeletonBlock className="h-12 w-full rounded-lg" />
                            </div>
                        ))}

                        {isRegister && (
                            <div className="grid grid-cols-2 gap-4">
                                <div className="space-y-2">
                                    <SkeletonLine className="w-24" />
                                    <SkeletonBlock className="h-12 rounded-lg" />
                                </div>
                                <div className="space-y-2">
                                    <SkeletonLine className="w-24" />
                                    <SkeletonBlock className="h-12 rounded-lg" />
                                </div>
                            </div>
                        )}

                        <SkeletonBlock className="mt-6 h-12 w-full rounded-full" />
                        <div className="my-6 flex items-center gap-4">
                            <div className="h-px flex-1 bg-ink/10" />
                            <SkeletonLine className="w-8" />
                            <div className="h-px flex-1 bg-ink/10" />
                        </div>
                        <SkeletonBlock className="h-12 w-full rounded-full" />
                    </div>
                </SkeletonCard>
            </div>
        </div>
    );
};

export default AuthSkeleton;
