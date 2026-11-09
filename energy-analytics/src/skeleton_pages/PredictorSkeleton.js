import React from 'react';
import { SkeletonBlock, SkeletonCard, SkeletonLine, SkeletonReason } from './SkeletonPrimitives';

const PredictorSkeleton = ({
    backendMode = false,
    title,
    subtitle
}) => {
    const resolvedTitle = title || (backendMode ? 'Waiting for prediction backend' : 'Preparing predictor workspace');
    const resolvedSubtitle = subtitle || (backendMode
        ? 'The prediction page is holding its layout until the combined Render backend finishes waking up and can serve forecasts.'
        : 'The predictor layout is being staged first so the heavier controls and visualizations can load smoothly.');

    return (
        <div className="space-y-4">
            <SkeletonReason
                title={resolvedTitle}
                subtitle={resolvedSubtitle}
            />

            <SkeletonCard className="p-6">
                <SkeletonLine className="mb-4 w-44" />
                <div className="grid gap-4 md:grid-cols-2">
                    {[...Array(4)].map((_, index) => (
                        <div key={index} className="space-y-2">
                            <SkeletonLine className="w-28" />
                            <SkeletonBlock className="h-12 w-full rounded-lg" />
                        </div>
                    ))}
                </div>
                <div className="mt-6 flex justify-end gap-3">
                    <SkeletonBlock className="h-11 w-28 rounded-full" />
                    <SkeletonBlock className="h-11 w-40 rounded-full" />
                </div>
            </SkeletonCard>

            <div className="grid gap-4 xl:grid-cols-[0.7fr_1.3fr]">
                <SkeletonCard className="p-5">
                    <SkeletonLine className="mb-5 w-36" />
                    <div className="space-y-4">
                        {[...Array(8)].map((_, index) => (
                            <div key={index} className="space-y-2">
                                <SkeletonLine className="w-20" />
                                <SkeletonBlock className="h-1.5 w-full rounded-full" />
                            </div>
                        ))}
                    </div>
                </SkeletonCard>

                <SkeletonCard className="p-5">
                    <div className="mb-5 flex items-center justify-between">
                        <SkeletonLine className="w-52" />
                        <div className="flex gap-2">
                            <SkeletonBlock className="h-8 w-20 rounded-full" />
                            <SkeletonBlock className="h-8 w-20 rounded-full" />
                        </div>
                    </div>
                    <SkeletonBlock className="h-[26rem] w-full rounded-xl" />
                </SkeletonCard>
            </div>
        </div>
    );
};

export default PredictorSkeleton;
