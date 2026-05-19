import React, { Suspense } from 'react';
import useBackendHealth from '../hooks/useBackendHealth';
import useNetworkQuality from '../hooks/useNetworkQuality';

const MinimalRouteLoader = () => (
    <div className="flex min-h-screen items-center justify-center bg-slate-950 text-white">
        <div className="h-12 w-12 rounded-full border-4 border-white/10 border-t-indigo-500 animate-spin" />
    </div>
);

const RouteSkeletonGate = ({ component: Component, skeleton: Skeleton, requiresBackend = false, skeletonProps = {} }) => {
    const network = useNetworkQuality();
    const backendHealth = useBackendHealth(requiresBackend);
    const shouldWaitForBackend = requiresBackend && (backendHealth.isChecking || !backendHealth.isOnline);
    const suspenseFallback = network.isLowBandwidth
        ? <Skeleton {...skeletonProps} />
        : <MinimalRouteLoader />;

    if (shouldWaitForBackend) {
        return <Skeleton {...skeletonProps} backendMode />;
    }

    return (
        <Suspense fallback={suspenseFallback}>
            <Component />
        </Suspense>
    );
};

export default RouteSkeletonGate;
