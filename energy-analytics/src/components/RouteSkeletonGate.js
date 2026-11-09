import React, { Suspense } from 'react';
import useBackendHealth from '../hooks/useBackendHealth';
import useNetworkQuality from '../hooks/useNetworkQuality';

const isChunkLoadError = (error) => {
    const message = String(error?.message || error || '').toLowerCase();
    return message.includes('chunkloaderror') || message.includes('loading chunk') || message.includes('failed to fetch dynamically imported module');
};

class RouteChunkBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, isChunkError: false };
    }

    static getDerivedStateFromError(error) {
        return {
            hasError: true,
            isChunkError: isChunkLoadError(error)
        };
    }

    componentDidCatch(error) {
        if (isChunkLoadError(error) && typeof window !== 'undefined') {
            const reloadKey = 'loadiq_chunk_reload_attempted';
            const alreadyReloaded = sessionStorage.getItem(reloadKey) === '1';

            if (!alreadyReloaded) {
                sessionStorage.setItem(reloadKey, '1');
                setTimeout(() => window.location.reload(), 100);
            }
        }
    }

    handleRetry = () => {
        if (typeof window !== 'undefined') {
            sessionStorage.removeItem('loadiq_chunk_reload_attempted');
            window.location.reload();
        }
    };

    render() {
        const { hasError, isChunkError } = this.state;

        if (hasError) {
            if (isChunkError) {
                return (
                    <div className="flex min-h-screen items-center justify-center bg-paper p-6 text-ink">
                        <div className="w-full max-w-md rounded-2xl border border-ink/15 bg-paper-50 p-7 text-center shadow-lift sm:p-9">
                            <div className="mx-auto mb-5 flex h-14 w-14 items-center justify-center rounded-xl border border-ink/10 bg-paper-100">
                                <div className="h-6 w-6 animate-spin rounded-full border-2 border-ink/10 border-t-moss" />
                            </div>
                            <p className="label text-ink/35">Asset refresh</p>
                            <h1 className="mt-2 text-xl font-medium tracking-[-0.03em]">Refreshing app assets</h1>
                            <p className="mt-3 text-xs leading-relaxed text-ink/50">
                                A page chunk could not load, so we&apos;re trying to refresh the app with the latest files.
                                If the app stays stuck, reload once more.
                            </p>
                            <button
                                onClick={this.handleRetry}
                                className="mt-6 inline-flex h-11 items-center justify-center rounded-full bg-ink px-6 text-[0.75rem] font-semibold text-paper transition-colors hover:bg-ink-800"
                            >
                                Reload app
                            </button>
                        </div>
                    </div>
                );
            }
        }

        return this.props.children;
    }
}

const MinimalRouteLoader = () => (
    <div className="flex min-h-screen items-center justify-center bg-paper">
        <div className="flex flex-col items-center gap-4">
            <div className="h-10 w-10 animate-spin rounded-full border-2 border-ink/10 border-t-moss" />
            <span className="label-sm text-ink/35">Loading interface</span>
        </div>
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
        <RouteChunkBoundary>
            <Suspense fallback={suspenseFallback}>
                <Component />
            </Suspense>
        </RouteChunkBoundary>
    );
};

export default RouteSkeletonGate;
