import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, AlertCircle, RefreshCw, Wifi, WifiOff, Zap } from 'lucide-react';
import useBackendHealth from '../hooks/useBackendHealth';
import { API_CONFIG } from '../config/api';
import { StatusDot } from './ui';

const BackendStatus = ({ compact = false }) => {
    const {
        isOnline,
        isChecking,
        lastCheck,
        latency,
        error,
        modelLoaded,
        loadedModels,
        totalModels,
        backendMode,
        backendSource,
        backendLabel,
        activeBackendUrl,
        consecutiveFailures,
        refresh
    } = useBackendHealth();

    const getStatusTone = () => {
        if (isChecking) return 'amber';
        if (isOnline && modelLoaded) return 'green';
        if (isOnline && !modelLoaded) return 'amber';
        return 'red';
    };

    const getStatusText = () => {
        if (isChecking) return 'Checking...';
        if (isOnline && backendSource === 'railway') return `Railway fallback (${loadedModels}/${totalModels})`;
        if (isOnline && modelLoaded) return 'Online';
        if (isOnline && !modelLoaded && totalModels > 1) return `Models loading (${loadedModels}/${totalModels})`;
        if (isOnline && !modelLoaded) return 'Model loading';
        return 'Offline';
    };

    const getLatencyClass = () => {
        if (!latency) return 'text-ink/35';
        if (latency < 500) return 'text-moss';
        if (latency < 1000) return 'text-amber-700';
        return 'text-clay';
    };

    if (compact) {
        return (
            <motion.div
                initial={{ opacity: 0, scale: 0.96 }}
                animate={{ opacity: 1, scale: 1 }}
                className="flex items-center gap-2 rounded-full border border-ink/10 bg-paper-50 px-3 py-1.5"
            >
                <StatusDot tone={getStatusTone()} pulse={isOnline} />
                <span className="label-sm text-ink/45">{getStatusText()}</span>
                {latency && (
                    <span className={`mono-num text-[0.58rem] font-medium ${getLatencyClass()}`}>
                        {latency}ms
                    </span>
                )}
            </motion.div>
        );
    }

    return (
        <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            className="surface p-5 sm:p-6"
        >
            <div className="flex items-center justify-between gap-4 border-b border-ink/10 pb-4">
                <div className="flex items-center gap-3">
                    <span className={`flex h-10 w-10 items-center justify-center rounded-xl ${isOnline ? 'bg-moss/10 text-moss' : 'bg-clay/10 text-clay'}`}>
                        {isOnline ? <Wifi size={18} /> : <WifiOff size={18} />}
                    </span>
                    <div>
                        <h3 className="text-sm font-medium">Backend status</h3>
                        <p className="label-sm mt-1 text-ink/35">LoadIQ neural network</p>
                    </div>
                </div>

                <button
                    onClick={refresh}
                    disabled={isChecking}
                    aria-label="Refresh backend status"
                    className="flex h-8 w-8 items-center justify-center rounded-lg border border-ink/10 text-ink/45 transition-colors hover:border-ink/25 hover:text-ink disabled:opacity-50"
                >
                    <RefreshCw className={`${isChecking ? 'animate-spin' : ''}`} size={14} />
                </button>
            </div>

            <div className="mt-4 grid grid-cols-2 gap-2.5">
                <div className="rounded-xl border border-ink/10 bg-paper-100 p-3.5">
                    <div className="flex items-center gap-2">
                        <StatusDot tone={getStatusTone()} pulse={isOnline} />
                        <span className="label-sm text-ink/35">Connection</span>
                    </div>
                    <p className={`mt-2 text-xs font-medium ${isOnline ? 'text-moss' : 'text-clay'}`}>
                        {getStatusText()}
                    </p>
                </div>

                <div className="rounded-xl border border-ink/10 bg-paper-100 p-3.5">
                    <div className="flex items-center gap-2 text-ink/35">
                        <Activity size={11} />
                        <span className="label-sm">Latency</span>
                    </div>
                    <p className={`mono-num mt-2 text-xs font-medium ${getLatencyClass()}`}>
                        {latency ? `${latency}ms` : '—'}
                    </p>
                </div>

                <div className="rounded-xl border border-ink/10 bg-paper-100 p-3.5">
                    <div className="flex items-center gap-2 text-ink/35">
                        <Zap size={11} />
                        <span className="label-sm">Model</span>
                    </div>
                    <p className={`mt-2 text-xs font-medium ${modelLoaded ? 'text-moss' : 'text-ink/40'}`}>
                        {modelLoaded
                            ? backendSource === 'railway'
                                ? `Ready (${loadedModels}/${totalModels})`
                                : totalModels > 1
                                    ? `Ready (${loadedModels}/${totalModels})`
                                    : 'Ready'
                            : totalModels > 1
                                ? `Loading (${loadedModels}/${totalModels})`
                                : 'Loading'}
                    </p>
                    {(backendLabel || backendMode) && (
                        <p className="label-sm mt-1.5 text-ink/30">
                            {(backendLabel || backendMode).replace(/_/g, ' ')}
                        </p>
                    )}
                </div>

                <div className="rounded-xl border border-ink/10 bg-paper-100 p-3.5">
                    <div className="flex items-center gap-2 text-ink/35">
                        <RefreshCw size={11} />
                        <span className="label-sm">Last check</span>
                    </div>
                    <p className="mono-num mt-2 text-[0.68rem] font-medium text-ink/60">
                        {lastCheck ? new Date(lastCheck).toLocaleTimeString() : '—'}
                    </p>
                </div>
            </div>

            <AnimatePresence>
                {error && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="mt-3 overflow-hidden"
                    >
                        <div className="flex items-start gap-2.5 rounded-xl border border-clay/25 bg-clay/[0.06] p-3">
                            <AlertCircle className="mt-0.5 shrink-0 text-clay" size={13} />
                            <div>
                                <p className="label-sm text-clay">Connection error</p>
                                <p className="mt-1 text-[0.68rem] leading-relaxed text-clay/85">{error}</p>
                                {consecutiveFailures > 0 && (
                                    <p className="mt-1 text-[0.62rem] text-clay/65">Failed attempts: {consecutiveFailures}</p>
                                )}
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            <div className="mt-4 border-t border-ink/10 pt-3">
                <p className="label-sm truncate text-ink/30">
                    Endpoint: {(activeBackendUrl || API_CONFIG.BASE_URL).replace(/^https?:\/\//, '')}
                </p>
            </div>
        </motion.div>
    );
};

export default BackendStatus;
