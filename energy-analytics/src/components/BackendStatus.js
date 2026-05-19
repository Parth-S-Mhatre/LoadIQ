import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, AlertCircle, RefreshCw, Wifi, WifiOff, Zap } from 'lucide-react';
import useBackendHealth from '../hooks/useBackendHealth';

const BackendStatus = ({ compact = false }) => {
    const { isOnline, isChecking, lastCheck, latency, error, modelLoaded, consecutiveFailures, refresh } = useBackendHealth();

    const getStatusColor = () => {
        if (isChecking) return 'bg-yellow-500';
        if (isOnline && modelLoaded) return 'bg-emerald-500';
        if (isOnline && !modelLoaded) return 'bg-amber-500';
        return 'bg-red-500';
    };

    const getStatusText = () => {
        if (isChecking) return 'Checking...';
        if (isOnline && modelLoaded) return 'Online';
        if (isOnline && !modelLoaded) return 'Model Loading';
        return 'Offline';
    };

    const getLatencyColor = () => {
        if (!latency) return 'text-slate-500';
        if (latency < 500) return 'text-emerald-400';
        if (latency < 1000) return 'text-yellow-400';
        return 'text-red-400';
    };

    if (compact) {
        return (
            <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="flex items-center gap-2 bg-slate-950/50 px-3 py-1.5 rounded-full border border-white/5"
            >
                <div className="relative">
                    <div className={`w-2 h-2 rounded-full ${getStatusColor()}`} />
                    {isOnline && (
                        <motion.div
                            className={`absolute inset-0 rounded-full ${getStatusColor()} opacity-50`}
                            animate={{ scale: [1, 1.5, 1], opacity: [0.5, 0, 0.5] }}
                            transition={{ duration: 2, repeat: Infinity }}
                        />
                    )}
                </div>
                <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">
                    {getStatusText()}
                </span>
                {latency && (
                    <span className={`text-[9px] font-black uppercase tracking-widest ${getLatencyColor()}`}>
                        {latency}ms
                    </span>
                )}
            </motion.div>
        );
    }

    return (
        <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-panel border border-white/5 rounded-2xl p-6 shadow-xl"
        >
            <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                    <div className={`p-2 rounded-xl ${isOnline ? 'bg-emerald-500/10' : 'bg-red-500/10'}`}>
                        {isOnline ? (
                            <Wifi className={isOnline ? 'text-emerald-400' : 'text-red-400'} size={20} />
                        ) : (
                            <WifiOff className="text-red-400" size={20} />
                        )}
                    </div>
                    <div>
                        <h3 className="text-sm font-black text-white uppercase tracking-wider">Backend Status</h3>
                        <p className="text-[9px] font-black text-slate-500 uppercase tracking-widest">
                            LoadIQ Neural Network
                        </p>
                    </div>
                </div>

                <button
                    onClick={refresh}
                    disabled={isChecking}
                    className="p-2 bg-white/5 hover:bg-white/10 rounded-xl transition-all disabled:opacity-50"
                >
                    <RefreshCw
                        className={`text-slate-400 ${isChecking ? 'animate-spin' : ''}`}
                        size={16}
                    />
                </button>
            </div>

            <div className="grid grid-cols-2 gap-4">
                {/* Connection Status */}
                <div className="bg-slate-950/40 p-4 rounded-xl border border-white/5">
                    <div className="flex items-center gap-2 mb-2">
                        <div className="relative">
                            <div className={`w-2.5 h-2.5 rounded-full ${getStatusColor()}`} />
                            {isOnline && (
                                <motion.div
                                    className={`absolute inset-0 rounded-full ${getStatusColor()}`}
                                    animate={{ scale: [1, 2, 1], opacity: [0.5, 0, 0.5] }}
                                    transition={{ duration: 2, repeat: Infinity }}
                                />
                            )}
                        </div>
                        <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest">
                            Connection
                        </span>
                    </div>
                    <p className={`text-lg font-black uppercase tracking-tight ${isOnline ? 'text-emerald-400' : 'text-red-400'}`}>
                        {getStatusText()}
                    </p>
                </div>

                {/* Latency */}
                <div className="bg-slate-950/40 p-4 rounded-xl border border-white/5">
                    <div className="flex items-center gap-2 mb-2">
                        <Activity className="text-slate-500" size={10} />
                        <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest">
                            Latency
                        </span>
                    </div>
                    <p className={`text-lg font-black uppercase tracking-tight ${getLatencyColor()}`}>
                        {latency ? `${latency}ms` : '---'}
                    </p>
                </div>

                {/* Model Status */}
                <div className="bg-slate-950/40 p-4 rounded-xl border border-white/5">
                    <div className="flex items-center gap-2 mb-2">
                        <Zap className="text-slate-500" size={10} />
                        <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest">
                            Model
                        </span>
                    </div>
                    <p className={`text-lg font-black uppercase tracking-tight ${modelLoaded ? 'text-emerald-400' : 'text-slate-600'}`}>
                        {modelLoaded ? 'Ready' : 'Loading'}
                    </p>
                </div>

                {/* Last Check */}
                <div className="bg-slate-950/40 p-4 rounded-xl border border-white/5">
                    <div className="flex items-center gap-2 mb-2">
                        <RefreshCw className="text-slate-500" size={10} />
                        <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest">
                            Last Check
                        </span>
                    </div>
                    <p className="text-[10px] font-black text-slate-400 uppercase tracking-tight">
                        {lastCheck ? new Date(lastCheck).toLocaleTimeString() : '---'}
                    </p>
                </div>
            </div>

            {/* Error Message */}
            <AnimatePresence>
                {error && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="mt-4 p-3 bg-red-500/10 border border-red-500/20 rounded-xl flex items-start gap-2"
                    >
                        <AlertCircle className="text-red-400 flex-shrink-0 mt-0.5" size={14} />
                        <div>
                            <p className="text-[9px] font-black text-red-400 uppercase tracking-widest mb-1">
                                Connection Error
                            </p>
                            <p className="text-[10px] text-red-300 font-medium">
                                {error}
                            </p>
                            {consecutiveFailures > 0 && (
                                <p className="text-[9px] text-red-400/70 font-medium mt-1">
                                    Failed attempts: {consecutiveFailures}
                                </p>
                            )}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Backend URL Info */}
            <div className="mt-4 pt-4 border-t border-white/5">
                <p className="text-[8px] font-black text-slate-600 uppercase tracking-widest">
                    Endpoint: loadiq.onrender.com
                </p>
            </div>
        </motion.div>
    );
};

export default BackendStatus;
