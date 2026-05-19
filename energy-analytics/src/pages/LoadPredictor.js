import React, { lazy, Suspense, useState, useMemo, useCallback, useRef } from "react";
import {
  Play, Zap, TrendingUp, TrendingDown,
  RefreshCw, Cpu, AlertTriangle,
  Wifi, WifiOff, Sliders, Clock, Database
} from "lucide-react";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Brush,
  Legend
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import useResponsive from '../hooks/useResponsive';
import useBackendHealth from '../hooks/useBackendHealth';
import { useAuth } from '../context/AuthContext';
import { useDisclaimer } from '../context/DisclaimerContext';
import { AnalyticsService } from '../services/AnalyticsService';

const ThreeLoadChart = lazy(() => import('../components/3d/ThreeLoadChart'));

const LoadPredictor = () => {
  const [historicalLoads, setHistoricalLoads] = useState(Array(24).fill(32000));
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [forecastHorizon, setForecastHorizon] = useState(6); // hours ahead
  const [chartMode, setChartMode] = useState('combined'); // 'historical', 'forecast', 'combined', '3d'
  const [showAdvanced, setShowAdvanced] = useState(false);
  const animationRunRef = useRef(0);

  const { isLargeDesktop } = useResponsive();
  const backendHealth = useBackendHealth();
  const { user } = useAuth();
  const { disclaimerDismissed, dismissDisclaimer } = useDisclaimer();
  const backendStatusTone = backendHealth.isChecking
    ? 'amber'
    : backendHealth.isOnline
      ? 'green'
      : 'red';
  const backendStatusLabel = backendHealth.isChecking
    ? 'Checking'
    : backendHealth.isOnline
      ? 'Connected'
      : 'Status unavailable';

  // Statistics calculation
  const stats = useMemo(() => {
    const allLoads = [...historicalLoads, ...predictions];
    const avg = allLoads.reduce((a, b) => a + b, 0) / allLoads.length || 0;
    const max = Math.max(...allLoads);
    const min = Math.min(...historicalLoads);
    const lastActual = historicalLoads[historicalLoads.length - 1];
    const firstPred = predictions[0] || null;
    const delta = firstPred !== null ? firstPred - lastActual : 0;

    return {
      avg: avg.toFixed(0),
      peak: max.toFixed(0),
      min: min.toFixed(0),
      lastActual,
      firstPrediction: firstPred,
      delta,
      direction: delta > 0 ? 'up' : delta < 0 ? 'down' : 'stable'
    };
  }, [historicalLoads, predictions]);

  // Prepare chart data
  const chartData = useMemo(() => {
    const data = [];
    
    // Add historical data points
    historicalLoads.forEach((val, i) => {
      data.push({
        hour: i - 23 >= 0 ? `+${i - 23}h` : `${i - 23}h`,
        actual: val,
        predicted: null,
        type: 'historical'
      });
    });

    // Add prediction data points
    predictions.forEach((pred, i) => {
      data.push({
        hour: `+${historicalLoads.length - 23 + i + 1}h`,
        actual: null,
        predicted: pred,
        type: 'forecast'
      });
    });

    return data;
  }, [historicalLoads, predictions]);

  // Quick presets
  const applyPreset = useCallback((type) => {
    let newLoads;
    if (type === 'low') {
      newLoads = Array(24).fill(18000).map(v => v + Math.random() * 12000);
    } else if (type === 'high') {
      newLoads = Array(24).fill(52000).map(v => v + Math.random() * 18000);
    } else if (type === 'peak-evening') {
      newLoads = Array(24).fill(28000).map((v, i) => {
        if (i >= 16 && i <= 21) return 48000 + Math.random() * 14000;
        return v + Math.random() * 8000;
      });
    } else {
      newLoads = Array(24).fill(25000).map(() => 15000 + Math.random() * 45000);
    }
    setHistoricalLoads(newLoads);
    setPredictions([]); // reset prediction
  }, []);

  const handleInputChange = (index, value) => {
    const updated = [...historicalLoads];
    updated[index] = Number(value) || 0;
    setHistoricalLoads(updated);
    setPredictions([]); // reset predictions on change
  };

  const animatePredictions = useCallback(async (values, animationRunId) => {
    const animatedValues = [];

    for (const predicted of values) {
      if (animationRunRef.current !== animationRunId) {
        return null;
      }

      animatedValues.push(predicted);
      setPredictions([...animatedValues]);
      await new Promise((resolve) => setTimeout(resolve, 140));
    }

    return animatedValues;
  }, []);

  const fetchPredictionSeries = useCallback(async () => {
    const response = await AnalyticsService.getBatchPredictions({
      last24Hours: historicalLoads,
      horizon: forecastHorizon
    });

    return Array.isArray(response?.predictions)
      ? response.predictions.map((item) => item.value).filter(Number.isFinite)
      : [];
  }, [forecastHorizon, historicalLoads]);

  const runPrediction = async () => {
    animationRunRef.current += 1;
    const currentAnimationRun = animationRunRef.current;
    setLoading(true);
    setError(null);
    setPredictions([]);

    try {
      const resolvedPredictions = await fetchPredictionSeries();
      const newPreds = await animatePredictions(resolvedPredictions, currentAnimationRun);

      if (!newPreds || !newPreds.length) {
        return;
      }

      // Save to analytics history in localStorage
      const historyItem = {
        timestamp: new Date().toISOString(),
        prediction: newPreds[newPreds.length - 1],
        forecastHorizon,
        inputData: historicalLoads
      };

      const existingHistory = JSON.parse(localStorage.getItem(`analyticsHistory_${user?.uid}`) || '[]');
      existingHistory.unshift(historyItem);
      localStorage.setItem(`analyticsHistory_${user?.uid}`, JSON.stringify(existingHistory.slice(0, 50))); // Keep last 50

      // NoSQL disabled: keep history in localStorage only.

    } catch (err) {
      console.error(err);
      setError(err.message.includes('fetch')
        ? "Connection failed — check backend"
        : err.message.startsWith('Client error:')
          ? err.message
          : "Prediction error: " + err.message);
    } finally {
      if (animationRunRef.current === currentAnimationRun) {
        setLoading(false);
      }
    }
  };

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const entry = payload[0];
      return (
        <div className="bg-slate-900/95 backdrop-blur-lg border border-slate-700/70 rounded-xl p-4 shadow-xl min-w-[180px]">
          <p className="text-xs text-slate-400 mb-2 font-medium">{label}</p>
          <div className="flex items-center gap-3">
            <div className={`w-3 h-3 rounded-full ${entry.dataKey === 'actual' ? 'bg-indigo-500' : 'bg-emerald-500'}`} />
            <span className="text-white font-semibold">
              {entry.dataKey === 'actual' ? 'Actual' : 'Forecast'}: 
              <span className="ml-1.5 text-lg">{entry.value.toLocaleString()}</span> MW
            </span>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="relative min-h-[80vh] bg-gradient-to-b from-slate-950/80 to-slate-900/60 backdrop-blur-sm rounded-3xl border border-slate-700/40 shadow-2xl overflow-hidden"
    >
      {/* Loading Overlay */}
      <AnimatePresence>
        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 z-50 flex items-center justify-center bg-slate-950/90 backdrop-blur-xl"
          >
            <motion.div
              initial={{ scale: 0.8, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              className="bg-slate-900/90 border border-indigo-500/30 rounded-3xl p-8 max-w-sm w-full text-center shadow-2xl"
            >
              <div className="relative mb-6">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                  className="w-16 h-16 border-4 border-indigo-500/20 border-t-indigo-500 rounded-full mx-auto"
                />
                <div className="absolute inset-0 flex items-center justify-center">
                  <Cpu className="text-indigo-400" size={24} />
                </div>
              </div>
              <h3 className="text-xl font-bold text-white mb-2">Running Analytics</h3>
              <p className="text-slate-300 text-sm mb-4">
                Processing load predictions on the backend...
              </p>
              <div className="flex items-center justify-center gap-2">
                <div className="flex gap-1">
                  {[0, 1, 2].map((i) => (
                    <motion.div
                      key={i}
                      className="w-2 h-2 bg-indigo-500 rounded-full"
                      animate={{
                        scale: [1, 1.5, 1],
                        opacity: [0.5, 1, 0.5]
                      }}
                      transition={{
                        duration: 1.5,
                        repeat: Infinity,
                        delay: i * 0.2
                      }}
                    />
                  ))}
                </div>
              </div>
              <p className="text-xs text-slate-400 mt-4">
                {predictions.length > 0 ? `${predictions.length}/${forecastHorizon} predictions completed` : 'Initializing...'}
              </p>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Disclaimer */}
      <AnimatePresence>
        {!disclaimerDismissed && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 z-50 flex items-center justify-center bg-slate-950/95 backdrop-blur-xl p-6"
          >
            <motion.div 
              initial={{ scale: 0.92, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              className="bg-slate-900/90 border border-indigo-500/30 rounded-3xl p-10 max-w-md w-full text-center shadow-2xl"
            >
              <Cpu className="mx-auto text-indigo-400 mb-6" size={64} />
              <h3 className="text-2xl font-black text-white mb-4 tracking-tight">Load Prediction Engine</h3>
              <p className="text-slate-300 text-sm mb-8 leading-relaxed">
                This interface provides forecasted load values based on historical patterns. 
                Always validate critical decisions with real-time grid data.
              </p>
              <button
                onClick={() => dismissDisclaimer()}
                className="w-full py-5 bg-gradient-to-r from-indigo-600 to-indigo-500 hover:from-indigo-500 hover:to-blue-600 text-white font-bold tracking-wider rounded-2xl shadow-lg shadow-indigo-700/30 transition-all"
              >
                ACKNOWLEDGE & CONTINUE
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Header */}
      <div className="p-6 md:p-8 border-b border-slate-700/50">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="flex items-center gap-4">
            <div className="p-4 bg-indigo-900/30 rounded-2xl border border-indigo-500/20">
              <Zap className="text-indigo-400" size={32} />
            </div>
            <div>
              <h1 className="text-3xl md:text-4xl font-black text-white tracking-tight">
                LOAD <span className="text-indigo-400">FORECASTER</span>
              </h1>
              <div className="flex items-center gap-4 mt-2 text-sm">
                <div className="flex items-center gap-2">
                  {backendStatusTone === 'green' ? (
                    <Wifi className="text-emerald-400" size={16} />
                  ) : (
                    <WifiOff className={backendStatusTone === 'amber' ? "text-amber-400" : "text-red-400"} size={16} />
                  )}
                  <span className={backendStatusTone === 'green' ? "text-emerald-400" : backendStatusTone === 'amber' ? "text-amber-400" : "text-red-400"}>
                    {backendStatusLabel}
                  </span>
                </div>
                {backendHealth.latency && (
                  <span className="text-slate-400 text-xs">• {backendHealth.latency} ms</span>
                )}
              </div>
            </div>
          </div>

          <div className="flex flex-wrap gap-3">
            <div className="px-5 py-3 bg-slate-800/60 rounded-2xl border border-slate-700/50 text-sm font-medium">
              Horizon: <span className="text-indigo-300 font-bold">{forecastHorizon}h</span>
            </div>
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="px-5 py-3 bg-slate-800/60 hover:bg-slate-700/60 rounded-2xl border border-slate-600 flex items-center gap-2 text-sm font-medium transition-colors"
            >
              <Sliders size={16} />
              {showAdvanced ? "Simple" : "Advanced"}
            </button>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="grid lg:grid-cols-12 gap-6 p-6 md:p-8">
        {/* Left panel - Inputs */}
        <div className="lg:col-span-4 space-y-6">
          {/* Quick presets */}
          <div className="bg-slate-900/40 rounded-2xl border border-slate-700/50 p-6">
            <h3 className="text-sm font-black uppercase tracking-widest text-slate-400 mb-4 flex items-center gap-2">
              <Clock size={16} /> Quick Scenarios
            </h3>
            <div className="grid grid-cols-2 gap-3">
              {[
                { label: 'Low', action: () => applyPreset('low') },
                { label: 'High', action: () => applyPreset('high') },
                { label: 'Evening Peak', action: () => applyPreset('peak-evening') },
                { label: 'Random', action: () => applyPreset('random') }
              ].map(p => (
                <button
                  key={p.label}
                  onClick={p.action}
                  className="py-3 px-4 bg-slate-800/60 hover:bg-indigo-900/40 border border-slate-600 hover:border-indigo-500/50 rounded-xl text-sm font-medium transition-all"
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          {/* Load inputs */}
          <div className="bg-slate-900/40 rounded-2xl border border-slate-700/50 p-6">
            <div className="flex justify-between items-center mb-5">
              <h3 className="text-sm font-black uppercase tracking-widest text-slate-400 flex items-center gap-2">
                <Database size={16} /> Last 24 Hours
              </h3>
              <span className="text-xs text-slate-500">MW</span>
            </div>

            <div className={`space-y-4 max-h-[420px] overflow-y-auto pr-2 custom-scrollbar ${isLargeDesktop ? 'h-[420px]' : 'h-[340px]'}`}>
              {historicalLoads.map((val, i) => (
                <div key={i} className="space-y-1.5">
                  <div className="flex justify-between text-xs text-slate-500">
                    <span>Hour {i - 23 >= 0 ? `+${i - 23}` : i - 23}</span>
                    <span className="text-indigo-300 font-medium">{val.toLocaleString()}</span>
                  </div>
                  <input
                    type="range"
                    min={10000}
                    max={90000}
                    step={500}
                    value={val}
                    onChange={(e) => handleInputChange(i, e.target.value)}
                    className="w-full accent-indigo-500 cursor-pointer"
                  />
                </div>
              ))}
            </div>
          </div>

          {/* Forecast controls */}
          <div className="bg-gradient-to-r from-indigo-950/40 to-purple-950/20 rounded-2xl border border-indigo-500/20 p-6">
            <div className="flex flex-col gap-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-slate-300">Forecast Horizon</span>
                <span className="text-indigo-300 font-bold">{forecastHorizon} hours</span>
              </div>
              <input
                type="range"
                min={1}
                max={24}
                value={forecastHorizon}
                onChange={e => setForecastHorizon(Number(e.target.value))}
                className="accent-indigo-500"
              />
              
              <motion.button
                whileHover={{ scale: loading ? 1 : 1.03 }}
                whileTap={{ scale: 0.98 }}
                onClick={runPrediction}
                disabled={loading}
                className={`mt-3 w-full py-5 rounded-2xl font-black text-lg tracking-wider flex items-center justify-center gap-3 transition-all shadow-xl
                  ${loading 
                    ? 'bg-slate-700 cursor-wait' 
                    : 'bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-500 hover:to-blue-500 shadow-indigo-700/40'}`}
              >
                {loading ? (
                  <RefreshCw className="animate-spin" size={22} />
                ) : (
                  <Play size={22} />
                )}
                {loading ? "Predicting..." : "Run Forecast"}
              </motion.button>
            </div>
          </div>

          {/* Error */}
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="p-5 bg-red-950/60 border border-red-700/40 rounded-2xl flex items-start gap-3 text-sm"
              >
                <AlertTriangle className="text-red-400 mt-0.5" size={20} />
                <div>
                  <p className="font-semibold text-red-300 mb-1">Error</p>
                  <p className="text-red-200/90">{error}</p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Right panel - Visualization & Results */}
        <div className="lg:col-span-8 space-y-6">
          {/* Chart */}
          <div className="bg-slate-950/50 border border-slate-700/50 rounded-3xl p-6 h-[500px] relative">
            <div className="absolute top-5 left-6 z-10 flex items-center gap-3">
              <div className="w-3 h-3 rounded-full bg-indigo-500 animate-pulse" />
              <h3 className="text-lg font-black text-white tracking-tight">Load Forecast Visualization</h3>
            </div>

            <div className="absolute top-5 right-6 z-10 flex gap-2 bg-slate-900/70 backdrop-blur-sm p-1.5 rounded-2xl border border-slate-700/50">
              {['combined', '3d'].map(mode => (
                <button
                  key={mode}
                  onClick={() => setChartMode(mode)}
                  className={`px-5 py-2 text-xs font-bold rounded-xl transition-all ${
                    chartMode === mode 
                      ? 'bg-indigo-600 text-white shadow-md' 
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                  }`}
                >
                  {mode === '3d' ? '3D View' : '2D Chart'}
                </button>
              ))}
            </div>

            {chartMode === '3d' ? (
              <Suspense
                fallback={
                  <div className="flex h-full items-center justify-center rounded-3xl border border-slate-800 bg-slate-950/60">
                    <div className="h-12 w-12 rounded-full border-4 border-slate-700 border-t-indigo-500 animate-spin" />
                  </div>
                }
              >
                <ThreeLoadChart 
                  data={historicalLoads} 
                  predicted={predictions} 
                />
              </Suspense>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData} margin={{ top: 40, right: 30, left: 20, bottom: 20 }}>
                  <defs>
                    <linearGradient id="colorActual" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#6366f1" stopOpacity={0.35}/>
                      <stop offset="95%" stopColor="#6366f1" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="colorForecast" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={0.4}/>
                      <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="4 4" stroke="#ffffff" opacity={0.04} />
                  <XAxis dataKey="hour" stroke="#64748b" fontSize={11} />
                  <YAxis 
                    stroke="#64748b" 
                    fontSize={11} 
                    tickFormatter={(v) => `${(v/1000).toFixed(0)}k`} 
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                  
                  <Area 
                    type="monotone" 
                    dataKey="actual" 
                    stroke="#6366f1" 
                    fill="url(#colorActual)" 
                    name="Historical Load" 
                    strokeWidth={2.5}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="predicted" 
                    stroke="#10b981" 
                    fill="url(#colorForecast)" 
                    name="Forecast" 
                    strokeWidth={2.5}
                    strokeDasharray="6 4"
                  />
                  
                  <ReferenceLine 
                    x="Hnow" 
                    stroke="#f59e0b" 
                    strokeDasharray="3 3" 
                    label={{ value: "Now", position: 'top', fill: '#f59e0b', fontSize: 12 }} 
                  />
                  
                  <Brush 
                    dataKey="hour" 
                    height={30} 
                    stroke="#6366f1" 
                    fill="#1e293b" 
                    fillOpacity={0.4} 
                  />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>

          {/* Results cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
            <div className="bg-gradient-to-br from-slate-900 to-slate-950 border border-slate-700/50 rounded-2xl p-6">
              <p className="text-xs font-black uppercase tracking-widest text-slate-500 mb-2">Next Hour Forecast</p>
              {predictions.length > 0 ? (
                <div className="flex items-baseline gap-2">
                  <span className="text-4xl md:text-5xl font-black text-white tracking-tighter">
                    {(predictions[0] / 1000).toFixed(1)}
                  </span>
                  <span className="text-2xl font-bold text-slate-400">GW</span>
                </div>
              ) : (
                <div className="text-4xl font-black text-slate-700">—</div>
              )}
            </div>

            <div className="bg-gradient-to-br from-slate-900 to-slate-950 border border-slate-700/50 rounded-2xl p-6">
              <p className="text-xs font-black uppercase tracking-widest text-slate-500 mb-2">Change</p>
              {stats.delta !== 0 ? (
                <div className="flex items-center gap-3">
                  {stats.direction === 'up' ? (
                    <TrendingUp className="text-red-400" size={32} />
                  ) : stats.direction === 'down' ? (
                    <TrendingDown className="text-emerald-400" size={32} />
                  ) : null}
                  <div>
                    <span className="text-4xl font-black text-white">
                      {Math.abs(stats.delta).toLocaleString()}
                    </span>
                    <span className="text-sm text-slate-400 ml-1.5">MW</span>
                  </div>
                </div>
              ) : (
                <div className="text-4xl font-black text-slate-600">Stable</div>
              )}
            </div>

            <div className="bg-gradient-to-br from-slate-900 to-slate-950 border border-slate-700/50 rounded-2xl p-6">
              <p className="text-xs font-black uppercase tracking-widest text-slate-500 mb-2">Peak in Window</p>
              <div className="flex items-baseline gap-2">
                <span className="text-4xl font-black text-white tracking-tighter">
                  {(stats.peak / 1000).toFixed(1)}
                </span>
                <span className="text-2xl font-bold text-slate-400">GW</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default LoadPredictor;
