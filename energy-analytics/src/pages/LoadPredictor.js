import React, { lazy, Suspense, useState, useMemo, useCallback, useRef } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Calendar,
  CheckCircle,
  Clock,
  Cpu,
  Database,
  Gauge,
  Layers,
  Play,
  RefreshCw,
  Sliders,
  Target,
  TrendingDown,
  TrendingUp,
  Upload,
  Wifi,
  WifiOff,
  Zap
} from "lucide-react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Brush,
  Legend,
  BarChart,
  Bar
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import useResponsive from '../hooks/useResponsive';
import useBackendHealth from '../hooks/useBackendHealth';
import { useAuth } from '../context/AuthContext';
import { useDisclaimer } from '../context/DisclaimerContext';
import { AnalyticsService } from '../services/AnalyticsService';
import { UserHistoryService } from '../services/UserHistoryService';

const ThreeLoadChart = lazy(() => import('../components/3d/ThreeLoadChart'));

const POSTMAN_MODEL2_SAMPLE = {
  hour: 14,
  day_of_week: 2,
  month: 6,
  model: 'ensemble',
  last_24_hours: [
    28100, 27950, 27820, 27740, 27690, 27780, 28120, 28600,
    29150, 29780, 30220, 30510, 30780, 30920, 30810, 30480,
    30020, 29610, 29240, 28950, 28730, 28560, 28410, 28290
  ],
  load_lag_1h: 28290,
  load_lag_24h: 28100,
  load_rolling_mean_24h: 29038.75,
  load_rolling_std_24h: 1140
};

const WEEKDAY_LABELS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

const formatMw = (value) => `${Math.round(Number(value) || 0).toLocaleString()} MW`;
const formatGw = (value) => `${((Number(value) || 0) / 1000).toFixed(2)} GW`;

const calculateLoadFeatures = (loads) => {
  const cleanLoads = loads.map(Number).filter(Number.isFinite);
  const mean = cleanLoads.reduce((sum, value) => sum + value, 0) / Math.max(cleanLoads.length, 1);
  const variance = cleanLoads.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / Math.max(cleanLoads.length, 1);
  const std = Math.sqrt(variance);
  const min = Math.min(...cleanLoads);
  const max = Math.max(...cleanLoads);
  const last = cleanLoads[cleanLoads.length - 1] || 0;
  const first = cleanLoads[0] || 0;
  const ramp = last - (cleanLoads[cleanLoads.length - 2] || last);
  const loadFactor = max > 0 ? (mean / max) * 100 : 0;
  const volatility = mean > 0 ? (std / mean) * 100 : 0;
  const trend = last - first;

  return {
    mean,
    std,
    min,
    max,
    last,
    first,
    ramp,
    loadFactor,
    volatility,
    trend
  };
};

const buildContextFromDate = () => {
  const now = new Date();
  return {
    hour: now.getHours(),
    day_of_week: (now.getDay() + 6) % 7,
    month: now.getMonth() + 1,
    model: 'ensemble'
  };
};

const LoadPredictor = () => {
  const [historicalLoads, setHistoricalLoads] = useState(POSTMAN_MODEL2_SAMPLE.last_24_hours);
  const [modelContext, setModelContext] = useState({
    hour: POSTMAN_MODEL2_SAMPLE.hour,
    day_of_week: POSTMAN_MODEL2_SAMPLE.day_of_week,
    month: POSTMAN_MODEL2_SAMPLE.month,
    model: POSTMAN_MODEL2_SAMPLE.model
  });
  const [rawJsonInput, setRawJsonInput] = useState(JSON.stringify(POSTMAN_MODEL2_SAMPLE, null, 2));
  const [predictions, setPredictions] = useState([]);
  const [lastRequestPayload, setLastRequestPayload] = useState(POSTMAN_MODEL2_SAMPLE);
  const [lastResponseMeta, setLastResponseMeta] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [jsonMessage, setJsonMessage] = useState(null);
  const [forecastHorizon, setForecastHorizon] = useState(6);
  const [chartMode, setChartMode] = useState('combined');
  const [showAdvanced, setShowAdvanced] = useState(true);
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

  const loadFeatures = useMemo(() => calculateLoadFeatures(historicalLoads), [historicalLoads]);

  const requestPayload = useMemo(() => ({
    hour: Number(modelContext.hour),
    day_of_week: Number(modelContext.day_of_week),
    month: Number(modelContext.month),
    model: modelContext.model || 'ensemble',
    last_24_hours: historicalLoads.map((value) => Number(value) || 0),
    load_lag_1h: Number(loadFeatures.last.toFixed(2)),
    load_lag_24h: Number(loadFeatures.first.toFixed(2)),
    load_rolling_mean_24h: Number(loadFeatures.mean.toFixed(2)),
    load_rolling_std_24h: Number(loadFeatures.std.toFixed(2))
  }), [historicalLoads, loadFeatures, modelContext]);

  const stats = useMemo(() => {
    const allLoads = [...historicalLoads, ...predictions];
    const avg = allLoads.reduce((a, b) => a + b, 0) / Math.max(allLoads.length, 1);
    const max = Math.max(...allLoads);
    const min = Math.min(...historicalLoads);
    const firstPred = predictions[0] || null;
    const delta = firstPred !== null ? firstPred - loadFeatures.last : 0;
    const forecastPeak = predictions.length ? Math.max(...predictions) : null;
    const peakStep = forecastPeak ? predictions.findIndex((value) => value === forecastPeak) + 1 : null;
    const risk = loadFeatures.volatility > 5 || Math.abs(delta) > 2500
      ? 'Elevated'
      : loadFeatures.volatility > 3 || Math.abs(delta) > 1200
        ? 'Watch'
        : 'Normal';

    return {
      avg,
      peak: max,
      min,
      firstPrediction: firstPred,
      delta,
      forecastPeak,
      peakStep,
      risk,
      direction: delta > 0 ? 'up' : delta < 0 ? 'down' : 'stable'
    };
  }, [historicalLoads, loadFeatures, predictions]);

  const chartData = useMemo(() => {
    const data = historicalLoads.map((value, index) => ({
      hour: index === 23 ? 'Now' : `-${23 - index}h`,
      actual: value,
      predicted: null,
      type: 'historical'
    }));

    predictions.forEach((value, index) => {
      data.push({
        hour: `+${index + 1}h`,
        actual: null,
        predicted: value,
        type: 'forecast'
      });
    });

    return data;
  }, [historicalLoads, predictions]);

  const profileRows = useMemo(() => [
    { name: 'Lag 1h', value: requestPayload.load_lag_1h },
    { name: 'Lag 24h', value: requestPayload.load_lag_24h },
    { name: 'Mean', value: requestPayload.load_rolling_mean_24h },
    { name: 'Std dev', value: requestPayload.load_rolling_std_24h }
  ], [requestPayload]);

  const applyPayload = useCallback((payload, sourceLabel = 'JSON payload') => {
    const loads = payload?.last_24_hours;
    if (!Array.isArray(loads) || loads.length !== 24) {
      throw new Error('Payload must include last_24_hours with exactly 24 numeric values.');
    }

    const nextLoads = loads.map(Number);
    if (nextLoads.some((value) => !Number.isFinite(value))) {
      throw new Error('Every last_24_hours value must be numeric.');
    }

    setHistoricalLoads(nextLoads);
    setModelContext({
      hour: Number.isFinite(Number(payload.hour)) ? Number(payload.hour) : modelContext.hour,
      day_of_week: Number.isFinite(Number(payload.day_of_week)) ? Number(payload.day_of_week) : modelContext.day_of_week,
      month: Number.isFinite(Number(payload.month)) ? Number(payload.month) : modelContext.month,
      model: payload.model || 'ensemble'
    });
    setRawJsonInput(JSON.stringify({ ...POSTMAN_MODEL2_SAMPLE, ...payload, last_24_hours: nextLoads }, null, 2));
    setPredictions([]);
    setJsonMessage(`${sourceLabel} applied.`);
    setError(null);
  }, [modelContext.day_of_week, modelContext.hour, modelContext.month]);

  const applyPreset = useCallback((type) => {
    let newLoads;
    if (type === 'postman') {
      applyPayload(POSTMAN_MODEL2_SAMPLE, 'Postman sample');
      return;
    }

    if (type === 'low') {
      newLoads = Array(24).fill(0).map((_, index) => 22000 + Math.sin(index / 3) * 1800 + index * 90);
    } else if (type === 'high') {
      newLoads = Array(24).fill(0).map((_, index) => 52000 + Math.sin(index / 2.4) * 4200 + index * 180);
    } else if (type === 'peak-evening') {
      newLoads = Array(24).fill(0).map((_, index) => {
        const eveningPeak = index >= 16 && index <= 21 ? 17000 - Math.abs(index - 19) * 2600 : 0;
        return 28500 + Math.sin(index / 2.8) * 2400 + eveningPeak;
      });
    } else {
      newLoads = Array(24).fill(0).map((_, index) => 26000 + Math.random() * 9000 + Math.sin(index / 2) * 2200);
    }

    setHistoricalLoads(newLoads.map((value) => Number(value.toFixed(0))));
    setPredictions([]);
    setJsonMessage(`${type.replace('-', ' ')} scenario loaded.`);
  }, [applyPayload]);

  const handleInputChange = (index, value) => {
    const updated = [...historicalLoads];
    updated[index] = Number(value) || 0;
    setHistoricalLoads(updated);
    setPredictions([]);
  };

  const handleJsonApply = () => {
    try {
      applyPayload(JSON.parse(rawJsonInput), 'Model2 payload');
    } catch (parseError) {
      setJsonMessage(null);
      setError(parseError.message || 'Unable to parse Model2 JSON.');
    }
  };

  const syncJsonFromControls = () => {
    const nextPayload = {
      ...requestPayload,
      horizon: forecastHorizon
    };
    setRawJsonInput(JSON.stringify(nextPayload, null, 2));
    setJsonMessage('JSON synced from current controls.');
  };

  const useCurrentCalendar = () => {
    setModelContext(buildContextFromDate());
    setPredictions([]);
  };

  const animatePredictions = useCallback(async (values, animationRunId) => {
    const animatedValues = [];

    for (const predicted of values) {
      if (animationRunRef.current !== animationRunId) {
        return null;
      }

      animatedValues.push(predicted);
      setPredictions([...animatedValues]);
      await new Promise((resolve) => setTimeout(resolve, 120));
    }

    return animatedValues;
  }, []);

  const fetchPredictionSeries = useCallback(async () => {
    const response = await AnalyticsService.getBatchPredictions({
      last24Hours: historicalLoads,
      horizon: forecastHorizon,
      contextFeatures: {
        hour: requestPayload.hour,
        day_of_week: requestPayload.day_of_week,
        month: requestPayload.month,
        model: requestPayload.model,
        load_lag_1h: requestPayload.load_lag_1h,
        load_lag_24h: requestPayload.load_lag_24h,
        load_rolling_mean_24h: requestPayload.load_rolling_mean_24h,
        load_rolling_std_24h: requestPayload.load_rolling_std_24h
      }
    });

    const predictionValues = Array.isArray(response?.predictions)
      ? response.predictions.map((item) => item.value).filter(Number.isFinite)
      : [];

    return {
      response,
      predictionValues
    };
  }, [forecastHorizon, historicalLoads, requestPayload]);

  const buildPredictionRecord = useCallback((predictionResponse, resolvedPredictions) => {
    const allLoads = [...historicalLoads, ...resolvedPredictions];
    const average = allLoads.reduce((sum, value) => sum + value, 0) / Math.max(allLoads.length, 1);
    const peak = Math.max(...allLoads);
    const firstPrediction = resolvedPredictions[0] ?? null;
    const delta = firstPrediction !== null ? firstPrediction - loadFeatures.last : 0;

    return {
      timestamp: new Date().toISOString(),
      type: 'prediction',
      model: requestPayload.model,
      forecastHorizon,
      historicalLoads,
      requestPayload: {
        ...requestPayload,
        horizon: forecastHorizon
      },
      predictions: resolvedPredictions,
      predictionSource: predictionResponse?.prediction_source || 'ml_model',
      fallbackUsed: Boolean(predictionResponse?.fallback_used),
      fallbackReason: predictionResponse?.reason || null,
      latestPrediction: resolvedPredictions[resolvedPredictions.length - 1] ?? null,
      nextHourPrediction: firstPrediction,
      average: Number(average.toFixed(2)),
      peak: Number(peak.toFixed(2)),
      minimum: Number(loadFeatures.min.toFixed(2)),
      lastActual: Number(loadFeatures.last.toFixed(2)),
      firstPrediction: firstPrediction !== null ? Number(firstPrediction.toFixed(2)) : null,
      delta: Number(delta.toFixed(2)),
      direction: delta > 0 ? 'up' : delta < 0 ? 'down' : 'stable'
    };
  }, [forecastHorizon, historicalLoads, loadFeatures, requestPayload]);

  const savePredictionHistory = useCallback(async (predictionRecord) => {
    const historyKey = `analyticsHistory_${user?.uid || 'guest'}`;
    const savedProfile = user?.uid
      ? JSON.parse(localStorage.getItem(`userProfile_${user.uid}`) || 'null')
      : null;
    const recordWithWorkspace = {
      ...predictionRecord,
      workspaceName: savedProfile?.workspaceName || '',
      organization: savedProfile?.organization || ''
    };
    localStorage.setItem('loadiq_latest_prediction_context', JSON.stringify(recordWithWorkspace));
    const existingHistory = JSON.parse(localStorage.getItem(historyKey) || '[]');
    existingHistory.unshift(recordWithWorkspace);
    localStorage.setItem(historyKey, JSON.stringify(existingHistory.slice(0, 50)));

    if (!user?.uid) {
      return;
    }

    try {
      await UserHistoryService.savePrediction(user.uid, recordWithWorkspace);
    } catch (storageError) {
      console.error('Failed to persist prediction history:', storageError);
    }
  }, [user?.uid]);

  const runPrediction = async () => {
    animationRunRef.current += 1;
    const currentAnimationRun = animationRunRef.current;
    setLoading(true);
    setError(null);
    setPredictions([]);
    setLastRequestPayload({ ...requestPayload, horizon: forecastHorizon });

    try {
      const { response, predictionValues } = await fetchPredictionSeries();
      setLastResponseMeta({
        source: response?.prediction_source || 'ml_model',
        fallbackUsed: Boolean(response?.fallback_used),
        reason: response?.reason || null,
        count: predictionValues.length
      });
      const newPreds = await animatePredictions(predictionValues, currentAnimationRun);

      if (!newPreds || !newPreds.length) {
        return;
      }

      if (animationRunRef.current === currentAnimationRun) {
        setLoading(false);
      }

      savePredictionHistory(buildPredictionRecord(response, newPreds)).catch((storageError) => {
        console.error('Failed to queue prediction history:', storageError);
      });
    } catch (err) {
      console.error(err);
      setError(err.message.includes('fetch')
        ? "Connection failed. Check the backend service."
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
      const entry = payload.find((item) => item.value !== null && item.value !== undefined);
      if (!entry) {
        return null;
      }

      return (
        <div className="bg-slate-950/95 border border-slate-700 rounded-lg p-4 shadow-xl min-w-[180px]">
          <p className="text-xs text-slate-400 mb-2">{label}</p>
          <div className="flex items-center gap-3">
            <div className={`w-3 h-3 rounded-full ${entry.dataKey === 'actual' ? 'bg-cyan-400' : 'bg-emerald-400'}`} />
            <span className="text-white font-semibold">
              {entry.dataKey === 'actual' ? 'Actual' : 'Forecast'}:
              <span className="ml-1.5 text-lg">{formatMw(entry.value)}</span>
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
      className="relative min-h-[80vh] bg-slate-950/90 border border-slate-700/60 shadow-2xl overflow-hidden rounded-lg"
    >
      <AnimatePresence>
        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 z-50 flex items-center justify-center bg-slate-950/90 backdrop-blur-xl p-6"
          >
            <motion.div
              initial={{ scale: 0.9, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              className="bg-slate-900 border border-cyan-500/30 rounded-lg p-8 max-w-sm w-full text-center shadow-2xl"
            >
              <div className="relative mb-6">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                  className="w-16 h-16 border-4 border-cyan-500/20 border-t-cyan-400 rounded-full mx-auto"
                />
                <div className="absolute inset-0 flex items-center justify-center">
                  <Cpu className="text-cyan-300" size={24} />
                </div>
              </div>
              <h3 className="text-xl font-bold text-white mb-2">Running Model2 Forecast</h3>
              <p className="text-slate-300 text-sm mb-4">
                Sending lag, rolling statistics, calendar context, and the 24-hour sequence.
              </p>
              <p className="text-xs text-slate-400">
                {predictions.length > 0 ? `${predictions.length}/${forecastHorizon} steps completed` : 'Preparing payload...'}
              </p>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

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
              className="bg-slate-900 border border-cyan-500/30 rounded-lg p-8 max-w-md w-full text-center shadow-2xl"
            >
              <Cpu className="mx-auto text-cyan-300 mb-6" size={56} />
              <h3 className="text-2xl font-black text-white mb-4">Load Prediction Engine</h3>
              <p className="text-slate-300 text-sm mb-8 leading-relaxed">
                Forecasts are generated from historical patterns and model features. Validate operational decisions with live grid data.
              </p>
              <button
                onClick={() => dismissDisclaimer()}
                className="w-full py-4 bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold rounded-lg transition-all"
              >
                Acknowledge and Continue
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="p-4 sm:p-6 md:p-8 border-b border-slate-700/60 bg-slate-900/60">
        <div className="flex flex-col xl:flex-row xl:items-center justify-between gap-6">
          <div className="flex items-center gap-4">
            <div className="p-4 bg-cyan-500/10 rounded-lg border border-cyan-400/20">
              <Zap className="text-cyan-300" size={32} />
            </div>
            <div>
              <h1 className="text-2xl sm:text-3xl md:text-4xl font-black text-white">Load Forecaster</h1>
              <div className="flex flex-wrap items-center gap-4 mt-2 text-sm">
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
                {backendHealth.latency && <span className="text-slate-400 text-xs">{backendHealth.latency} ms</span>}
                <span className="text-slate-400 text-xs">Model: {requestPayload.model}</span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 sm:gap-3">
            {[
              { label: 'Horizon', value: `${forecastHorizon}h`, icon: Clock },
              { label: 'Hour', value: `${requestPayload.hour}:00`, icon: Calendar },
              { label: 'Mean', value: formatGw(loadFeatures.mean), icon: BarChart3 },
              { label: 'Risk', value: stats.risk, icon: Gauge }
            ].map((item) => (
              <div key={item.label} className="rounded-lg border border-slate-700 bg-slate-950/70 p-4">
                <div className="flex items-center gap-2 text-slate-400 text-xs mb-2">
                  <item.icon size={14} />
                  <span>{item.label}</span>
                </div>
                <p className="text-white font-black text-xl">{item.value}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="grid xl:grid-cols-12 gap-4 sm:gap-6 p-4 sm:p-6 md:p-8">
        <div className="xl:col-span-4 space-y-6">
          <section className="bg-slate-900/70 rounded-lg border border-slate-700 p-5">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-bold text-slate-300 flex items-center gap-2">
                <Layers size={16} /> Model2 Scenarios
              </h3>
              <button
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="px-3 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg border border-slate-600 flex items-center gap-2 text-xs font-medium transition-colors"
              >
                <Sliders size={14} />
                {showAdvanced ? "Hide JSON" : "Show JSON"}
              </button>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {[
                { label: 'Postman Sample', type: 'postman' },
                { label: 'Low Demand', type: 'low' },
                { label: 'High Demand', type: 'high' },
                { label: 'Evening Peak', type: 'peak-evening' }
              ].map((preset) => (
                <button
                  key={preset.type}
                  onClick={() => applyPreset(preset.type)}
                  className="py-3 px-4 bg-slate-800 hover:bg-cyan-500/15 border border-slate-600 hover:border-cyan-400/50 rounded-lg text-sm font-medium text-slate-100 transition-all"
                >
                  {preset.label}
                </button>
              ))}
            </div>
          </section>

          <section className="bg-slate-900/70 rounded-lg border border-slate-700 p-5">
            <div className="flex justify-between items-center mb-5">
              <h3 className="text-sm font-bold text-slate-300 flex items-center gap-2">
                <Database size={16} /> Last 24 Hours
              </h3>
              <span className="text-xs text-slate-500">MW</span>
            </div>

            <div className={`space-y-4 max-h-[430px] overflow-y-auto pr-2 custom-scrollbar ${isLargeDesktop ? 'h-[430px]' : 'h-[340px]'}`}>
              {historicalLoads.map((value, index) => (
                <div key={index} className="space-y-1.5">
                  <div className="flex justify-between text-xs text-slate-500">
                    <span>{index === 23 ? 'Now' : `-${23 - index}h`}</span>
                    <span className="text-cyan-300 font-medium">{formatMw(value)}</span>
                  </div>
                  <input
                    type="range"
                    min={10000}
                    max={90000}
                    step={100}
                    value={value}
                    onChange={(event) => handleInputChange(index, event.target.value)}
                    className="w-full accent-cyan-400 cursor-pointer"
                  />
                </div>
              ))}
            </div>
          </section>

          <section className="bg-slate-900/70 rounded-lg border border-slate-700 p-5 space-y-5">
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <label className="space-y-2">
                <span className="text-xs text-slate-400">Hour</span>
                <input
                  type="number"
                  min={0}
                  max={23}
                  value={modelContext.hour}
                  onChange={(event) => setModelContext((prev) => ({ ...prev, hour: Number(event.target.value) }))}
                  className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-cyan-400 focus:outline-none"
                />
              </label>
              <label className="space-y-2">
                <span className="text-xs text-slate-400">Day</span>
                <select
                  value={modelContext.day_of_week}
                  onChange={(event) => setModelContext((prev) => ({ ...prev, day_of_week: Number(event.target.value) }))}
                  className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-cyan-400 focus:outline-none"
                >
                  {WEEKDAY_LABELS.map((label, index) => <option key={label} value={index}>{label}</option>)}
                </select>
              </label>
              <label className="space-y-2">
                <span className="text-xs text-slate-400">Month</span>
                <select
                  value={modelContext.month}
                  onChange={(event) => setModelContext((prev) => ({ ...prev, month: Number(event.target.value) }))}
                  className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white focus:border-cyan-400 focus:outline-none"
                >
                  {MONTH_LABELS.map((label, index) => <option key={label} value={index + 1}>{label}</option>)}
                </select>
              </label>
            </div>

            <button
              onClick={useCurrentCalendar}
              className="w-full py-3 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-100 font-semibold transition-all"
            >
              Use Current Date and Hour
            </button>

            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-slate-300">Forecast Horizon</span>
                <span className="text-cyan-300 font-bold">{forecastHorizon} hours</span>
              </div>
              <input
                type="range"
                min={1}
                max={24}
                value={forecastHorizon}
                onChange={(event) => setForecastHorizon(Number(event.target.value))}
                className="w-full accent-cyan-400"
              />
            </div>

            <motion.button
              whileHover={{ scale: loading ? 1 : 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={runPrediction}
              disabled={loading}
              className={`w-full py-4 rounded-lg font-black text-lg flex items-center justify-center gap-3 transition-all shadow-xl ${
                loading
                  ? 'bg-slate-700 cursor-wait text-slate-300'
                  : 'bg-cyan-500 hover:bg-cyan-400 text-slate-950 shadow-cyan-700/20'
              }`}
            >
              {loading ? <RefreshCw className="animate-spin" size={22} /> : <Play size={22} />}
              {loading ? "Predicting..." : "Run Forecast"}
            </motion.button>
          </section>

          {showAdvanced && (
            <section className="bg-slate-900/70 rounded-lg border border-slate-700 p-5">
              <div className="flex items-center justify-between gap-3 mb-4">
                <h3 className="text-sm font-bold text-slate-300 flex items-center gap-2">
                  <Upload size={16} /> Model2 JSON
                </h3>
                <button
                  onClick={syncJsonFromControls}
                  className="px-3 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-200"
                >
                  Sync
                </button>
              </div>
              <textarea
                value={rawJsonInput}
                onChange={(event) => setRawJsonInput(event.target.value)}
                className="h-56 w-full resize-none rounded-lg border border-slate-700 bg-slate-950 p-3 font-mono text-xs text-slate-200 focus:border-cyan-400 focus:outline-none"
                spellCheck={false}
              />
              <button
                onClick={handleJsonApply}
                className="mt-3 w-full py-3 rounded-lg bg-slate-800 hover:bg-cyan-500/20 border border-slate-700 hover:border-cyan-400/50 text-slate-100 font-semibold"
              >
                Apply JSON Payload
              </button>
              {jsonMessage && (
                <p className="mt-3 flex items-center gap-2 text-sm text-emerald-300">
                  <CheckCircle size={16} /> {jsonMessage}
                </p>
              )}
            </section>
          )}

          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="p-5 bg-red-950/60 border border-red-700/40 rounded-lg flex items-start gap-3 text-sm"
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

        <div className="xl:col-span-8 space-y-6">
          <section className="bg-slate-950/70 border border-slate-700 rounded-lg p-4 sm:p-5 h-[420px] sm:h-[520px] relative">
            <div className="absolute top-4 left-4 sm:top-5 sm:left-5 z-10 flex max-w-[calc(100%-7rem)] items-center gap-2 sm:gap-3">
              <div className="w-3 h-3 rounded-full bg-cyan-400 animate-pulse" />
              <h3 className="text-sm sm:text-lg font-black text-white">Forecast Workbench</h3>
            </div>

            <div className="absolute top-4 right-4 sm:top-5 sm:right-5 z-10 flex gap-1 sm:gap-2 bg-slate-900/80 backdrop-blur-sm p-1 rounded-lg border border-slate-700">
              {['combined', '3d'].map((mode) => (
                <button
                  key={mode}
                  onClick={() => setChartMode(mode)}
                  className={`px-3 sm:px-4 py-2 text-xs font-bold rounded-md transition-all ${
                    chartMode === mode
                      ? 'bg-cyan-500 text-slate-950'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                  }`}
                >
                  {mode === '3d' ? '3D' : '2D'}
                </button>
              ))}
            </div>

            {chartMode === '3d' ? (
              <Suspense
                fallback={
                  <div className="flex h-full items-center justify-center rounded-lg border border-slate-800 bg-slate-950/60">
                    <div className="h-12 w-12 rounded-full border-4 border-slate-700 border-t-cyan-400 animate-spin" />
                  </div>
                }
              >
                <ThreeLoadChart data={historicalLoads} predicted={predictions} />
              </Suspense>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData} margin={{ top: 54, right: 24, left: 12, bottom: 20 }}>
                  <defs>
                    <linearGradient id="colorActual" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.32} />
                      <stop offset="95%" stopColor="#22d3ee" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="colorForecast" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#34d399" stopOpacity={0.38} />
                      <stop offset="95%" stopColor="#34d399" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="4 4" stroke="#ffffff" opacity={0.05} />
                  <XAxis dataKey="hour" stroke="#64748b" fontSize={11} interval="preserveStartEnd" />
                  <YAxis stroke="#64748b" fontSize={11} tickFormatter={(value) => `${(value / 1000).toFixed(0)}k`} />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                  <Area type="monotone" dataKey="actual" stroke="#22d3ee" fill="url(#colorActual)" name="Historical Load" strokeWidth={2.5} />
                  <Area type="monotone" dataKey="predicted" stroke="#34d399" fill="url(#colorForecast)" name="Forecast" strokeWidth={2.5} strokeDasharray="6 4" />
                  <ReferenceLine x="Now" stroke="#f59e0b" strokeDasharray="3 3" label={{ value: "Now", position: 'top', fill: '#f59e0b', fontSize: 12 }} />
                  <Brush dataKey="hour" height={28} stroke="#22d3ee" fill="#0f172a" fillOpacity={0.5} />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </section>

          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
            {[
              {
                label: 'Next Hour',
                value: predictions.length ? formatGw(predictions[0]) : '-',
                sub: predictions.length ? formatMw(predictions[0]) : 'Run forecast',
                icon: Target
              },
              {
                label: 'Forecast Delta',
                value: stats.delta ? formatMw(Math.abs(stats.delta)) : 'Stable',
                sub: stats.direction === 'up' ? 'Demand rising' : stats.direction === 'down' ? 'Demand easing' : 'No movement',
                icon: stats.direction === 'down' ? TrendingDown : TrendingUp
              },
              {
                label: 'Peak Window',
                value: stats.forecastPeak ? formatGw(stats.forecastPeak) : formatGw(stats.peak),
                sub: stats.peakStep ? `Forecast step +${stats.peakStep}h` : 'Historical peak',
                icon: Activity
              },
              {
                label: 'Load Factor',
                value: `${loadFeatures.loadFactor.toFixed(1)}%`,
                sub: `Volatility ${loadFeatures.volatility.toFixed(1)}%`,
                icon: Gauge
              }
            ].map((card) => (
              <div key={card.label} className="bg-slate-900/75 border border-slate-700 rounded-lg p-5">
                <div className="flex items-center justify-between mb-4">
                  <p className="text-xs font-bold text-slate-500">{card.label}</p>
                  <card.icon className="text-cyan-300" size={18} />
                </div>
                <p className="text-3xl font-black text-white">{card.value}</p>
                <p className="text-sm text-slate-400 mt-2">{card.sub}</p>
              </div>
            ))}
          </div>

          <div className="grid lg:grid-cols-2 gap-6">
            <section className="bg-slate-900/70 rounded-lg border border-slate-700 p-5">
              <div className="flex items-center justify-between mb-5">
                <h3 className="text-sm font-bold text-slate-300 flex items-center gap-2">
                  <BarChart3 size={16} /> Feature Profile
                </h3>
                <span className="text-xs text-slate-500">{WEEKDAY_LABELS[requestPayload.day_of_week]} / {MONTH_LABELS[requestPayload.month - 1]}</span>
              </div>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={profileRows} margin={{ top: 10, right: 10, left: 4, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="4 4" stroke="#ffffff" opacity={0.05} />
                    <XAxis dataKey="name" stroke="#64748b" fontSize={11} />
                    <YAxis stroke="#64748b" fontSize={11} tickFormatter={(value) => `${(value / 1000).toFixed(0)}k`} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="value" fill="#22d3ee" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </section>

            <section className="bg-slate-900/70 rounded-lg border border-slate-700 p-5">
              <div className="flex items-center justify-between mb-5">
                <h3 className="text-sm font-bold text-slate-300 flex items-center gap-2">
                  <Cpu size={16} /> Request Snapshot
                </h3>
                {lastResponseMeta && (
                  <span className={`text-xs ${lastResponseMeta.fallbackUsed ? 'text-amber-300' : 'text-emerald-300'}`}>
                    {lastResponseMeta.fallbackUsed ? 'Fallback' : 'Model output'}
                  </span>
                )}
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-4">
                {[
                  ['Hour', `${lastRequestPayload.hour}:00`],
                  ['Day Index', lastRequestPayload.day_of_week],
                  ['Month', lastRequestPayload.month],
                  ['Horizon', `${lastRequestPayload.horizon || forecastHorizon}h`],
                  ['Lag 1h', formatMw(lastRequestPayload.load_lag_1h)],
                  ['Rolling Std', formatMw(lastRequestPayload.load_rolling_std_24h)]
                ].map(([label, value]) => (
                  <div key={label} className="rounded-lg border border-slate-700 bg-slate-950/70 p-3">
                    <p className="text-xs text-slate-500">{label}</p>
                    <p className="text-white font-semibold mt-1">{value}</p>
                  </div>
                ))}
              </div>
              <div className="rounded-lg border border-slate-700 bg-slate-950 p-3">
                <p className="text-xs text-slate-500 mb-2">Response source</p>
                <p className="text-sm text-slate-200">
                  {lastResponseMeta
                    ? `${lastResponseMeta.source} returned ${lastResponseMeta.count} forecast step${lastResponseMeta.count === 1 ? '' : 's'}.`
                    : 'Run a forecast to see backend response metadata.'}
                </p>
                {lastResponseMeta?.reason && <p className="text-xs text-amber-200 mt-2">{lastResponseMeta.reason}</p>}
              </div>
            </section>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default LoadPredictor;
