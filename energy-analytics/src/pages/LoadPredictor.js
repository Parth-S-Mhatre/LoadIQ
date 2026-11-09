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
import { Badge, EASE, StatusDot, buttonClass } from '../components/ui';

const ThreeLoadChart = lazy(() => import('../components/3d/ThreeLoadChart'));

const CHART = {
  grid: 'rgba(17, 24, 19, 0.09)',
  axis: '#8A9188',
  ink: '#111813',
  moss: '#0F3D2E',
  signal: '#A5C01D',
  clay: '#C75B32',
  sky: '#3A6B8A',
  amber: '#C99A2E',
};

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

const Panel = ({ children, className = '' }) => (
  <section className={`surface ${className}`.trim()}>{children}</section>
);

const PanelHeading = ({ icon: Icon, title, meta }) => (
  <div className="flex items-center justify-between gap-3">
    <h3 className="flex items-center gap-2 text-[0.8rem] font-medium text-ink/75">
      {Icon && <Icon size={15} className="text-ink/45" />}
      {title}
    </h3>
    {meta}
  </div>
);

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
    const userSuffix = user?.uid ? `_${user.uid}` : '_guest';
    const historyKey = `analyticsHistory${userSuffix}`;
    const savedProfile = user?.uid
      ? JSON.parse(localStorage.getItem(`userProfile_${user.uid}`) || 'null')
      : null;
    const recordWithWorkspace = {
      ...predictionRecord,
      workspaceName: savedProfile?.workspaceName || '',
      organization: savedProfile?.organization || ''
    };
    localStorage.setItem(`loadiq_latest_prediction_context${userSuffix}`, JSON.stringify(recordWithWorkspace));
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
        <div className="min-w-[10rem] rounded-xl border border-ink/15 bg-paper-50 p-3 shadow-panel">
          <p className="label-sm mb-2 text-ink/40">{label}</p>
          <div className="flex items-center gap-2.5">
            <span
              className="h-1.5 w-1.5 rounded-full"
              style={{ background: entry.dataKey === 'actual' ? CHART.ink : CHART.moss }}
            />
            <span className="text-xs font-medium text-ink/60">
              {entry.dataKey === 'actual' ? 'Actual' : entry.dataKey === 'predicted' ? 'Forecast' : 'Value'}
            </span>
            <span className="mono-num ml-auto text-sm font-medium text-ink">{formatMw(entry.value)}</span>
          </div>
        </div>
      );
    }
    return null;
  };

  const horizonPercent = ((forecastHorizon - 1) / 23) * 100;

  const headerStats = [
    { label: 'Horizon', value: `${forecastHorizon}h`, icon: Clock },
    { label: 'Hour', value: `${requestPayload.hour}:00`, icon: Calendar },
    { label: 'Mean', value: formatGw(loadFeatures.mean), icon: BarChart3 },
    { label: 'Risk', value: stats.risk, icon: Gauge }
  ];

  const summaryCards = [
    {
      label: 'Next hour',
      value: predictions.length ? formatGw(predictions[0]) : '—',
      sub: predictions.length ? formatMw(predictions[0]) : 'Run a forecast',
      icon: Target
    },
    {
      label: 'Forecast delta',
      value: stats.delta ? formatMw(Math.abs(stats.delta)) : 'Stable',
      sub: stats.direction === 'up' ? 'Demand rising' : stats.direction === 'down' ? 'Demand easing' : 'No movement',
      icon: stats.direction === 'down' ? TrendingDown : TrendingUp
    },
    {
      label: 'Peak window',
      value: stats.forecastPeak ? formatGw(stats.forecastPeak) : formatGw(stats.peak),
      sub: stats.peakStep ? `Forecast step +${stats.peakStep}h` : 'Historical peak',
      icon: Activity
    },
    {
      label: 'Load factor',
      value: `${loadFeatures.loadFactor.toFixed(1)}%`,
      sub: `Volatility ${loadFeatures.volatility.toFixed(1)}%`,
      icon: Gauge
    }
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: EASE }}
      className="relative space-y-4"
    >
      <AnimatePresence>
        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 z-50 flex items-center justify-center rounded-2xl bg-paper/85 p-6 backdrop-blur-md"
          >
            <motion.div
              initial={{ scale: 0.96, y: 14 }}
              animate={{ scale: 1, y: 0 }}
              className="w-full max-w-sm rounded-2xl border border-ink/15 bg-paper-50 p-8 text-center shadow-lift"
            >
              <div className="relative mx-auto mb-6 h-16 w-16">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                  className="absolute inset-0 rounded-full border-2 border-ink/10 border-t-moss"
                />
                <div className="absolute inset-0 flex items-center justify-center">
                  <Cpu className="text-moss" size={22} />
                </div>
              </div>
              <p className="label text-ink/40">Model 2 · sequence</p>
              <h3 className="mt-2 text-lg font-medium tracking-[-0.03em]">Running the forecast</h3>
              <p className="mt-2 text-xs leading-relaxed text-ink/45">
                Sending lag values, rolling statistics, calendar context and the 24-hour sequence.
              </p>
              <div className="mt-5 flex items-center justify-center gap-2 text-[0.68rem] text-ink/40">
                <StatusDot tone="lime" />
                {predictions.length > 0 ? `${predictions.length}/${forecastHorizon} steps completed` : 'Preparing payload'}
              </div>
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
            className="absolute inset-0 z-50 flex items-center justify-center rounded-2xl bg-ink/45 p-6 backdrop-blur-sm"
          >
            <motion.div
              initial={{ scale: 0.96, y: 14 }}
              animate={{ scale: 1, y: 0 }}
              className="w-full max-w-md overflow-hidden rounded-2xl border border-ink/15 bg-paper-50 shadow-lift"
            >
              <div className="flex items-center justify-between border-b border-ink/10 px-6 py-4">
                <div className="flex items-center gap-2.5">
                  <span className="flex h-7 w-7 items-center justify-center rounded-full bg-ink text-signal">
                    <Zap size={13} />
                  </span>
                  <span className="text-sm font-medium">Prediction engine</span>
                </div>
                <span className="label-sm text-ink/30">Model 2</span>
              </div>
              <div className="px-6 py-6">
                <h3 className="text-xl font-medium tracking-[-0.03em]">Before you run a forecast</h3>
                <p className="mt-3 text-sm leading-relaxed text-ink/55">
                  Forecasts are generated from historical patterns and model features. Validate
                  operational decisions against live grid data before acting.
                </p>
                <div className="mt-5 rounded-xl border border-ink/10 bg-paper-100 p-3.5">
                  <p className="label-sm text-ink/35">Good practice</p>
                  <p className="mt-1.5 text-[0.7rem] leading-relaxed text-ink/50">
                    Compare the ensemble output with your telemetry and confidence interval before dispatching.
                  </p>
                </div>
              </div>
              <div className="border-t border-ink/10 px-6 py-4">
                <button
                  onClick={() => dismissDisclaimer()}
                  className={buttonClass('primary', 'md', 'w-full')}
                >
                  Acknowledge and continue
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Header */}
      <div className="surface overflow-hidden">
        <div className="flex flex-col gap-6 border-b border-ink/10 px-5 py-6 sm:px-7 xl:flex-row xl:items-center xl:justify-between">
          <div className="flex items-start gap-4">
            <span className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl border border-ink/10 bg-ink text-signal">
              <Zap size={22} />
            </span>
            <div>
              <div className="flex flex-wrap items-center gap-2.5">
                <p className="label text-ink/35">Model 2 · sequence</p>
                <Badge tone="moss">Ensemble</Badge>
              </div>
              <h1 className="mt-2 text-xl font-medium tracking-[-0.04em] sm:text-2xl">Load forecaster</h1>
              <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-2 text-[0.68rem] text-ink/45">
                <span className="flex items-center gap-1.5">
                  {backendStatusTone === 'green' ? (
                    <Wifi size={13} className="text-moss" />
                  ) : (
                    <WifiOff size={13} className={backendStatusTone === 'amber' ? 'text-amber-600' : 'text-clay'} />
                  )}
                  {backendStatusLabel}
                </span>
                {backendHealth.latency && <span className="mono-num text-ink/35">{backendHealth.latency} ms</span>}
                <span className="label-sm text-ink/30">model {requestPayload.model}</span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 sm:gap-3">
            {headerStats.map((item) => (
              <div key={item.label} className="rounded-xl border border-ink/10 bg-paper-100 px-3.5 py-3">
                <div className="flex items-center gap-1.5 text-ink/35">
                  <item.icon size={12} />
                  <span className="label-sm">{item.label}</span>
                </div>
                <p className="mono-num mt-2 text-sm font-medium text-ink">{item.value}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-12">
        {/* Controls */}
        <div className="space-y-4 xl:col-span-4">
          <Panel className="p-5">
            <PanelHeading
              icon={Layers}
              title="Model 2 scenarios"
              meta={
                <button
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className={buttonClass('soft', 'sm', 'h-8 px-3 text-[0.68rem]')}
                >
                  <Sliders size={13} />
                  {showAdvanced ? 'Hide JSON' : 'Show JSON'}
                </button>
              }
            />
            <div className="mt-4 grid grid-cols-2 gap-2">
              {[
                { label: 'Postman sample', type: 'postman' },
                { label: 'Low demand', type: 'low' },
                { label: 'High demand', type: 'high' },
                { label: 'Evening peak', type: 'peak-evening' }
              ].map((preset) => (
                <button
                  key={preset.type}
                  onClick={() => applyPreset(preset.type)}
                  className="rounded-lg border border-ink/12 bg-paper-50 px-3 py-2.5 text-left text-[0.72rem] font-medium text-ink/70 transition-all duration-300 hover:border-ink/30 hover:bg-white hover:text-ink"
                >
                  {preset.label}
                </button>
              ))}
            </div>
          </Panel>

          <Panel className="p-5">
            <PanelHeading
              icon={Database}
              title="Last 24 hours"
              meta={<span className="label-sm text-ink/30">MW</span>}
            />

            <div className={`mt-4 space-y-4 overflow-y-auto pr-2 custom-scrollbar ${isLargeDesktop ? 'h-[430px]' : 'h-[340px]'}`}>
              {historicalLoads.map((value, index) => {
                const percent = ((value - 10000) / 80000) * 100;
                return (
                  <div key={index} className="space-y-1.5">
                    <div className="flex items-baseline justify-between text-[0.68rem]">
                      <span className="label-sm text-ink/35">{index === 23 ? 'Now' : `-${23 - index}h`}</span>
                      <span className="mono-num font-medium text-ink/70">{formatMw(value)}</span>
                    </div>
                    <input
                      type="range"
                      min={10000}
                      max={90000}
                      step={100}
                      value={value}
                      onChange={(event) => handleInputChange(index, event.target.value)}
                      className="range"
                      style={{
                        background: `linear-gradient(to right, #0F3D2E ${percent}%, rgba(17,24,19,0.12) ${percent}%, rgba(17,24,19,0.12) 100%)`,
                      }}
                      aria-label={`Load at ${index === 23 ? 'now' : `minus ${23 - index} hours`}`}
                    />
                  </div>
                );
              })}
            </div>
          </Panel>

          <Panel className="space-y-5 p-5">
            <div className="grid grid-cols-3 gap-2.5">
              <label className="space-y-1.5">
                <span className="label-sm block text-ink/40">Hour</span>
                <input
                  type="number"
                  min={0}
                  max={23}
                  value={modelContext.hour}
                  onChange={(event) => setModelContext((prev) => ({ ...prev, hour: Number(event.target.value) }))}
                  className="h-10 w-full rounded-lg border border-ink/15 bg-paper-50 px-2.5 text-sm text-ink transition-all focus:border-ink focus:outline-none focus:ring-2 focus:ring-signal-300/60"
                />
              </label>
              <label className="space-y-1.5">
                <span className="label-sm block text-ink/40">Day</span>
                <select
                  value={modelContext.day_of_week}
                  onChange={(event) => setModelContext((prev) => ({ ...prev, day_of_week: Number(event.target.value) }))}
                  className="h-10 w-full cursor-pointer rounded-lg border border-ink/15 bg-paper-50 px-2.5 text-sm text-ink transition-all focus:border-ink focus:outline-none focus:ring-2 focus:ring-signal-300/60"
                >
                  {WEEKDAY_LABELS.map((label, index) => <option key={label} value={index}>{label}</option>)}
                </select>
              </label>
              <label className="space-y-1.5">
                <span className="label-sm block text-ink/40">Month</span>
                <select
                  value={modelContext.month}
                  onChange={(event) => setModelContext((prev) => ({ ...prev, month: Number(event.target.value) }))}
                  className="h-10 w-full cursor-pointer rounded-lg border border-ink/15 bg-paper-50 px-2.5 text-sm text-ink transition-all focus:border-ink focus:outline-none focus:ring-2 focus:ring-signal-300/60"
                >
                  {MONTH_LABELS.map((label, index) => <option key={label} value={index + 1}>{label}</option>)}
                </select>
              </label>
            </div>

            <button
              onClick={useCurrentCalendar}
              className={buttonClass('outline', 'md', 'w-full')}
            >
              <Calendar size={14} />
              Use current date and hour
            </button>

            <div>
              <div className="mb-2 flex items-baseline justify-between">
                <span className="text-[0.78rem] font-medium text-ink/60">Forecast horizon</span>
                <span className="mono-num text-sm font-medium">{forecastHorizon} hours</span>
              </div>
              <input
                type="range"
                min={1}
                max={24}
                value={forecastHorizon}
                onChange={(event) => setForecastHorizon(Number(event.target.value))}
                className="range"
                style={{
                  background: `linear-gradient(to right, #A5C01D ${horizonPercent}%, rgba(17,24,19,0.12) ${horizonPercent}%, rgba(17,24,19,0.12) 100%)`,
                }}
                aria-label="Forecast horizon in hours"
              />
              <div className="mt-2 flex justify-between">
                <span className="label-sm text-ink/25">1h</span>
                <span className="label-sm text-ink/25">24h</span>
              </div>
            </div>

            <button
              onClick={runPrediction}
              disabled={loading}
              className={buttonClass('primary', 'lg', 'w-full')}
            >
              {loading ? <RefreshCw className="animate-spin" size={17} /> : <Play size={17} />}
              {loading ? 'Predicting…' : 'Run forecast'}
            </button>
          </Panel>

          {showAdvanced && (
            <Panel className="p-5">
              <PanelHeading
                icon={Upload}
                title="Model 2 JSON"
                meta={
                  <button
                    onClick={syncJsonFromControls}
                    className="label-sm rounded-md border border-ink/12 px-2.5 py-1.5 text-ink/50 transition-colors hover:border-ink/30 hover:text-ink"
                  >
                    Sync
                  </button>
                }
              />
              <textarea
                value={rawJsonInput}
                onChange={(event) => setRawJsonInput(event.target.value)}
                className="ui-textarea mt-4 h-56 resize-none"
                spellCheck={false}
              />
              <button
                onClick={handleJsonApply}
                className={buttonClass('soft', 'md', 'mt-3 w-full')}
              >
                Apply JSON payload
              </button>
              {jsonMessage && (
                <p className="mt-3 flex items-center gap-2 text-[0.72rem] text-moss">
                  <CheckCircle size={14} /> {jsonMessage}
                </p>
              )}
            </Panel>
          )}

          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="flex items-start gap-3 rounded-xl border border-clay/25 bg-clay/[0.06] p-4"
              >
                <AlertTriangle className="mt-0.5 shrink-0 text-clay" size={18} />
                <div>
                  <p className="label-sm text-clay">Error</p>
                  <p className="mt-1 text-xs leading-relaxed text-clay/90">{error}</p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Results */}
        <div className="space-y-4 xl:col-span-8">
          <Panel className="relative h-[420px] overflow-hidden p-4 sm:h-[520px] sm:p-5">
            <div className="absolute left-4 top-4 z-10 flex max-w-[calc(100%-8rem)] items-center gap-2.5 sm:left-5 sm:top-5">
              <StatusDot tone="lime" pulse={loading} />
              <h3 className="text-sm font-medium tracking-[-0.02em] sm:text-base">Forecast workbench</h3>
            </div>

            <div className="absolute right-4 top-4 z-10 flex gap-1 rounded-full border border-ink/10 bg-paper-100 p-1 sm:right-5 sm:top-5">
              {['combined', '3d'].map((mode) => (
                <button
                  key={mode}
                  onClick={() => setChartMode(mode)}
                  className={`relative rounded-full px-3.5 py-1.5 text-[0.68rem] font-semibold transition-colors ${
                    chartMode === mode ? 'text-paper' : 'text-ink/45 hover:text-ink/75'
                  }`}
                >
                  {chartMode === mode && (
                    <motion.span
                      layoutId="predictor-chart-toggle"
                      className="absolute inset-0 rounded-full bg-ink"
                      transition={{ type: 'spring', stiffness: 400, damping: 34 }}
                    />
                  )}
                  <span className="relative z-10">{mode === '3d' ? '3D' : '2D'}</span>
                </button>
              ))}
            </div>

            {chartMode === '3d' ? (
              <Suspense
                fallback={
                  <div className="flex h-full items-center justify-center">
                    <div className="h-10 w-10 animate-spin rounded-full border-2 border-ink/10 border-t-moss" />
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
                      <stop offset="5%" stopColor={CHART.ink} stopOpacity={0.22} />
                      <stop offset="95%" stopColor={CHART.ink} stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="colorForecast" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={CHART.moss} stopOpacity={0.28} />
                      <stop offset="95%" stopColor={CHART.moss} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 6" stroke={CHART.grid} vertical={false} />
                  <XAxis
                    dataKey="hour"
                    stroke={CHART.axis}
                    fontSize={10}
                    interval="preserveStartEnd"
                    tickLine={false}
                    axisLine={false}
                  />
                  <YAxis
                    stroke={CHART.axis}
                    fontSize={10}
                    tickLine={false}
                    axisLine={false}
                    width={40}
                    tickFormatter={(value) => `${(value / 1000).toFixed(0)}k`}
                  />
                  <Tooltip
                    content={<CustomTooltip />}
                    cursor={{ stroke: 'rgba(17,24,19,0.18)', strokeDasharray: '3 4' }}
                  />
                  <Legend
                    verticalAlign="bottom"
                    iconType="plainline"
                    wrapperStyle={{ fontSize: '10px', paddingTop: '8px', textTransform: 'uppercase', letterSpacing: '0.12em' }}
                  />
                  <Area
                    type="monotone"
                    dataKey="actual"
                    stroke={CHART.ink}
                    fill="url(#colorActual)"
                    name="Historical load"
                    strokeWidth={2}
                    animationDuration={900}
                  />
                  <Area
                    type="monotone"
                    dataKey="predicted"
                    stroke={CHART.moss}
                    fill="url(#colorForecast)"
                    name="Forecast"
                    strokeWidth={2}
                    strokeDasharray="6 4"
                    animationDuration={1100}
                  />
                  <ReferenceLine
                    x="Now"
                    stroke={CHART.signal}
                    strokeWidth={1.5}
                    strokeDasharray="3 3"
                    label={{ value: 'Now', position: 'top', fill: CHART.moss, fontSize: 10 }}
                  />
                  <Brush
                    dataKey="hour"
                    height={24}
                    stroke={CHART.moss}
                    fill="#F2F0E9"
                    fillOpacity={0.7}
                    travellerWidth={8}
                  />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </Panel>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
            {summaryCards.map((card) => (
              <div key={card.label} className="surface card-interactive p-5">
                <div className="flex items-center justify-between">
                  <p className="label text-ink/35">{card.label}</p>
                  <card.icon className="text-ink/35" size={15} />
                </div>
                <p className="mono-num mt-6 text-xl font-medium tracking-[-0.05em]">{card.value}</p>
                <p className="mt-1.5 text-[0.68rem] text-ink/40">{card.sub}</p>
              </div>
            ))}
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <Panel className="p-5">
              <PanelHeading
                icon={BarChart3}
                title="Feature profile"
                meta={
                  <span className="label-sm text-ink/30">
                    {WEEKDAY_LABELS[requestPayload.day_of_week]} / {MONTH_LABELS[requestPayload.month - 1]}
                  </span>
                }
              />
              <div className="mt-5 h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={profileRows} margin={{ top: 10, right: 10, left: 4, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 6" stroke={CHART.grid} vertical={false} />
                    <XAxis dataKey="name" stroke={CHART.axis} fontSize={10} tickLine={false} axisLine={false} />
                    <YAxis
                      stroke={CHART.axis}
                      fontSize={10}
                      tickLine={false}
                      axisLine={false}
                      width={38}
                      tickFormatter={(value) => `${(value / 1000).toFixed(0)}k`}
                    />
                    <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(17,24,19,0.035)' }} />
                    <Bar dataKey="value" fill={CHART.moss} radius={[3, 3, 0, 0]} animationDuration={900} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Panel>

            <Panel className="p-5">
              <PanelHeading
                icon={Cpu}
                title="Request snapshot"
                meta={
                  lastResponseMeta && (
                    <span className={`label-sm ${lastResponseMeta.fallbackUsed ? 'text-amber-700' : 'text-moss'}`}>
                      {lastResponseMeta.fallbackUsed ? 'Fallback' : 'Model output'}
                    </span>
                  )
                }
              />
              <div className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-2">
                {[
                  ['Hour', `${lastRequestPayload.hour}:00`],
                  ['Day index', lastRequestPayload.day_of_week],
                  ['Month', lastRequestPayload.month],
                  ['Horizon', `${lastRequestPayload.horizon || forecastHorizon}h`],
                  ['Lag 1h', formatMw(lastRequestPayload.load_lag_1h)],
                  ['Rolling std', formatMw(lastRequestPayload.load_rolling_std_24h)]
                ].map(([label, value]) => (
                  <div key={label} className="rounded-lg border border-ink/10 bg-paper-100 p-3">
                    <p className="label-sm text-ink/35">{label}</p>
                    <p className="mono-num mt-1.5 text-xs font-medium text-ink">{value}</p>
                  </div>
                ))}
              </div>
              <div className="mt-4 rounded-lg border border-ink/10 bg-paper-100 p-3.5">
                <p className="label-sm text-ink/35">Response source</p>
                <p className="mt-1.5 text-xs leading-relaxed text-ink/55">
                  {lastResponseMeta
                    ? `${lastResponseMeta.source} returned ${lastResponseMeta.count} forecast step${lastResponseMeta.count === 1 ? '' : 's'}.`
                    : 'Run a forecast to see backend response metadata.'}
                </p>
                {lastResponseMeta?.reason && (
                  <p className="mt-2 text-[0.68rem] leading-relaxed text-amber-700">{lastResponseMeta.reason}</p>
                )}
              </div>
            </Panel>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default LoadPredictor;
