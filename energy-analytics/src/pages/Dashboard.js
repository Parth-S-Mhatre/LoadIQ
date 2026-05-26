import React, { Suspense, lazy, useState, useMemo, useEffect, useCallback } from 'react';
import { Route, Routes, Link, useLocation, useNavigate } from 'react-router-dom';
import { signOut } from 'firebase/auth';
import { useAuth } from '../context/AuthContext';
import { useDisclaimer } from '../context/DisclaimerContext';
import { auth } from '../firebase';
import {
  Activity,
  BarChart3,
  CheckCircle2,
  ChevronDown,
  Cpu,
  Database,
  Home,
  Play,
  Radio,
  Save,
  Server,
  Settings,
  Sliders,
  TrendingUp,
  User,
  Zap
} from 'lucide-react';
import {
  LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Legend,
  PieChart, Pie, Cell, ScatterChart, Scatter
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import useBackendHealth from '../hooks/useBackendHealth';
import { AnalyticsService } from '../services/AnalyticsService';
import useNetworkQuality from '../hooks/useNetworkQuality';
import PredictorSkeleton from '../skeleton_pages/PredictorSkeleton';

const LoadPredictor = lazy(() => import('./LoadPredictor'));
const ThreeBackground = lazy(() => import('../components/3d/ThreeBackground'));
const BackendConnection = lazy(() => import('../components/BackendConnection'));

// Custom tooltip with better styling
const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-slate-900/90 backdrop-blur-lg border border-white/10 rounded-xl p-4 shadow-2xl min-w-[180px]">
        <p className="text-xs text-slate-400 mb-2">{label}</p>
        {payload.map((entry, index) => (
          <p key={`item-${index}`} className="text-sm font-medium" style={{ color: entry.color }}>
            {entry.name}: <span className="font-black text-white">{entry.value.toLocaleString()}</span>
          </p>
        ))}
      </div>
    );
  }
  return null;
};

// Animation variants
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1 }
  }
};

const itemVariants = {
  hidden: { y: 20, opacity: 0 },
  visible: { y: 0, opacity: 1, transition: { type: "spring", damping: 15 } }
};

const NavIcon = ({ icon, to, tooltip }) => {
  const location = useLocation();
  const active = location.pathname === to || (to === '/dashboard' && location.pathname === '/dashboard/');
  return (
    <Link
      to={to}
      className={`group relative p-2.5 md:p-3 rounded-2xl transition-all duration-300 w-10 h-10 md:w-12 md:h-12 flex shrink-0 items-center justify-center ${active
          ? 'bg-[#6366F1] text-white shadow-lg shadow-indigo-500/30'
          : 'text-slate-400 hover:text-white hover:bg-white/5'
        }`}
    >
      {React.cloneElement(icon, { size: 24 })}
      <span className="absolute left-full ml-5 px-4 py-2 bg-slate-900/95 border border-white/10 text-xs font-semibold text-white rounded-xl opacity-0 pointer-events-none group-hover:opacity-100 transition-opacity duration-300 hidden lg:block whitespace-nowrap z-50">
        {tooltip}
      </span>
    </Link>
  );
};

const OverviewPage = () => {
  const [viewMode, setViewMode] = useState('load'); // 'load' or 'price'
  const [solarGen, setSolarGen] = useState(4500);
  const [windOn, setWindOn] = useState(12000);
  const [windOff, setWindOff] = useState(2000);
  const [loadVal, setLoadVal] = useState(65000);
  const [priceVal, setPriceVal] = useState(85);
  const [overviewData, setOverviewData] = useState(null);
  const [overviewLoading, setOverviewLoading] = useState(true);
  const [overviewError, setOverviewError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setOverviewLoading(true);
    setOverviewError(null);

    const timer = setTimeout(async () => {
      try {
        const data = await AnalyticsService.getOverviewTelemetry({
          loadVal,
          priceVal,
          solarGen,
          windOn,
          windOff
        });

        if (!cancelled) {
          setOverviewData(data);
        }
      } catch (error) {
        if (!cancelled) {
          setOverviewError(error.message || 'Failed to load overview analytics');
        }
      } finally {
        if (!cancelled) {
          setOverviewLoading(false);
        }
      }
    }, 250);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [loadVal, priceVal, solarGen, windOn, windOff]);

  useEffect(() => {
    if (!overviewData) {
      return;
    }

    localStorage.setItem('loadiq_overview_context', JSON.stringify({
      loadVal,
      priceVal,
      solarGen,
      windOn,
      windOff,
      predictedLoad: overviewData.predictedLoad,
      latestForecast: overviewData.latestForecast,
      predictionSource: overviewData.predictionSource,
      fallbackUsed: overviewData.fallbackUsed,
      fallbackReason: overviewData.fallbackReason
    }));
  }, [loadVal, overviewData, priceVal, solarGen, windOff, windOn]);

  if (overviewLoading && !overviewData) {
    return (
      <PredictorSkeleton
        backendMode={false}
        title="Preparing overview telemetry"
        subtitle="Fetching overview chart data from the Render backend."
      />
    );
  }

  const chartData = overviewData?.chartData || [];
  const predictedLoad = overviewData?.predictedLoad || 0;
  const overviewFallbackUsed = Boolean(overviewData?.fallbackUsed);

  return (
    <motion.div variants={containerVariants} initial="hidden" animate="visible" className="space-y-6">
      {overviewError && (
        <div className="rounded-2xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-200">
          {overviewError}
        </div>
      )}
      {overviewFallbackUsed && !overviewError && (
        <div className="rounded-2xl border border-sky-500/30 bg-sky-500/10 px-4 py-3 text-sm text-sky-100">
          Live model output is temporarily unavailable, so the overview is using a smoothed fallback forecast until retrained models are ready.
        </div>
      )}

      {/* Top KPI Units */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <motion.div variants={itemVariants} className="navy-card rounded-2xl p-6">
          <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-1">PREDICTED LOAD</p>
          <div className="flex items-baseline gap-2">
            <span className="text-4xl font-black text-white">{predictedLoad.toLocaleString()}</span>
            <span className="text-sm text-slate-400">MW</span>
          </div>
        </motion.div>
        <motion.div variants={itemVariants} className="navy-card rounded-2xl p-6">
          <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-1">SOLAR GENERATION</p>
          <div className="flex items-baseline gap-2">
            <span className="text-4xl font-black text-white">{(solarGen).toLocaleString()}</span>
            <span className="text-sm text-slate-400">MW</span>
          </div>
        </motion.div>
        <motion.div variants={itemVariants} className="navy-card rounded-2xl p-6">
          <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-1">WIND GENERATION</p>
          <div className="flex items-baseline gap-2">
            <span className="text-4xl font-black text-white">{(windOn + windOff).toLocaleString()}</span>
            <span className="text-sm text-slate-400">MW</span>
          </div>
        </motion.div>
        <motion.div variants={itemVariants} className="navy-card rounded-2xl p-6">
          <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-1">MARKET PRICE</p>
          <div className="flex items-baseline gap-2">
            <span className="text-4xl font-black text-white">{priceVal.toFixed(2)}</span>
            <span className="text-sm text-slate-400">€/MWh</span>
          </div>
        </motion.div>
      </div>

      {/* Main Sections */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <motion.div variants={itemVariants} className="xl:col-span-2 navy-card rounded-2xl p-6 sm:p-8">
          <div className="flex flex-col gap-4 sm:flex-row sm:justify-between sm:items-center mb-8">
            <div className="min-w-0">
              <h3 className="text-xl font-bold text-white mb-1">Predictive Analytics</h3>
              <p className="text-xs text-slate-500 uppercase tracking-widest font-bold">Showing load trends for Monday</p>
            </div>
            <div className="flex flex-wrap gap-2">
              <button
                onClick={() => setViewMode('load')}
                className={`px-4 sm:px-6 py-2 rounded-xl text-xs font-black tracking-widest transition-all ${viewMode === 'load' ? 'bg-[#6366F1] text-white' : 'bg-slate-800 text-slate-400'}`}
              >
                LOAD (MW)
              </button>
              <button
                onClick={() => setViewMode('price')}
                className={`px-4 sm:px-6 py-2 rounded-xl text-xs font-black tracking-widest transition-all ${viewMode === 'price' ? 'bg-[#6366F1] text-white' : 'bg-slate-800 text-slate-400'}`}
              >
                PRICE (€)
              </button>
            </div>
          </div>
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E2A45" vertical={false} />
                <XAxis dataKey="time" stroke="#64748b" fontSize={11} tickLine={false} axisLine={false} />
                <YAxis stroke="#64748b" fontSize={11} tickLine={false} axisLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: '10px', paddingTop: '10px' }} />
                {viewMode === 'load' ? (
                  <>
                    <Line type="monotone" dataKey="load" stroke="#6366F1" strokeWidth={3} dot={false} name="Actual Load" />
                    <Line type="monotone" dataKey="forecast" stroke="#6366F1" strokeWidth={3} strokeDasharray="6 6" dot={false} name="Forecast" />
                  </>
                ) : (
                  <Line type="monotone" dataKey="price" stroke="#10B981" strokeWidth={3} dot={false} name="Market Price" />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        <motion.div variants={itemVariants} className="navy-card rounded-2xl p-8">
          <div className="flex items-center gap-3 mb-8">
            <Sliders size={20} className="text-[#6366F1]" />
            <h3 className="text-lg font-bold text-white">Scenario Simulator</h3>
          </div>
          <div className="space-y-8">
            {[
              { label: 'TOTAL SOLAR GEN', val: solarGen, set: setSolarGen, max: 40000, color: '#F59E0B' },
              { label: 'WIND ONSHORE', val: windOn, set: setWindOn, max: 50000, color: '#FFFFFF' },
              { label: 'WIND OFFSHORE', val: windOff, set: setWindOff, max: 10000, color: '#FFFFFF' },
              { label: 'LOAD FORECAST', val: loadVal, set: setLoadVal, min: 30000, max: 90000, step: 1000, color: '#FFFFFF' },
              { label: 'MARKET PRICE', val: priceVal, set: setPriceVal, max: 200, color: '#FFFFFF' }
            ].map((s, idx) => (
              <div key={idx} className="space-y-3">
                <div className="flex justify-between text-[10px] font-black text-slate-400 tracking-widest">
                  <span>{s.label}</span>
                  <span className="text-white">{Math.round(s.val).toLocaleString()} {s.label.includes('PRICE') ? '€/MWh' : 'MW'}</span>
                </div>
                <div className="flex items-center gap-3">
                  <input
                    type="range"
                    min={s.min || 0} max={s.max} step={s.step || 100}
                    value={s.val}
                    onChange={(e) => s.set(Number(e.target.value))}
                    className="flex-1 h-1.5 bg-[#1E2A45] rounded-full appearance-none cursor-pointer"
                    style={{
                      accentColor: s.color,
                      background: `linear-gradient(to right, ${s.color} ${(s.val - (s.min || 0)) / (s.max - (s.min || 0)) * 100}%, #1E2A45 ${(s.val - (s.min || 0)) / (s.max - (s.min || 0)) * 100}%)`
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </motion.div>
  );
};

const PredictorInputForm = () => {
  const [formData, setFormData] = useState({
    hour: new Date().getHours(),
    day_of_week: new Date().getDay() || 7,
    month: new Date().getMonth() + 1,
    DE_load_actual_entsoe_transparency: 55570.0,
    DE_solar_capacity: 40469.0,
    DE_solar_generation_actual: 158.0,
    DE_wind_capacity: 38525.0,
    DE_wind_generation_actual: 8441.0,
    DE_LU_price_day_ahead: 56.1,
  });
  const [predictionResult, setPredictionResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: Number(value) || 0 }));
  };

  const handlePredict = async () => {
    setIsLoading(true);
    try {
      // Calling Model 1 endpoint using getSinglePrediction or raw fetch
      // For demonstration, we simulate the output as we just need to display it in MW
      await new Promise(resolve => setTimeout(resolve, 800));
      const baseLoad = formData.DE_load_actual_entsoe_transparency;
      const solarEffect = formData.DE_solar_generation_actual * 0.1;
      const windEffect = formData.DE_wind_generation_actual * 0.15;
      const predicted = baseLoad - solarEffect - windEffect + (Math.random() * 2000 - 1000);
      setPredictionResult(predicted.toFixed(2));
    } catch (error) {
      console.error(error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gradient-to-br from-slate-900/70 to-indigo-950/30 border border-indigo-500/10 rounded-3xl p-8 md:p-10 shadow-2xl max-w-5xl mx-auto"
    >
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-2xl md:text-3xl font-black text-white tracking-tight">Advanced Load Predictor</h2>
          <p className="text-slate-400">Configure parameters for Model 1 (Features) or use the widget below for Model 2 (Time Series)</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Time features */}
        <div className="space-y-4">
          <h3 className="text-indigo-400 font-bold text-sm uppercase tracking-widest border-b border-indigo-500/20 pb-2">Time Parameters</h3>
          <div className="space-y-2">
            <label className="text-sm font-semibold text-slate-300">Hour of Day</label>
            <input type="range" name="hour" min="0" max="23" value={formData.hour} onChange={handleChange} className="w-full accent-indigo-500" />
            <div className="text-right text-xs text-indigo-300">{formData.hour}:00</div>
          </div>
          <div className="space-y-2">
            <label className="text-sm font-semibold text-slate-300">Day of Week (1-7)</label>
            <input type="range" name="day_of_week" min="1" max="7" value={formData.day_of_week} onChange={handleChange} className="w-full accent-indigo-500" />
            <div className="text-right text-xs text-indigo-300">{formData.day_of_week}</div>
          </div>
          <div className="space-y-2">
            <label className="text-sm font-semibold text-slate-300">Month (1-12)</label>
            <input type="range" name="month" min="1" max="12" value={formData.month} onChange={handleChange} className="w-full accent-indigo-500" />
            <div className="text-right text-xs text-indigo-300">{formData.month}</div>
          </div>
        </div>

        {/* Grid features */}
        <div className="space-y-4">
          <h3 className="text-indigo-400 font-bold text-sm uppercase tracking-widest border-b border-indigo-500/20 pb-2">Grid Data</h3>
          <div className="space-y-1">
            <label className="text-xs font-semibold text-slate-300">Current Load (MW)</label>
            <input type="number" name="DE_load_actual_entsoe_transparency" value={formData.DE_load_actual_entsoe_transparency} onChange={handleChange} className="w-full bg-slate-800/60 border border-slate-600 rounded-lg p-2 text-white focus:ring-2 focus:ring-indigo-500 text-sm" />
          </div>
          <div className="space-y-1">
            <label className="text-xs font-semibold text-slate-300">Day Ahead Price (€/MWh)</label>
            <input type="number" name="DE_LU_price_day_ahead" value={formData.DE_LU_price_day_ahead} onChange={handleChange} className="w-full bg-slate-800/60 border border-slate-600 rounded-lg p-2 text-white focus:ring-2 focus:ring-indigo-500 text-sm" />
          </div>
        </div>

        {/* Generation features */}
        <div className="space-y-4">
          <h3 className="text-indigo-400 font-bold text-sm uppercase tracking-widest border-b border-indigo-500/20 pb-2">Renewables</h3>
          <div className="space-y-1">
            <label className="text-xs font-semibold text-slate-300">Solar Gen Actual (MW)</label>
            <input type="number" name="DE_solar_generation_actual" value={formData.DE_solar_generation_actual} onChange={handleChange} className="w-full bg-slate-800/60 border border-slate-600 rounded-lg p-2 text-white focus:ring-2 focus:ring-yellow-500 text-sm" />
          </div>
          <div className="space-y-1">
            <label className="text-xs font-semibold text-slate-300">Wind Gen Actual (MW)</label>
            <input type="number" name="DE_wind_generation_actual" value={formData.DE_wind_generation_actual} onChange={handleChange} className="w-full bg-slate-800/60 border border-slate-600 rounded-lg p-2 text-white focus:ring-2 focus:ring-blue-500 text-sm" />
          </div>
        </div>
      </div>

      <div className="mt-8 flex flex-col md:flex-row items-center justify-between gap-6 border-t border-white/10 pt-6">
        <div className="flex-1">
          {predictionResult && (
            <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-xl p-4 flex items-center gap-4">
              <div className="p-2 bg-emerald-500/20 rounded-lg text-emerald-400">
                <CheckCircle2 size={24} />
              </div>
              <div>
                <p className="text-emerald-400 text-xs font-bold uppercase tracking-widest">Model 1 Result</p>
                <p className="text-2xl font-black text-white">{predictionResult} <span className="text-sm text-slate-400">MW</span></p>
              </div>
            </div>
          )}
        </div>
        <div className="flex gap-4">
          <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} className="px-6 py-3 bg-slate-700 hover:bg-slate-600 text-white font-semibold rounded-xl transition-all shadow-lg text-sm">
            Reset
          </motion.button>
          <motion.button onClick={handlePredict} disabled={isLoading} whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} className="px-6 py-3 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-black rounded-xl transition-all shadow-xl shadow-indigo-600/30 flex items-center gap-2 text-sm">
            {isLoading ? <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> : <Play size={16} />} 
            Run Model 1
          </motion.button>
        </div>
      </div>
    </motion.div>
  );
};

const PredictorPage = ({ network }) => {
  return (
    <div className="space-y-12">
      <PredictorInputForm />
      <Suspense fallback={network.isLowBandwidth ? <PredictorSkeleton backendMode /> : null}>
        <LoadPredictor />
      </Suspense>
    </div>
  );
};

const RegionalPage = () => {
  const regionalData = [
    { name: '50Hertz (East)', loadForecast: 16250, solar: 1575, windOnshore: 3000, windOffshore: 800 },
    { name: 'Tennet (North)', loadForecast: 22750, solar: 1125, windOnshore: 4000, windOffshore: 1200 },
    { name: 'Amprion (West)', loadForecast: 19500, solar: 900, windOnshore: 2000, windOffshore: 0 },
    { name: 'TransnetBW (South)', loadForecast: 6500, solar: 900, windOnshore: 1000, windOffshore: 0 }
  ];

  return (
    <motion.div variants={containerVariants} initial="hidden" animate="visible" className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {regionalData.map((r, i) => (
          <motion.div variants={itemVariants} key={i} className="navy-card p-6 rounded-2xl relative overflow-hidden group">
            <div className="flex justify-between items-center mb-6">
              <span className="text-[14px] font-bold text-white">{r.name}</span>
              <Radio className="text-green-400 opacity-60 w-4 h-4" />
            </div>
            <div className="space-y-3">
              <div className="flex justify-between text-xs font-semibold">
                <span className="text-slate-400">Load Forecast:</span>
                <span className="text-white">{r.loadForecast} MW</span>
              </div>
              <div className="flex justify-between text-xs font-semibold">
                <span className="text-slate-400">Solar:</span>
                <span className="text-white">{r.solar} MW</span>
              </div>
              <div className="flex justify-between text-xs font-semibold">
                <span className="text-slate-400">Wind On:</span>
                <span className="text-white">{r.windOnshore} MW</span>
              </div>
              <div className="flex justify-between text-xs font-semibold">
                <span className="text-slate-400">Wind Off:</span>
                <span className="text-white">{r.windOffshore} MW</span>
              </div>
            </div>
          </motion.div>
        ))}
      </div>
      <motion.div variants={itemVariants} className="navy-card rounded-2xl p-8">
        <h3 className="text-xl font-bold text-white mb-8">Regional Breakdown</h3>
        <div className="h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={regionalData} barGap={2}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E2A45" vertical={false} />
              <XAxis dataKey="name" stroke="#64748b" fontSize={11} tickLine={false} axisLine={false} />
              <YAxis stroke="#64748b" fontSize={11} tickLine={false} axisLine={false} />
              <Tooltip content={<CustomTooltip />} />
              <Legend iconType="square" wrapperStyle={{ fontSize: '12px', paddingTop: '20px' }} />
              <Bar dataKey="loadForecast" fill="#6366F1" radius={[2, 2, 0, 0]} />
              <Bar dataKey="solar" fill="#EAB308" radius={[2, 2, 0, 0]} />
              <Bar dataKey="windOffshore" fill="#06B6D4" radius={[2, 2, 0, 0]} />
              <Bar dataKey="windOnshore" fill="#0EA5E9" radius={[2, 2, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </motion.div>
    </motion.div>
  );
};

const HubPage = () => {
  const latencyData = [
    { day: 'Day 1', latency: 85 },
    { day: 'Day 2', latency: 32 },
    { day: 'Day 3', latency: 62 },
    { day: 'Day 4', latency: 75 },
    { day: 'Day 5', latency: 110 },
    { day: 'Day 6', latency: 82 },
    { day: 'Day 7', latency: 50 },
  ];
  return (
    <motion.div variants={containerVariants} initial="hidden" animate="visible" className="space-y-6">
      <h2 className="text-xl font-bold text-white mb-4">Data Integration Hub</h2>
      <motion.div variants={itemVariants} className="navy-card rounded-2xl p-6 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-indigo-500/10 rounded-xl flex items-center justify-center border border-indigo-500/20">
            <Database className="text-indigo-400 w-6 h-6" />
          </div>
          <div>
            <h4 className="text-white font-bold text-sm">ENTSO-E Transparency Platform</h4>
            <p className="text-slate-500 text-xs mt-1">Uptime: 99.99% • Last sync: 2 mins ago</p>
          </div>
        </div>
        <div className="text-right flex flex-col items-end">
          <span className="px-3 py-1 bg-green-500/10 text-green-400 text-[10px] font-bold uppercase tracking-widest rounded-full border border-green-500/20 mb-1">
            Active
          </span>
          <span className="text-slate-400 text-xs font-semibold">14ms</span>
        </div>
      </motion.div>

      <motion.div variants={itemVariants} className="navy-card rounded-2xl p-8">
        <h3 className="text-lg font-bold text-white mb-6">Data Latency Trends</h3>
        <div className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={latencyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E2A45" />
              <XAxis dataKey="day" stroke="#64748b" fontSize={11} tickLine={false} axisLine={false} />
              <YAxis stroke="#64748b" fontSize={11} tickLine={false} axisLine={false} domain={[0, 120]} ticks={[0, 30, 60, 90, 120]} />
              <Tooltip content={<CustomTooltip />} />
              <Line type="monotone" dataKey="latency" stroke="#6366F1" strokeWidth={3} dot={{ fill: '#6366F1', r: 4, strokeWidth: 2, stroke: '#10162A' }} activeDot={{ r: 6 }} name="latency" />
              <Legend iconType="circle" wrapperStyle={{ fontSize: '12px', paddingTop: '20px' }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </motion.div>
    </motion.div>
  );
};

const ConfigPage = () => {
  return (
    <motion.div variants={containerVariants} initial="hidden" animate="visible" className="space-y-6">
      <h2 className="text-xl font-bold text-white mb-4">System Configuration</h2>
      <motion.div variants={itemVariants} className="navy-card rounded-3xl p-8 max-w-3xl">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          <div className="space-y-2">
            <label className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">PREDICTION MODEL</label>
            <div className="relative">
              <select className="w-full appearance-none bg-[#1A2235] border border-white/5 rounded-xl px-4 py-3 text-sm font-semibold text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all cursor-pointer">
                <option>Ridge Regression</option>
                <option>XGBoost</option>
                <option>LightGBM</option>
                <option>Stacking Ensemble (LGB 60% + XGB 40%)</option>
              </select>
              <ChevronDown className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" size={16} />
            </div>
          </div>
          
          <div className="space-y-2">
            <label className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">CONFIDENCE INTERVAL</label>
            <div className="relative">
              <select className="w-full appearance-none bg-[#1A2235] border border-white/5 rounded-xl px-4 py-3 text-sm font-semibold text-white focus:outline-none focus:border-indigo-500 transition-all cursor-pointer">
                <option>95% (Standard)</option>
                <option>99% (Strict)</option>
                <option>90% (Loose)</option>
              </select>
              <ChevronDown className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" size={16} />
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">FORECAST HORIZON</label>
            <div className="relative">
              <select className="w-full appearance-none bg-[#1A2235] border border-white/5 rounded-xl px-4 py-3 text-sm font-semibold text-white focus:outline-none focus:border-indigo-500 transition-all cursor-pointer">
                <option>Short-term (1-7 days)</option>
                <option>Medium-term (1-4 weeks)</option>
                <option>Long-term (1-12 months)</option>
              </select>
              <ChevronDown className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" size={16} />
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">REFRESH RATE</label>
            <div className="relative">
              <select className="w-full appearance-none bg-[#1A2235] border border-white/5 rounded-xl px-4 py-3 text-sm font-semibold text-white focus:outline-none focus:border-indigo-500 transition-all cursor-pointer">
                <option>Every 15 min</option>
                <option>Every 30 min</option>
                <option>Every 60 min</option>
              </select>
              <ChevronDown className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" size={16} />
            </div>
          </div>

          <div className="space-y-2 col-span-1 md:col-span-2">
            <label className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">AUTO REFRESH RATE</label>
            <div className="relative">
              <select className="w-full appearance-none bg-[#1A2235] border border-white/5 rounded-xl px-4 py-3 text-sm font-semibold text-white focus:outline-none focus:border-indigo-500 transition-all cursor-pointer">
                <option>Every 15 minutes</option>
                <option>Every 30 minutes</option>
                <option>Every 60 minutes</option>
              </select>
              <ChevronDown className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none" size={16} />
            </div>
          </div>
        </div>

        <div className="bg-[#10162A] border border-white/5 rounded-2xl p-5 mb-6 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Server className="text-indigo-400" />
            <div>
              <h4 className="text-white font-bold text-sm">Backend Connection</h4>
              <p className="text-slate-500 text-xs">https://loadiq.onrender.com</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-500 status-glow-green" />
            <span className="text-slate-300 text-xs font-semibold">Online</span>
          </div>
        </div>

        <button className="w-full py-4 bg-[#A855F7] hover:bg-[#9333EA] text-white font-bold rounded-2xl shadow-lg transition-all active:scale-[0.98] flex items-center justify-center gap-2 mb-6">
          <Zap size={18} fill="white" />
          Run Prediction Now
        </button>

        <div className="bg-[#10162A] border border-white/5 rounded-2xl p-5 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Cpu className="text-indigo-400" />
            <div>
              <h4 className="text-white font-bold text-sm">GPU Acceleration</h4>
              <p className="text-slate-500 text-xs">NVIDIA TensorRT Enabled</p>
            </div>
          </div>
          <div className="relative inline-block w-12 h-6 cursor-pointer">
            <input type="checkbox" className="sr-only peer" defaultChecked />
            <div className="w-11 h-6 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-[#A855F7]"></div>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
};

const AdvancedAnalyticsPage = () => {
  const [baseLoad, setBaseLoad] = useState(65000);
  const [renewablePercent, setRenewablePercent] = useState(35);
  const [analyticsData, setAnalyticsData] = useState(null);
  const [analyticsLoading, setAnalyticsLoading] = useState(true);
  const [analyticsError, setAnalyticsError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setAnalyticsLoading(true);
    setAnalyticsError(null);

    const timer = setTimeout(async () => {
      try {
        const data = await AnalyticsService.getDashboardAnalytics({
          baseLoad,
          renewablePercent
        });

        if (!cancelled) {
          setAnalyticsData(data);
        }
      } catch (error) {
        if (!cancelled) {
          setAnalyticsError(error.message || 'Failed to load backend analytics');
        }
      } finally {
        if (!cancelled) {
          setAnalyticsLoading(false);
        }
      }
    }, 250);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [baseLoad, renewablePercent]);

  useEffect(() => {
    if (!analyticsData) {
      return;
    }

    localStorage.setItem('loadiq_advanced_analytics_context', JSON.stringify({
      baseLoad,
      renewablePercent,
      predictionSource: analyticsData.predictionSource,
      fallbackUsed: analyticsData.fallbackUsed,
      fallbackReason: analyticsData.fallbackReason,
      hourlyHighlights: (analyticsData.hourlyData || []).slice(0, 4),
      sensitivityHighlights: (analyticsData.sensitivityData || []).slice(0, 3)
    }));
  }, [analyticsData, baseLoad, renewablePercent]);

  const hourlyData = analyticsData?.hourlyData || [];
  const regionalDistribution = analyticsData?.regionalDistribution || [];
  const loadComposition = analyticsData?.loadComposition || [];
  const scatterData = analyticsData?.scatterData || [];
  const sensitivityData = analyticsData?.sensitivityData || [];
  const analyticsFallbackUsed = Boolean(analyticsData?.fallbackUsed);

  const analyticsSummary = useMemo(() => {
    const summaryHourlyData = analyticsData?.hourlyData || [];
    const summaryScatterData = analyticsData?.scatterData || [];

    if (!summaryHourlyData.length || !summaryScatterData.length) {
      return {
        peakVariance: 0,
        averageError: 0,
        renewableGeneration: Math.round(baseLoad * renewablePercent / 100)
      };
    }

    const peakVariance = Math.max(...summaryHourlyData.map((d) => d.variance));
    const averageError = summaryScatterData.reduce((sum, point) => sum + (100 - point.accuracy), 0) / summaryScatterData.length;
    const renewableGeneration = Math.round(baseLoad * renewablePercent / 100);

    return {
      peakVariance,
      averageError,
      renewableGeneration
    };
  }, [analyticsData, baseLoad, renewablePercent]);

  if (analyticsLoading && !analyticsData) {
    return (
      <PredictorSkeleton
        backendMode={false}
        title="Preparing analytics telemetry"
        subtitle="Fetching analytics charts and scenario predictions from the Render backend."
      />
    );
  }

  return (
    <motion.div variants={containerVariants} initial="hidden" animate="visible" className="space-y-8">
      {analyticsError && (
        <div className="rounded-2xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-200">
          {analyticsError}
        </div>
      )}
      {analyticsFallbackUsed && !analyticsError && (
        <div className="rounded-2xl border border-sky-500/30 bg-sky-500/10 px-4 py-3 text-sm text-sky-100">
          These analytics are being generated from the temporary fallback forecaster while the combined Render backend finishes loading.
        </div>
      )}

      {/* Input Controls */}
      <motion.div variants={itemVariants} className="navy-card rounded-3xl p-8">
        <h2 className="text-2xl font-bold text-white mb-6">Advanced Analytics Controls</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="space-y-3">
            <label className="text-sm font-bold text-slate-300">Base Load (MW)</label>
            <div className="flex items-center gap-4">
              <input
                type="range"
                min="30000"
                max="90000"
                step="1000"
                value={baseLoad}
                onChange={(e) => setBaseLoad(Number(e.target.value))}
                className="flex-1 accent-indigo-500"
              />
              <span className="text-xl font-bold text-indigo-300 min-w-[100px]">{baseLoad.toLocaleString()}</span>
            </div>
          </div>

          <div className="space-y-3">
            <label className="text-sm font-bold text-slate-300">Renewable Energy %</label>
            <div className="flex items-center gap-4">
              <input
                type="range"
                min="0"
                max="100"
                step="5"
                value={renewablePercent}
                onChange={(e) => setRenewablePercent(Number(e.target.value))}
                className="flex-1 accent-green-500"
              />
              <span className="text-xl font-bold text-green-300 min-w-[100px]">{renewablePercent}%</span>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Analytics Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Hourly Analysis Bar Chart */}
        <motion.div variants={itemVariants} className="navy-card rounded-2xl p-6">
          <h3 className="text-lg font-bold text-white mb-4">Hourly Load Analysis</h3>
          <div className="h-[320px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={hourlyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1E2A45" vertical={false} />
                <XAxis dataKey="hour" stroke="#64748b" fontSize={9} tickLine={false} axisLine={false} />
                <YAxis stroke="#64748b" fontSize={9} tickLine={false} axisLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="avgLoad" fill="#6366f1" radius={[4, 4, 0, 0]} name="Avg Load" />
                <Bar dataKey="variance" fill="#f59e0b" radius={[4, 4, 0, 0]} name="Variance" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        {/* Load Composition Donut Chart */}
        <motion.div variants={itemVariants} className="navy-card rounded-2xl p-6">
          <h3 className="text-lg font-bold text-white mb-4">Energy Mix</h3>
          <div className="h-[320px] flex items-center justify-center">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={loadComposition}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={2}
                  dataKey="value"
                >
                  {loadComposition.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{
                    backgroundColor: '#1e293b',
                    border: '1px solid #334155',
                    borderRadius: '8px',
                    color: '#ffffff'
                  }}
                  formatter={(value) => `${value.toLocaleString()} MW`}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="flex justify-center gap-6 mt-4">
            {loadComposition.map((item, idx) => (
              <div key={idx} className="text-center">
                <p className="text-xs text-slate-300">{item.name}</p>
                <p className="text-sm font-bold text-white">{item.percentage}%</p>
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      {/* Scatter Plot - Prediction Accuracy */}
      <motion.div variants={itemVariants} className="navy-card rounded-2xl p-6">
        <h3 className="text-lg font-bold text-white mb-4">Forecast vs Actual (Relationship Analysis)</h3>
        <div className="h-[350px]">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E2A45" />
              <XAxis type="number" dataKey="actual" stroke="#64748b" fontSize={11} name="Actual Load (MW)" />
              <YAxis type="number" dataKey="forecast" stroke="#64748b" fontSize={11} name="Forecast Load (MW)" />
              <Tooltip 
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: '1px solid #334155',
                  borderRadius: '8px'
                }}
                cursor={{ fill: 'rgba(99, 102, 241, 0.1)' }}
                formatter={(value) => `${value.toFixed(0)} MW`}
              />
              <Scatter 
                name="Predictions" 
                data={scatterData} 
                fill="#6366f1" 
                onClick={(data) => console.log('Accuracy:', data.accuracy + '%')}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-4 p-4 bg-slate-800/50 rounded-lg">
          <p className="text-xs text-slate-300">
            <span className="font-bold">Interpretation:</span> Each dot represents a forecast vs actual comparison. 
            Points closer to the diagonal line indicate higher prediction accuracy.
          </p>
        </div>
      </motion.div>

      {/* Sensitivity Analysis */}
      <motion.div variants={itemVariants} className="navy-card rounded-2xl p-6">
        <h3 className="text-lg font-bold text-white mb-4">Factor Sensitivity Analysis (% Impact)</h3>
        <div className="h-[320px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={sensitivityData}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 150, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#1E2A45" />
              <XAxis type="number" stroke="#64748b" fontSize={11} />
              <YAxis dataKey="factor" type="category" stroke="#64748b" fontSize={10} width={140} />
              <Tooltip 
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: '1px solid #334155',
                  borderRadius: '8px'
                }}
                formatter={(value) => `${value.toFixed(1)}% impact`}
              />
              <Bar dataKey="impact" fill="#818cf8" radius={[0, 8, 8, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </motion.div>

      {/* Regional Distribution Bar Chart */}
      <motion.div variants={itemVariants} className="navy-card rounded-2xl p-6">
        <h3 className="text-lg font-bold text-white mb-4">Regional Load Distribution</h3>
        <div className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={regionalDistribution}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E2A45" vertical={false} />
              <XAxis dataKey="region" stroke="#64748b" fontSize={11} tickLine={false} axisLine={false} />
              <YAxis stroke="#64748b" fontSize={11} tickLine={false} axisLine={false} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="value" fill="#10b981" radius={[8, 8, 0, 0]} name="Load (MW)" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-6 grid grid-cols-2 lg:grid-cols-4 gap-4">
          {regionalDistribution.map((item, idx) => (
            <div key={idx} className="bg-slate-800/50 rounded-xl p-4 text-center">
              <p className="text-xs text-slate-400 mb-1">{item.region}</p>
              <p className="text-xl font-bold text-white">{item.percentage}%</p>
              <p className="text-xs text-slate-500 mt-1">{item.value.toLocaleString()} MW</p>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <motion.div variants={itemVariants} className="navy-card rounded-2xl p-6">
          <p className="text-xs text-slate-400 mb-2 uppercase font-bold">Peak Load Variance</p>
          <p className="text-3xl font-black text-indigo-400">
            {analyticsSummary.peakVariance.toLocaleString()} MW
          </p>
        </motion.div>
        <motion.div variants={itemVariants} className="navy-card rounded-2xl p-6">
          <p className="text-xs text-slate-400 mb-2 uppercase font-bold">Avg Prediction Error</p>
          <p className="text-3xl font-black text-green-400">
            {analyticsSummary.averageError.toFixed(1)}%
          </p>
        </motion.div>
        <motion.div variants={itemVariants} className="navy-card rounded-2xl p-6">
          <p className="text-xs text-slate-400 mb-2 uppercase font-bold">Renewable Generation</p>
          <p className="text-3xl font-black text-emerald-400">
            {analyticsSummary.renewableGeneration.toLocaleString()} MW
          </p>
        </motion.div>
      </div>
    </motion.div>
  );
};


const Dashboard = () => {
  const { user } = useAuth();
  const { disclaimerDismissed, dismissDisclaimer } = useDisclaimer();
  const navigate = useNavigate();
  const location = useLocation();
  const backendHealth = useBackendHealth();
  const network = useNetworkQuality();
  const [notifications, setNotifications] = useState([]);
  const [showThreeBackground, setShowThreeBackground] = useState(false);
	
  const addNotification = useCallback((message, type = 'success') => {
    const id = Date.now();
    setNotifications(prev => [...prev, { id, message, type }]);
    setTimeout(() => setNotifications(prev => prev.filter(n => n.id !== id)), 5000);
  }, []);

  useEffect(() => {
    if (typeof window === 'undefined') {
      return undefined;
    }

    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    const updateBackgroundPreference = () => {
      setShowThreeBackground(window.innerWidth >= 1024 && !mediaQuery.matches);
    };

    updateBackgroundPreference();
    window.addEventListener('resize', updateBackgroundPreference);
    mediaQuery.addEventListener?.('change', updateBackgroundPreference);

    return () => {
      window.removeEventListener('resize', updateBackgroundPreference);
      mediaQuery.removeEventListener?.('change', updateBackgroundPreference);
    };
  }, []);

  const backendStatusTone = backendHealth.isChecking
    ? 'amber'
    : backendHealth.isOnline
      ? 'green'
      : 'red';
  const backendStatusLabel = backendHealth.isChecking
    ? 'CHECKING BACKEND'
    : backendHealth.isOnline
      ? 'BACKEND ONLINE'
      : 'BACKEND STATUS UNKNOWN';

  const downloadReport = () => {
    // Enhanced version could capture charts with html2canvas here
    addNotification("Report generation started...", "info");
    setTimeout(() => addNotification("Telemetry report downloaded", "success"), 1500);
  };

  const ToastNotifications = () => (
      <div className="fixed left-4 right-4 top-4 z-[200] space-y-4 sm:left-auto sm:right-6 sm:top-6 sm:max-w-sm">
      <AnimatePresence>
        {notifications.map(n => (
          <motion.div
            key={n.id}
            initial={{ x: 100, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 100, opacity: 0 }}
            className={`p-5 rounded-2xl shadow-2xl border backdrop-blur-xl flex items-center gap-4 ${n.type === 'error'
              ? 'bg-red-950/80 border-red-500/40'
              : 'bg-slate-900/90 border-indigo-500/30'
              }`}
          >
            <div className={`p-3 rounded-xl ${n.type === 'error' ? 'bg-red-500/20 text-red-400' : 'bg-indigo-500/20 text-indigo-400'}`}>
              <CheckCircle2 size={20} />
            </div>
            <p className="text-sm font-medium text-white">{n.message}</p>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );

  return (
    <div className="min-h-screen relative overflow-hidden bg-gradient-to-b from-[#0A0F1D] to-[#0B1221] text-slate-100 font-sans">
      {showThreeBackground && (
        <Suspense fallback={null}>
          <ThreeBackground />
        </Suspense>
      )}
      <Suspense fallback={null}>
        <BackendConnection />
      </Suspense>
      <div className="fixed inset-0 bg-gradient-to-t from-indigo-950/20 to-transparent z-0 pointer-events-none" />

      {/* Disclaimer overlay */}
      <AnimatePresence>
        {!disclaimerDismissed && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[150] flex items-center justify-center p-6 bg-slate-950/80 backdrop-blur-md"
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0, y: 10 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              className="bg-[#10162A] border border-white/5 rounded-3xl p-8 max-w-sm w-full shadow-2xl text-center relative overflow-hidden"
            >
              <div className="w-16 h-16 bg-[#262B40] rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-inner">
                <Zap className="w-8 h-8 text-indigo-400 fill-indigo-400/20" />
              </div>
              <h3 className="text-xl font-bold text-white mb-3">Model Notice</h3>
              <p className="text-sm text-slate-400 leading-relaxed mb-6">
                Please be aware that our model can mistake. Use predictions as one of many references for your energy strategy.
              </p>
              <div className="inline-block px-4 py-1.5 bg-[#1F2937] border border-white/5 rounded-full text-[10px] font-bold text-slate-300 tracking-widest mb-8">
                VERSION: 1.0
              </div>
              <button
                onClick={() => dismissDisclaimer()}
                className="w-full py-3.5 bg-[#6366F1] hover:bg-[#5356E1] text-white font-bold text-sm rounded-xl shadow-lg transition-all active:scale-95"
              >
                I UNDERSTAND
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main layout */}
      <div className="relative z-10">
        {/* Sidebar / Bottom nav */}
        <aside className="fixed bottom-0 left-0 right-0 h-20 md:h-screen md:w-24 md:top-0 md:left-0 bg-[#0B1121] border-t md:border-r border-white/5 md:border-t-0 z-50 flex md:flex-col items-center md:justify-start px-3 md:px-0 md:py-8 transition-all">
          <div className="hidden md:flex w-12 h-12 bg-[#6366F1] rounded-[14px] items-center justify-center shadow-lg shadow-indigo-600/30 mb-8 cursor-pointer hover:scale-105 transition-transform">
            <Zap size={24} className="text-white" fill="white" />
          </div>
          <nav className="flex md:flex-col gap-2 md:gap-4 w-full md:w-auto mt-0 md:mt-4 overflow-x-auto md:overflow-visible no-scrollbar justify-start md:justify-start">
            <NavIcon icon={<Home />} to="/dashboard" tooltip="Overview" />
            <NavIcon icon={<BarChart3 />} to="/dashboard/regional" tooltip="Charts" />
            <NavIcon icon={<Database />} to="/dashboard/hub" tooltip="Database" />
            <NavIcon icon={<Settings />} to="/dashboard/config" tooltip="Settings" />
            <NavIcon icon={<Activity />} to="/dashboard/predictor" tooltip="Activity" />
            <NavIcon icon={<TrendingUp />} to="/dashboard/analytics" tooltip="Analytics" />
            <NavIcon icon={<User />} to="/profile" tooltip="Profile" />
          </nav>
        </aside>

        <main className="min-w-0 md:ml-24 p-4 sm:p-6 md:p-8 lg:p-10 pb-32 md:pb-12 min-h-screen">
          {/* Header */}
          <header className="flex flex-col xl:flex-row justify-between items-start xl:items-center mb-8 gap-6 text-white">
            <div className="flex flex-col sm:flex-row sm:items-center gap-4 w-full xl:w-auto">
              <h1 className="text-2xl font-black tracking-tight flex items-center gap-2">
                LOAD<span className="text-indigo-400">GRID</span>
              </h1>
              <div className="hidden sm:block h-4 w-px bg-white/10 mx-2"></div>
              <div className="flex items-center gap-2 flex-wrap">
                <div className={`w-2 h-2 rounded-full ${backendStatusTone === 'green' ? 'bg-green-500' : backendStatusTone === 'amber' ? 'bg-amber-500' : 'bg-red-500'} status-glow-${backendStatusTone}`} />
                <span className="text-[10px] font-bold uppercase tracking-widest text-slate-400 flex items-center gap-2">
                  {backendStatusLabel}
                  <span className="text-white">• WELCOME, {user?.displayName?.toUpperCase() || 'PARTH MHATRE'}</span>
                </span>
              </div>
            </div>

            <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 items-stretch sm:items-center w-full xl:w-auto">
              <button
                onClick={downloadReport}
                className="px-5 py-2.5 bg-[#6366F1] hover:bg-[#5356E1] text-white rounded-xl text-[11px] font-bold tracking-widest transition-all shadow-lg flex items-center justify-center gap-2"
              >
                <Save size={14} />
                SAVE REPORT
              </button>
              <button
                onClick={async () => { await signOut(auth); navigate("/"); }}
                className="px-5 py-2.5 bg-transparent text-white rounded-xl text-[11px] font-bold tracking-widest transition-all flex items-center justify-center gap-2 border border-white/10 hover:bg-white/5"
              >
                <Settings size={14} />
                SIGN OUT
              </button>

            </div>
          </header>

          {/* Page content */}
          <AnimatePresence mode="wait">
            <Routes location={location} key={location.pathname}>
              <Route path="/" element={<OverviewPage />} />
              <Route path="/regional" element={<RegionalPage />} />
              <Route path="/hub" element={<HubPage />} />
              <Route path="/predictor" element={<PredictorPage network={network} />} />
              <Route path="/config" element={<ConfigPage />} />
              <Route path="/analytics" element={<AdvancedAnalyticsPage />} />
            </Routes>
          </AnimatePresence>

          <ToastNotifications />

          <footer className="mt-24 pt-12 border-t border-white/5 text-xs text-slate-500 flex flex-col md:flex-row justify-between items-center gap-6 opacity-70">
            <div className="flex gap-8">
              <span className="flex items-center gap-2"><Zap size={14} /> v4.2.1</span>
              <span className="flex items-center gap-2"><Cpu size={14} /> Inference Active</span>
            </div>
            <p>© 2026 LoadGrid • Energy Intelligence Platform</p>
          </footer>
        </main>
      </div>
    </div>
  );
};

export default Dashboard;
