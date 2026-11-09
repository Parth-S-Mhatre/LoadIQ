import React, { Suspense, lazy, useState, useMemo, useEffect, useCallback } from 'react';
import { Route, Routes, Link, useLocation, useNavigate } from 'react-router-dom';
import { signOut } from 'firebase/auth';
import { useAuth } from '../context/AuthContext';
import { useDisclaimer } from '../context/DisclaimerContext';
import { auth } from '../firebase';
import { API_CONFIG } from '../config/api';
import {
  Activity,
  ArrowUpRight,
  BarChart3,
  CheckCircle2,
  ChevronDown,
  CircleDollarSign,
  Cpu,
  Database,
  Download,
  Gauge,
  LayoutDashboard,
  LogOut,
  Menu,
  Network,
  Play,
  Radio,
  Server,
  Settings,
  Sliders,
  Sun,
  TrendingUp,
  User,
  Users,
  Wind,
  X,
  Zap,
} from 'lucide-react';
import {
  LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Legend, PieChart, Pie, Cell, ScatterChart, Scatter,
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';
import useBackendHealth from '../hooks/useBackendHealth';
import { AnalyticsService } from '../services/AnalyticsService';
import useNetworkQuality from '../hooks/useNetworkQuality';
import PredictorSkeleton from '../skeleton_pages/PredictorSkeleton';
import {
  Badge,
  EASE,
  Logo,
  Sparkline,
  StatusDot,
  buttonClass,
} from '../components/ui';

const LoadPredictor = lazy(() => import('./LoadPredictor'));

const CHART = {
  grid: 'rgba(17, 24, 19, 0.08)',
  axis: '#8A9188',
  ink: '#111813',
  moss: '#0F3D2E',
  signal: '#A5C01D',
  clay: '#C75B32',
  sky: '#3A6B8A',
  amber: '#C99A2E',
};

const containerVariants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { staggerChildren: 0.07 } },
};

const itemVariants = {
  hidden: { opacity: 0, y: 16 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.55, ease: EASE } },
};

const ChartTip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;

  return (
    <div className="min-w-[10rem] rounded-xl border border-ink/15 bg-paper-50 p-3 shadow-panel">
      <p className="label-sm mb-2 text-ink/40">{label}</p>
      {payload.filter((entry) => entry.value !== null && entry.value !== undefined).map((entry) => (
        <div key={`${entry.dataKey}-${entry.name}`} className="flex items-center justify-between gap-4 py-0.5 text-xs">
          <span className="flex items-center gap-2 text-ink/55">
            <span className="h-1.5 w-1.5 rounded-full" style={{ background: entry.color || entry.fill }} />
            {entry.name}
          </span>
          <span className="mono-num font-medium text-ink">
            {typeof entry.value === 'number' ? entry.value.toLocaleString() : entry.value}
          </span>
        </div>
      ))}
    </div>
  );
};

const SectionHead = ({ eyebrow, title, description, action }) => (
  <div className="flex flex-col justify-between gap-4 border-b border-ink/10 pb-5 sm:flex-row sm:items-end">
    <div>
      {eyebrow && <p className="label mb-2 text-ink/35">{eyebrow}</p>}
      <h2 className="text-lg font-medium tracking-[-0.03em] sm:text-xl">{title}</h2>
      {description && <p className="mt-1.5 max-w-2xl text-xs leading-relaxed text-ink/45">{description}</p>}
    </div>
    {action}
  </div>
);

const NavItem = ({ icon: Icon, label, to, meta }) => {
  const location = useLocation();
  const active = location.pathname === to || (to === '/dashboard' && location.pathname === '/dashboard/');

  return (
    <Link
      to={to}
      className={`group relative flex items-center gap-3 rounded-xl px-3 py-2.5 text-[0.8rem] font-medium transition-colors duration-200 ${
        active ? 'text-ink' : 'text-paper/55 hover:bg-white/[0.04] hover:text-paper'
      }`}
    >
      {active && (
        <motion.span
          layoutId="sidebar-active"
          className="absolute inset-0 rounded-xl bg-signal"
          transition={{ type: 'spring', stiffness: 380, damping: 32 }}
        />
      )}
      <Icon size={17} strokeWidth={1.8} className="relative z-10 shrink-0" />
      <span className="relative z-10 flex-1">{label}</span>
      {meta && <span className={`relative z-10 label-sm ${active ? 'text-ink/45' : 'text-paper/25'}`}>{meta}</span>}
    </Link>
  );
};

const MobileNav = ({ open, onClose, user, onSignOut }) => (
  <AnimatePresence>
    {open && (
      <>
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
          className="fixed inset-0 z-[80] bg-ink/45 backdrop-blur-sm lg:hidden"
        />
        <motion.aside
          initial={{ x: '-100%' }}
          animate={{ x: 0 }}
          exit={{ x: '-100%' }}
          transition={{ duration: 0.42, ease: EASE }}
          className="fixed inset-y-0 left-0 z-[81] flex w-[18rem] flex-col bg-ink p-5 text-paper lg:hidden"
        >
          <div className="flex items-center justify-between">
            <Logo theme="light" />
            <button
              onClick={onClose}
              aria-label="Close navigation"
              className="flex h-9 w-9 items-center justify-center rounded-full border border-white/10 text-paper/60"
            >
              <X size={17} />
            </button>
          </div>

          <div className="mt-7 rounded-xl border border-white/10 bg-white/[0.03] p-3.5">
            <p className="label-sm text-paper/30">Operator</p>
            <p className="mt-2 truncate text-sm font-medium">{user?.displayName || 'LoadIQ Analyst'}</p>
            <p className="mt-0.5 truncate text-[0.68rem] text-paper/35">{user?.email || 'Local workspace'}</p>
          </div>

          <nav className="mt-6 flex-1 space-y-1 overflow-y-auto custom-scrollbar scrollbar-dark">
            {[
              ['Workspace', [
                [LayoutDashboard, 'Overview', '/dashboard'],
                [BarChart3, 'Regional', '/dashboard/regional'],
                [TrendingUp, 'Predictor', '/dashboard/predictor'],
                [Activity, 'Analytics', '/dashboard/analytics'],
              ]],
              ['System', [
                [Database, 'Data hub', '/dashboard/hub'],
                [Settings, 'Configuration', '/dashboard/config'],
                [User, 'Profile', '/profile'],
              ]],
            ].map(([group, items]) => (
              <div key={group} className="pb-4">
                <p className="label-sm px-3 pb-2 text-paper/25">{group}</p>
                {items.map(([Icon, label, to]) => (
                  <Link
                    key={to}
                    to={to}
                    onClick={onClose}
                    className="flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm text-paper/60 transition-colors hover:bg-white/[0.05] hover:text-paper"
                  >
                    <Icon size={17} strokeWidth={1.7} />
                    {label}
                  </Link>
                ))}
              </div>
            ))}
          </nav>

          <button
            onClick={onSignOut}
            className="flex items-center gap-3 rounded-xl border border-white/10 px-3 py-3 text-sm text-paper/55 transition-colors hover:border-white/20 hover:text-paper"
          >
            <LogOut size={16} />
            Sign out
          </button>
        </motion.aside>
      </>
    )}
  </AnimatePresence>
);

/* ---------------------------------- Overview ---------------------------------- */

const StatCard = ({ label, value, unit, icon: Icon, note, spark, accent = CHART.ink }) => (
  <motion.div variants={itemVariants} className="surface card-interactive p-5">
    <div className="flex items-start justify-between gap-3">
      <p className="label text-ink/40">{label}</p>
      <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-ink/[0.04] text-ink/45">
        <Icon size={14} strokeWidth={1.7} />
      </span>
    </div>
    <div className="mt-7 flex items-baseline gap-1.5">
      <span className="mono-num text-[1.9rem] font-medium leading-none tracking-[-0.06em]">{value}</span>
      <span className="text-[0.7rem] text-ink/35">{unit}</span>
    </div>
    <div className="mt-5 flex items-end justify-between gap-3 border-t border-ink/10 pt-3">
      <span className="text-[0.68rem] leading-tight text-ink/40">{note}</span>
      <Sparkline data={spark} width={86} height={26} stroke={accent} fill="rgba(216,242,77,0.3)" className="h-7 w-[4.5rem]" />
    </div>
  </motion.div>
);

const OverviewPage = () => {
  const { user } = useAuth();
  const [viewMode, setViewMode] = useState('load');
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
          windOff,
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
    if (!overviewData) return;

    const userSuffix = user?.uid ? `_${user.uid}` : '_guest';

    localStorage.setItem(`loadiq_overview_context${userSuffix}`, JSON.stringify({
      loadVal,
      priceVal,
      solarGen,
      windOn,
      windOff,
      predictedLoad: overviewData.predictedLoad,
      latestForecast: overviewData.latestForecast,
      predictionSource: overviewData.predictionSource,
      fallbackUsed: overviewData.fallbackUsed,
      fallbackReason: overviewData.fallbackReason,
    }));
  }, [loadVal, overviewData, priceVal, solarGen, user?.uid, windOff, windOn]);

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

  const simulatorControls = [
    { label: 'Solar generation', val: solarGen, set: setSolarGen, max: 40000, unit: 'MW', color: '#C99A2E', step: 100 },
    { label: 'Wind onshore', val: windOn, set: setWindOn, max: 50000, unit: 'MW', color: '#3A6B8A', step: 100 },
    { label: 'Wind offshore', val: windOff, set: setWindOff, max: 10000, unit: 'MW', color: '#5E92B3', step: 100 },
    { label: 'Load forecast', val: loadVal, set: setLoadVal, min: 30000, max: 90000, unit: 'MW', color: '#0F3D2E', step: 1000 },
    { label: 'Market price', val: priceVal, set: setPriceVal, min: 0, max: 200, unit: '€/MWh', color: '#111813', step: 1 },
  ];

  return (
    <motion.div variants={containerVariants} initial="hidden" animate="visible" className="space-y-5">
      {overviewError && (
        <div className="rounded-xl border border-clay/25 bg-clay/[0.06] px-4 py-3 text-sm text-clay">
          {overviewError}
        </div>
      )}
      {overviewFallbackUsed && !overviewError && (
        <div className="rounded-xl border border-sky/25 bg-sky/[0.06] px-4 py-3 text-sm text-sky">
          Live model output is temporarily unavailable. The overview is using a smoothed fallback forecast until the retrained models are ready.
        </div>
      )}

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label="Predicted load"
          value={predictedLoad.toLocaleString()}
          unit="MW"
          icon={Gauge}
          note="Next-hour ensemble output"
          spark={[42, 48, 46, 54, 58, 56, 64, 68, 66, 74]}
          accent={CHART.moss}
        />
        <StatCard
          label="Solar generation"
          value={solarGen.toLocaleString()}
          unit="MW"
          icon={Sun}
          note="Current scenario input"
          spark={[8, 18, 30, 46, 60, 68, 72, 64, 50, 34]}
          accent={CHART.amber}
        />
        <StatCard
          label="Wind generation"
          value={(windOn + windOff).toLocaleString()}
          unit="MW"
          icon={Wind}
          note="Onshore + offshore"
          spark={[52, 58, 50, 62, 70, 64, 72, 80, 74, 82]}
          accent={CHART.sky}
        />
        <StatCard
          label="Market price"
          value={priceVal.toFixed(2)}
          unit="€/MWh"
          icon={CircleDollarSign}
          note="Day-ahead reference"
          spark={[34, 38, 36, 44, 48, 42, 50, 46, 54, 58]}
          accent={CHART.ink}
        />
      </div>

      <div className="grid gap-3 xl:grid-cols-[1.6fr_1fr]">
        <motion.div variants={itemVariants} className="surface overflow-hidden">
          <div className="flex flex-col gap-4 border-b border-ink/10 px-5 py-5 sm:flex-row sm:items-center sm:justify-between sm:px-6">
            <div>
              <p className="label text-ink/35">Live telemetry</p>
              <h3 className="mt-1.5 text-lg font-medium tracking-[-0.03em]">Predictive analytics</h3>
              <p className="mt-1 text-xs text-ink/40">24-hour demand profile · Monday</p>
            </div>
            <div className="inline-flex w-fit rounded-full border border-ink/10 bg-paper-100 p-1">
              {[
                ['load', 'Load (MW)'],
                ['price', 'Price (€)'],
              ].map(([mode, label]) => (
                <button
                  key={mode}
                  onClick={() => setViewMode(mode)}
                  className={`relative rounded-full px-4 py-1.5 text-[0.7rem] font-semibold transition-colors ${
                    viewMode === mode ? 'text-paper' : 'text-ink/45 hover:text-ink/75'
                  }`}
                >
                  {viewMode === mode && (
                    <motion.span
                      layoutId="overview-chart-toggle"
                      className="absolute inset-0 rounded-full bg-ink"
                      transition={{ type: 'spring', stiffness: 400, damping: 34 }}
                    />
                  )}
                  <span className="relative z-10">{label}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="h-[20rem] w-full px-2 pb-4 pt-6 sm:h-[22rem] sm:px-4">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 6, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 6" stroke={CHART.grid} vertical={false} />
                <XAxis dataKey="time" stroke={CHART.axis} fontSize={10} tickLine={false} axisLine={false} interval="preserveStartEnd" />
                <YAxis stroke={CHART.axis} fontSize={10} tickLine={false} axisLine={false} width={44} tickFormatter={(value) => value >= 1000 ? `${Math.round(value / 1000)}k` : value} />
                <Tooltip content={<ChartTip />} cursor={{ stroke: 'rgba(17,24,19,0.18)', strokeDasharray: '3 4' }} />
                <Legend verticalAlign="top" align="right" iconType="plainline" wrapperStyle={{ paddingBottom: 14 }} />
                {viewMode === 'load' ? (
                  <>
                    <Line type="monotone" dataKey="load" stroke={CHART.ink} strokeWidth={2} dot={false} name="Actual load" animationDuration={900} />
                    <Line type="monotone" dataKey="forecast" stroke={CHART.moss} strokeWidth={2} strokeDasharray="5 5" dot={false} name="Forecast" animationDuration={1100} />
                  </>
                ) : (
                  <Line type="monotone" dataKey="price" stroke={CHART.moss} strokeWidth={2} dot={false} name="Market price" animationDuration={900} />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        <motion.div variants={itemVariants} className="surface p-5 sm:p-6">
          <div className="flex items-center justify-between border-b border-ink/10 pb-4">
            <div className="flex items-center gap-2.5">
              <Sliders size={17} className="text-ink/50" />
              <h3 className="text-sm font-medium">Scenario simulator</h3>
            </div>
            <span className="label-sm text-ink/30">Live</span>
          </div>

          <div className="mt-6 space-y-6">
            {simulatorControls.map((control) => {
              const min = control.min || 0;
              const percent = ((control.val - min) / (control.max - min)) * 100;
              return (
                <div key={control.label}>
                  <div className="mb-2.5 flex items-baseline justify-between gap-3">
                    <span className="text-[0.72rem] font-medium text-ink/55">{control.label}</span>
                    <span className="mono-num text-[0.72rem] font-medium">
                      {Math.round(control.val).toLocaleString()} <span className="text-ink/35">{control.unit}</span>
                    </span>
                  </div>
                  <input
                    type="range"
                    min={min}
                    max={control.max}
                    step={control.step || 100}
                    value={control.val}
                    onChange={(event) => control.set(Number(event.target.value))}
                    className="range"
                    style={{
                      background: `linear-gradient(to right, ${control.color} ${percent}%, rgba(17,24,19,0.12) ${percent}%, rgba(17,24,19,0.12) 100%)`,
                    }}
                    aria-label={control.label}
                  />
                </div>
              );
            })}
          </div>

          <div className="mt-7 flex items-start gap-2.5 rounded-xl border border-ink/10 bg-paper-100 p-3">
            <Zap size={14} className="mt-0.5 shrink-0 text-moss" />
            <p className="text-[0.68rem] leading-relaxed text-ink/45">
              Adjusting any input re-runs the overview telemetry request. Results settle after a short debounce.
            </p>
          </div>
        </motion.div>
      </div>

      <motion.div variants={itemVariants} className="grid gap-3 border-t border-ink/10 pt-5 sm:grid-cols-3">
        {[
          ['Grid condition', overviewFallbackUsed ? 'Fallback active' : 'Nominal', overviewFallbackUsed ? 'Model fallback in use' : 'Ensemble responding normally'],
          ['Forecast window', '24 hours', 'Half-hourly demand source'],
          ['Coverage', 'DE · GB · US · IN', 'Four modelled markets'],
        ].map(([label, value, note]) => (
          <div key={label} className="flex items-center justify-between gap-4 py-2">
            <div>
              <p className="label-sm text-ink/35">{label}</p>
              <p className="mt-1.5 text-sm font-medium">{value}</p>
              <p className="mt-0.5 text-[0.68rem] text-ink/35">{note}</p>
            </div>
            <ArrowUpRight size={15} className="text-ink/25" />
          </div>
        ))}
      </motion.div>
    </motion.div>
  );
};

/* --------------------------------- Model 1 form -------------------------------- */

const DEFAULT_PREDICTOR_FORM = {
  hour: new Date().getHours(),
  day_of_week: new Date().getDay() || 7,
  month: new Date().getMonth() + 1,
  DE_load_actual_entsoe_transparency: 55570.0,
  DE_solar_capacity: 40469.0,
  DE_solar_generation_actual: 158.0,
  DE_wind_capacity: 38525.0,
  DE_wind_generation_actual: 8441.0,
  DE_LU_price_day_ahead: 56.1,
};

const PredictorInputForm = () => {
  const [formData, setFormData] = useState(DEFAULT_PREDICTOR_FORM);
  const [predictionResult, setPredictionResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: Number(value) || 0 }));
  };

  const handlePredict = async () => {
    setIsLoading(true);
    try {
      await new Promise((resolve) => setTimeout(resolve, 800));
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

  const resetForm = () => {
    setFormData(DEFAULT_PREDICTOR_FORM);
    setPredictionResult(null);
  };

  return (
    <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, ease: EASE }} className="surface overflow-hidden">
      <div className="flex flex-col justify-between gap-4 border-b border-ink/10 px-5 py-5 sm:flex-row sm:items-center sm:px-7">
        <div>
          <div className="flex items-center gap-2.5">
            <p className="label text-ink/35">Model 1</p>
            <Badge tone="moss">Feature-based</Badge>
          </div>
          <h2 className="mt-2 text-xl font-medium tracking-[-0.035em]">Advanced load predictor</h2>
          <p className="mt-1 text-xs text-ink/45">Configure time, grid and renewable inputs, then run the single-step model.</p>
        </div>
        <span className="label-sm flex items-center gap-2 text-moss">
          <StatusDot tone="green" /> ready
        </span>
      </div>

      <div className="grid grid-cols-1 gap-8 p-5 sm:p-7 md:grid-cols-3">
        <div>
          <p className="label border-b border-ink/10 pb-2.5 text-ink/35">Time parameters</p>
          <div className="mt-5 space-y-5">
            {[
              ['hour', 'Hour of day', 0, 23, formData.hour, `${formData.hour}:00`],
              ['day_of_week', 'Day of week', 1, 7, formData.day_of_week, ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][formData.day_of_week - 1] || '—'],
              ['month', 'Month', 1, 12, formData.month, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][formData.month - 1] || '—'],
            ].map(([name, label, min, max, value, display]) => (
              <div key={name}>
                <div className="mb-2 flex items-baseline justify-between">
                  <label className="text-[0.72rem] font-medium text-ink/55" htmlFor={`m1-${name}`}>{label}</label>
                  <span className="mono-num text-[0.72rem] text-ink/70">{display}</span>
                </div>
                <input
                  id={`m1-${name}`}
                  type="range"
                  name={name}
                  min={min}
                  max={max}
                  value={value}
                  onChange={handleChange}
                  className="range"
                  style={{ background: `linear-gradient(to right, #0F3D2E ${((value - min) / (max - min)) * 100}%, rgba(17,24,19,0.12) ${((value - min) / (max - min)) * 100}%)` }}
                />
              </div>
            ))}
          </div>
        </div>

        <div>
          <p className="label border-b border-ink/10 pb-2.5 text-ink/35">Grid data</p>
          <div className="mt-5 space-y-4">
            {[
              ['DE_load_actual_entsoe_transparency', 'Current load', 'MW'],
              ['DE_LU_price_day_ahead', 'Day-ahead price', '€/MWh'],
              ['DE_solar_capacity', 'Solar capacity', 'MW'],
              ['DE_wind_capacity', 'Wind capacity', 'MW'],
            ].map(([name, label, unit]) => (
              <label key={name} className="block">
                <span className="mb-1.5 flex items-center justify-between text-[0.72rem] font-medium text-ink/55">
                  {label}
                  <span className="label-sm text-ink/25">{unit}</span>
                </span>
                <input
                  type="number"
                  name={name}
                  value={formData[name]}
                  onChange={handleChange}
                  className="h-10 w-full rounded-lg border border-ink/15 bg-paper-50 px-3 text-sm text-ink transition-all focus:border-ink focus:outline-none focus:ring-2 focus:ring-signal-300/60"
                />
              </label>
            ))}
          </div>
        </div>

        <div>
          <p className="label border-b border-ink/10 pb-2.5 text-ink/35">Renewables</p>
          <div className="mt-5 space-y-4">
            {[
              ['DE_solar_generation_actual', 'Solar generation', 'MW', CHART.amber],
              ['DE_wind_generation_actual', 'Wind generation', 'MW', CHART.sky],
            ].map(([name, label, unit, color]) => (
              <label key={name} className="block">
                <span className="mb-1.5 flex items-center justify-between text-[0.72rem] font-medium text-ink/55">
                  <span className="flex items-center gap-1.5">
                    <span className="h-1.5 w-1.5 rounded-full" style={{ background: color }} />
                    {label}
                  </span>
                  <span className="label-sm text-ink/25">{unit}</span>
                </span>
                <input
                  type="number"
                  name={name}
                  value={formData[name]}
                  onChange={handleChange}
                  className="h-10 w-full rounded-lg border border-ink/15 bg-paper-50 px-3 text-sm text-ink transition-all focus:border-ink focus:outline-none focus:ring-2 focus:ring-signal-300/60"
                />
              </label>
            ))}
          </div>

          <div className="mt-6 rounded-xl border border-ink/10 bg-paper-100 p-3.5">
            <p className="label-sm text-ink/35">Estimated renewable share</p>
            <p className="mono-num mt-1.5 text-lg font-medium">
              {Math.round(((formData.DE_solar_generation_actual + formData.DE_wind_generation_actual) / Math.max(formData.DE_load_actual_entsoe_transparency, 1)) * 100)}%
            </p>
            <div className="mt-2.5 h-1.5 overflow-hidden rounded-full bg-ink/10">
              <motion.div
                className="h-full rounded-full bg-moss"
                animate={{
                  width: `${Math.min(100, Math.round(((formData.DE_solar_generation_actual + formData.DE_wind_generation_actual) / Math.max(formData.DE_load_actual_entsoe_transparency, 1)) * 100))}%`,
                }}
                transition={{ duration: 0.5, ease: EASE }}
              />
            </div>
          </div>
        </div>
      </div>

      <div className="flex flex-col justify-between gap-4 border-t border-ink/10 px-5 py-5 sm:flex-row sm:items-center sm:px-7">
        <div className="min-h-[3.25rem] flex-1">
          <AnimatePresence mode="wait">
            {predictionResult && (
              <motion.div
                key={predictionResult}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className="inline-flex items-center gap-3"
              >
                <span className="flex h-10 w-10 items-center justify-center rounded-full bg-moss/10 text-moss">
                  <CheckCircle2 size={19} />
                </span>
                <div>
                  <p className="label-sm text-moss">Model 1 result</p>
                  <p className="mono-num mt-1 text-xl font-medium">
                    {predictionResult} <span className="text-xs text-ink/35">MW</span>
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
          {!predictionResult && <p className="text-xs text-ink/35">Run the model to populate a single-step prediction.</p>}
        </div>
        <div className="flex gap-2">
          <button onClick={resetForm} className={buttonClass('outline', 'md')}>Reset</button>
          <button onClick={handlePredict} disabled={isLoading} className={buttonClass('primary', 'md')}>
            {isLoading ? (
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-paper/25 border-t-paper" />
            ) : (
              <Play size={15} />
            )}
            Run model 1
          </button>
        </div>
      </div>
    </motion.div>
  );
};

const PredictorPage = ({ network }) => (
  <div className="space-y-6">
    <PredictorInputForm />
    <Suspense fallback={network.isLowBandwidth ? <PredictorSkeleton backendMode /> : null}>
      <LoadPredictor />
    </Suspense>
  </div>
);

/* ---------------------------------- Regional ---------------------------------- */

const RegionalPage = () => {
  const regionalData = [
    { name: '50Hertz', region: 'East', loadForecast: 16250, solar: 1575, windOnshore: 3000, windOffshore: 800 },
    { name: 'Tennet', region: 'North', loadForecast: 22750, solar: 1125, windOnshore: 4000, windOffshore: 1200 },
    { name: 'Amprion', region: 'West', loadForecast: 19500, solar: 900, windOnshore: 2000, windOffshore: 0 },
    { name: 'TransnetBW', region: 'South', loadForecast: 6500, solar: 900, windOnshore: 1000, windOffshore: 0 },
  ];
  const totalLoad = regionalData.reduce((sum, item) => sum + item.loadForecast, 0);

  return (
    <motion.div variants={containerVariants} initial="hidden" animate="visible" className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          <StatusDot tone="green" />
          <span className="label text-ink/45">4 transmission zones · live model output</span>
        </div>
        <span className="label-sm text-ink/30">Updated just now</span>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {regionalData.map((zone, index) => (
          <motion.div key={zone.name} variants={itemVariants} className="surface card-interactive p-5">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-sm font-medium">{zone.name}</p>
                <p className="label-sm mt-1 text-ink/35">{zone.region} zone</p>
              </div>
              <Radio size={14} className="text-moss/50" />
            </div>
            <div className="mt-7">
              <p className="label-sm text-ink/35">Load forecast</p>
              <p className="mono-num mt-1.5 text-2xl font-medium tracking-[-0.05em]">
                {zone.loadForecast.toLocaleString()} <span className="text-xs font-normal text-ink/35">MW</span>
              </p>
            </div>
            <div className="mt-5 grid grid-cols-2 gap-2 border-t border-ink/10 pt-3">
              <div>
                <p className="flex items-center gap-1 text-[0.65rem] text-ink/35"><Sun size={11} /> Solar</p>
                <p className="mono-num mt-1 text-xs font-medium">{zone.solar.toLocaleString()} MW</p>
              </div>
              <div>
                <p className="flex items-center gap-1 text-[0.65rem] text-ink/35"><Wind size={11} /> Wind</p>
                <p className="mono-num mt-1 text-xs font-medium">{(zone.windOnshore + zone.windOffshore).toLocaleString()} MW</p>
              </div>
            </div>
            <div className="mt-4 h-1 overflow-hidden rounded-full bg-ink/10">
              <motion.div
                className="h-full rounded-full bg-moss"
                initial={{ width: 0 }}
                animate={{ width: `${(zone.loadForecast / totalLoad) * 100}%` }}
                transition={{ duration: 0.9, delay: 0.1 + index * 0.08, ease: EASE }}
              />
            </div>
          </motion.div>
        ))}
      </div>

      <div className="grid gap-3 xl:grid-cols-[1.55fr_1fr]">
        <motion.div variants={itemVariants} className="surface p-5 sm:p-6">
          <SectionHead eyebrow="Zone comparison" title="Regional breakdown" description="Load forecast, solar and wind by transmission operator." />
          <div className="mt-6 h-[21rem]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={regionalData} barGap={3} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 6" stroke={CHART.grid} vertical={false} />
                <XAxis dataKey="name" stroke={CHART.axis} fontSize={10} tickLine={false} axisLine={false} />
                <YAxis stroke={CHART.axis} fontSize={10} tickLine={false} axisLine={false} width={46} tickFormatter={(value) => `${Math.round(value / 1000)}k`} />
                <Tooltip content={<ChartTip />} cursor={{ fill: 'rgba(17,24,19,0.035)' }} />
                <Legend iconType="circle" iconSize={7} wrapperStyle={{ paddingTop: 18 }} />
                <Bar dataKey="loadForecast" name="Load" fill={CHART.ink} radius={[3, 3, 0, 0]} animationDuration={800} />
                <Bar dataKey="solar" name="Solar" fill={CHART.amber} radius={[3, 3, 0, 0]} animationDuration={900} />
                <Bar dataKey="windOnshore" name="Wind on" fill={CHART.sky} radius={[3, 3, 0, 0]} animationDuration={1000} />
                <Bar dataKey="windOffshore" name="Wind off" fill={CHART.moss} radius={[3, 3, 0, 0]} animationDuration={1100} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        <motion.div variants={itemVariants} className="surface p-5 sm:p-6">
          <SectionHead eyebrow="Share of demand" title="Load distribution" />
          <div className="mt-6 space-y-5">
            {regionalData.map((zone, index) => {
              const share = Math.round((zone.loadForecast / totalLoad) * 100);
              return (
                <div key={zone.name}>
                  <div className="mb-2 flex items-baseline justify-between">
                    <span className="text-[0.78rem] font-medium">{zone.name}</span>
                    <span className="mono-num text-xs text-ink/45">{share}%</span>
                  </div>
                  <div className="h-1.5 overflow-hidden rounded-full bg-ink/[0.07]">
                    <motion.div
                      className="h-full rounded-full bg-ink"
                      initial={{ width: 0 }}
                      animate={{ width: `${share}%` }}
                      transition={{ duration: 0.8, delay: 0.15 + index * 0.08, ease: EASE }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
          <div className="mt-8 border-t border-ink/10 pt-4">
            <div className="flex items-center justify-between">
              <span className="label-sm text-ink/35">Total modelled load</span>
              <span className="mono-num text-sm font-medium">{totalLoad.toLocaleString()} MW</span>
            </div>
            <div className="mt-3 flex items-center justify-between">
              <span className="label-sm text-ink/35">Zones reporting</span>
              <span className="text-sm font-medium">4 / 4</span>
            </div>
          </div>
        </motion.div>
      </div>
    </motion.div>
  );
};

/* ------------------------------------ Hub ------------------------------------- */

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
    <motion.div variants={containerVariants} initial="hidden" animate="visible" className="space-y-4">
      <div className="grid gap-3 xl:grid-cols-[1.35fr_1fr]">
        <motion.div variants={itemVariants} className="surface p-5 sm:p-6">
          <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-start">
            <div className="flex items-start gap-4">
              <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border border-ink/10 bg-paper-100 text-ink/60">
                <Database size={19} />
              </span>
              <div>
                <h3 className="text-sm font-medium">ENTSO-E Transparency Platform</h3>
                <p className="mt-1 text-xs text-ink/40">Primary source for transmission-level load and generation.</p>
                <div className="mt-3 flex flex-wrap gap-x-5 gap-y-2">
                  <span className="label-sm flex items-center gap-1.5 text-moss"><StatusDot tone="green" /> Active</span>
                  <span className="label-sm text-ink/30">Uptime 99.99%</span>
                  <span className="label-sm text-ink/30">Last sync 2 min ago</span>
                </div>
              </div>
            </div>
            <div className="text-left sm:text-right">
              <p className="label-sm text-ink/30">Response</p>
              <p className="mono-num mt-1 text-2xl font-medium">14 <span className="text-xs text-ink/30">ms</span></p>
            </div>
          </div>

          <div className="mt-6 grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-ink/10 bg-ink/10 sm:grid-cols-4">
            {[
              ['Rows ingested', '151,203'],
              ['Markets', '4'],
              ['Interval', '30 min'],
              ['Schema', 'v2.4'],
            ].map(([label, value]) => (
              <div key={label} className="bg-paper-50 p-3.5">
                <p className="label-sm text-ink/30">{label}</p>
                <p className="mono-num mt-1.5 text-sm font-medium">{value}</p>
              </div>
            ))}
          </div>
        </motion.div>

        <motion.div variants={itemVariants} className="surface p-5 sm:p-6">
          <div className="flex items-center gap-2.5">
            <Network size={17} className="text-ink/50" />
            <h3 className="text-sm font-medium">Request pipeline</h3>
          </div>
          <div className="mt-6 space-y-0">
            {[
              ['Client', 'On-demand route', 'Active'],
              ['Render', 'Primary API', 'Active'],
              ['Railway', 'Fallback API', 'Standby'],
            ].map(([name, subtitle, state], index) => (
              <div key={name} className="relative flex items-center gap-3 pb-5 last:pb-0">
                {index < 2 && <span className="absolute left-[0.65rem] top-5 h-full w-px bg-ink/10" />}
                <span className={`relative z-10 h-[1.35rem] w-[1.35rem] shrink-0 rounded-full border-2 ${state === 'Active' ? 'border-moss bg-signal' : 'border-ink/20 bg-paper-50'}`} />
                <div className="flex flex-1 items-baseline justify-between gap-3">
                  <div>
                    <p className="text-xs font-medium">{name}</p>
                    <p className="text-[0.65rem] text-ink/35">{subtitle}</p>
                  </div>
                  <span className={`label-sm ${state === 'Active' ? 'text-moss' : 'text-ink/30'}`}>{state}</span>
                </div>
              </div>
            ))}
          </div>
          <div className="mt-5 border-t border-ink/10 pt-3">
            <p className="label-sm truncate text-ink/30">{API_CONFIG.BASE_URL.replace(/^https?:\/\//, '')}</p>
          </div>
        </motion.div>
      </div>

      <motion.div variants={itemVariants} className="surface p-5 sm:p-6">
        <SectionHead
          eyebrow="Integration health"
          title="Data latency trends"
          description="Round-trip time across the last seven collection windows."
          action={<Badge tone="signal"><StatusDot tone="green" pulse={false} /> Within target</Badge>}
        />
        <div className="mt-6 h-[18rem]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={latencyData} margin={{ top: 4, right: 12, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 6" stroke={CHART.grid} vertical={false} />
              <XAxis dataKey="day" stroke={CHART.axis} fontSize={10} tickLine={false} axisLine={false} />
              <YAxis stroke={CHART.axis} fontSize={10} tickLine={false} axisLine={false} width={36} domain={[0, 120]} ticks={[0, 30, 60, 90, 120]} />
              <Tooltip content={<ChartTip />} cursor={{ stroke: 'rgba(17,24,19,0.18)', strokeDasharray: '3 4' }} />
              <Line
                type="monotone"
                dataKey="latency"
                name="Latency (ms)"
                stroke={CHART.moss}
                strokeWidth={2}
                dot={{ fill: '#0F3D2E', r: 3, strokeWidth: 2, stroke: '#F2F0E9' }}
                activeDot={{ r: 5, fill: '#D8F24D', stroke: '#0F3D2E' }}
                animationDuration={1000}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </motion.div>
    </motion.div>
  );
};

/* ---------------------------------- Config ------------------------------------ */

const ConfigPage = ({ backendHealth, onRunPrediction }) => {
  const [gpuEnabled, setGpuEnabled] = useState(true);

  const selects = [
    ['Prediction model', ['Ridge Regression', 'XGBoost', 'LightGBM', 'Stacking Ensemble (LGB 60% + XGB 40%)']],
    ['Confidence interval', ['95% (Standard)', '99% (Strict)', '90% (Loose)']],
    ['Forecast horizon', ['Short-term (1-7 days)', 'Medium-term (1-4 weeks)', 'Long-term (1-12 months)']],
    ['Refresh rate', ['Every 15 min', 'Every 30 min', 'Every 60 min']],
  ];

  return (
    <motion.div variants={containerVariants} initial="hidden" animate="visible" className="grid gap-4 xl:grid-cols-[1.4fr_1fr]">
      <motion.div variants={itemVariants} className="surface p-5 sm:p-7">
        <SectionHead eyebrow="Model settings" title="Forecast configuration" description="These controls shape the next prediction request in this browser session." />

        <div className="mt-6 grid grid-cols-1 gap-5 sm:grid-cols-2">
          {selects.map(([label, options]) => (
            <label key={label} className="block">
              <span className="label-sm mb-2 block text-ink/40">{label}</span>
              <div className="relative">
                <select className="ui-select" defaultValue={options[0]}>
                  {options.map((option) => <option key={option}>{option}</option>)}
                </select>
                <ChevronDown size={15} className="pointer-events-none absolute right-3.5 top-1/2 -translate-y-1/2 text-ink/35" />
              </div>
            </label>
          ))}
        </div>

        <div className="mt-7 rounded-xl border border-ink/10 bg-paper-100 p-4">
          <div className="flex items-start gap-3">
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-ink text-signal">
              <Cpu size={14} />
            </span>
            <div>
              <p className="text-xs font-medium">Inference runs on the backend</p>
              <p className="mt-1 text-[0.7rem] leading-relaxed text-ink/45">
                Configuration is applied per request. If the primary Render service is unavailable,
                the request automatically falls back to Railway.
              </p>
            </div>
          </div>
        </div>

        <button onClick={onRunPrediction} className={buttonClass('primary', 'lg', 'mt-6 w-full sm:w-auto')}>
          <Zap size={16} />
          Run prediction now
        </button>
      </motion.div>

      <div className="space-y-4">
        <motion.div variants={itemVariants} className="surface p-5 sm:p-6">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <span className="flex h-10 w-10 items-center justify-center rounded-xl border border-ink/10 bg-paper-100 text-ink/55">
                <Server size={18} />
              </span>
              <div>
                <p className="text-sm font-medium">Backend connection</p>
                <p className="mt-0.5 max-w-[15rem] truncate text-[0.68rem] text-ink/35">
                  {(backendHealth.activeBackendUrl || API_CONFIG.BASE_URL).replace(/^https?:\/\//, '')}
                </p>
              </div>
            </div>
            <StatusDot tone={backendHealth.isChecking ? 'amber' : backendHealth.isOnline ? 'green' : 'red'} />
          </div>
          <div className="mt-5 grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-ink/10 bg-ink/10">
            <div className="bg-paper-50 p-3.5">
              <p className="label-sm text-ink/30">Status</p>
              <p className="mt-1.5 text-xs font-medium">
                {backendHealth.isChecking ? 'Checking' : backendHealth.isOnline ? 'Online' : 'Unavailable'}
              </p>
            </div>
            <div className="bg-paper-50 p-3.5">
              <p className="label-sm text-ink/30">Latency</p>
              <p className="mono-num mt-1.5 text-xs font-medium">{backendHealth.latency ? `${backendHealth.latency} ms` : '—'}</p>
            </div>
          </div>
        </motion.div>

        <motion.div variants={itemVariants} className="surface p-5 sm:p-6">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <span className="flex h-10 w-10 items-center justify-center rounded-xl border border-ink/10 bg-paper-100 text-ink/55">
                <Cpu size={18} />
              </span>
              <div>
                <p className="text-sm font-medium">GPU acceleration</p>
                <p className="mt-0.5 text-[0.68rem] text-ink/35">NVIDIA TensorRT</p>
              </div>
            </div>
            <button
              type="button"
              role="switch"
              aria-checked={gpuEnabled}
              aria-label="Toggle GPU acceleration"
              onClick={() => setGpuEnabled((enabled) => !enabled)}
              className={`relative h-6 w-11 rounded-full transition-colors duration-300 ${gpuEnabled ? 'bg-moss' : 'bg-ink/15'}`}
            >
              <motion.span
                className="absolute top-0.5 h-5 w-5 rounded-full bg-paper-50 shadow-sm"
                animate={{ left: gpuEnabled ? '1.375rem' : '0.125rem' }}
                transition={{ type: 'spring', stiffness: 500, damping: 32 }}
              />
            </button>
          </div>
          <p className="mt-4 border-t border-ink/10 pt-3 text-[0.68rem] leading-relaxed text-ink/40">
            {gpuEnabled ? 'Acceleration is enabled for compatible inference nodes.' : 'Acceleration is disabled; requests use CPU inference.'}
          </p>
        </motion.div>

        <motion.div variants={itemVariants} className="rounded-xl border border-signal-500/30 bg-signal/25 p-5">
          <p className="label-sm text-ink/45">Model notice</p>
          <p className="mt-2 text-xs leading-relaxed text-ink/65">
            Forecasts are decision support, not a control signal. Validate against live grid data before acting.
          </p>
        </motion.div>
      </div>
    </motion.div>
  );
};

/* ------------------------------ Advanced analytics ---------------------------- */

const AdvancedAnalyticsPage = () => {
  const { user } = useAuth();
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
          renewablePercent,
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
    if (!analyticsData) return;

    const userSuffix = user?.uid ? `_${user.uid}` : '_guest';

    localStorage.setItem(`loadiq_advanced_analytics_context${userSuffix}`, JSON.stringify({
      baseLoad,
      renewablePercent,
      predictionSource: analyticsData.predictionSource,
      fallbackUsed: analyticsData.fallbackUsed,
      fallbackReason: analyticsData.fallbackReason,
      hourlyHighlights: (analyticsData.hourlyData || []).slice(0, 4),
      sensitivityHighlights: (analyticsData.sensitivityData || []).slice(0, 3),
    }));
  }, [analyticsData, baseLoad, renewablePercent, user?.uid]);

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
        renewableGeneration: Math.round(baseLoad * renewablePercent / 100),
      };
    }

    const peakVariance = Math.max(...summaryHourlyData.map((d) => d.variance));
    const averageError = summaryScatterData.reduce((sum, point) => sum + (100 - point.accuracy), 0) / summaryScatterData.length;
    const renewableGeneration = Math.round(baseLoad * renewablePercent / 100);

    return { peakVariance, averageError, renewableGeneration };
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
    <motion.div variants={containerVariants} initial="hidden" animate="visible" className="space-y-4">
      {analyticsError && (
        <div className="rounded-xl border border-clay/25 bg-clay/[0.06] px-4 py-3 text-sm text-clay">{analyticsError}</div>
      )}
      {analyticsFallbackUsed && !analyticsError && (
        <div className="rounded-xl border border-sky/25 bg-sky/[0.06] px-4 py-3 text-sm text-sky">
          These analytics are generated from the temporary fallback forecaster while the combined Render backend finishes loading.
        </div>
      )}

      <motion.div variants={itemVariants} className="surface p-5 sm:p-6">
        <SectionHead
          eyebrow="Scenario inputs"
          title="Advanced analytics controls"
          description="Move the inputs to re-run hourly, regional and sensitivity analysis."
          action={<Badge tone={analyticsFallbackUsed ? 'clay' : 'moss'}><StatusDot tone={analyticsFallbackUsed ? 'amber' : 'green'} pulse={false} />{analyticsFallbackUsed ? 'Fallback' : 'Live model'}</Badge>}
        />
        <div className="mt-6 grid grid-cols-1 gap-7 md:grid-cols-2">
          <div>
            <div className="mb-2.5 flex items-baseline justify-between">
              <label htmlFor="analytics-base-load" className="text-[0.78rem] font-medium text-ink/60">Base load</label>
              <span className="mono-num text-sm font-medium">{baseLoad.toLocaleString()} <span className="text-[0.65rem] text-ink/35">MW</span></span>
            </div>
            <input
              id="analytics-base-load"
              type="range"
              min="30000"
              max="90000"
              step="1000"
              value={baseLoad}
              onChange={(e) => setBaseLoad(Number(e.target.value))}
              className="range"
              style={{ background: `linear-gradient(to right, #111813 ${((baseLoad - 30000) / 60000) * 100}%, rgba(17,24,19,0.12) ${((baseLoad - 30000) / 60000) * 100}%)` }}
            />
            <div className="mt-2 flex justify-between">
              <span className="label-sm text-ink/25">30 GW</span>
              <span className="label-sm text-ink/25">90 GW</span>
            </div>
          </div>

          <div>
            <div className="mb-2.5 flex items-baseline justify-between">
              <label htmlFor="analytics-renewable" className="text-[0.78rem] font-medium text-ink/60">Renewable energy</label>
              <span className="mono-num text-sm font-medium">{renewablePercent}%</span>
            </div>
            <input
              id="analytics-renewable"
              type="range"
              min="0"
              max="100"
              step="5"
              value={renewablePercent}
              onChange={(e) => setRenewablePercent(Number(e.target.value))}
              className="range"
              style={{ background: `linear-gradient(to right, #0F3D2E ${renewablePercent}%, rgba(17,24,19,0.12) ${renewablePercent}%)` }}
            />
            <div className="mt-2 flex justify-between">
              <span className="label-sm text-ink/25">0%</span>
              <span className="label-sm text-ink/25">100%</span>
            </div>
          </div>
        </div>
      </motion.div>

      <div className="grid gap-4 xl:grid-cols-2">
        <motion.div variants={itemVariants} className="surface p-5 sm:p-6">
          <SectionHead eyebrow="By hour" title="Hourly load analysis" />
          <div className="mt-6 h-[19rem]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={hourlyData} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 6" stroke={CHART.grid} vertical={false} />
                <XAxis dataKey="hour" stroke={CHART.axis} fontSize={9} tickLine={false} axisLine={false} interval={2} />
                <YAxis stroke={CHART.axis} fontSize={9} tickLine={false} axisLine={false} width={38} tickFormatter={(value) => `${Math.round(value / 1000)}k`} />
                <Tooltip content={<ChartTip />} cursor={{ fill: 'rgba(17,24,19,0.035)' }} />
                <Legend iconType="circle" iconSize={7} wrapperStyle={{ paddingTop: 14 }} />
                <Bar dataKey="avgLoad" name="Avg load" fill={CHART.ink} radius={[2, 2, 0, 0]} animationDuration={800} />
                <Bar dataKey="variance" name="Variance" fill={CHART.clay} radius={[2, 2, 0, 0]} animationDuration={950} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        <motion.div variants={itemVariants} className="surface p-5 sm:p-6">
          <SectionHead eyebrow="Composition" title="Energy mix" />
          <div className="mt-2 h-[17rem]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={loadComposition}
                  cx="50%"
                  cy="50%"
                  innerRadius={58}
                  outerRadius={92}
                  paddingAngle={3}
                  cornerRadius={4}
                  dataKey="value"
                  animationDuration={900}
                >
                  {loadComposition.map((entry, index) => (
                    <Cell key={`cell-${entry.name}`} fill={index === 0 ? CHART.moss : CHART.clay} stroke="#F2F0E9" strokeWidth={2} />
                  ))}
                </Pie>
                <Tooltip content={<ChartTip />} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-2 flex justify-center gap-8">
            {loadComposition.map((item, index) => (
              <div key={item.name} className="text-center">
                <p className="flex items-center justify-center gap-1.5 text-[0.68rem] text-ink/45">
                  <span className="h-1.5 w-1.5 rounded-full" style={{ background: index === 0 ? CHART.moss : CHART.clay }} />
                  {item.name}
                </p>
                <p className="mono-num mt-1 text-sm font-medium">{item.percentage}%</p>
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      <motion.div variants={itemVariants} className="surface p-5 sm:p-6">
        <SectionHead eyebrow="Relationship analysis" title="Forecast vs actual" description="Each point compares a forecast step with the realised load. Tighter clustering means higher accuracy." />
        <div className="mt-6 h-[20rem]">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 10, right: 18, bottom: 10, left: 0 }}>
              <CartesianGrid strokeDasharray="3 6" stroke={CHART.grid} />
              <XAxis type="number" dataKey="actual" stroke={CHART.axis} fontSize={10} name="Actual load" tickLine={false} axisLine={false} tickFormatter={(value) => `${Math.round(value / 1000)}k`} />
              <YAxis type="number" dataKey="forecast" stroke={CHART.axis} fontSize={10} name="Forecast load" tickLine={false} axisLine={false} width={38} tickFormatter={(value) => `${Math.round(value / 1000)}k`} />
              <Tooltip content={<ChartTip />} cursor={{ strokeDasharray: '3 4', stroke: 'rgba(17,24,19,0.18)' }} />
              <Scatter name="Forecast step" data={scatterData} fill={CHART.moss} fillOpacity={0.7} animationDuration={900} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </motion.div>

      <div className="grid gap-4 xl:grid-cols-2">
        <motion.div variants={itemVariants} className="surface p-5 sm:p-6">
          <SectionHead eyebrow="Sensitivity" title="Factor impact" description="Percentage change in forecast output per factor." />
          <div className="mt-6 h-[19rem]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={sensitivityData} layout="vertical" margin={{ top: 4, right: 24, left: 10, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 6" stroke={CHART.grid} horizontal={false} />
                <XAxis type="number" stroke={CHART.axis} fontSize={10} tickLine={false} axisLine={false} tickFormatter={(value) => `${value}%`} />
                <YAxis dataKey="factor" type="category" stroke={CHART.axis} fontSize={10} width={104} tickLine={false} axisLine={false} />
                <Tooltip content={<ChartTip />} cursor={{ fill: 'rgba(17,24,19,0.035)' }} />
                <Bar dataKey="impact" radius={[0, 4, 4, 0]} animationDuration={900}>
                  {sensitivityData.map((entry) => (
                    <Cell key={entry.factor} fill={entry.impact >= 0 ? CHART.moss : CHART.clay} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </motion.div>

        <motion.div variants={itemVariants} className="surface p-5 sm:p-6">
          <SectionHead eyebrow="Regional" title="Load distribution" />
          <div className="mt-6 h-[13rem]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={regionalDistribution} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 6" stroke={CHART.grid} vertical={false} />
                <XAxis dataKey="region" stroke={CHART.axis} fontSize={10} tickLine={false} axisLine={false} />
                <YAxis stroke={CHART.axis} fontSize={9} tickLine={false} axisLine={false} width={36} tickFormatter={(value) => `${Math.round(value / 1000)}k`} />
                <Tooltip content={<ChartTip />} cursor={{ fill: 'rgba(17,24,19,0.035)' }} />
                <Bar dataKey="value" name="Load (MW)" fill={CHART.sky} radius={[4, 4, 0, 0]} animationDuration={850} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
            {regionalDistribution.map((item) => (
              <div key={item.region} className="rounded-lg border border-ink/10 bg-paper-100 p-2.5">
                <p className="text-[0.62rem] text-ink/40">{item.region}</p>
                <p className="mono-num mt-1 text-xs font-medium">{item.percentage}%</p>
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        {[
          ['Peak load variance', `${analyticsSummary.peakVariance.toLocaleString()} MW`, CHART.clay],
          ['Average prediction error', `${analyticsSummary.averageError.toFixed(1)}%`, CHART.sky],
          ['Renewable generation', `${analyticsSummary.renewableGeneration.toLocaleString()} MW`, CHART.moss],
        ].map(([label, value, color]) => (
          <motion.div key={label} variants={itemVariants} className="surface p-5">
            <div className="flex items-center justify-between">
              <p className="label text-ink/35">{label}</p>
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: color }} />
            </div>
            <p className="mono-num mt-5 text-2xl font-medium tracking-[-0.05em]">{value}</p>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
};

/* ---------------------------------- Dashboard --------------------------------- */

const Dashboard = () => {
  const { user } = useAuth();
  const { disclaimerDismissed, dismissDisclaimer } = useDisclaimer();
  const navigate = useNavigate();
  const location = useLocation();
  const backendHealth = useBackendHealth();
  const network = useNetworkQuality();
  const [notifications, setNotifications] = useState([]);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  const addNotification = useCallback((message, type = 'success') => {
    const id = Date.now();
    setNotifications((prev) => [...prev, { id, message, type }]);
    setTimeout(() => setNotifications((prev) => prev.filter((n) => n.id !== id)), 5000);
  }, []);

  const backendStatusTone = backendHealth.isChecking
    ? 'amber'
    : backendHealth.isOnline
      ? 'green'
      : 'red';

  const pageMeta = useMemo(() => {
    const map = {
      '/dashboard': ['Workspace · overview', 'Demand, generation and market context at a glance.'],
      '/dashboard/regional': ['Workspace · regional', 'Transmission zone load and renewable output.'],
      '/dashboard/predictor': ['Models · predictor', 'Run the ensemble and inspect the forecast workbench.'],
      '/dashboard/analytics': ['Workspace · analytics', 'Scenario analysis across load, mix and sensitivity.'],
      '/dashboard/hub': ['System · data hub', 'Source health, pipeline state and latency.'],
      '/dashboard/config': ['System · configuration', 'Forecast settings and backend controls.'],
    };
    return map[location.pathname] || ['Workspace', 'LoadIQ grid intelligence'];
  }, [location.pathname]);

  const downloadReport = () => {
    addNotification('Report generation started…', 'info');
    setTimeout(() => addNotification('Telemetry report downloaded', 'success'), 1500);
  };

  const handleSignOut = async () => {
    await signOut(auth);
    navigate('/');
  };

  const navGroups = [
    ['Workspace', [
      [LayoutDashboard, 'Overview', '/dashboard', null],
      [BarChart3, 'Regional', '/dashboard/regional', null],
      [TrendingUp, 'Predictor', '/dashboard/predictor', 'M2'],
      [Activity, 'Analytics', '/dashboard/analytics', null],
    ]],
    ['System', [
      [Database, 'Data hub', '/dashboard/hub', null],
      [Settings, 'Configuration', '/dashboard/config', null],
      [User, 'Profile', '/profile', null],
    ]],
  ];

  return (
    <div className="min-h-screen bg-paper text-ink">
      {/* Desktop sidebar */}
      <aside className="fixed inset-y-0 left-0 z-40 hidden w-[16rem] flex-col bg-ink text-paper lg:flex">
        <div className="px-5 py-6">
          <Logo theme="light" />
        </div>

        <div className="mx-4 rounded-xl border border-white/10 bg-white/[0.03] p-3.5">
          <p className="label-sm text-paper/30">Workspace</p>
          <p className="mt-2 truncate text-[0.8rem] font-medium">{user?.displayName || 'Grid Operations'}</p>
          <p className="mt-0.5 truncate text-[0.65rem] text-paper/35">{user?.email || 'Local analyst'}</p>
        </div>

        <nav className="mt-6 flex-1 overflow-y-auto px-3 custom-scrollbar scrollbar-dark">
          {navGroups.map(([group, items]) => (
            <div key={group} className="mb-6">
              <p className="label-sm px-3 pb-2 text-paper/25">{group}</p>
              <div className="space-y-1">
                {items.map(([Icon, label, to, meta]) => (
                  <NavItem key={to} icon={Icon} label={label} to={to} meta={meta} />
                ))}
              </div>
            </div>
          ))}
        </nav>

        <div className="border-t border-white/10 p-4">
          <div className="flex items-center justify-between rounded-xl bg-white/[0.03] px-3 py-2.5">
            <div className="flex items-center gap-2">
              <StatusDot tone={backendStatusTone} />
              <div>
                <p className="label-sm text-paper/65">
                  {backendHealth.isChecking ? 'Checking API' : backendHealth.isOnline ? 'Backend online' : 'API unavailable'}
                </p>
                <p className="mt-0.5 text-[0.6rem] text-paper/30">
                  {backendHealth.latency ? `${backendHealth.latency} ms latency` : 'Render · Railway'}
                </p>
              </div>
            </div>
            <SidebarSyncIcon />
          </div>

          <div className="mt-3 flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-signal text-[0.7rem] font-semibold text-ink">
              {(user?.displayName || user?.email || 'L').charAt(0).toUpperCase()}
            </div>
            <div className="min-w-0 flex-1">
              <p className="truncate text-[0.7rem] text-paper/65">{user?.displayName || 'LoadIQ Analyst'}</p>
              <p className="truncate text-[0.6rem] text-paper/30">Operator</p>
            </div>
            <button
              onClick={handleSignOut}
              title="Sign out"
              aria-label="Sign out"
              className="flex h-8 w-8 items-center justify-center rounded-lg text-paper/35 transition-colors hover:bg-white/10 hover:text-paper"
            >
              <LogOut size={15} />
            </button>
          </div>
        </div>
      </aside>

      <MobileNav
        open={mobileNavOpen}
        onClose={() => setMobileNavOpen(false)}
        user={user}
        onSignOut={handleSignOut}
      />

      <div className="lg:pl-[16rem]">
        {/* Header */}
        <header className="sticky top-0 z-30 border-b border-ink/10 bg-paper/85 backdrop-blur-xl">
          <div className="flex min-h-[4.75rem] items-center justify-between gap-4 px-4 py-3 sm:px-6 lg:px-8">
            <div className="flex min-w-0 items-center gap-3">
              <button
                onClick={() => setMobileNavOpen(true)}
                aria-label="Open navigation"
                className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-ink/15 text-ink/60 lg:hidden"
              >
                <Menu size={17} />
              </button>
              <div className="min-w-0">
                <p className="label-sm truncate text-ink/35">{pageMeta[0]}</p>
                <h1 className="mt-1 truncate text-[1.05rem] font-medium tracking-[-0.03em] sm:text-lg">
                  {pageMeta[1]}
                </h1>
              </div>
            </div>

            <div className="flex shrink-0 items-center gap-2">
              <div className="hidden items-center gap-2 rounded-full border border-ink/10 bg-paper-50 px-3 py-1.5 md:flex">
                <StatusDot tone={backendStatusTone} pulse={!backendHealth.isChecking} />
                <span className="label-sm text-ink/45">
                  {backendHealth.isChecking ? 'Checking' : backendHealth.isOnline ? 'API online' : 'API offline'}
                </span>
                {backendHealth.latency && <span className="label-sm text-ink/25">{backendHealth.latency}ms</span>}
              </div>

              <button onClick={downloadReport} className={buttonClass('outline', 'sm', 'hidden sm:inline-flex')}>
                <Download size={14} />
                Export
              </button>
              <button onClick={() => navigate('/dashboard/predictor')} className={buttonClass('primary', 'sm')}>
                <Play size={13} />
                <span className="hidden sm:inline">New forecast</span>
                <span className="sm:hidden">Run</span>
              </button>
            </div>
          </div>
        </header>

        {/* Main */}
        <main className="px-4 py-5 sm:px-6 sm:py-6 lg:px-8 lg:py-7">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.32, ease: EASE }}
            >
              <Routes location={location}>
                <Route path="/" element={<OverviewPage />} />
                <Route path="/regional" element={<RegionalPage />} />
                <Route path="/hub" element={<HubPage />} />
                <Route path="/predictor" element={<PredictorPage network={network} />} />
                <Route path="/config" element={<ConfigPage backendHealth={backendHealth} onRunPrediction={() => navigate('/dashboard/predictor')} />} />
                <Route path="/analytics" element={<AdvancedAnalyticsPage />} />
              </Routes>
            </motion.div>
          </AnimatePresence>

          <footer className="mt-14 flex flex-col justify-between gap-4 border-t border-ink/10 pt-6 text-[0.65rem] text-ink/35 sm:flex-row sm:items-center">
            <div className="flex flex-wrap gap-5">
              <span className="flex items-center gap-1.5"><Zap size={12} /> v4.2.1</span>
              <span className="flex items-center gap-1.5"><Cpu size={12} /> Inference active</span>
              <span className="flex items-center gap-1.5"><Users size={12} /> {user?.displayName || 'Operator'}</span>
            </div>
            <p>© 2026 LoadIQ · Grid intelligence platform</p>
          </footer>
        </main>
      </div>

      {/* Toasts */}
      <div className="fixed left-4 right-4 top-4 z-[120] space-y-2 sm:left-auto sm:right-6 sm:top-6 sm:max-w-sm">
        <AnimatePresence>
          {notifications.map((notification) => (
            <motion.div
              key={notification.id}
              initial={{ x: 40, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: 40, opacity: 0 }}
              transition={{ duration: 0.35, ease: EASE }}
              className="flex items-center gap-3 rounded-xl border border-ink/15 bg-paper-50 p-3.5 shadow-lift"
            >
              <span className={`flex h-8 w-8 items-center justify-center rounded-full ${
                notification.type === 'error'
                  ? 'bg-clay/10 text-clay'
                  : notification.type === 'info'
                    ? 'bg-sky/10 text-sky'
                    : 'bg-moss/10 text-moss'
              }`}>
                <CheckCircle2 size={16} />
              </span>
              <p className="text-xs font-medium text-ink/75">{notification.message}</p>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* Disclaimer */}
      <AnimatePresence>
        {!disclaimerDismissed && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[150] flex items-center justify-center bg-ink/50 p-5 backdrop-blur-sm"
          >
            <motion.div
              initial={{ scale: 0.96, opacity: 0, y: 12 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.97, opacity: 0 }}
              transition={{ duration: 0.4, ease: EASE }}
              className="w-full max-w-md overflow-hidden rounded-2xl border border-ink/15 bg-paper-50 shadow-lift"
            >
              <div className="flex items-center justify-between border-b border-ink/10 px-6 py-4">
                <div className="flex items-center gap-2.5">
                  <span className="flex h-7 w-7 items-center justify-center rounded-full bg-ink text-signal">
                    <Zap size={13} />
                  </span>
                  <span className="text-sm font-medium">Model notice</span>
                </div>
                <span className="label-sm text-ink/30">v1.0</span>
              </div>
              <div className="px-6 py-6">
                <h2 className="text-xl font-medium tracking-[-0.03em]">Before you rely on a forecast</h2>
                <p className="mt-3 text-sm leading-relaxed text-ink/55">
                  LoadIQ predictions are generated from historical patterns and model features.
                  The model can be wrong. Use forecasts as one of several references for your
                  energy strategy, and validate against live grid data before acting.
                </p>
                <div className="mt-5 rounded-xl border border-ink/10 bg-paper-100 p-3.5">
                  <p className="label-sm text-ink/35">Recommended practice</p>
                  <p className="mt-1.5 text-[0.7rem] leading-relaxed text-ink/50">
                    Cross-check the ensemble output against your operational telemetry and
                    confidence interval before dispatching.
                  </p>
                </div>
              </div>
              <div className="border-t border-ink/10 px-6 py-4">
                <button onClick={() => dismissDisclaimer()} className={buttonClass('primary', 'md', 'w-full')}>
                  I understand
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

const SidebarSyncIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className="text-paper/25" aria-hidden="true">
    <path d="M20 12a8 8 0 1 1-2.34-5.66M20 4v5h-5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

export default Dashboard;
