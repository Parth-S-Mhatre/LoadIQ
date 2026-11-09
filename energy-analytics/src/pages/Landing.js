import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import {
  AnimatePresence,
  motion,
  useReducedMotion,
  useScroll,
  useSpring,
  useTransform,
} from "framer-motion";
import {
  Activity,
  ArrowRight,
  Bot,
  Check,
  FileText,
  Github,
  Layers,
  Leaf,
  Linkedin,
  Map,
  Menu,
  Radio,
  Shield,
  Twitter,
  X,
} from "lucide-react";
import plotActualVsPredicted from "../Images_result/plot_actual_vs_predicted.png";
import plotErrorByHour from "../Images_result/plot_error_by_hour.png";
import plotFeatureImportance from "../Images_result/plot_feature_importance.png";
import plotResiduals from "../Images_result/plot_residuals.png";
import heroLoop from "../assets/LoadIQ_hero_loop.mp4";
import heroPoster from "../assets/loadiq-hero-poster.jpg";
import {
  Badge,
  CountUp,
  EASE,
  Logo,
  Reveal,
  SectionIntro,
  Sparkline,
  StatusDot,
  Ticker,
  buttonClass,
} from "../components/ui";

const IMAGES = {
  earth: "https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=1800&q=80",
  solar: "https://images.unsplash.com/photo-1509391366360-2e959784a276?auto=format&fit=crop&w=1600&q=80",
  solarAerial: "https://images.unsplash.com/photo-1508514177221-188b1cf16e9d?auto=format&fit=crop&w=1600&q=80",
  wind: "https://images.unsplash.com/photo-1466611653911-95081537e5b7?auto=format&fit=crop&w=1600&q=80",
  pylons: "https://images.unsplash.com/photo-1473341304170-971dccb5ac1e?auto=format&fit=crop&w=1600&q=80",
};

const NAV_LINKS = [
  { label: "Capabilities", href: "#features" },
  { label: "Method", href: "#methodology" },
  { label: "Performance", href: "#performance" },
  { label: "Roadmap", href: "#future" },
];

const SPARK_DATA = [28100, 27950, 27820, 27740, 27690, 27780, 28120, 28600, 29150, 29780, 30220, 30510, 30780, 30920, 30810, 30480];

const FEATURES = [
  {
    index: "01",
    icon: Activity,
    title: "Six years of grid memory",
    desc: "Half-hourly transmission data shaped with calendar signals, lag windows and rolling statistics. The model learns daily ramps, weekday rhythm and seasonal drift — not isolated points.",
  },
  {
    index: "02",
    icon: Shield,
    title: "One forecast, two engines",
    desc: "LightGBM and XGBoost run as a weighted ensemble, with ridge kept as a benchmark. When Render is waking up, Railway quietly takes the request and hands it back.",
  },
  {
    index: "03",
    icon: Leaf,
    title: "Light by construction",
    desc: "Routes, charts and model payloads load only when needed. Skeleton states hold the layout steady instead of preloading every dashboard, dataset and visual layer.",
  },
];

const METHOD_STEPS = [
  {
    title: "Dataset summary",
    desc: "Multi-year, half-hourly demand records preserve intraday ramps, weekday cycles and broader seasonal variation. That long view gives the model enough context to recognise routine demand shape and sudden stress periods.",
  },
  {
    title: "Feature engineering",
    desc: "Hour, day, month, lag values and rolling statistics are built from the most recent demand window. Features such as last_24_hours, rolling mean and rolling deviation help the model read momentum, recent peaks and short-term volatility.",
  },
  {
    title: "Production model",
    desc: "The deployed forecast path weights LightGBM with XGBoost support, while ridge remains a lightweight comparison path. Render is primary; Railway is the hot fallback when the long-term service is busy.",
  },
  {
    title: "Smart serving",
    desc: "Model artifacts load on demand, backend routes fire only when a page needs them, and inactive heavy paths stay out of memory. The result is a lighter platform under real usage.",
  },
  {
    title: "Operator context",
    desc: "The assistant is grounded in electricity behaviour across the UK, USA, Germany and India. It reads simulator and dashboard context before responding, then explains grid health and expected demand in plain language.",
  },
];

const FUTURE = [
  {
    icon: Radio,
    title: "Real-time API",
    desc: "High-frequency streaming for instant grid balancing and automated trading execution.",
  },
  {
    icon: Map,
    title: "Station mapping",
    desc: "Geospatial tracking of station status, output and maintenance schedules in real time.",
  },
  {
    icon: Layers,
    title: "Sector analytics",
    desc: "Granular consumption heatmaps for schools, industry and residential zones.",
  },
  {
    icon: Bot,
    title: "AI energy agent",
    desc: "A LangChain-powered assistant delivering personalised saving and optimisation strategies.",
  },
  {
    icon: FileText,
    title: "Scientific reporting",
    desc: "Automated deep dives on power factor, sinusoidal wave analysis and frequency stability.",
  },
];

const METRICS = [
  { label: "Model accuracy", value: 98.5, decimals: 1, suffix: "%", sub: "MAPE below 2%" },
  { label: "Inference time", value: 45, decimals: 0, suffix: "ms", sub: "Per request" },
  { label: "Data points", value: 2.4, decimals: 1, suffix: "M+", sub: "Historical records" },
  { label: "Uptime", value: 99.9, decimals: 1, suffix: "%", sub: "System availability" },
];

const EVALUATION_ROWS = [
  { model: "Ridge regression", mae: "850.80", rmse: "1141.82", r2: "0.99", mape: "1.59", muted: true },
  { model: "XGBoost", mae: "206.17", rmse: "313.58", r2: "1.00", mape: "0.42", best: true },
  { model: "LightGBM", mae: "202.05", rmse: "300.67", r2: "1.00", mape: "0.40", best: true },
  { model: "Stacking ensemble · LGB 60% + XGB 40%", mae: "187.23", rmse: "284.67", r2: "1.00", mape: "0.37", active: true },
];

const PLOTS = [
  { src: plotActualVsPredicted, title: "Actual vs predicted", caption: "Forecast fidelity across the holdout window" },
  { src: plotFeatureImportance, title: "Feature importance", caption: "What moves the forecast most" },
  { src: plotErrorByHour, title: "Error by hour", caption: "Residual behaviour through the day" },
  { src: plotResiduals, title: "Residual analysis", caption: "Distribution of model error" },
];

const fadeUp = {
  hidden: { opacity: 0, y: 28 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.9, ease: EASE } },
};

const stagger = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { staggerChildren: 0.1, delayChildren: 0.08 } },
};

export default function Landing() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const reduceMotion = useReducedMotion();
  const [showCookies, setShowCookies] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  const { scrollY, scrollYProgress } = useScroll();
  const progress = useSpring(scrollYProgress, { stiffness: 120, damping: 30, mass: 0.2 });
  const heroTextY = useTransform(scrollY, [0, 700], [0, 32]);

  useEffect(() => {
    const consent = localStorage.getItem("cookies-consent");
    if (!consent) {
      const timer = setTimeout(() => setShowCookies(true), 1600);
      return () => clearTimeout(timer);
    }
    return undefined;
  }, []);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    if (!mobileMenuOpen) return undefined;
    const onKey = (event) => {
      if (event.key === "Escape") setMobileMenuOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [mobileMenuOpen]);

  const acceptCookies = () => {
    localStorage.setItem("cookies-consent", "true");
    setShowCookies(false);
  };

  const goToConsole = () => navigate(user ? "/dashboard" : "/login");

  const scrollTo = (href) => {
    setMobileMenuOpen(false);
    document.querySelector(href)?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <div className="relative min-h-screen bg-paper text-ink">
      <motion.div
        className="fixed left-0 right-0 top-0 z-[70] h-[2px] origin-left bg-ink"
        style={{ scaleX: progress }}
      />

      {/* Navigation */}
      <header
        className={`fixed inset-x-0 top-0 z-50 border-b transition-all duration-500 ${
          scrolled
            ? "border-ink/10 bg-paper/88 backdrop-blur-xl"
            : "border-transparent bg-transparent"
        }`}
      >
        <div className="shell flex h-[4.5rem] items-center justify-between">
          <button onClick={() => navigate("/")} className="shrink-0" aria-label="LoadIQ home">
            <Logo />
          </button>

          <nav className="hidden items-center gap-9 lg:flex" aria-label="Primary">
            {NAV_LINKS.map((link) => (
              <a
                key={link.href}
                href={link.href}
                className="group relative text-[0.8125rem] font-medium text-ink/60 transition-colors hover:text-ink"
              >
                {link.label}
                <span className="absolute -bottom-1.5 left-0 h-px w-full origin-left scale-x-0 bg-ink transition-transform duration-300 group-hover:scale-x-100" />
              </a>
            ))}
          </nav>

          <div className="hidden items-center gap-3 sm:flex">
            <div className="mr-2 hidden items-center gap-2 xl:flex">
              <StatusDot tone="green" />
              <span className="label-sm text-ink/45">All systems nominal</span>
            </div>
            <button onClick={goToConsole} className={buttonClass("primary", "md")}>
              {user ? "Open console" : "Launch dashboard"}
              <ArrowRight size={15} className="transition-transform duration-300 group-hover:translate-x-0.5" />
            </button>
          </div>

          <button
            className="flex h-10 w-10 items-center justify-center rounded-full border border-ink/15 text-ink sm:hidden"
            onClick={() => setMobileMenuOpen((open) => !open)}
            aria-label={mobileMenuOpen ? "Close menu" : "Open menu"}
            aria-expanded={mobileMenuOpen}
          >
            {mobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>

        <AnimatePresence>
          {mobileMenuOpen && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.45, ease: EASE }}
              className="overflow-hidden border-t border-ink/10 bg-paper sm:hidden"
            >
              <div className="shell flex flex-col py-5">
                {NAV_LINKS.map((link) => (
                  <button
                    key={link.href}
                    onClick={() => scrollTo(link.href)}
                    className="flex items-center justify-between border-b border-ink/10 py-4 text-left text-lg font-medium"
                  >
                    {link.label}
                    <ArrowRight size={17} className="text-ink/40" />
                  </button>
                ))}
                <button onClick={goToConsole} className={buttonClass("primary", "lg", "mt-5 w-full")}>
                  {user ? "Open console" : "Launch dashboard"}
                  <ArrowRight size={16} />
                </button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </header>

      {/* Hero */}
      <section className="relative overflow-hidden pb-14 pt-32 sm:pt-36 lg:pb-20 lg:pt-40">
        <div className="pointer-events-none absolute inset-0 grid-lines opacity-60 [mask-image:linear-gradient(to_bottom,black,transparent_78%)]" />
        <div className="shell relative grid items-start gap-14 lg:grid-cols-12 lg:gap-10">
          <motion.div
            style={reduceMotion ? undefined : { y: heroTextY }}
            className="lg:col-span-7 xl:col-span-7"
          >
            <motion.div initial="hidden" animate="visible" variants={stagger}>
              <motion.div variants={fadeUp} className="mb-7 flex flex-wrap items-center gap-3">
                <Badge tone="signal">
                  <span className="relative flex h-1.5 w-1.5">
                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-ink opacity-50" />
                    <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-ink" />
                  </span>
                  Short-term demand intelligence
                </Badge>
                <span className="label-sm text-ink/40">v4.2.1</span>
              </motion.div>

              <motion.h1
                variants={fadeUp}
                className="text-[3.1rem] leading-[0.94] tracking-[-0.055em] sm:text-[4.5rem] lg:text-[5.4rem] xl:text-[6.2rem]"
              >
                The grid,
                <br />
                before it
                <span className="display-serif ml-3 italic text-moss">moves.</span>
              </motion.h1>

              <motion.p
                variants={fadeUp}
                className="mt-8 max-w-xl text-[0.98rem] leading-[1.75] text-ink/60 sm:text-[1.05rem]"
              >
                LoadIQ forecasts electricity demand across the UK, USA, Germany and India
                from six years of half-hourly transmission data. On-demand routes and a
                Render-to-Railway fallback keep the console fast when the network isn&apos;t.
              </motion.p>

              <motion.div variants={fadeUp} className="mt-9 flex flex-col gap-3 sm:flex-row sm:items-center">
                <button onClick={goToConsole} className={buttonClass("primary", "lg")}>
                  Start analysing
                  <ArrowRight size={17} className="transition-transform duration-300 group-hover:translate-x-1" />
                </button>
                <button onClick={() => scrollTo("#methodology")} className={buttonClass("outline", "lg")}>
                  How the forecast works
                </button>
              </motion.div>

              <motion.div variants={fadeUp} className="mt-12 grid max-w-xl grid-cols-3 gap-5 border-t border-ink/10 pt-6">
                {[
                  ["4", "Markets modelled"],
                  ["6 yrs", "Half-hourly history"],
                  ["<1.5%", "Reported MAPE"],
                ].map(([value, label]) => (
                  <div key={label}>
                    <p className="mono-num text-2xl font-medium tracking-tight">{value}</p>
                    <p className="label-sm mt-2 text-ink/40">{label}</p>
                  </div>
                ))}
              </motion.div>
            </motion.div>
          </motion.div>

          {/* Hero visual */}
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1.1, delay: 0.25, ease: EASE }}
            className="relative lg:col-span-5"
          >
            <div className="relative overflow-hidden rounded-[1.5rem] border border-ink/15 bg-ink shadow-[0_40px_90px_-50px_rgba(17,24,19,0.9)]">
              <div className="relative h-[26rem] overflow-hidden sm:h-[30rem] lg:h-[34rem]">
                {reduceMotion ? (
                  <img
                    src={heroPoster}
                    alt="A lime energy demand curve sweeping across a dark green grid surface"
                    className="absolute inset-0 h-full w-full object-cover"
                    loading="eager"
                  />
                ) : (
                  <video
                    src={heroLoop}
                    poster={heroPoster}
                    autoPlay
                    muted
                    loop
                    playsInline
                    preload="metadata"
                    aria-hidden="true"
                    className="absolute inset-0 h-full w-full object-cover"
                  />
                )}
                <div className="absolute inset-0 bg-gradient-to-t from-ink via-ink/25 to-transparent" />
                <div className="absolute inset-0 grain" />

                <div className="absolute left-5 right-5 top-5 flex items-center justify-between">
                  <Badge tone="light">
                    <span className="h-1.5 w-1.5 rounded-full bg-signal-300 animate-blink" />
                    Live grid snapshot
                  </Badge>
                  <span className="label-sm text-paper/50">GB · 14:00 UTC</span>
                </div>

                <div className="absolute inset-x-5 bottom-5">
                  <div className="rounded-2xl border border-white/10 bg-ink-950/70 p-4 backdrop-blur-md sm:p-5">
                    <div className="flex items-end justify-between gap-4">
                      <div>
                        <p className="label-sm text-paper/45">Predicted load · next hour</p>
                        <p className="mono-num mt-2 text-3xl font-medium text-paper sm:text-4xl">
                          28,120 <span className="text-sm font-normal text-paper/50">MW</span>
                        </p>
                      </div>
                      <div className="text-right">
                        <span className="label-sm inline-flex items-center gap-1.5 rounded-full border border-signal/30 bg-signal/10 px-2.5 py-1 text-signal-200">
                          <ArrowRight size={11} className="-rotate-45" />
                          1.8%
                        </span>
                        <p className="label-sm mt-2 text-paper/35">vs. last hour</p>
                      </div>
                    </div>
                    <div className="mt-4">
                      <Sparkline
                        data={SPARK_DATA}
                        width={420}
                        height={54}
                        stroke="#D8F24D"
                        fill="rgba(216,242,77,0.22)"
                        className="h-12 w-full"
                      />
                    </div>
                    <div className="mt-3 flex items-center justify-between border-t border-white/10 pt-3">
                      <span className="label-sm text-paper/35">Ensemble · LGB + XGB</span>
                      <span className="label-sm flex items-center gap-1.5 text-signal-200">
                        <Check size={11} /> model live
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <motion.div
              initial={{ opacity: 0, x: 18 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8, delay: 0.9, ease: EASE }}
              className="absolute -left-3 top-[42%] hidden rounded-2xl border border-ink/10 bg-paper-50 p-3.5 shadow-panel sm:block"
            >
              <p className="label-sm text-ink/40">Forecast accuracy</p>
              <p className="mono-num mt-1 text-xl font-medium">98.5%</p>
              <div className="mt-2 flex gap-1" aria-hidden="true">
                {[38, 62, 48, 74, 58, 88, 72, 96].map((height, index) => (
                  <span
                    key={index}
                    className="w-1 rounded-sm bg-moss/70"
                    style={{ height: `${Math.max(6, height / 6)}px`, opacity: 0.45 + index * 0.06 }}
                  />
                ))}
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Ticker */}
      <div className="border-y border-ink/10 bg-ink text-paper">
        <Ticker
          items={[
            "UK · USA · GERMANY · INDIA",
            "LIGHTGBM + XGBOOST ENSEMBLE",
            "RENDER PRIMARY · RAILWAY FALLBACK",
            "SKELETON-FIRST ROUTES",
            "30-MINUTE TRANSMISSION DATA",
            "OPEN ENERGY INTELLIGENCE",
          ]}
        />
      </div>

      {/* Capabilities */}
      <section id="features" className="relative scroll-mt-24 py-24 sm:py-32">
        <div className="shell">
          <div className="grid gap-10 border-b border-ink/10 pb-14 lg:grid-cols-12 lg:items-end">
            <SectionIntro
              eyebrow="Why it is different"
              title={
                <>
                  Most platforms load everything.
                  <span className="text-ink/35"> We load what you need.</span>
                </>
              }
              className="lg:col-span-8"
            />
            <p className="text-sm leading-relaxed text-ink/55 lg:col-span-4 lg:pb-2">
              Every chart, dataset and visual layer has a cost. LoadIQ treats compute and
              carbon as the same budget — rendering on demand, holding layout with skeletons,
              and grounding the assistant in actual grid context.
            </p>
          </div>

          <div className="grid gap-px overflow-hidden rounded-[1.5rem] border border-ink/10 bg-ink/10 lg:grid-cols-3">
            {FEATURES.map((feature, index) => (
              <Reveal key={feature.title} delay={index * 0.08} className="group relative bg-paper-50 p-7 sm:p-9 lg:min-h-[23rem]">
                <div className="flex items-start justify-between">
                  <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-ink/10 bg-paper transition-colors duration-500 group-hover:border-moss/30 group-hover:bg-signal/40">
                    <feature.icon size={19} strokeWidth={1.7} />
                  </div>
                  <span className="label text-ink/25">{feature.index}</span>
                </div>
                <h3 className="mt-10 text-[1.35rem] leading-snug tracking-[-0.03em]">{feature.title}</h3>
                <p className="mt-4 text-sm leading-[1.7] text-ink/55">{feature.desc}</p>
                <div className="absolute inset-x-9 bottom-8 h-px origin-left scale-x-0 bg-ink/20 transition-transform duration-700 group-hover:scale-x-100" />
              </Reveal>
            ))}
          </div>

          <div className="mt-6 grid gap-6 lg:grid-cols-12">
            <Reveal className="relative overflow-hidden rounded-[1.5rem] lg:col-span-7" y={30}>
              <img
                src={IMAGES.solarAerial}
                alt="Aerial view of a large solar farm"
                className="h-[20rem] w-full object-cover transition-transform duration-[1.4s] ease-out hover:scale-[1.03] sm:h-[24rem]"
                loading="lazy"
              />
              <div className="absolute inset-0 bg-gradient-to-t from-ink/85 via-ink/10 to-transparent" />
              <div className="absolute inset-x-6 bottom-6 flex flex-wrap items-end justify-between gap-4">
                <div>
                  <p className="label-sm text-paper/55">Renewable input layer</p>
                  <p className="mt-2 max-w-sm text-xl font-medium leading-snug text-paper sm:text-2xl">
                    Demand forecasts only matter when supply can answer them.
                  </p>
                </div>
                <span className="label-sm rounded-full border border-paper/20 px-3 py-2 text-paper/70">
                  Solar · Wind · Price
                </span>
              </div>
            </Reveal>

            <Reveal delay={0.1} className="flex flex-col justify-between rounded-[1.5rem] border border-ink/10 bg-paper-50 p-7 sm:p-9 lg:col-span-5">
              <div>
                <p className="label text-ink/40">Built for the operator</p>
                <p className="mt-6 text-[1.6rem] leading-[1.3] tracking-[-0.035em]">
                  A working console, not a demo.
                  <span className="display-serif italic text-moss"> Every number is traceable.</span>
                </p>
              </div>
              <div className="mt-10 space-y-0">
                {[
                  ["Source", "ENTSO-E transparency"],
                  ["Refresh", "On demand, then cached locally"],
                  ["Fallback", "Automatic Render → Railway"],
                ].map(([key, value], index) => (
                  <div key={key} className={`flex items-baseline justify-between gap-4 py-3.5 ${index !== 2 ? "border-b border-ink/10" : ""}`}>
                    <span className="label-sm text-ink/40">{key}</span>
                    <span className="text-right text-xs font-medium text-ink/75">{value}</span>
                  </div>
                ))}
              </div>
            </Reveal>
          </div>
        </div>
      </section>

      {/* Smart engineering / walkthrough */}
      <section className="relative overflow-hidden bg-ink py-24 text-paper sm:py-32">
        <div className="pointer-events-none absolute inset-0 dot-grid-light opacity-40" />
        <div className="pointer-events-none absolute -right-40 top-0 h-[34rem] w-[34rem] rounded-full bg-moss/40 blur-[120px]" />
        <div className="shell relative grid gap-14 lg:grid-cols-12 lg:gap-16">
          <div className="lg:col-span-5">
            <Reveal>
              <div className="mb-5 flex items-center gap-3">
                <span className="h-1.5 w-1.5 rounded-full bg-signal" />
                <span className="label text-paper/45">Engineering note</span>
              </div>
              <h2 className="text-[2.4rem] leading-[1.02] tracking-[-0.045em] sm:text-5xl">
                Smart engineering behind
                <span className="display-serif italic text-signal"> every forecast.</span>
              </h2>
              <p className="mt-7 max-w-lg text-sm leading-[1.8] text-paper/55 sm:text-[0.98rem]">
                Backend-driven pages arrive on demand. Skeleton states hold the interface
                steady during network or service delays. Heavier rendering stays conditional,
                and the platform remains honest about the energy it consumes.
              </p>
            </Reveal>

            <Reveal delay={0.12} className="mt-10 grid gap-3 sm:grid-cols-2">
              {[
                { label: "On-demand routes", value: "No prefetch" },
                { label: "Skeleton paging", value: "Stable layout" },
                { label: "Render → Railway", value: "Auto handoff" },
                { label: "Median-fill inference", value: "No zeroing" },
              ].map((item) => (
                <div key={item.label} className="rounded-xl border border-white/10 bg-white/[0.03] p-4">
                  <p className="label-sm text-paper/35">{item.label}</p>
                  <p className="mt-2 text-sm font-medium text-paper/85">{item.value}</p>
                </div>
              ))}
            </Reveal>

            <Reveal delay={0.2} className="mt-8 overflow-hidden rounded-2xl border border-white/10">
              <img
                src={IMAGES.pylons}
                alt="High-voltage transmission towers at sunset"
                className="h-56 w-full object-cover opacity-80 transition-transform duration-[1.4s] hover:scale-[1.04]"
                loading="lazy"
              />
            </Reveal>
          </div>

          <Reveal delay={0.1} y={36} className="lg:col-span-7">
            <div className="overflow-hidden rounded-[1.5rem] border border-white/15 bg-paper shadow-[0_50px_100px_-40px_rgba(0,0,0,0.8)]">
              <div className="flex items-center justify-between border-b border-ink/10 bg-paper-100 px-4 py-3">
                <div className="flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full bg-clay/70" />
                  <span className="h-2 w-2 rounded-full bg-amber-400/80" />
                  <span className="h-2 w-2 rounded-full bg-moss/60" />
                </div>
                <span className="label-sm text-ink/40">LoadIQ · product walkthrough</span>
                <span className="label-sm hidden text-ink/30 sm:inline">02:14</span>
              </div>
              <div className="relative aspect-video w-full bg-ink">
                <iframe
                  src="https://app.heygen.com/embeds/794cd7ca4fd1481a90bf858901b627c4"
                  title="LoadIQ AI video walkthrough"
                  frameBorder="0"
                  allow="encrypted-media; fullscreen; picture-in-picture"
                  allowFullScreen
                  loading="lazy"
                  className="absolute inset-0 h-full w-full"
                />
              </div>
            </div>
            <div className="mt-5 flex flex-wrap items-center gap-x-8 gap-y-3 border-t border-white/10 pt-5">
              {["Narrated walkthrough", "Live dashboard", "Model context"].map((item) => (
                <span key={item} className="label-sm flex items-center gap-2 text-paper/40">
                  <Check size={12} className="text-signal" /> {item}
                </span>
              ))}
            </div>
          </Reveal>
        </div>
      </section>

      {/* Methodology */}
      <section id="methodology" className="scroll-mt-24 py-24 sm:py-32">
        <div className="shell">
          <div className="grid gap-10 lg:grid-cols-12">
            <div className="lg:col-span-5">
              <div className="lg:sticky lg:top-28">
                <SectionIntro
                  eyebrow="Under the hood"
                  title={
                    <>
                      From raw demand
                      <br />
                      to a usable signal.
                    </>
                  }
                  description="The production path is deliberately narrow: build the features, run the ensemble, keep a benchmark, and return a forecast the operator can reason about."
                />
                <div className="mt-10 overflow-hidden rounded-[1.25rem] border border-ink/10">
                  <img
                    src={IMAGES.solar}
                    alt="Solar panels under a blue sky"
                    className="h-52 w-full object-cover image-wash"
                    loading="lazy"
                  />
                </div>
                <div className="mt-4 flex items-center justify-between">
                  <span className="label-sm text-ink/35">Pipeline · v2026.05</span>
                  <span className="label-sm flex items-center gap-2 text-moss">
                    <StatusDot tone="green" /> healthy
                  </span>
                </div>
              </div>
            </div>

            <div className="lg:col-span-7">
              <div className="border-t border-ink/10">
                {METHOD_STEPS.map((step, index) => (
                  <Reveal
                    key={step.title}
                    delay={index * 0.04}
                    className="group grid grid-cols-[2.75rem_1fr] gap-5 border-b border-ink/10 py-8 transition-colors duration-500 hover:bg-paper-50 sm:grid-cols-[4rem_1fr] sm:gap-8 sm:py-10"
                  >
                    <span className="mono-num pt-1 text-xs text-ink/30 transition-colors group-hover:text-moss">
                      {String(index + 1).padStart(2, "0")}
                    </span>
                    <div>
                      <div className="flex items-center justify-between gap-4">
                        <h3 className="text-lg font-medium tracking-[-0.025em] sm:text-xl">{step.title}</h3>
                        <span className="h-px w-8 bg-ink/15 transition-all duration-500 group-hover:w-14 group-hover:bg-ink/40" />
                      </div>
                      <p className="mt-3 max-w-2xl text-sm leading-[1.75] text-ink/55">{step.desc}</p>
                    </div>
                  </Reveal>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Performance metrics */}
      <section id="performance" className="scroll-mt-24 border-y border-ink/10 bg-signal">
        <div className="shell grid grid-cols-2 gap-px bg-ink/15 md:grid-cols-4">
          {METRICS.map((metric) => (
            <div key={metric.label} className="bg-signal px-5 py-10 sm:px-7 sm:py-14">
              <p className="mono-num text-[2.1rem] font-medium leading-none tracking-[-0.06em] sm:text-[2.8rem]">
                <CountUp value={metric.value} decimals={metric.decimals} suffix={metric.suffix} />
              </p>
              <p className="label mt-5 text-ink/55">{metric.label}</p>
              <p className="mt-1.5 text-xs text-ink/40">{metric.sub}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Model evaluation */}
      <section className="py-24 sm:py-32">
        <div className="shell">
          <div className="flex flex-col justify-between gap-8 border-b border-ink/10 pb-12 lg:flex-row lg:items-end">
            <SectionIntro
              eyebrow="Evaluation metrics"
              title={
                <>
                  Measured, not
                  <span className="display-serif italic"> asserted.</span>
                </>
              }
              description="Results across baseline and ensemble models. The blended path is the production default because it smooths variance at peak transitions without giving up accuracy."
            />
            <div className="flex shrink-0 gap-8 border-t border-ink/10 pt-5 lg:border-t-0 lg:pt-0">
              <div>
                <p className="mono-num text-2xl font-medium">0.37%</p>
                <p className="label-sm mt-2 text-ink/40">Ensemble MAPE</p>
              </div>
              <div className="border-l border-ink/10 pl-8">
                <p className="mono-num text-2xl font-medium">187.23</p>
                <p className="label-sm mt-2 text-ink/40">Ensemble MAE</p>
              </div>
            </div>
          </div>

          <div className="mt-12 overflow-x-auto">
            <table className="w-full min-w-[680px] border-collapse text-left">
              <thead>
                <tr className="border-b border-ink/15">
                  {["Model", "MAE", "RMSE", "R²", "MAPE"].map((heading, index) => (
                    <th
                      key={heading}
                      className={`label pb-4 font-medium text-ink/40 ${index > 0 ? "text-right" : ""}`}
                    >
                      {heading}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {EVALUATION_ROWS.map((row) => (
                  <tr
                    key={row.model}
                    className={`border-b border-ink/10 transition-colors hover:bg-paper-100 ${
                      row.active ? "bg-paper-100/70" : ""
                    }`}
                  >
                    <td className={`py-5 pr-6 text-sm ${row.active ? "font-semibold" : "font-medium text-ink/75"}`}>
                      <span className="flex items-center gap-3">
                        {row.active && <span className="h-1.5 w-1.5 rounded-full bg-moss" />}
                        {row.model}
                      </span>
                    </td>
                    {[row.mae, row.rmse, row.r2, row.mape].map((value, index) => (
                      <td
                        key={index}
                        className={`mono-num py-5 pl-6 text-right text-sm ${
                          row.best || row.active ? "text-moss" : "text-ink/55"
                        } ${row.active ? "font-semibold" : ""}`}
                      >
                        {value}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-6 flex flex-col items-start justify-between gap-4 rounded-2xl border border-ink/10 bg-paper-50 px-6 py-5 sm:flex-row sm:items-center">
            <div>
              <p className="label text-ink/40">5-fold TimeSeriesSplit · LightGBM</p>
              <p className="mt-2 text-sm text-ink/55">
                Mean absolute error <span className="mono-num font-medium text-ink">400.1 ± 544.3 MW</span>
                <span className="mx-3 text-ink/20">/</span>
                R² <span className="mono-num font-medium text-ink">0.7836 ± 0.4291</span>
              </p>
            </div>
            <Badge tone="moss">
              <Check size={11} /> Chronological split
            </Badge>
          </div>

          <div className="mt-14 grid gap-5 sm:grid-cols-2">
            {PLOTS.map((plot, index) => (
              <Reveal key={plot.title} delay={index * 0.05} className="group overflow-hidden rounded-[1.25rem] border border-ink/10 bg-paper-50">
                <div className="flex items-center justify-between border-b border-ink/10 px-5 py-3.5">
                  <span className="text-sm font-medium">{plot.title}</span>
                  <span className="label-sm text-ink/30">{String(index + 1).padStart(2, "0")}</span>
                </div>
                <div className="overflow-hidden bg-white p-3">
                  <img
                    src={plot.src}
                    alt={plot.title}
                    className="aspect-[16/10] w-full object-contain transition-transform duration-700 group-hover:scale-[1.025]"
                    loading="lazy"
                  />
                </div>
                <p className="px-5 py-3.5 text-xs text-ink/45">{plot.caption}</p>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* Future */}
      <section id="future" className="scroll-mt-24 border-t border-ink/10 bg-paper-100 py-24 sm:py-32">
        <div className="shell">
          <div className="grid gap-10 lg:grid-cols-12 lg:items-end">
            <SectionIntro
              eyebrow="Roadmap"
              title={
                <>
                  What comes after
                  <span className="display-serif italic"> the forecast.</span>
                </>
              }
              className="lg:col-span-7"
            />
            <p className="text-sm leading-relaxed text-ink/55 lg:col-span-4 lg:col-start-9">
              The next layer is operational: streaming data, station-level context and
              reporting that an energy team can act on without translating it first.
            </p>
          </div>

          <div className="mt-14 grid gap-px overflow-hidden rounded-[1.5rem] border border-ink/10 bg-ink/10 md:grid-cols-2 lg:grid-cols-3">
            {FUTURE.map((feature, index) => (
              <Reveal
                key={feature.title}
                delay={index * 0.04}
                className={`group relative bg-paper-50 p-7 transition-colors duration-500 hover:bg-white sm:p-8 ${
                  index === 0 ? "lg:col-span-2" : ""
                }`}
              >
                <div className="flex items-start justify-between">
                  <feature.icon size={20} strokeWidth={1.6} className="text-ink/70 transition-transform duration-500 group-hover:-translate-y-0.5" />
                  <span className="label-sm text-ink/25">{String(index + 1).padStart(2, "0")}</span>
                </div>
                <h3 className={`mt-8 font-medium tracking-[-0.03em] ${index === 0 ? "text-2xl sm:text-3xl" : "text-xl"}`}>
                  {feature.title}
                </h3>
                <p className="mt-3 max-w-md text-sm leading-relaxed text-ink/50">{feature.desc}</p>
                {index === 0 && (
                  <div className="mt-10 flex flex-wrap gap-2">
                    {["WebSocket feed", "Balance signals", "Trade triggers"].map((tag) => (
                      <span key={tag} className="label-sm rounded-full border border-ink/10 px-3 py-1.5 text-ink/40">
                        {tag}
                      </span>
                    ))}
                  </div>
                )}
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* Closing CTA */}
      <section className="relative overflow-hidden bg-ink py-20 text-paper sm:py-28">
        <img
          src={IMAGES.wind}
          alt="Wind turbines in a field at sunset"
          className="absolute inset-0 h-full w-full object-cover opacity-25"
          loading="lazy"
        />
        <div className="absolute inset-0 bg-gradient-to-r from-ink via-ink/90 to-ink/40" />
        <div className="absolute inset-0 grain" />
        <div className="shell relative">
          <Reveal className="max-w-3xl">
            <p className="label text-signal">Built like the future it supports</p>
            <h2 className="mt-6 text-[2.5rem] leading-[1.02] tracking-[-0.05em] sm:text-5xl lg:text-6xl">
              A platform built to promote clean energy
              <span className="display-serif italic text-paper/60"> should be built cleanly, too.</span>
            </h2>
            <p className="mt-7 max-w-xl text-sm leading-[1.8] text-paper/55 sm:text-base">
              LoadIQ is not just a forecasting tool. It treats compute and carbon with equal
              respect — because the software behind the transition should not be wasteful.
            </p>
            <div className="mt-9 flex flex-col gap-3 sm:flex-row">
              <button onClick={goToConsole} className={buttonClass("signal", "lg")}>
                Enter the console
                <ArrowRight size={17} className="transition-transform duration-300 group-hover:translate-x-1" />
              </button>
              <button onClick={() => scrollTo("#methodology")} className={buttonClass("darkOutline", "lg")}>
                Read the method
              </button>
            </div>
          </Reveal>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-ink-950 pb-10 pt-16 text-paper">
        <div className="shell">
          <div className="grid gap-12 border-b border-white/10 pb-14 lg:grid-cols-12">
            <div className="lg:col-span-6">
              <Logo theme="light" />
              <p className="mt-6 max-w-sm text-sm leading-relaxed text-paper/40">
                Advancing artificial intelligence in the energy sector. Building the future
                of smart grid analytics, one forecast at a time.
              </p>
              <div className="mt-7 flex gap-2">
                {[
                  { icon: Linkedin, href: "https://www.linkedin.com/in/parthmhatre41/", label: "LinkedIn" },
                  { icon: Github, href: "https://github.com/Parth-S-Mhatre", label: "GitHub" },
                  { icon: Twitter, href: "http://x.com/ParthMhatre41", label: "Twitter" },
                ].map((social) => (
                  <a
                    key={social.label}
                    href={social.href}
                    target="_blank"
                    rel="noreferrer"
                    aria-label={social.label}
                    className="flex h-9 w-9 items-center justify-center rounded-full border border-white/10 text-paper/50 transition-colors hover:border-signal/40 hover:text-signal"
                  >
                    <social.icon size={15} />
                  </a>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-8 sm:grid-cols-3 lg:col-span-6">
              <div>
                <p className="label text-paper/30">Product</p>
                <div className="mt-4 flex flex-col gap-3 text-sm text-paper/55">
                  {NAV_LINKS.slice(0, 3).map((link) => (
                    <a key={link.href} href={link.href} className="transition-colors hover:text-paper">
                      {link.label}
                    </a>
                  ))}
                </div>
              </div>
              <div>
                <p className="label text-paper/30">Platform</p>
                <div className="mt-4 flex flex-col gap-3 text-sm text-paper/55">
                  <button onClick={goToConsole} className="text-left transition-colors hover:text-paper">Console</button>
                  <button onClick={() => navigate(user ? "/profile" : "/login")} className="text-left transition-colors hover:text-paper">Workspace</button>
                  <a href="#future" className="transition-colors hover:text-paper">Roadmap</a>
                </div>
              </div>
              <div>
                <p className="label text-paper/30">Crafted by</p>
                <p className="mt-4 text-sm font-medium text-paper/80">Parth Mhatre</p>
                <p className="mt-2 text-xs leading-relaxed text-paper/35">Energy intelligence,<br />built for operators.</p>
              </div>
            </div>
          </div>

          <div className="flex flex-col justify-between gap-4 pt-8 text-[0.7rem] text-paper/30 sm:flex-row sm:items-center">
            <p>© 2026 LoadIQ · All rights reserved</p>
            <div className="flex items-center gap-5">
              <span className="flex items-center gap-2"><StatusDot tone="lime" /> API operational</span>
              <span>Made for the energy transition</span>
            </div>
          </div>
        </div>
      </footer>

      {/* Cookies */}
      <AnimatePresence>
        {showCookies && (
          <motion.div
            initial={{ y: 40, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: 30, opacity: 0 }}
            transition={{ type: "spring", stiffness: 260, damping: 28 }}
            className="fixed bottom-4 left-4 right-4 z-[100] sm:left-auto sm:right-6 sm:max-w-sm"
          >
            <div className="rounded-2xl border border-ink/15 bg-paper-50 p-5 shadow-lift">
              <div className="flex items-start gap-3">
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-ink text-signal">
                  <Shield size={16} />
                </div>
                <div>
                  <p className="text-sm font-semibold">Cookies & privacy</p>
                  <p className="mt-1.5 text-xs leading-relaxed text-ink/50">
                    We use local storage to remember your session, workspace and grid context.
                    No advertising trackers.
                  </p>
                  <div className="mt-4 flex gap-2">
                    <button onClick={acceptCookies} className={buttonClass("primary", "sm", "flex-1")}>
                      Accept all
                    </button>
                    <button onClick={() => setShowCookies(false)} className={buttonClass("outline", "sm")}>
                      Decline
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
