import React, { lazy, Suspense, useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { motion } from "framer-motion";
import { ArrowRight, Activity, Shield, Leaf, Cpu, BarChart2, Target, Zap, Map, FileText, Bot, Layers, Radio, Menu, X } from "lucide-react";
const ThreeBackground = lazy(() => import("../components/3d/ThreeBackground"));

export default function Landing() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [showCookies, setShowCookies] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const consent = localStorage.getItem("cookies-consent");
    if (!consent) {
      const timer = setTimeout(() => setShowCookies(true), 1500);
      return () => clearTimeout(timer);
    }
  }, []);

  const acceptCookies = () => {
    localStorage.setItem("cookies-consent", "true");
    setShowCookies(false);
  };

  const fadeInUp = {
    hidden: { opacity: 0, y: 30 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.8, ease: "easeOut" } }
  };

  const staggerContainer = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.2 } }
  };

  return (
    <div className="relative min-h-screen bg-[#080B14] text-white selection:bg-[#6366F1]/30 overflow-x-hidden font-sans">

      {/* 3D Background Layer */}
      <Suspense fallback={null}>
        <ThreeBackground />
      </Suspense>

      {/* Overlay to ensure text readability */}
      <div className="fixed inset-0 bg-[#080B14]/40 pointer-events-none z-0"></div>

      {/* Navigation */}
      <nav className="fixed top-0 w-full z-50 backdrop-blur-md border-b border-white/5 bg-[#080B14]/70 transition-all duration-300">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          <div className="flex items-center gap-3 cursor-pointer" onClick={() => navigate("/")}>
            <div className="w-10 h-10 bg-gradient-to-tr from-indigo-600 to-cyan-400 rounded-xl flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <Zap className="text-white fill-current" size={20} />
            </div>
            <div>
              <p className="font-bold text-xl tracking-tight text-white">LoadIQ</p>
              <p className="text-[10px] items-center font-semibold tracking-widest text-indigo-400 uppercase">Energy Analytics</p>
            </div>
          </div>

          <div className="hidden md:flex gap-8">
            {["Features", "Methodology", "Performance", "Future"].map((item) => (
              <a
                key={item}
                href={`#${item.toLowerCase()}`}
                className="relative text-sm font-medium text-slate-300 hover:text-white transition-colors py-2 group"
              >
                {item}
                <span className="absolute bottom-0 left-0 w-0 h-0.5 bg-indigo-500 transition-all duration-300 ease-out group-hover:w-full" />
              </a>
            ))}
          </div>

          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate(user ? "/dashboard" : "/login")}
              className="hidden sm:block bg-white text-slate-950 px-6 py-2.5 rounded-full font-bold text-sm hover:bg-indigo-50 shadow-lg hover:scale-105 transition-all"
            >
              Launch Dashboard
            </button>

            <button
              className="md:hidden text-white p-2 hover:bg-white/5 rounded-lg transition-colors"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              {mobileMenuOpen ? <X size={28} /> : <Menu size={28} />}
            </button>
          </div>
        </div>

        {/* Mobile Menu Overlay */}
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{
            opacity: mobileMenuOpen ? 1 : 0,
            height: mobileMenuOpen ? "auto" : 0
          }}
          className="md:hidden overflow-hidden bg-[#080B14]/95 backdrop-blur-xl border-t border-white/5"
        >
          <div className="flex flex-col p-6 space-y-4">
            {["Features", "Methodology", "Performance", "Future"].map((item) => (
              <a
                key={item}
                href={`#${item.toLowerCase()}`}
                onClick={() => setMobileMenuOpen(false)}
                className="text-lg font-semibold text-slate-300 hover:text-indigo-400 py-3 border-b border-white/5 transition-colors"
              >
                {item}
              </a>
            ))}
            <button
              onClick={() => {
                setMobileMenuOpen(false);
                navigate(user ? "/dashboard" : "/login");
              }}
              className="w-full bg-indigo-600 text-white py-4 rounded-2xl font-bold mt-4 shadow-lg shadow-indigo-600/20 active:scale-95 transition-all"
            >
              Launch Dashboard
            </button>
          </div>
        </motion.div>
      </nav>

      {/* Hero Section */}
      <section className="relative pt-32 pb-20 px-6 z-10 flex flex-col items-center text-center max-w-5xl mx-auto min-h-screen justify-center">
        <motion.div
          initial="hidden"
          animate="visible"
          variants={staggerContainer}
          className="space-y-8"
        >
          <motion.div variants={fadeInUp} className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-indigo-500/10 border border-indigo-500/20 mx-auto backdrop-blur-md">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-500"></span>
            </span>
            <span className="text-xs font-bold uppercase tracking-widest text-indigo-300">Next-Gen Forecasting Engine</span>
          </motion.div>

          <motion.h1 variants={fadeInUp} className="text-5xl md:text-8xl font-black tracking-tighter leading-tight md:leading-[1.1]">
            Predict the <br className="hidden md:block" />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-cyan-400 to-emerald-400">
              Energy Pulse.
            </span>
          </motion.h1>

          <motion.p variants={fadeInUp} className="text-base md:text-xl text-slate-400 max-w-2xl mx-auto leading-relaxed px-4">
            Advanced LSTM neural networks meet real-time grid analytics.
            Optimize distribution, reduce waste, and anticipate demand surges with precision.
          </motion.p>

          <motion.div variants={fadeInUp} className="flex flex-col sm:flex-row gap-4 justify-center items-center pt-8 w-full px-6">
            <button
              onClick={() => navigate(user ? "/dashboard" : "/login")}
              className="w-full sm:w-auto group relative px-8 py-4 bg-indigo-600 rounded-full font-bold text-white shadow-2xl shadow-indigo-600/30 overflow-hidden transition-all hover:scale-105 hover:shadow-indigo-600/50"
            >
              <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300"></div>
              <span className="relative flex items-center justify-center gap-2">
                Start Analyzing <ArrowRight size={18} />
              </span>
            </button>
            <button
              onClick={() => document.getElementById('methodology')?.scrollIntoView({ behavior: 'smooth' })}
              className="w-full sm:w-auto px-8 py-4 glass-panel border border-slate-700/50 rounded-full font-bold text-slate-300 hover:text-white hover:bg-slate-800 hover:border-slate-600 transition-all backdrop-blur-md"
            >
              See How It Works
            </button>
          </motion.div>
        </motion.div>
      </section>

      {/* Features Grid */}
      <section id="features" className="py-24 px-6 relative z-10">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-5xl font-bold mb-4">Intelligence at Scale</h2>
            <p className="text-slate-400 text-sm md:text-base">Powered by state-of-the-art infrastructure.</p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              { icon: <Activity />, title: "Time-Series Analysis", desc: "LSTM & GRU architectures process historical load data to forecast future demand with up to 98% accuracy." },
              { icon: <Shield />, title: "Anomaly Detection", desc: "Real-time outlier detection algorithms instantly flag potential grid failures or data inconsistencies." },
              { icon: <Leaf />, title: "Sustainability First", desc: "Optimize renewable integration by aligning consumption patterns with solar and wind generation peaks." }
            ].map((f, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.2, duration: 0.6 }}
                className="p-8 rounded-[2rem] glass-panel border border-white/5 hover:border-[#6366F1]/30 hover:bg-[#10162A] transition-all group backdrop-blur-sm"
              >
                <div className="w-14 h-14 bg-indigo-500/10 rounded-2xl flex items-center justify-center text-indigo-400 mb-6 group-hover:rotate-12 transition-transform duration-300">
                  {React.cloneElement(f.icon, { size: 28 })}
                </div>
                <h3 className="text-xl font-bold mb-3">{f.title}</h3>
                <p className="text-slate-400 leading-relaxed text-sm">{f.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Interactive 3D/Tech Section */}
      <section className="py-24 px-6 relative z-10 glass-panel border-y border-white/5 overflow-hidden">
        <div className="max-w-7xl mx-auto grid lg:grid-cols-2 gap-16 items-center">
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-4xl md:text-6xl font-black mb-6 tracking-tight">
              See the Grid in <br />
              <span className="text-indigo-400">Three Dimensions.</span>
            </h2>
            <p className="text-slate-300 text-lg mb-8 leading-relaxed">
              Our 3D visualization engine provides unprecedented situational awareness. Rotate, zoom, and inspect load distribution across temporal and spatial dimensions.
            </p>
            <div className="flex flex-wrap gap-4">
              <div className="px-5 py-3 rounded-xl bg-slate-950 border border-slate-800 flex items-center gap-3">
                <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                <span className="text-sm font-bold text-slate-300">WebGL 2.0</span>
              </div>
              <div className="px-5 py-3 rounded-xl bg-slate-950 border border-slate-800 flex items-center gap-3">
                <div className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse"></div>
                <span className="text-sm font-bold text-slate-300">React Three Fiber</span>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
            className="aspect-video bg-gradient-to-tr from-indigo-900/20 to-slate-900 rounded-3xl border border-indigo-500/20 relative overflow-hidden flex items-center justify-center group shadow-2xl"
          >
            <div className="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1550751827-4bd374c3f58b?q=80&w=2070&auto=format&fit=crop')] bg-cover bg-center opacity-40 mix-blend-overlay transition-transform duration-700 group-hover:scale-110"></div>
            <div className="relative z-10 text-center">
              <div className="w-20 h-20 bg-white/10 backdrop-blur-md rounded-full flex items-center justify-center mx-auto mb-4 border border-white/20 group-hover:scale-110 transition-transform cursor-pointer shadow-2xl">
                <div className="w-0 h-0 border-t-[10px] border-t-transparent border-l-[18px] border-l-white border-b-[10px] border-b-transparent ml-1"></div>
              </div>
              <p className="font-bold text-white tracking-widest text-sm uppercase">Interactive Demo</p>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Methodology & Training */}
      <section id="methodology" className="py-24 px-6 relative z-10">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-16">
            <span className="text-indigo-400 font-bold tracking-widest text-xs uppercase mb-3 block">Under the Hood</span>
            <h2 className="text-3xl md:text-5xl font-bold">Model Training Methodology</h2>
          </div>

          <div className="space-y-6">
            {[
              { title: "Data Ingestion", desc: "Aggregating data from ENTSO-E, weather APIs, and historical CSV logs.", color: "bg-blue-500" },
              { title: "Preprocessing", desc: "Normalization (MinMax Scaling), handling missing values, and lag feature creation.", color: "bg-indigo-500" },
              { title: "Model Architecture", desc: "Two stacked LSTM layers (50 units) + Dropout (0.2) + Dense Output Layer.", color: "bg-violet-500" },
              { title: "Training", desc: "Adam Optimizer, MSE Loss Function, 50 Epochs with Early Stopping.", color: "bg-purple-500" },
              { title: "Validation", desc: "Walk-forward validation on holdout test set to ensure generalizability.", color: "bg-fuchsia-500" }
            ].map((step, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="flex gap-6 items-start p-6 rounded-2xl glass-panel border border-white/5 hover:bg-[#10162A] transition-colors"
              >
                <div className={`w-3 h-3 mt-2 rounded-full ${step.color} shadow-[0_0_10px_currentColor] flex-shrink-0`}></div>
                <div>
                  <h4 className="text-lg font-bold text-white mb-2">{step.title}</h4>
                  <p className="text-slate-400 text-sm leading-relaxed">{step.desc}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Performance Stats */}
      <section id="performance" className="py-24 px-6 relative z-10 bg-indigo-950/20 border-y border-indigo-500/10 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-y-12 gap-x-8 text-center">
          {[
            { label: "Model Accuracy", value: "98.5%", sub: "MAPE < 2%" },
            { label: "Inference Time", value: "45ms", sub: "Per Request" },
            { label: "Data Points", value: "2.4M+", sub: "Historical Records" },
            { label: "Uptime", value: "99.9%", sub: "System Availability" }
          ].map((stat, i) => (
            <motion.div
              key={i}
              initial={{ scale: 0.5, opacity: 0 }}
              whileInView={{ scale: 1, opacity: 1 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1, type: "spring" }}
            >
              <div className="text-3xl md:text-6xl font-black text-white mb-2">{stat.value}</div>
              <div className="text-[10px] md:text-sm font-bold text-indigo-400 uppercase tracking-widest mb-1">{stat.label}</div>
              <div className="text-[10px] text-slate-500">{stat.sub}</div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Future Scope & Objectives */}
      <section id="future" className="py-24 px-6 relative z-10">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <span className="text-indigo-400 font-bold tracking-widest text-xs uppercase mb-3 block">Roadmap</span>
            <h2 className="text-3xl md:text-5xl font-bold mb-4">Future Capabilities</h2>
            <p className="text-slate-400 max-w-2xl mx-auto text-sm md:text-base">Expanding the LoadGrid ecosystem with next-generation tools.</p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[
              {
                icon: <Radio />,
                title: "Real-Time API",
                desc: "High-frequency data streaming for instant grid balancing and automated trading execution."
              },
              {
                icon: <Map />,
                title: "Station Mapping",
                desc: "Geospatial integration for tracking energy station status, output, and maintenance schedules in real-time."
              },
              {
                icon: <Layers />,
                title: "Sector Analytics",
                desc: "Granular consumption heatmaps specifically tailored for Schools, Industries, and Residential zones."
              },
              {
                icon: <Bot />,
                title: "AI Energy Agent",
                desc: "LangChain-powered assistant delivering personalized electricity saving tips and optimization strategies."
              },
              {
                icon: <FileText />,
                title: "Scientific Reporting",
                desc: "Automated deep-dive reports covering power factor, sinusoidal wave analysis, and frequency stability graphs."
              }
            ].map((feature, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                whileHover={{ y: -10, scale: 1.02 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1, duration: 0.4 }}
                className="group relative p-8 rounded-[2rem] glass-panel border border-white/5 hover:border-[#6366F1]/50 transition-colors backdrop-blur-sm overflow-hidden"
              >
                <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/0 via-indigo-500/0 to-cyan-500/0 group-hover:from-indigo-500/5 group-hover:to-cyan-500/10 transition-all duration-500 ease-out" />

                <motion.div
                  className="relative w-14 h-14 bg-indigo-500/10 rounded-2xl flex items-center justify-center text-indigo-400 mb-6 border border-white/5 shadow-inner"
                  whileHover={{ rotateY: 360, backgroundColor: "rgba(99, 102, 241, 0.2)", color: "#ffffff", scale: 1.1 }}
                  transition={{ duration: 0.8, type: "spring", stiffness: 200 }}
                  style={{ perspective: 1000 }}
                >
                  {React.cloneElement(feature.icon, { size: 28 })}
                </motion.div>

                <h3 className="relative text-xl font-bold text-white mb-3 group-hover:text-indigo-400 transition-all">
                  {feature.title}
                </h3>
                <p className="relative text-slate-400 text-sm leading-relaxed group-hover:text-slate-300 transition-colors">
                  {feature.desc}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 bg-[#080B14] border-t border-white/5 py-12 px-6">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-10">
          <div className="flex flex-col items-center md:items-start gap-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center shadow-lg shadow-indigo-500/20">
                <Zap className="text-white fill-current" size={16} />
              </div>
              <span className="font-bold text-lg tracking-tight text-white">LoadIQ</span>
            </div>
            <p className="text-slate-500 text-xs text-center md:text-left leading-relaxed">
              Advancing artificial intelligence in the energy sector.<br />
              Building the future of smart grid analytics.
            </p>
          </div>

          <div className="flex flex-col items-center gap-4">
            <p className="text-slate-400 text-[10px] font-bold uppercase tracking-[0.2em]">Crafted by</p>
            <p className="text-white font-bold tracking-tight">Parth Mhatre</p>
            <div className="flex gap-6 mt-2">
              <a href="https://www.linkedin.com/in/parthmhatre41/" target="_blank" rel="noreferrer" className="text-slate-500 hover:text-indigo-400 transition-colors hover:scale-110"><GlobeIcon size={20} /></a>
              <a href="https://github.com/Parth-S-Mhatre" target="_blank" rel="noreferrer" className="text-slate-500 hover:text-white transition-colors hover:scale-110"><GithubIcon size={20} /></a>
              <a href="http://x.com/ParthMhatre41" target="_blank" rel="noreferrer" className="text-slate-500 hover:text-cyan-400 transition-colors hover:scale-110"><TwitterIcon size={20} /></a>
            </div>
          </div>

          <div className="text-slate-600 text-[10px] font-medium tracking-widest uppercase text-center md:text-right">
            © 2026 LoadIQ • All Rights Reserved
          </div>
        </div>
      </footer>

      {/* Cookies Consent Banner */}
      {showCookies && (
        <motion.div
          initial={{ y: 100, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ type: "spring", stiffness: 300, damping: 30 }}
          className="fixed bottom-6 left-6 right-6 md:left-auto md:right-8 md:max-w-md z-[100]"
        >
          <div className="glass-panel border border-white/10 p-6 rounded-[2rem] shadow-2xl">
            <div className="flex items-start gap-4">
              <div className="w-12 h-12 bg-indigo-500/20 rounded-2xl flex-shrink-0 flex items-center justify-center">
                <Shield className="text-indigo-400" size={24} />
              </div>
              <div className="flex-1">
                <h3 className="text-white font-bold mb-1 text-sm">Cookies & Privacy</h3>
                <p className="text-slate-400 text-[10px] leading-relaxed mb-4">
                  We use cookies to improve your experience and analyze grid traffic. By continuing to use LoadIQ, you agree to our privacy policy.
                </p>
                <div className="flex gap-3">
                  <button
                    onClick={acceptCookies}
                    className="flex-1 px-6 py-2 bg-indigo-600 hover:bg-indigo-700 text-white text-[10px] font-bold rounded-xl transition-all shadow-lg shadow-indigo-500/20 active:scale-95"
                  >
                    Accept All
                  </button>
                  <button
                    onClick={() => setShowCookies(false)}
                    className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-[10px] font-bold rounded-xl transition-all active:scale-95"
                  >
                    Decline
                  </button>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}

// Icons
const GlobeIcon = (props) => (
  <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20" /><path d="M2 12h20" /></svg>
)
const GithubIcon = (props) => (
  <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.403 5.403 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4" /><path d="M9 18c-4.51 2-5-2-7-2" /></svg>
)
const TwitterIcon = (props) => (
  <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 4s-.7 2.1-2 3.4c1.6 10-9.4 17.3-18 11.6 2.2.1 4.4-.6 6-2C3 15.5.5 9.6 3 5c2.2 2.6 5.6 4.1 9 4-.9-4.2 4-6.6 7-3.8 1.1 0 3-1.2 3-1.2z" /></svg>
)
