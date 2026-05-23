import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { signInWithEmailAndPassword, signInWithPopup, signInWithRedirect } from "firebase/auth";
import { auth, googleProvider } from "../firebase";
import { useAuth } from "../context/AuthContext";
import { getFriendlyAuthError } from "../utils/authErrors";
import { motion, useMotionValue, useTransform } from "framer-motion";
import { Zap, Home, Mail, Lock, ArrowRight } from "lucide-react";

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const { user, isNewUser } = useAuth();

  useEffect(() => {
    if (user) {
      navigate(isNewUser ? "/profile" : "/dashboard");
    }
  }, [user, isNewUser, navigate]);

  useEffect(() => {
    const inviteParams = new URLSearchParams(location.search);
    const workspaceName = inviteParams.get("workspace") || "";
    const organization = inviteParams.get("organization") || "";
    const invitedBy = inviteParams.get("invitedBy") || "";

    if (!workspaceName && !organization && !invitedBy) {
      return;
    }

    localStorage.setItem("pendingInviteContext", JSON.stringify({
      workspaceName,
      organization,
      invitedBy
    }));
  }, [location.search]);

  const handleLogin = async (e) => {
    e.preventDefault();
    if (!email || !password) {
      setError("Please fill in all fields.");
      return;
    }
    setError("");
    setIsLoading(true);
    try {
      await signInWithEmailAndPassword(auth, email, password);
      // Navigation will be handled by useEffect
    } catch (err) {
      setError("Invalid credentials. Please attempt again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleGoogleSignIn = async () => {
    setError("");
    setIsLoading(true);
    try {
      await signInWithPopup(auth, googleProvider);
      // Navigation will be handled by useEffect
    } catch (err) {
      if (err?.code === "auth/popup-blocked") {
        setError(getFriendlyAuthError(err));
        await signInWithRedirect(auth, googleProvider);
        return;
      }
      setError(getFriendlyAuthError(err));
    } finally {
      setIsLoading(false);
    }
  };

  // 3D Tilt Effect
  const x = useMotionValue(0);
  const y = useMotionValue(0);

  const rotateX = useTransform(y, [-300, 300], [10, -10]);
  const rotateY = useTransform(x, [-300, 300], [-10, 10]);

  const handleMouseMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    x.set(e.clientX - rect.left - rect.width / 2);
    y.set(e.clientY - rect.top - rect.height / 2);
  };

  const handleMouseLeave = () => {
    x.set(0);
    y.set(0);
  };

  return (
    <div className="min-h-screen bg-[#080B14] flex flex-col items-center justify-center p-4 sm:p-6 pt-24 sm:pt-6 relative overflow-x-hidden overflow-y-auto font-sans">
      {/* Background Orbs */}
      <div className="absolute inset-0 z-0 overflow-hidden pointer-events-none">
        <motion.div
          animate={{ scale: [1, 1.1, 1], opacity: [0.15, 0.25, 0.15] }}
          transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
          className="absolute top-1/4 -left-1/2 sm:-left-1/4 w-[420px] h-[420px] sm:w-[800px] sm:h-[800px] bg-indigo-600/30 rounded-full blur-[90px] sm:blur-[120px]"
        />
        <motion.div
          animate={{ scale: [1, 1.2, 1], opacity: [0.1, 0.2, 0.1] }}
          transition={{ duration: 10, repeat: Infinity, ease: "easeInOut", delay: 1 }}
          className="absolute bottom-1/4 -right-1/2 sm:-right-1/4 w-[360px] h-[360px] sm:w-[600px] sm:h-[600px] bg-purple-600/20 rounded-full blur-[90px] sm:blur-[120px]"
        />
        <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-10 mix-blend-overlay"></div>
      </div>

      <motion.button
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        onClick={() => navigate("/")}
        className="absolute top-4 left-4 sm:top-8 sm:left-8 flex items-center gap-3 text-xs font-bold uppercase tracking-widest text-slate-400 hover:text-white transition-all group z-10"
      >
        <div className="w-10 h-10 rounded-2xl bg-[#10162A] border border-white/5 flex items-center justify-center group-hover:bg-[#6366F1] group-hover:border-[#6366F1] transition-all shadow-lg group-hover:shadow-[0_0_20px_rgba(99,102,241,0.4)]">
          <Home size={16} />
        </div>
        <span className="hidden sm:inline">Return Home</span>
      </motion.button>

      <div
        className="relative z-10 w-full max-w-[420px] perspective-[1000px]"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      >
        <motion.div
          style={{ rotateX, rotateY, transformStyle: "preserve-3d" }}
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ type: "spring", bounce: 0.4, duration: 1 }}
          className="glass-panel rounded-[2rem] overflow-hidden"
        >
          {/* Inner 3D Content Wrapper */}
          <div style={{ transform: "translateZ(30px)" }} className="p-6 sm:p-10">
            <div className="text-center mb-8">
              <motion.div
                whileHover={{ rotate: 180 }}
                transition={{ duration: 0.5 }}
                className="w-16 h-16 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-[0_0_30px_rgba(99,102,241,0.5)] transform -rotate-12"
              >
                <Zap size={32} className="text-white fill-white" />
              </motion.div>
              <h1 className="text-3xl font-black text-white tracking-tight mb-2">Welcome Back</h1>
              <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">Sign in to LoadGrid Terminal</p>
            </div>

            {error && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-6 p-4 bg-red-500/10 border border-red-500/20 rounded-xl text-xs font-bold text-red-400 text-center"
              >
                {error}
              </motion.div>
            )}

            <form onSubmit={handleLogin} className="space-y-5">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="space-y-2"
              >
                <label className="text-[10px] font-bold text-slate-400 uppercase tracking-widest ml-1">Email Address</label>
                <div className="relative group">
                  <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                    <Mail size={16} className="text-slate-500 group-focus-within:text-indigo-400 transition-colors duration-300" />
                  </div>
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="w-full h-12 glass-input rounded-xl pl-11 pr-4 text-sm focus:bg-[rgba(15,23,42,0.8)] transition-all duration-300 hover:border-indigo-500/30"
                    placeholder="operator@loadgrid.ai"
                  />
                </div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="space-y-2"
              >
                <label className="text-[10px] font-bold text-slate-400 uppercase tracking-widest ml-1">Password</label>
                <div className="relative group">
                  <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                    <Lock size={16} className="text-slate-500 group-focus-within:text-indigo-400 transition-colors duration-300" />
                  </div>
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="w-full h-12 glass-input rounded-xl pl-11 pr-4 text-sm focus:bg-[rgba(15,23,42,0.8)] transition-all duration-300 hover:border-indigo-500/30"
                    placeholder="••••••••••••"
                  />
                </div>
              </motion.div>

              <motion.button
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                type="submit"
                disabled={isLoading}
                style={{ transform: "translateZ(20px)" }}
                className="w-full h-12 mt-4 bg-[#6366F1] hover:bg-[#5356E1] text-white rounded-xl text-xs font-bold uppercase tracking-widest transition-all duration-300 shadow-[0_4px_14px_0_rgba(99,102,241,0.39)] hover:shadow-[0_6px_20px_rgba(99,102,241,0.23)] hover:-translate-y-0.5 flex items-center justify-center gap-2 group relative overflow-hidden active:scale-95"
              >
                {isLoading ? (
                  <div className="w-5 h-5 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                ) : (
                  <>
                    Sign In
                    <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform duration-300" />
                  </>
                )}
              </motion.button>
            </form>

            <div className="my-6 flex items-center gap-4">
              <div className="h-px bg-white/10 flex-1"></div>
              <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">or</span>
              <div className="h-px bg-white/10 flex-1"></div>
            </div>

            <motion.button
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              onClick={handleGoogleSignIn}
              disabled={isLoading}
              style={{ transform: "translateZ(15px)" }}
              className="w-full h-12 bg-[#10162A] hover:bg-[#151B30] border border-white/5 text-white rounded-xl text-xs font-bold uppercase tracking-widest transition-all duration-300 flex items-center justify-center gap-3 hover:-translate-y-0.5 active:scale-95 disabled:opacity-60"
            >
              <svg className="w-4 h-4" viewBox="0 0 24 24">
                <path fill="currentColor" d="M21.35 11.1h-9.17v2.73h6.51c-.33 3.82-3.5 5.44-6.5 5.44C8.36 19.27 5 16.25 5 12c0-4.1 3.2-7.27 7.2-7.27c3.09 0 4.9 1.97 4.9 1.97L19 4.72S16.56 2 12.1 2C6.42 2 2.03 6.8 2.03 12c0 5.05 4.13 10 10.22 10c5.35 0 9.25-3.67 9.25-9.09c0-1.15-.15-1.81-.15-1.81Z" />
              </svg>
              Google
            </motion.button>

            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
              className="text-center text-xs font-semibold text-slate-400 mt-8"
              style={{ transform: "translateZ(10px)" }}
            >
              Don't have an account?{" "}
              <motion.button
                onClick={() => navigate(`/register${location.search}`)}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="text-indigo-400 hover:text-indigo-300 transition-colors duration-300 font-bold"
              >
                Register
              </motion.button>
            </motion.p>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
