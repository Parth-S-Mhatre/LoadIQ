import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { signInWithEmailAndPassword, signInWithPopup, signInWithRedirect } from "firebase/auth";
import { auth, googleProvider } from "../firebase";
import { useAuth } from "../context/AuthContext";
import { getFriendlyAuthError } from "../utils/authErrors";
import { motion } from "framer-motion";
import { AlertCircle, ArrowRight, Eye, EyeOff, Lock, Mail } from "lucide-react";
import AuthShell from "../components/AuthShell";
import { EASE, buttonClass } from "../components/ui";

const GoogleMark = () => (
  <svg className="h-4 w-4" viewBox="0 0 24 24" aria-hidden="true">
    <path fill="currentColor" d="M21.35 11.1h-9.17v2.73h6.51c-.33 3.82-3.5 5.44-6.5 5.44C8.36 19.27 5 16.25 5 12c0-4.1 3.2-7.27 7.2-7.27c3.09 0 4.9 1.97 4.9 1.97L19 4.72S16.56 2 12.1 2C6.42 2 2.03 6.8 2.03 12c0 5.05 4.13 10 10.22 10c5.35 0 9.25-3.67 9.25-9.09c0-1.15-.15-1.81-.15-1.81Z" />
  </svg>
);

export default function Login() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
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

  return (
    <AuthShell
      eyebrow="Operator access"
      title={
        <>
          Welcome back to
          <span className="display-serif italic text-moss"> the console.</span>
        </>
      }
      description="Sign in to open your workspace, resume forecast history and pick up the grid where you left it."
      image="https://images.unsplash.com/photo-1473341304170-971dccb5ac1e?auto=format&fit=crop&w=1400&q=80"
      imageAlt="High-voltage transmission towers against a sunset"
      imageCaption="Grid operations · live context"
      quote="The forecast is only useful if the operator can act on it before the peak."
    >
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35, ease: EASE }}
          role="alert"
          className="mb-5 flex items-start gap-2.5 rounded-xl border border-clay/25 bg-clay/[0.06] px-3.5 py-3 text-[0.78rem] leading-relaxed text-clay"
        >
          <AlertCircle size={15} className="mt-0.5 shrink-0" />
          {error}
        </motion.div>
      )}

      <form onSubmit={handleLogin} className="space-y-4">
        <div>
          <label htmlFor="login-email" className="label-sm mb-2 block text-ink/45">Email address</label>
          <div className="relative">
            <Mail size={16} className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-ink/30" />
            <input
              id="login-email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="ui-input pl-10"
              placeholder="operator@loadiq.ai"
              autoComplete="email"
            />
          </div>
        </div>

        <div>
          <div className="mb-2 flex items-center justify-between">
            <label htmlFor="login-password" className="label-sm text-ink/45">Password</label>
            <span className="text-[0.65rem] text-ink/30">Minimum 6 characters</span>
          </div>
          <div className="relative">
            <Lock size={16} className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-ink/30" />
            <input
              id="login-password"
              type={showPassword ? "text" : "password"}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="ui-input pl-10 pr-11"
              placeholder="Enter your password"
              autoComplete="current-password"
            />
            <button
              type="button"
              onClick={() => setShowPassword((visible) => !visible)}
              aria-label={showPassword ? "Hide password" : "Show password"}
              className="absolute right-2 top-1/2 flex h-8 w-8 -translate-y-1/2 items-center justify-center rounded-lg text-ink/35 transition-colors hover:bg-ink/[0.05] hover:text-ink/70"
            >
              {showPassword ? <EyeOff size={15} /> : <Eye size={15} />}
            </button>
          </div>
        </div>

        <button type="submit" disabled={isLoading} className={buttonClass("primary", "lg", "mt-2 w-full")}>
          {isLoading ? (
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-paper/25 border-t-paper" />
          ) : (
            <>
              Sign in to workspace
              <ArrowRight size={16} className="transition-transform duration-300 group-hover:translate-x-0.5" />
            </>
          )}
        </button>
      </form>

      <div className="my-6 flex items-center gap-3">
        <div className="h-px flex-1 bg-ink/10" />
        <span className="label-sm text-ink/30">or continue with</span>
        <div className="h-px flex-1 bg-ink/10" />
      </div>

      <button
        type="button"
        onClick={handleGoogleSignIn}
        disabled={isLoading}
        className={buttonClass("outline", "lg", "w-full bg-paper-50")}
      >
        <GoogleMark />
        Continue with Google
      </button>

      <p className="mt-7 text-center text-[0.8rem] text-ink/45">
        Don&apos;t have an account?{" "}
        <button
          onClick={() => navigate(`/register${location.search}`)}
          className="font-semibold text-ink underline decoration-signal-400 decoration-2 underline-offset-4 transition-colors hover:text-moss"
        >
          Create one
        </button>
      </p>
    </AuthShell>
  );
}
