import React, { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { createUserWithEmailAndPassword, updateProfile, signInWithPopup, signInWithRedirect } from "firebase/auth";
import { auth, googleProvider } from "../firebase";
import { useAuth } from "../context/AuthContext";
import { getFriendlyAuthError } from "../utils/authErrors";
import { motion } from "framer-motion";
import { AlertCircle, ArrowRight, Eye, EyeOff, Lock, Mail, UserRound, Users } from "lucide-react";
import AuthShell from "../components/AuthShell";
import { EASE, buttonClass } from "../components/ui";

const GoogleMark = () => (
  <svg className="h-4 w-4" viewBox="0 0 24 24" aria-hidden="true">
    <path fill="currentColor" d="M21.35 11.1h-9.17v2.73h6.51c-.33 3.82-3.5 5.44-6.5 5.44C8.36 19.27 5 16.25 5 12c0-4.1 3.2-7.27 7.2-7.27c3.09 0 4.9 1.97 4.9 1.97L19 4.72S16.56 2 12.1 2C6.42 2 2.03 6.8 2.03 12c0 5.05 4.13 10 10.22 10c5.35 0 9.25-3.67 9.25-9.09c0-1.15-.15-1.81-.15-1.81Z" />
  </svg>
);

export default function Register() {
  const [formData, setFormData] = useState({ fullName: "", email: "", password: "", confirmPassword: "" });
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const { user, isNewUser } = useAuth();
  const inviteParams = new URLSearchParams(location.search);
  const inviteWorkspace = inviteParams.get("workspace") || "";
  const inviteOrganization = inviteParams.get("organization") || "";
  const inviteSender = inviteParams.get("invitedBy") || "";

  useEffect(() => {
    if (user) {
      navigate(isNewUser ? "/profile" : "/dashboard");
    }
  }, [user, isNewUser, navigate]);

  useEffect(() => {
    if (!inviteWorkspace && !inviteOrganization && !inviteSender) {
      return;
    }

    localStorage.setItem("pendingInviteContext", JSON.stringify({
      workspaceName: inviteWorkspace,
      organization: inviteOrganization,
      invitedBy: inviteSender
    }));
  }, [inviteWorkspace, inviteOrganization, inviteSender]);

  const handleRegister = async (e) => {
    e.preventDefault();
    if (!formData.email || !formData.password || !formData.fullName) {
      setError("All fields are required.");
      return;
    }
    if (formData.password !== formData.confirmPassword) {
      setError("Passwords do not match.");
      return;
    }
    setError("");
    setIsLoading(true);
    try {
      const cred = await createUserWithEmailAndPassword(auth, formData.email, formData.password);
      await updateProfile(cred.user, { displayName: formData.fullName });
    } catch (err) {
      setError("Registration failed. Please try again.");
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

  const updateField = (field, value) => setFormData((prev) => ({ ...prev, [field]: value }));

  return (
    <AuthShell
      eyebrow={inviteWorkspace ? "Workspace invitation" : "New operator"}
      title={
        <>
          Create your
          <span className="display-serif italic text-moss"> operator workspace.</span>
        </>
      }
      description="One account for forecasts, model history and shared grid context. Setup takes under a minute."
      image="https://images.unsplash.com/photo-1508514177221-188b1cf16e9d?auto=format&fit=crop&w=1400&q=80"
      imageAlt="Aerial view of rows of solar panels"
      imageCaption={inviteWorkspace ? `Invited to ${inviteWorkspace}` : "Workspace onboarding"}
      quote="Every megawatt saved starts with a better question."
    >
      {(inviteWorkspace || inviteOrganization || inviteSender) && (
        <motion.div
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, ease: EASE }}
          className="mb-5 rounded-xl border border-moss/20 bg-moss/[0.06] p-3.5"
        >
          <div className="flex items-center gap-2 text-xs font-semibold text-moss">
            <Users size={14} />
            Workspace invite detected
          </div>
          <p className="mt-1.5 text-xs leading-relaxed text-ink/55">
            Join {inviteWorkspace || inviteOrganization || "a LoadIQ workspace"}
            {inviteSender ? `, shared by ${inviteSender}` : ""}. Your workspace fields will be pre-filled.
          </p>
        </motion.div>
      )}

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

      <form onSubmit={handleRegister} className="space-y-4">
        <div>
          <label htmlFor="register-name" className="label-sm mb-2 block text-ink/45">Full name</label>
          <div className="relative">
            <UserRound size={16} className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-ink/30" />
            <input
              id="register-name"
              type="text"
              value={formData.fullName}
              onChange={(e) => updateField("fullName", e.target.value)}
              className="ui-input pl-10"
              placeholder="Parth Mhatre"
              autoComplete="name"
            />
          </div>
        </div>

        <div>
          <label htmlFor="register-email" className="label-sm mb-2 block text-ink/45">Email address</label>
          <div className="relative">
            <Mail size={16} className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-ink/30" />
            <input
              id="register-email"
              type="email"
              value={formData.email}
              onChange={(e) => updateField("email", e.target.value)}
              className="ui-input pl-10"
              placeholder="operator@loadiq.ai"
              autoComplete="email"
            />
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div>
            <label htmlFor="register-password" className="label-sm mb-2 block text-ink/45">Password</label>
            <div className="relative">
              <Lock size={15} className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-ink/30" />
              <input
                id="register-password"
                type={showPassword ? "text" : "password"}
                value={formData.password}
                onChange={(e) => updateField("password", e.target.value)}
                className="ui-input pl-10 pr-10 text-sm"
                placeholder="Minimum 6 characters"
                autoComplete="new-password"
              />
              <button
                type="button"
                onClick={() => setShowPassword((visible) => !visible)}
                aria-label={showPassword ? "Hide password" : "Show password"}
                className="absolute right-1.5 top-1/2 flex h-8 w-8 -translate-y-1/2 items-center justify-center rounded-lg text-ink/35 transition-colors hover:bg-ink/[0.05] hover:text-ink/70"
              >
                {showPassword ? <EyeOff size={14} /> : <Eye size={14} />}
              </button>
            </div>
          </div>

          <div>
            <label htmlFor="register-confirm" className="label-sm mb-2 block text-ink/45">Confirm</label>
            <div className="relative">
              <Lock size={15} className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-ink/30" />
              <input
                id="register-confirm"
                type={showPassword ? "text" : "password"}
                value={formData.confirmPassword}
                onChange={(e) => updateField("confirmPassword", e.target.value)}
                className="ui-input pl-10 text-sm"
                placeholder="Repeat password"
                autoComplete="new-password"
              />
            </div>
          </div>
        </div>

        <button type="submit" disabled={isLoading} className={buttonClass("primary", "lg", "mt-2 w-full")}>
          {isLoading ? (
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-paper/25 border-t-paper" />
          ) : (
            <>
              Create account
              <ArrowRight size={16} className="transition-transform duration-300 group-hover:translate-x-0.5" />
            </>
          )}
        </button>

        <p className="text-center text-[0.68rem] leading-relaxed text-ink/35">
          By continuing you agree to responsible use of forecast data. Predictions are a reference, not a control signal.
        </p>
      </form>

      <div className="my-5 flex items-center gap-3">
        <div className="h-px flex-1 bg-ink/10" />
        <span className="label-sm text-ink/30">or sign up with</span>
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

      <p className="mt-6 text-center text-[0.8rem] text-ink/45">
        Already registered?{" "}
        <button
          onClick={() => navigate(`/login${location.search}`)}
          className="font-semibold text-ink underline decoration-signal-400 decoration-2 underline-offset-4 transition-colors hover:text-moss"
        >
          Sign in
        </button>
      </p>
    </AuthShell>
  );
}
