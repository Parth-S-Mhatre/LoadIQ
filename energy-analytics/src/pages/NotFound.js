import React from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowLeft, ArrowUpRight, Compass, Home } from "lucide-react";
import { EASE, Logo, buttonClass } from "../components/ui";

const NotFound = () => {
    const navigate = useNavigate();

    return (
        <div className="relative flex min-h-screen flex-col overflow-hidden bg-paper text-ink">
            <div className="pointer-events-none absolute inset-0 grid-lines opacity-60 [mask-image:linear-gradient(to_bottom,black,transparent_75%)]" />
            <div className="pointer-events-none absolute -right-32 top-24 h-[28rem] w-[28rem] rounded-full bg-signal/25 blur-[110px]" />

            <header className="relative z-10 border-b border-ink/10">
                <div className="mx-auto flex h-[4.5rem] max-w-7xl items-center justify-between px-5 sm:px-8">
                    <button onClick={() => navigate("/")} aria-label="LoadIQ home">
                        <Logo />
                    </button>
                    <span className="label-sm text-ink/35">Error 404</span>
                </div>
            </header>

            <main className="relative z-10 flex flex-1 items-center">
                <div className="mx-auto w-full max-w-7xl px-5 py-16 sm:px-8">
                    <div className="grid items-center gap-12 lg:grid-cols-12">
                        <motion.div
                            initial={{ opacity: 0, y: 24 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.8, ease: EASE }}
                            className="lg:col-span-7"
                        >
                            <div className="mb-6 flex items-center gap-3">
                                <span className="h-1.5 w-1.5 rounded-full bg-clay" />
                                <span className="label text-ink/45">Route not found</span>
                            </div>

                            <h1 className="text-[3.5rem] leading-[0.92] tracking-[-0.055em] sm:text-[5rem] lg:text-[6rem]">
                                Off the
                                <span className="display-serif italic text-moss"> grid.</span>
                            </h1>

                            <p className="mt-7 max-w-lg text-[0.95rem] leading-[1.75] text-ink/55">
                                The page you requested isn&apos;t part of this workspace. It may have moved,
                                or the link may be incomplete. Head back to the console and pick up where
                                you left off.
                            </p>

                            <div className="mt-9 flex flex-col gap-3 sm:flex-row">
                                <button onClick={() => navigate("/dashboard")} className={buttonClass("primary", "lg")}>
                                    <Home size={16} />
                                    Return to console
                                </button>
                                <button onClick={() => navigate("/")} className={buttonClass("outline", "lg")}>
                                    <ArrowLeft size={16} />
                                    Back to home
                                </button>
                            </div>

                            <div className="mt-12 grid max-w-lg grid-cols-2 gap-px overflow-hidden rounded-xl border border-ink/10 bg-ink/10 sm:grid-cols-3">
                                {[
                                    ["Console", "/dashboard"],
                                    ["Predictor", "/dashboard/predictor"],
                                    ["Profile", "/profile"],
                                ].map(([label, path]) => (
                                    <button
                                        key={label}
                                        onClick={() => navigate(path)}
                                        className="group flex items-center justify-between gap-2 bg-paper-50 px-4 py-3.5 text-left text-[0.72rem] font-medium text-ink/60 transition-colors hover:bg-white hover:text-ink"
                                    >
                                        {label}
                                        <ArrowUpRight size={13} className="text-ink/25 transition-transform duration-300 group-hover:-translate-y-0.5" />
                                    </button>
                                ))}
                            </div>
                        </motion.div>

                        <motion.div
                            initial={{ opacity: 0, scale: 0.96 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ duration: 0.9, delay: 0.12, ease: EASE }}
                            className="lg:col-span-5"
                        >
                            <div className="relative overflow-hidden rounded-[1.5rem] border border-ink/15 bg-ink p-8 text-paper sm:p-10">
                                <div className="absolute inset-0 dot-grid-light opacity-30" />
                                <div className="relative">
                                    <div className="flex items-center justify-between">
                                        <span className="flex h-11 w-11 items-center justify-center rounded-xl border border-white/10 bg-white/[0.05] text-signal">
                                            <Compass size={20} />
                                        </span>
                                        <span className="mono-num text-[3.5rem] font-medium leading-none tracking-[-0.07em] text-paper/15">
                                            404
                                        </span>
                                    </div>

                                    <div className="mt-10 space-y-0 border-t border-white/10">
                                        {[
                                            ["Signal", "Lost"],
                                            ["Last known route", "Unknown"],
                                            ["Suggested action", "Return to console"],
                                        ].map(([key, value], index) => (
                                            <div
                                                key={key}
                                                className={`flex items-baseline justify-between gap-4 py-3.5 ${index < 2 ? "border-b border-white/10" : ""}`}
                                            >
                                                <span className="label-sm text-paper/35">{key}</span>
                                                <span className="text-right text-[0.72rem] font-medium text-paper/70">{value}</span>
                                            </div>
                                        ))}
                                    </div>

                                    <div className="mt-8 flex items-center gap-2 border-t border-white/10 pt-4">
                                        <span className="h-1.5 w-1.5 animate-blink rounded-full bg-signal" />
                                        <span className="label-sm text-paper/40">Grid still operational</span>
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    </div>
                </div>
            </main>

            <footer className="relative z-10 border-t border-ink/10 px-5 py-5 sm:px-8">
                <div className="mx-auto flex max-w-7xl flex-col justify-between gap-2 text-[0.62rem] text-ink/35 sm:flex-row sm:items-center">
                    <span>© 2026 LoadIQ · Grid intelligence platform</span>
                    <span className="label-sm">UK · USA · Germany · India</span>
                </div>
            </footer>
        </div>
    );
};

export default NotFound;
