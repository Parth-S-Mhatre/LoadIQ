import React from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { Home, AlertCircle } from "lucide-react";
import ThreeBackground from "../components/3d/ThreeBackground";

const NotFound = () => {
    const navigate = useNavigate();

    return (
        <div className="min-h-screen relative flex items-center justify-center p-6 overflow-hidden bg-[#080B14] text-white">
            <ThreeBackground />

            <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="relative z-10 text-center space-y-8 max-w-md"
            >
                <div className="w-24 h-24 bg-red-500/20 rounded-3xl flex items-center justify-center mx-auto mb-4 border border-red-500/30 shadow-2xl shadow-red-500/20">
                    <AlertCircle size={48} className="text-red-500" />
                </div>

                <div className="space-y-2">
                    <h1 className="text-6xl font-black italic tracking-tighter">404</h1>
                    <h2 className="text-2xl font-bold uppercase tracking-widest text-slate-400">Page Lost in Grid</h2>
                    <p className="text-slate-500 text-sm leading-relaxed">
                        The load parameters for this route are invalid. The page you are looking for does not exist or has been relocated.
                    </p>
                </div>

                <button
                    onClick={() => navigate("/")}
                    className="flex items-center gap-2 px-8 py-4 bg-indigo-600 hover:bg-indigo-700 text-white font-bold rounded-full transition-all shadow-xl shadow-indigo-600/30 mx-auto group"
                >
                    <Home size={18} className="transition-transform group-hover:-translate-y-0.5" />
                    Back to Home
                </button>
            </motion.div>

            {/* Grid Overlay */}
            <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 pointer-events-none"></div>
        </div>
    );
};

export default NotFound;
