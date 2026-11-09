import React from "react";
import { Navigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { Zap } from "lucide-react";

const ProtectedRoute = ({ children }) => {
    const { user, loading } = useAuth();

    if (loading) {
        return (
            <div className="flex min-h-screen flex-col items-center justify-center bg-paper text-ink">
                <div className="mb-4 flex h-14 w-14 animate-pulse items-center justify-center rounded-xl border border-ink/10 bg-ink text-signal">
                    <Zap size={26} />
                </div>
                <p className="label text-ink/40">Authenticating session</p>
            </div>
        );
    }

    if (!user) {
        return <Navigate to="/login" replace />;
    }

    return children;
};

export default ProtectedRoute;
