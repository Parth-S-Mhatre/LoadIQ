import React, { lazy } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import { DisclaimerProvider } from "./context/DisclaimerContext";
import ProtectedRoute from "./components/ProtectedRoute";
import RouteSkeletonGate from "./components/RouteSkeletonGate";
import LandingSkeleton from "./skeleton_pages/LandingSkeleton";
import AuthSkeleton from "./skeleton_pages/AuthSkeleton";
import DashboardSkeleton from "./skeleton_pages/DashboardSkeleton";
import EnergyChatbot from './components/EnergyChatbot';
import ProfileSkeleton from "./skeleton_pages/ProfileSkeleton";

const Landing = lazy(() => import("./pages/Landing"));
const Login = lazy(() => import("./pages/Login"));
const Register = lazy(() => import("./pages/Register"));
const Dashboard = lazy(() => import("./pages/Dashboard"));
const UserProfile = lazy(() => import("./pages/UserProfile"));
const NotFound = lazy(() => import("./pages/NotFound"));

function App() {
  return (
    <AuthProvider>
      <DisclaimerProvider>
        <Router>
          <Routes>
            <Route
              path="/"
              element={<RouteSkeletonGate component={Landing} skeleton={LandingSkeleton} />}
            />
            <Route
              path="/login"
              element={<RouteSkeletonGate component={Login} skeleton={AuthSkeleton} skeletonProps={{ mode: "login" }} />}
            />
            <Route
              path="/register"
              element={<RouteSkeletonGate component={Register} skeleton={AuthSkeleton} skeletonProps={{ mode: "register" }} />}
            />

          {/* PROTECTED ROUTES */}
          <Route
            path="/dashboard/*"
            element={
              <ProtectedRoute>
                <RouteSkeletonGate component={Dashboard} skeleton={DashboardSkeleton} />
              </ProtectedRoute>
            }
          />
          <Route
            path="/profile"
            element={
              <ProtectedRoute>
                <RouteSkeletonGate component={UserProfile} skeleton={ProfileSkeleton} />
              </ProtectedRoute>
            }
          />

          {/* 404 CATCH-ALL */}
          <Route
            path="*"
            element={<RouteSkeletonGate component={NotFound} skeleton={LandingSkeleton} />}
          />
        </Routes>
        <EnergyChatbot />
      </Router>
      </DisclaimerProvider>
    </AuthProvider>
  );
}

export default App;
