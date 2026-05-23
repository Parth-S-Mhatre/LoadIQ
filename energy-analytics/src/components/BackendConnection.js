import React, { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle, AlertTriangle, Loader, WifiOff } from 'lucide-react';
import { API_CONFIG, buildUrl, fetchWithTimeout } from '../config/api';

const BackendConnection = ({ onConnectionChange }) => {
  const [connectionStatus, setConnectionStatus] = useState('connecting'); // 'connecting', 'connected', 'timeout', 'error'
  const [isVisible, setIsVisible] = useState(true);
  const [retryCount, setRetryCount] = useState(0);
  const retryCountRef = useRef(0);

  const checkConnection = useCallback(async () => {
    try {
      setConnectionStatus('connecting');
      const healthEndpoints = [
        API_CONFIG.ENDPOINTS.HEALTH,
        API_CONFIG.ENDPOINTS.HEALTH_CHECK
      ];
      let lastError = null;

      for (const endpoint of healthEndpoints) {
        try {
          const response = await fetchWithTimeout(
            buildUrl(endpoint),
            {},
            API_CONFIG.TIMEOUTS.HEALTH_CHECK
          );

          if (response.ok) {
            setConnectionStatus('connected');
            onConnectionChange?.(true);
            // Hide after successful connection
            setTimeout(() => setIsVisible(false), 3000);
            return;
          }

          lastError = new Error(`Backend responded with ${response.status} at ${endpoint}`);
        } catch (error) {
          lastError = error;
        }
      }

      throw lastError || new Error('Backend responded with error');
    } catch (error) {
      const nextRetryCount = retryCountRef.current + 1;
      retryCountRef.current = nextRetryCount;
      setRetryCount(nextRetryCount);

      if (error.message === 'Request timeout') {
        setConnectionStatus('timeout');
      } else {
        setConnectionStatus('error');
      }
      onConnectionChange?.(false);

      if (error.message === 'Request timeout' && nextRetryCount < 3) {
        setTimeout(checkConnection, 5000);
      }
    }
  }, [onConnectionChange]);

  useEffect(() => {
    checkConnection();
  }, [checkConnection]);

  const getStatusConfig = () => {
    switch (connectionStatus) {
      case 'connecting':
        return {
          icon: Loader,
          color: 'text-blue-500',
          bgColor: 'bg-blue-500/10',
          borderColor: 'border-blue-500/30',
          title: 'Connecting to Backend',
          message: 'Establishing connection to LoadIQ server...',
          animation: 'animate-spin'
        };
      case 'connected':
        return {
          icon: CheckCircle,
          color: 'text-green-500',
          bgColor: 'bg-green-500/10',
          borderColor: 'border-green-500/30',
          title: 'Backend Connected',
          message: 'Successfully connected to LoadIQ analytics server!',
          animation: ''
        };
      case 'timeout':
        return {
          icon: AlertTriangle,
          color: 'text-amber-500',
          bgColor: 'bg-amber-500/10',
          borderColor: 'border-amber-500/30',
          title: 'Connection Timeout',
          message: 'Server is waking up. Please try again later.',
          animation: ''
        };
      case 'error':
        return {
          icon: WifiOff,
          color: 'text-red-500',
          bgColor: 'bg-red-500/10',
          borderColor: 'border-red-500/30',
          title: 'Connection Failed',
          message: 'Unable to connect to backend server.',
          animation: ''
        };
      default:
        return {};
    }
  };

  const statusConfig = getStatusConfig();

  if (!isVisible) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, scale: 0.9, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        exit={{ opacity: 0, scale: 0.9, y: -20 }}
        className={`fixed top-4 left-4 right-4 sm:left-auto z-50 sm:max-w-sm ${statusConfig.bgColor} backdrop-blur-xl border ${statusConfig.borderColor} rounded-2xl p-4 sm:p-6 shadow-2xl`}
      >
        <div className="flex items-start gap-4">
          <motion.div
            animate={connectionStatus === 'connecting' ? { rotate: 360 } : {}}
            transition={{
              duration: 1,
              repeat: connectionStatus === 'connecting' ? Infinity : 0,
              ease: "linear"
            }}
            className={`flex-shrink-0 w-12 h-12 rounded-xl bg-slate-800/50 flex items-center justify-center ${statusConfig.color}`}
          >
            <statusConfig.icon size={24} className={statusConfig.animation} />
          </motion.div>

          <div className="flex-1 min-w-0">
            <h3 className={`font-bold text-lg ${statusConfig.color} mb-2`}>
              {statusConfig.title}
            </h3>
            <p className="text-slate-300 text-sm leading-relaxed mb-4">
              {statusConfig.message}
            </p>

            {connectionStatus === 'connecting' && (
              <div className="flex items-center gap-2">
                <div className="flex gap-1">
                  {[0, 1, 2].map((i) => (
                    <motion.div
                      key={i}
                      className="w-2 h-2 bg-blue-500 rounded-full"
                      animate={{
                        scale: [1, 1.5, 1],
                        opacity: [0.5, 1, 0.5]
                      }}
                      transition={{
                        duration: 1.5,
                        repeat: Infinity,
                        delay: i * 0.2
                      }}
                    />
                  ))}
                </div>
                <span className="text-xs text-slate-400">Connecting...</span>
              </div>
            )}

            {connectionStatus === 'timeout' && retryCount < 3 && (
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-amber-500/30 border-t-amber-500 rounded-full animate-spin" />
                <span className="text-xs text-slate-400">Retrying in 5 seconds...</span>
              </div>
            )}

            {connectionStatus === 'connected' && (
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.5, type: "spring", damping: 15 }}
                className="flex items-center gap-2 text-green-400"
              >
                <CheckCircle size={16} />
                <span className="text-xs font-medium">Ready for analytics</span>
              </motion.div>
            )}
          </div>

          <button
            onClick={() => setIsVisible(false)}
            className="flex-shrink-0 w-8 h-8 rounded-lg bg-slate-700/50 hover:bg-slate-600/50 flex items-center justify-center text-slate-400 hover:text-white transition-all"
          >
            ×
          </button>
        </div>

        {/* Progress bar for connecting state */}
        {connectionStatus === 'connecting' && (
          <motion.div
            className="mt-4 h-1 bg-slate-700/50 rounded-full overflow-hidden"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <motion.div
              className="h-full bg-gradient-to-r from-blue-500 to-indigo-500 rounded-full"
              initial={{ width: "0%" }}
              animate={{ width: "100%" }}
              transition={{ duration: 3, ease: "easeInOut" }}
            />
          </motion.div>
        )}
      </motion.div>
    </AnimatePresence>
  );
};

export default BackendConnection;
