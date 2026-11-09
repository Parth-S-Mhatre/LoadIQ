import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { CheckCircle, AlertTriangle, Loader, WifiOff } from 'lucide-react';
import useBackendHealth from '../hooks/useBackendHealth';
import { EASE, StatusDot } from './ui';

const BackendConnection = ({ onConnectionChange }) => {
  const [isVisible, setIsVisible] = useState(true);
  const { isOnline, isChecking, error, backendSource, backendLabel, activeBackendUrl, modelLoaded } = useBackendHealth();

  useEffect(() => {
    onConnectionChange?.(isOnline);

    if (isOnline) {
      const timer = setTimeout(() => setIsVisible(false), 3000);
      return () => clearTimeout(timer);
    }

    return undefined;
  }, [isOnline, onConnectionChange]);

  const getStatusConfig = () => {
    if (isChecking) {
      return {
        icon: Loader,
        tone: 'sky',
        accent: 'text-sky',
        chip: 'bg-sky/10',
        title: 'Connecting to backend',
        message: 'Checking the Render primary first, then the Railway fallback if needed.',
        animation: 'animate-spin'
      };
    }

    if (isOnline && backendSource === 'railway') {
      return {
        icon: AlertTriangle,
        tone: 'amber',
        accent: 'text-amber-700',
        chip: 'bg-amber-500/10',
        title: 'Railway fallback active',
        message: 'Render is still coming back online. Railway is serving requests for now and the app will switch back automatically.',
        animation: ''
      };
    }

    if (isOnline) {
      return {
        icon: CheckCircle,
        tone: 'green',
        accent: 'text-moss',
        chip: 'bg-moss/10',
        title: 'Backend connected',
        message: `Connected to ${backendLabel || 'the Render backend'} and ready for analytics.`,
        animation: ''
      };
    }

    return {
      icon: WifiOff,
      tone: 'red',
      accent: 'text-clay',
      chip: 'bg-clay/10',
      title: 'Backend unavailable',
      message: error || 'Unable to reach Render or the Railway fallback right now.',
      animation: ''
    };
  };

  const statusConfig = getStatusConfig();

  if (!isVisible) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: -16 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -16 }}
        transition={{ duration: 0.4, ease: EASE }}
        className="fixed left-4 right-4 top-4 z-[120] sm:left-auto sm:right-6 sm:max-w-sm"
      >
        <div className="rounded-2xl border border-ink/15 bg-paper-50 p-4 shadow-lift">
          <div className="flex items-start gap-3.5">
            <span className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl ${statusConfig.chip} ${statusConfig.accent}`}>
              <motion.span
                animate={isChecking ? { rotate: 360 } : {}}
                transition={{ duration: 1, repeat: isChecking ? Infinity : 0, ease: 'linear' }}
                className="flex items-center justify-center"
              >
                <statusConfig.icon size={19} className={statusConfig.animation} />
              </motion.span>
            </span>

            <div className="min-w-0 flex-1">
              <div className="flex items-start justify-between gap-3">
                <h3 className={`text-sm font-medium ${statusConfig.accent}`}>{statusConfig.title}</h3>
                <button
                  onClick={() => setIsVisible(false)}
                  aria-label="Dismiss connection notice"
                  className="-mr-1 -mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-ink/35 transition-colors hover:bg-ink/[0.06] hover:text-ink/70"
                >
                  ×
                </button>
              </div>
              <p className="mt-1.5 text-[0.72rem] leading-relaxed text-ink/50">{statusConfig.message}</p>

              {isChecking && (
                <div className="mt-3 flex items-center gap-2">
                  <div className="flex gap-1">
                    {[0, 1, 2].map((i) => (
                      <motion.div
                        key={i}
                        className="h-1.5 w-1.5 rounded-full bg-sky"
                        animate={{ scale: [1, 1.4, 1], opacity: [0.45, 1, 0.45] }}
                        transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.2 }}
                      />
                    ))}
                  </div>
                  <span className="label-sm text-ink/35">Checking connection</span>
                </div>
              )}

              {isOnline && backendSource === 'railway' && (
                <div className="mt-3 flex items-center gap-1.5 text-amber-700">
                  <CheckCircle size={13} />
                  <span className="label-sm">Fallback running on Railway</span>
                </div>
              )}

              {isOnline && backendSource !== 'railway' && (
                <div className="mt-3 flex items-center gap-1.5 text-moss">
                  <CheckCircle size={13} />
                  <span className="label-sm">Ready for analytics</span>
                </div>
              )}
            </div>
          </div>

          <div className="mt-3.5 flex flex-wrap items-center justify-between gap-2 border-t border-ink/10 pt-3">
            <span className="label-sm truncate text-ink/30">
              {String(activeBackendUrl || backendLabel || 'unknown').replace(/^https?:\/\//, '')}
            </span>
            {modelLoaded ? (
              <span className="label-sm flex items-center gap-1.5 text-moss">
                <StatusDot tone="green" pulse={false} /> Models live
              </span>
            ) : (
              <span className="label-sm text-ink/30">Awaiting models</span>
            )}
          </div>

          {isChecking && (
            <div className="mt-3 h-0.5 overflow-hidden rounded-full bg-ink/10">
              <motion.div
                className="h-full rounded-full bg-moss"
                initial={{ width: '0%' }}
                animate={{ width: '100%' }}
                transition={{ duration: 3, ease: 'easeInOut' }}
              />
            </div>
          )}
        </div>
      </motion.div>
    </AnimatePresence>
  );
};

export default BackendConnection;
