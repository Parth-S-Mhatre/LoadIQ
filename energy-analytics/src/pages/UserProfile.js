import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { jsPDF } from 'jspdf';
import {
  User,
  Settings,
  History,
  Bell,
  Save,
  AlertTriangle,
  CheckCircle,
  X,
  TrendingUp,
  BarChart3,
  Clock,
  Zap,
  Building2,
  Users,
  Copy,
  Share2,
  Download,
  Activity,
  Sparkles,
  Link2,
  ArrowLeft
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { UserHistoryService } from '../services/UserHistoryService';
import { Badge, EASE, Logo, StatusDot, buttonClass } from '../components/ui';

const createDefaultProfile = (user) => ({
  fullName: user?.displayName || '',
  email: user?.email || '',
  phone: '',
  location: '',
  organization: '',
  workspaceName: '',
  role: '',
  experience: 'intermediate',
  notifications: true
});

const getStoredDate = (value) => {
  if (!value) {
    return new Date().toISOString();
  }

  if (typeof value?.toDate === 'function') {
    return value.toDate().toISOString();
  }

  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? new Date().toISOString() : parsed.toISOString();
};

const isOfflineFirestoreError = (error) => {
  return error?.code === 'unavailable' || String(error?.message || '').toLowerCase().includes('client is offline');
};

const FIREBASE_SAVE_TIMEOUT_MS = 4500;

const withTimeout = (promise, timeoutMs, message) => {
  let timeoutId;
  const timeoutPromise = new Promise((_, reject) => {
    timeoutId = setTimeout(() => reject(new Error(message)), timeoutMs);
  });

  return Promise.race([promise, timeoutPromise]).finally(() => clearTimeout(timeoutId));
};

const normalizePredictionSeries = (item) => {
  if (Array.isArray(item?.predictionSeries) && item.predictionSeries.length) {
    return item.predictionSeries
      .map((entry, index) => ({
        step: entry?.step ?? index + 1,
        value: Number(entry?.value)
      }))
      .filter((entry) => Number.isFinite(entry.value));
  }

  if (Array.isArray(item?.predictions) && item.predictions.length) {
    return item.predictions
      .map((value, index) => ({
        step: index + 1,
        value: Number(value)
      }))
      .filter((entry) => Number.isFinite(entry.value));
  }

  if (Number.isFinite(Number(item?.prediction))) {
    return [{
      step: 1,
      value: Number(item.prediction)
    }];
  }

  return [];
};

const normalizeHistoryEntry = (item, index) => {
  const predictionSeries = normalizePredictionSeries(item);
  const latestPrediction = item?.latestPrediction
    ?? item?.prediction
    ?? predictionSeries[predictionSeries.length - 1]?.value
    ?? 0;
  const nextHourPrediction = item?.nextHourPrediction
    ?? item?.statistics?.firstPrediction
    ?? predictionSeries[0]?.value
    ?? latestPrediction;

  return {
    id: item?.id || `history-${index}`,
    timestamp: getStoredDate(item?.timestampIso || item?.timestamp),
    forecastHorizon: item?.forecastHorizon || predictionSeries.length || 1,
    predictionSeries,
    latestPrediction: Number(latestPrediction) || 0,
    nextHourPrediction: Number(nextHourPrediction) || 0,
    predictionSource: item?.predictionSource || item?.prediction_source || 'ml_model',
    fallbackUsed: Boolean(item?.fallbackUsed || item?.fallback_used),
    fallbackReason: item?.fallbackReason || item?.reason || null,
    historicalLoads: Array.isArray(item?.historicalLoads) ? item.historicalLoads : (item?.inputData || []),
    statistics: {
      average: item?.statistics?.average ?? item?.average ?? null,
      peak: item?.statistics?.peak ?? item?.peak ?? null,
      minimum: item?.statistics?.minimum ?? item?.minimum ?? null,
      lastActual: item?.statistics?.lastActual ?? item?.lastActual ?? null,
      firstPrediction: item?.statistics?.firstPrediction ?? item?.firstPrediction ?? nextHourPrediction,
      delta: item?.statistics?.delta ?? item?.delta ?? 0,
      direction: item?.statistics?.direction ?? item?.direction ?? 'stable'
    }
  };
};

const Panel = ({ children, className = '' }) => (
  <div className={`surface ${className}`.trim()}>{children}</div>
);

const FieldLabel = ({ children }) => (
  <span className="label-sm mb-2 block text-ink/40">{children}</span>
);

const inputClass = 'h-11 w-full rounded-lg border border-ink/15 bg-paper-50 px-3.5 text-sm text-ink transition-all duration-200 placeholder:text-ink/30 hover:border-ink/25 focus:border-ink focus:bg-white focus:outline-none focus:ring-2 focus:ring-signal-300/60';

const UserProfile = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [isNewUser, setIsNewUser] = useState(false);
  const [formData, setFormData] = useState(createDefaultProfile(user));
  const [analyticsHistory, setAnalyticsHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('profile');
  const [notifications, setNotifications] = useState([]);
  const [inviteContext, setInviteContext] = useState(null);
  const [saveDoneMessage, setSaveDoneMessage] = useState('');

  useEffect(() => {
    let isMounted = true;

    const loadProfileAndHistory = async () => {
      if (!user?.uid) {
        return;
      }

      setHistoryLoading(true);

      const localProfile = JSON.parse(localStorage.getItem(`userProfile_${user.uid}`) || 'null');

      let cloudProfile = null;
      try {
        cloudProfile = await withTimeout(
          UserHistoryService.getUserProfile(user.uid),
          FIREBASE_SAVE_TIMEOUT_MS,
          'Firebase profile fetch timed out.'
        );
      } catch (error) {
        if (!isOfflineFirestoreError(error) && !String(error?.message || '').includes('timed out')) {
          console.error('Failed to load cloud profile:', error);
        }
      }

      const pendingInvite = JSON.parse(localStorage.getItem('pendingInviteContext') || 'null');
      const mergedProfile = {
        ...createDefaultProfile(user),
        ...(cloudProfile || {}),
        ...(localProfile || {})
      };

      if (pendingInvite) {
        mergedProfile.organization = mergedProfile.organization || pendingInvite.organization || '';
        mergedProfile.workspaceName = mergedProfile.workspaceName || pendingInvite.workspaceName || '';
        setInviteContext(pendingInvite);
      } else {
        setInviteContext(null);
      }

      let history = [];
      try {
        history = await withTimeout(
          UserHistoryService.getUserAnalytics(user.uid),
          FIREBASE_SAVE_TIMEOUT_MS,
          'Firebase history fetch timed out.'
        );
      } catch (error) {
        if (!isOfflineFirestoreError(error) && !String(error?.message || '').includes('timed out')) {
          console.error('Failed to load cloud history:', error);
        }
      }

      const normalizedHistory = Array.isArray(history) && history.length
        ? history.map(normalizeHistoryEntry)
        : JSON.parse(localStorage.getItem(`analyticsHistory_${user.uid}`) || '[]').map(normalizeHistoryEntry);

      if (!isMounted) {
        return;
      }

      setFormData(mergedProfile);
      setIsNewUser(!localProfile && !cloudProfile);
      setAnalyticsHistory(normalizedHistory);
      setHistoryLoading(false);
    };

    loadProfileAndHistory();

    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }

    return () => {
      isMounted = false;
    };
  }, [user]);

  useEffect(() => {
    if (!user?.uid) {
      return undefined;
    }

    const syncPendingProfile = async () => {
      const pendingProfile = JSON.parse(localStorage.getItem(`pendingUserProfileSync_${user.uid}`) || 'null');
      if (!pendingProfile) {
        return;
      }

      try {
        await withTimeout(
          UserHistoryService.saveUserProfile(user.uid, pendingProfile),
          FIREBASE_SAVE_TIMEOUT_MS,
          'Firebase sync timed out.'
        );
        localStorage.removeItem(`pendingUserProfileSync_${user.uid}`);
        addNotification('Profile synced to Firebase.', 'success');
      } catch (error) {
        const isExpectedSyncDelay = isOfflineFirestoreError(error) || String(error?.message || '').includes('Firebase sync timed out');
        if (!isExpectedSyncDelay) {
          console.error('Failed to sync pending profile:', error);
        }
      }
    };

    syncPendingProfile();
    window.addEventListener('online', syncPendingProfile);
    return () => window.removeEventListener('online', syncPendingProfile);
  }, [user?.uid]);

  useEffect(() => {
    const handleBeforeUnload = () => {
      if (formData.notifications && 'Notification' in window && Notification.permission === 'granted') {
        new Notification('LoadIQ Analytics', {
          body: 'Your analytics workspace is ready when you return.',
          icon: '/favicon.ico'
        });
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [formData.notifications]);

  const analyticsSummary = useMemo(() => {
    if (!analyticsHistory.length) {
      return {
        totalRuns: 0,
        averageNextHour: 0,
        averageHorizon: 0,
        latestPrediction: 0,
        fallbackRate: 0
      };
    }

    const totalRuns = analyticsHistory.length;
    const averageNextHour = analyticsHistory.reduce((sum, item) => sum + item.nextHourPrediction, 0) / totalRuns;
    const averageHorizon = analyticsHistory.reduce((sum, item) => sum + item.forecastHorizon, 0) / totalRuns;
    const latestPrediction = analyticsHistory[0]?.latestPrediction || 0;
    const fallbackRate = (analyticsHistory.filter((item) => item.fallbackUsed).length / totalRuns) * 100;

    return {
      totalRuns,
      averageNextHour,
      averageHorizon,
      latestPrediction,
      fallbackRate
    };
  }, [analyticsHistory]);

  const workspaceInviteLink = useMemo(() => {
    if (typeof window === 'undefined') {
      return '';
    }

    const params = new URLSearchParams({
      workspace: formData.workspaceName || formData.organization || 'LoadIQ Workspace',
      organization: formData.organization || '',
      invitedBy: formData.fullName || user?.displayName || user?.email || 'LoadIQ teammate'
    });

    return `${window.location.origin}/register?${params.toString()}`;
  }, [formData.workspaceName, formData.organization, formData.fullName, user?.displayName, user?.email]);

  const handleInputChange = (field, value) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const addNotification = (message, type = 'info') => {
    const id = Date.now();
    setNotifications((prev) => [...prev, { id, message, type }]);
    setTimeout(() => setNotifications((prev) => prev.filter((entry) => entry.id !== id)), 5000);
  };

  const playSaveDoneAnimation = (message = 'Profile saved') => {
    setSaveDoneMessage(message);
    setTimeout(() => setSaveDoneMessage(''), 1400);
  };

  const waitForDoneAnimation = () => new Promise((resolve) => setTimeout(resolve, 850));

  const saveProfile = async () => {
    if (!user?.uid) {
      return;
    }

    setIsLoading(true);

    try {
      const profileToStore = {
        ...formData,
        uid: user.uid,
        userId: user.uid
      };

      localStorage.setItem(`userProfile_${user.uid}`, JSON.stringify(profileToStore));
      await withTimeout(
        UserHistoryService.saveUserProfile(user.uid, formData),
        FIREBASE_SAVE_TIMEOUT_MS,
        'Firebase save timed out.'
      );
      localStorage.removeItem(`pendingUserProfileSync_${user.uid}`);
      localStorage.removeItem('pendingInviteContext');
      setInviteContext(null);
      setIsNewUser(false);
      playSaveDoneAnimation('Profile saved');
      addNotification('Profile and workspace settings saved.', 'success');

      if (isNewUser) {
        await waitForDoneAnimation();
        navigate('/dashboard');
      }
    } catch (error) {
      if (isOfflineFirestoreError(error) || String(error?.message || '').includes('Firebase save timed out')) {
        localStorage.setItem(`pendingUserProfileSync_${user.uid}`, JSON.stringify({
          ...formData,
          uid: user.uid,
          userId: user.uid,
          pendingAt: new Date().toISOString()
        }));
        localStorage.removeItem('pendingInviteContext');
        setInviteContext(null);
        setIsNewUser(false);
        playSaveDoneAnimation('Saved locally');
        addNotification('Saved locally. Firebase is offline, so it will need to sync when the connection returns.', 'info');

        if (isNewUser) {
          await waitForDoneAnimation();
          navigate('/dashboard');
        }
      } else {
        console.error('Failed to save profile:', error);
        addNotification('Failed to save profile settings.', 'error');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const reportIssue = async () => {
    try {
      await UserHistoryService.saveReport(user?.uid || 'anonymous', {
        title: 'Profile workspace issue',
        description: 'User requested support from the profile dashboard.'
      });
      addNotification('Issue reported successfully. We will follow up soon.', 'success');
    } catch (error) {
      addNotification('Failed to submit support request.', 'error');
    }
  };

  const toggleNotifications = () => {
    setFormData((prev) => ({ ...prev, notifications: !prev.notifications }));
  };

  const exportHistory = () => {
    try {
      const doc = new jsPDF({ unit: 'pt', format: 'a4' });
      const pageWidth = doc.internal.pageSize.getWidth();
      const pageHeight = doc.internal.pageSize.getHeight();
      const margin = 42;
      let y = 48;

      const addText = (text, x, size = 10, style = 'normal', color = [30, 41, 59]) => {
        doc.setFont('helvetica', style);
        doc.setFontSize(size);
        doc.setTextColor(...color);
        doc.text(String(text), x, y);
      };

      const ensureSpace = (height = 36) => {
        if (y + height <= pageHeight - margin) {
          return;
        }

        doc.addPage();
        y = margin;
      };

      doc.setFillColor(17, 24, 19);
      doc.rect(0, 0, pageWidth, 96, 'F');
      doc.setTextColor(255, 255, 255);
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(22);
      doc.text('LoadIQ Forecast History Report', margin, y);
      y += 24;
      doc.setFont('helvetica', 'normal');
      doc.setFontSize(10);
      doc.setTextColor(216, 242, 77);
      doc.text(`Generated ${new Date().toLocaleString()}`, margin, y);
      y = 126;

      addText('Workspace Summary', margin, 14, 'bold', [17, 24, 19]);
      y += 24;
      const summaryRows = [
        ['Analyst', formData.fullName || user?.displayName || 'Not set'],
        ['Email', formData.email || user?.email || 'Not set'],
        ['Workspace', formData.workspaceName || 'Not set'],
        ['Organization', formData.organization || 'Not set'],
        ['Total runs', analyticsSummary.totalRuns],
        ['Latest prediction', `${Math.round(analyticsSummary.latestPrediction).toLocaleString()} MW`],
        ['Average next hour', `${Math.round(analyticsSummary.averageNextHour).toLocaleString()} MW`],
        ['Fallback rate', `${analyticsSummary.fallbackRate.toFixed(0)}%`]
      ];

      summaryRows.forEach(([label, value]) => {
        ensureSpace(24);
        addText(label, margin, 9, 'bold', [71, 85, 105]);
        addText(value, margin + 150, 10, 'normal', [17, 24, 19]);
        y += 20;
      });

      y += 18;
      addText('Forecast Sessions', margin, 14, 'bold', [17, 24, 19]);
      y += 24;

      if (!analyticsHistory.length) {
        addText('No analytics history has been saved yet.', margin, 10, 'normal', [71, 85, 105]);
      }

      analyticsHistory.forEach((item, index) => {
        ensureSpace(120);
        doc.setDrawColor(226, 232, 240);
        doc.setFillColor(246, 244, 238);
        doc.roundedRect(margin, y - 14, pageWidth - margin * 2, 100, 8, 8, 'FD');

        addText(`Session ${index + 1}`, margin + 16, 11, 'bold', [17, 24, 19]);
        addText(new Date(item.timestamp).toLocaleString(), margin + 110, 9, 'normal', [71, 85, 105]);
        y += 22;

        const sessionLines = [
          `Horizon: ${item.forecastHorizon}h`,
          `Source: ${item.predictionSource}`,
          `Next hour: ${Math.round(item.nextHourPrediction).toLocaleString()} MW`,
          `Latest step: ${Math.round(item.latestPrediction).toLocaleString()} MW`,
          `Peak: ${Math.round(item.statistics.peak || 0).toLocaleString()} MW`,
          `Delta: ${Math.round(item.statistics.delta || 0).toLocaleString()} MW`
        ];

        const wrapped = doc.splitTextToSize(sessionLines.join('  |  '), pageWidth - margin * 2 - 32);
        doc.setFont('helvetica', 'normal');
        doc.setFontSize(9);
        doc.setTextColor(51, 65, 85);
        doc.text(wrapped, margin + 16, y);
        y += wrapped.length * 12 + 20;

        if (item.predictionSeries.length) {
          const seriesPreview = item.predictionSeries
            .slice(0, 8)
            .map((entry) => `+${entry.step}h ${Math.round(entry.value).toLocaleString()} MW`)
            .join(', ');
          const seriesWrapped = doc.splitTextToSize(`Series: ${seriesPreview}`, pageWidth - margin * 2 - 32);
          doc.text(seriesWrapped, margin + 16, y);
        }

        y += 58;
      });

      doc.save(`loadiq_history_${user?.uid || 'guest'}.pdf`);
      addNotification('Analytics history PDF exported.', 'success');
    } catch (error) {
      console.error('Failed to export PDF history:', error);
      addNotification('Failed to export analytics history PDF.', 'error');
    }
  };

  const copyInviteLink = async () => {
    try {
      await navigator.clipboard.writeText(workspaceInviteLink);
      addNotification('Invite link copied to clipboard.', 'success');
    } catch (error) {
      addNotification('Failed to copy invite link.', 'error');
    }
  };

  const shareInviteLink = async () => {
    try {
      if (navigator.share) {
        await navigator.share({
          title: 'Join my LoadIQ workspace',
          text: `Join ${formData.workspaceName || formData.organization || 'our LoadIQ workspace'} and collaborate on energy analysis.`,
          url: workspaceInviteLink
        });
        addNotification('Invite link shared.', 'success');
        return;
      }

      await copyInviteLink();
    } catch (error) {
      addNotification('Failed to share invite link.', 'error');
    }
  };

  const tabs = [
    { id: 'profile', label: 'Profile', icon: User },
    { id: 'history', label: 'History', icon: History },
    { id: 'workspace', label: 'Workspace', icon: Users },
    { id: 'settings', label: 'Settings', icon: Settings },
    { id: 'notifications', label: 'Notifications', icon: Bell }
  ];

  const summaryStats = [
    ['Runs', analyticsSummary.totalRuns],
    ['Latest', Math.round(analyticsSummary.latestPrediction).toLocaleString()],
    ['Avg next hour', Math.round(analyticsSummary.averageNextHour).toLocaleString()],
    ['Fallback rate', `${analyticsSummary.fallbackRate.toFixed(0)}%`]
  ];

  return (
    <div className="min-h-screen bg-paper text-ink">
      <AnimatePresence>
        {saveDoneMessage && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[80] flex items-center justify-center bg-ink/40 px-6 backdrop-blur-sm"
          >
            <motion.div
              initial={{ scale: 0.9, y: 18 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.95, y: -8 }}
              transition={{ type: 'spring', stiffness: 260, damping: 22 }}
              className="w-full max-w-sm rounded-2xl border border-ink/15 bg-paper-50 p-8 text-center shadow-lift"
            >
              <motion.div
                initial={{ scale: 0, rotate: -18 }}
                animate={{ scale: 1, rotate: 0 }}
                transition={{ type: 'spring', stiffness: 320, damping: 16, delay: 0.08 }}
                className="mx-auto mb-5 flex h-16 w-16 items-center justify-center rounded-full bg-signal text-ink"
              >
                <CheckCircle size={32} strokeWidth={2.2} />
              </motion.div>
              <motion.h2
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.16 }}
                className="text-xl font-medium tracking-[-0.03em]"
              >
                {saveDoneMessage}
              </motion.h2>
              <motion.p
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.24 }}
                className="mt-2 text-sm text-ink/45"
              >
                Your workspace details are ready.
              </motion.p>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Header */}
      <header className="border-b border-ink/10 bg-paper/85 backdrop-blur-xl">
        <div className="mx-auto max-w-7xl px-4 py-5 sm:px-6">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex items-center gap-4">
              <button onClick={() => navigate('/dashboard')} aria-label="LoadIQ home" className="shrink-0">
                <Logo />
              </button>
              <span className="hidden h-9 w-px bg-ink/10 sm:block" />
              <div className="hidden sm:block">
                <p className="label-sm text-ink/35">Operator workspace</p>
                <p className="mt-1 text-sm font-medium">Account &amp; history</p>
              </div>
            </div>

            <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
              <button onClick={() => navigate('/dashboard')} className={buttonClass('outline', 'md')}>
                <ArrowLeft size={15} />
                Back to dashboard
              </button>
              <div className="flex items-center gap-2.5 rounded-full border border-ink/10 bg-paper-50 px-3 py-2">
                <StatusDot tone="green" />
                <span className="label-sm text-ink/45">Signed in</span>
              </div>
            </div>
          </div>

          <div className="mt-8 grid gap-6 border-t border-ink/10 pt-7 lg:grid-cols-[1.2fr_1fr] lg:items-end">
            <div className="flex items-start gap-4">
              <span className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl border border-ink/10 bg-ink text-signal">
                <Sparkles size={22} />
              </span>
              <div>
                <p className="label text-ink/35">Grid intelligence · personal</p>
                <h1 className="mt-2 text-[1.7rem] leading-tight tracking-[-0.04em] sm:text-[2rem]">
                  {formData.fullName ? `${formData.fullName.split(' ')[0]}'s workspace` : 'Analysis workspace'}
                </h1>
                <p className="mt-1.5 max-w-xl text-sm leading-relaxed text-ink/50">
                  Profile, forecast history and collaboration settings in one place.
                </p>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-ink/10 bg-ink/10 sm:grid-cols-4">
              {summaryStats.map(([label, value]) => (
                <div key={label} className="bg-paper-50 px-3.5 py-3">
                  <p className="label-sm text-ink/35">{label}</p>
                  <p className="mono-num mt-1.5 text-sm font-medium">{value}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 sm:py-8">
        {/* Tabs */}
        <div className="mb-6 flex gap-1.5 overflow-x-auto rounded-full border border-ink/10 bg-paper-50 p-1.5 no-scrollbar sm:w-fit">
          {tabs.map((tab) => {
            const active = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`relative flex shrink-0 items-center gap-2 rounded-full px-4 py-2.5 text-[0.75rem] font-medium transition-colors duration-200 ${
                  active ? 'text-paper' : 'text-ink/50 hover:text-ink'
                }`}
              >
                {active && (
                  <motion.span
                    layoutId="profile-tab-active"
                    className="absolute inset-0 rounded-full bg-ink"
                    transition={{ type: 'spring', stiffness: 380, damping: 32 }}
                  />
                )}
                <tab.icon size={15} className="relative z-10" />
                <span className="relative z-10 whitespace-nowrap">{tab.label}</span>
              </button>
            );
          })}
        </div>

        <AnimatePresence mode="wait">
          {activeTab === 'profile' && (
            <motion.div
              key="profile"
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.4, ease: EASE }}
              className="space-y-4"
            >
              {isNewUser && (
                <div className="rounded-xl border border-amber-600/25 bg-amber-50 p-4 sm:p-5">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="mt-0.5 shrink-0 text-amber-700" size={18} />
                    <div>
                      <h3 className="text-sm font-medium text-amber-900">Complete your analyst setup</h3>
                      <p className="mt-1 text-xs leading-relaxed text-amber-800/70">
                        Fill out your profile and workspace details so your forecast history and collaboration link stay organized.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {inviteContext && (
                <div className="rounded-xl border border-moss/20 bg-moss/[0.06] p-4 sm:p-5">
                  <div className="flex items-start gap-3">
                    <Users className="mt-0.5 shrink-0 text-moss" size={18} />
                    <div>
                      <h3 className="text-sm font-medium text-moss">Workspace invite detected</h3>
                      <p className="mt-1 text-xs leading-relaxed text-ink/55">
                        You were invited to join{' '}
                        <span className="font-semibold text-ink">{inviteContext.workspaceName || 'a LoadIQ workspace'}</span>
                        {inviteContext.invitedBy ? ` by ${inviteContext.invitedBy}` : ''}. These fields are pre-filled below.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
                <Panel className="p-5 sm:p-6 xl:col-span-2">
                  <div className="border-b border-ink/10 pb-4">
                    <p className="label text-ink/35">Personal details</p>
                    <p className="mt-1.5 text-xs text-ink/45">These fields are stored locally and synced to Firebase.</p>
                  </div>

                  <div className="mt-5 grid grid-cols-1 gap-4 md:grid-cols-2">
                    <label>
                      <FieldLabel>Full name *</FieldLabel>
                      <input
                        type="text"
                        value={formData.fullName}
                        onChange={(e) => handleInputChange('fullName', e.target.value)}
                        className={inputClass}
                        placeholder="Enter your full name"
                      />
                    </label>
                    <label>
                      <FieldLabel>Email *</FieldLabel>
                      <input
                        type="email"
                        value={formData.email}
                        onChange={(e) => handleInputChange('email', e.target.value)}
                        className={inputClass}
                        placeholder="Enter your email"
                      />
                    </label>
                    <label>
                      <FieldLabel>Phone</FieldLabel>
                      <input
                        type="tel"
                        value={formData.phone}
                        onChange={(e) => handleInputChange('phone', e.target.value)}
                        className={inputClass}
                        placeholder="Phone number"
                      />
                    </label>
                    <label>
                      <FieldLabel>Location</FieldLabel>
                      <input
                        type="text"
                        value={formData.location}
                        onChange={(e) => handleInputChange('location', e.target.value)}
                        className={inputClass}
                        placeholder="City, Country"
                      />
                    </label>
                    <label>
                      <FieldLabel>Organization</FieldLabel>
                      <input
                        type="text"
                        value={formData.organization}
                        onChange={(e) => handleInputChange('organization', e.target.value)}
                        className={inputClass}
                        placeholder="Company or institution"
                      />
                    </label>
                    <label>
                      <FieldLabel>Workspace name</FieldLabel>
                      <input
                        type="text"
                        value={formData.workspaceName}
                        onChange={(e) => handleInputChange('workspaceName', e.target.value)}
                        className={inputClass}
                        placeholder="Grid Operations Team"
                      />
                    </label>
                    <label>
                      <FieldLabel>Role</FieldLabel>
                      <select
                        value={formData.role}
                        onChange={(e) => handleInputChange('role', e.target.value)}
                        className={`${inputClass} cursor-pointer`}
                      >
                        <option value="">Select your role</option>
                        <option value="analyst">Energy Analyst</option>
                        <option value="engineer">Power Engineer</option>
                        <option value="manager">Grid Manager</option>
                        <option value="researcher">Researcher</option>
                        <option value="student">Student</option>
                        <option value="other">Other</option>
                      </select>
                    </label>
                    <label>
                      <FieldLabel>Experience level</FieldLabel>
                      <select
                        value={formData.experience}
                        onChange={(e) => handleInputChange('experience', e.target.value)}
                        className={`${inputClass} cursor-pointer`}
                      >
                        <option value="beginner">Beginner</option>
                        <option value="intermediate">Intermediate</option>
                        <option value="advanced">Advanced</option>
                        <option value="expert">Expert</option>
                      </select>
                    </label>
                  </div>
                </Panel>

                <Panel className="space-y-5 p-5 sm:p-6">
                  <div>
                    <p className="label text-ink/35">Workspace snapshot</p>
                    <h3 className="mt-2 text-lg font-medium leading-snug tracking-[-0.03em]">
                      {formData.workspaceName || formData.organization || 'Set up your workspace'}
                    </h3>
                    <p className="mt-2 text-xs leading-relaxed text-ink/45">
                      Keep profile details, history and collaboration context aligned for repeat analysis sessions.
                    </p>
                  </div>

                  <div className="grid grid-cols-2 gap-2.5">
                    <div className="rounded-xl border border-ink/10 bg-paper-100 p-3.5">
                      <Building2 className="mb-2.5 text-ink/35" size={16} />
                      <p className="label-sm text-ink/35">Org</p>
                      <p className="mt-1 break-words text-xs font-medium text-ink">{formData.organization || 'Not set'}</p>
                    </div>
                    <div className="rounded-xl border border-ink/10 bg-paper-100 p-3.5">
                      <Activity className="mb-2.5 text-ink/35" size={16} />
                      <p className="label-sm text-ink/35">Role</p>
                      <p className="mt-1 break-words text-xs font-medium capitalize text-ink">{formData.role || 'Not set'}</p>
                    </div>
                  </div>

                  <div className="rounded-xl border border-ink/10 bg-paper-100 p-3.5">
                    <p className="label-sm text-ink/35">Experience</p>
                    <p className="mt-1.5 text-xs font-medium capitalize text-ink">{formData.experience || 'Not set'}</p>
                  </div>

                  <button
                    onClick={saveProfile}
                    disabled={isLoading || !formData.fullName || !formData.email}
                    className={buttonClass('primary', 'lg', 'w-full')}
                  >
                    {isLoading ? (
                      <span className="h-4 w-4 animate-spin rounded-full border-2 border-paper/25 border-t-paper" />
                    ) : (
                      <Save size={16} />
                    )}
                    {isNewUser ? 'Complete workspace setup' : 'Save profile'}
                  </button>
                </Panel>
              </div>
            </motion.div>
          )}

          {activeTab === 'history' && (
            <motion.div
              key="history"
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.4, ease: EASE }}
              className="space-y-4"
            >
              <div className="flex flex-col gap-4 border-b border-ink/10 pb-5 md:flex-row md:items-end md:justify-between">
                <div>
                  <p className="label text-ink/35">Saved sessions</p>
                  <h2 className="mt-1.5 text-xl font-medium tracking-[-0.03em]">Prediction history</h2>
                  <p className="mt-1.5 text-xs text-ink/45">
                    Model 2 forecast sessions with horizon, source and fallback context.
                  </p>
                </div>
                <button onClick={exportHistory} className={buttonClass('outline', 'md')}>
                  <Download size={15} />
                  Export history
                </button>
              </div>

              {historyLoading ? (
                <Panel className="p-10 text-center">
                  <div className="mx-auto mb-4 h-10 w-10 animate-spin rounded-full border-2 border-ink/10 border-t-moss" />
                  <p className="text-sm text-ink/50">Loading forecast history…</p>
                </Panel>
              ) : analyticsHistory.length === 0 ? (
                <Panel className="p-10 text-center sm:p-14">
                  <BarChart3 className="mx-auto mb-4 text-ink/25" size={40} />
                  <p className="text-sm font-medium text-ink/65">No analytics history yet</p>
                  <p className="mt-1.5 text-xs text-ink/40">
                    Run forecasts in the Load Predictor and they will appear here automatically.
                  </p>
                </Panel>
              ) : (
                <div className="space-y-3">
                  {analyticsHistory.map((item, index) => (
                    <motion.div
                      key={item.id}
                      initial={{ opacity: 0, y: 12 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.4, delay: Math.min(index * 0.03, 0.24), ease: EASE }}
                    >
                      <Panel className="p-5">
                        <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
                          <div className="space-y-3">
                            <div className="flex items-center gap-3">
                              <span className="flex h-10 w-10 items-center justify-center rounded-xl border border-ink/10 bg-paper-100 text-moss">
                                <TrendingUp size={17} />
                              </span>
                              <div>
                                <p className="text-sm font-medium">Load forecast session</p>
                                <p className="mt-0.5 text-[0.68rem] text-ink/40">{new Date(item.timestamp).toLocaleString()}</p>
                              </div>
                            </div>
                            <div className="flex flex-wrap gap-1.5">
                              <Badge tone="neutral">Horizon {item.forecastHorizon}h</Badge>
                              <Badge tone="neutral">Source {item.predictionSource}</Badge>
                              <Badge tone={item.fallbackUsed ? 'clay' : 'moss'}>
                                <StatusDot tone={item.fallbackUsed ? 'amber' : 'green'} pulse={false} />
                                {item.fallbackUsed ? 'Fallback used' : 'Model output'}
                              </Badge>
                            </div>
                          </div>

                          <div className="grid min-w-0 grid-cols-2 gap-2 sm:grid-cols-4 lg:min-w-[26rem]">
                            {[
                              ['Next hour', `${Math.round(item.nextHourPrediction).toLocaleString()} MW`],
                              ['Latest step', `${Math.round(item.latestPrediction).toLocaleString()} MW`],
                              ['Peak', `${Math.round(item.statistics.peak || 0).toLocaleString()} MW`],
                            ].map(([label, value]) => (
                              <div key={label} className="rounded-lg border border-ink/10 bg-paper-100 p-3">
                                <p className="label-sm text-ink/35">{label}</p>
                                <p className="mono-num mt-1.5 text-xs font-medium">{value}</p>
                              </div>
                            ))}
                            <div className="rounded-lg border border-ink/10 bg-paper-100 p-3">
                              <p className="label-sm text-ink/35">Delta</p>
                              <p className={`mono-num mt-1.5 text-xs font-medium ${
                                item.statistics.delta > 0 ? 'text-clay' : item.statistics.delta < 0 ? 'text-moss' : 'text-ink'
                              }`}>
                                {Math.round(item.statistics.delta || 0).toLocaleString()} MW
                              </p>
                            </div>
                          </div>
                        </div>

                        {item.predictionSeries.length > 0 && (
                          <div className="mt-4 grid grid-cols-2 gap-2 border-t border-ink/10 pt-4 md:grid-cols-4 xl:grid-cols-6">
                            {item.predictionSeries.slice(0, 6).map((entry) => (
                              <div key={`${item.id}-${entry.step}`} className="rounded-lg bg-paper-100 p-2.5">
                                <p className="label-sm text-ink/30">Step {entry.step}</p>
                                <p className="mono-num mt-1.5 text-xs font-medium">{Math.round(entry.value).toLocaleString()} MW</p>
                              </div>
                            ))}
                          </div>
                        )}

                        {item.fallbackReason && (
                          <p className="mt-3 text-[0.7rem] leading-relaxed text-amber-700">
                            Fallback reason: {item.fallbackReason}
                          </p>
                        )}
                      </Panel>
                    </motion.div>
                  ))}
                </div>
              )}
            </motion.div>
          )}

          {activeTab === 'workspace' && (
            <motion.div
              key="workspace"
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.4, ease: EASE }}
              className="grid grid-cols-1 gap-4 xl:grid-cols-2"
            >
              <Panel className="space-y-5 p-5 sm:p-7">
                <div>
                  <p className="label text-ink/35">Invite teammates</p>
                  <h3 className="mt-2 text-xl font-medium tracking-[-0.03em]">Share your analysis workspace</h3>
                  <p className="mt-2 text-xs leading-relaxed text-ink/45">
                    Send a registration link that carries your workspace and organization context so new
                    teammates can join the same analysis setup.
                  </p>
                </div>

                <div className="rounded-xl border border-ink/10 bg-paper-100 p-4">
                  <div className="flex items-start gap-3">
                    <Link2 className="mt-0.5 shrink-0 text-moss" size={16} />
                    <p className="break-all font-mono text-[0.7rem] leading-relaxed text-ink/65">{workspaceInviteLink}</p>
                  </div>
                </div>

                <div className="flex flex-wrap gap-2">
                  <button onClick={copyInviteLink} className={buttonClass('primary', 'md')}>
                    <Copy size={15} />
                    Copy link
                  </button>
                  <button onClick={shareInviteLink} className={buttonClass('outline', 'md')}>
                    <Share2 size={15} />
                    Share invite
                  </button>
                </div>

                <div className="rounded-xl border border-signal-500/30 bg-signal/25 p-4">
                  <p className="label-sm text-ink/45">How invites work</p>
                  <p className="mt-1.5 text-[0.7rem] leading-relaxed text-ink/60">
                    Teammates opening the link land on registration with your workspace context preloaded,
                    then continue into the shared setup after sign-up.
                  </p>
                </div>
              </Panel>

              <Panel className="space-y-5 p-5 sm:p-7">
                <div>
                  <p className="label text-ink/35">Workspace state</p>
                  <h3 className="mt-2 text-xl font-medium tracking-[-0.03em]">
                    {formData.workspaceName || 'No workspace name yet'}
                  </h3>
                </div>

                <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
                  {[
                    { icon: Building2, label: 'Organization', value: formData.organization || 'Not set' },
                    { icon: Users, label: 'Invited by', value: inviteContext?.invitedBy || formData.fullName || 'You' },
                    { icon: Clock, label: 'Average horizon', value: `${analyticsSummary.averageHorizon.toFixed(1)} hours` },
                    { icon: Zap, label: 'Saved runs', value: analyticsSummary.totalRuns },
                  ].map((item) => (
                    <div key={item.label} className="rounded-xl border border-ink/10 bg-paper-100 p-4">
                      <item.icon className="mb-2.5 text-ink/35" size={16} />
                      <p className="label-sm text-ink/35">{item.label}</p>
                      <p className="mt-1.5 break-words text-xs font-medium text-ink">{item.value}</p>
                    </div>
                  ))}
                </div>

                <div className="border-t border-ink/10 pt-4">
                  <div className="flex items-center justify-between">
                    <span className="label-sm text-ink/35">Members</span>
                    <span className="mono-num text-xs font-medium">1 active</span>
                  </div>
                  <div className="mt-3 flex items-center gap-3">
                    <span className="flex h-8 w-8 items-center justify-center rounded-full bg-signal text-[0.65rem] font-semibold text-ink">
                      {(formData.fullName || user?.email || 'L').charAt(0).toUpperCase()}
                    </span>
                    <div className="min-w-0">
                      <p className="truncate text-xs font-medium">{formData.fullName || 'You'}</p>
                      <p className="truncate text-[0.65rem] text-ink/40">{formData.email || user?.email || 'No email set'}</p>
                    </div>
                    <Badge tone="moss" className="ml-auto">Owner</Badge>
                  </div>
                </div>
              </Panel>
            </motion.div>
          )}

          {activeTab === 'settings' && (
            <motion.div
              key="settings"
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.4, ease: EASE }}
              className="grid grid-cols-1 gap-4 xl:grid-cols-2"
            >
              <Panel className="p-5 sm:p-6">
                <p className="label text-ink/35">Preferences</p>
                <h3 className="mt-1.5 text-lg font-medium tracking-[-0.03em]">Notification settings</h3>

                <div className="mt-5 flex items-center justify-between gap-6 border-t border-ink/10 pt-5">
                  <div>
                    <p className="text-sm font-medium text-ink/75">Browser notifications</p>
                    <p className="mt-1 text-xs leading-relaxed text-ink/45">
                      Receive updates when long-running analytics complete.
                    </p>
                  </div>
                  <button
                    onClick={toggleNotifications}
                    role="switch"
                    aria-checked={formData.notifications}
                    aria-label="Toggle browser notifications"
                    className={`relative h-6 w-11 shrink-0 rounded-full transition-colors duration-300 ${
                      formData.notifications ? 'bg-moss' : 'bg-ink/15'
                    }`}
                  >
                    <motion.span
                      className="absolute top-0.5 h-5 w-5 rounded-full bg-paper-50 shadow-sm"
                      animate={{ left: formData.notifications ? '1.375rem' : '0.125rem' }}
                      transition={{ type: 'spring', stiffness: 500, damping: 32 }}
                    />
                  </button>
                </div>
              </Panel>

              <Panel className="p-5 sm:p-6">
                <p className="label text-ink/35">Support</p>
                <h3 className="mt-1.5 text-lg font-medium tracking-[-0.03em]">Report an issue</h3>
                <p className="mt-2 text-xs leading-relaxed text-ink/45">
                  Report a workspace or analytics issue and keep the rest of the system running while we investigate.
                </p>
                <button onClick={reportIssue} className={buttonClass('danger', 'md', 'mt-5')}>
                  <AlertTriangle size={15} />
                  Report issue
                </button>
              </Panel>

              <Panel className="p-5 sm:p-6 xl:col-span-2">
                <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
                  <div>
                    <p className="label text-ink/35">Session</p>
                    <p className="mt-1.5 text-sm font-medium">{user?.email || 'Local workspace'}</p>
                    <p className="mt-1 text-[0.68rem] text-ink/40">
                      Account state is managed by Firebase Auth.
                    </p>
                  </div>
                  <button onClick={() => navigate('/dashboard')} className={buttonClass('outline', 'md')}>
                    Return to console
                  </button>
                </div>
              </Panel>
            </motion.div>
          )}

          {activeTab === 'notifications' && (
            <motion.div
              key="notifications"
              initial={{ opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.4, ease: EASE }}
              className="grid grid-cols-1 gap-4 xl:grid-cols-2"
            >
              <Panel className="p-5 sm:p-6">
                <div className="flex items-center gap-2.5">
                  <Bell className="text-ink/45" size={17} />
                  <h3 className="text-lg font-medium tracking-[-0.03em]">Notification center</h3>
                </div>
                <p className="mt-2 text-xs leading-relaxed text-ink/45">
                  Forecast completion alerts use your browser permission state and saved workspace preferences.
                </p>

                <div className="mt-5 rounded-xl border border-ink/10 bg-paper-100 p-4">
                  <p className="label-sm text-ink/35">Permission status</p>
                  <p className="mt-1.5 text-sm font-medium">
                    {'Notification' in window
                      ? Notification.permission === 'granted'
                        ? 'Granted'
                        : Notification.permission === 'denied'
                          ? 'Denied'
                          : 'Not requested'
                      : 'Not supported'}
                  </p>
                </div>
              </Panel>

              <Panel className="p-5 sm:p-6">
                <p className="label text-ink/35">Delivery</p>
                <h3 className="mt-1.5 text-lg font-medium tracking-[-0.03em]">What you will receive</h3>
                <div className="mt-4 space-y-0">
                  {[
                    ['Forecast completion', 'When a multi-step run finishes'],
                    ['Fallback notices', 'If Render hands off to Railway'],
                    ['Workspace updates', 'Invites and profile sync state'],
                  ].map(([label, note], index) => (
                    <div key={label} className={`flex items-center justify-between gap-4 py-3 ${index < 2 ? 'border-b border-ink/10' : ''}`}>
                      <div>
                        <p className="text-xs font-medium">{label}</p>
                        <p className="mt-0.5 text-[0.65rem] text-ink/40">{note}</p>
                      </div>
                      <StatusDot tone="green" pulse={false} />
                    </div>
                  ))}
                </div>
              </Panel>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <AnimatePresence>
        {notifications.map((notification) => (
          <motion.div
            key={notification.id}
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 30 }}
            transition={{ duration: 0.35, ease: EASE }}
            className="fixed left-4 right-4 top-4 z-[90] sm:left-auto sm:right-6 sm:top-6 sm:max-w-sm"
          >
            <div className="flex items-center gap-3 rounded-xl border border-ink/15 bg-paper-50 p-3.5 shadow-lift">
              <span className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-full ${
                notification.type === 'success'
                  ? 'bg-moss/10 text-moss'
                  : notification.type === 'error'
                    ? 'bg-clay/10 text-clay'
                    : 'bg-sky/10 text-sky'
              }`}>
                {notification.type === 'success' ? (
                  <CheckCircle size={16} />
                ) : notification.type === 'error' ? (
                  <X size={16} />
                ) : (
                  <Bell size={16} />
                )}
              </span>
              <p className="text-xs font-medium text-ink/75">{notification.message}</p>
            </div>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
};

export default UserProfile;
