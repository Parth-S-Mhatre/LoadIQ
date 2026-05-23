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
      const cloudProfile = await UserHistoryService.getUserProfile(user.uid);
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

      const history = await UserHistoryService.getUserAnalytics(user.uid);
      const normalizedHistory = history.length
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

      doc.setFillColor(15, 23, 42);
      doc.rect(0, 0, pageWidth, 96, 'F');
      doc.setTextColor(255, 255, 255);
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(22);
      doc.text('LoadIQ Forecast History Report', margin, y);
      y += 24;
      doc.setFont('helvetica', 'normal');
      doc.setFontSize(10);
      doc.setTextColor(203, 213, 225);
      doc.text(`Generated ${new Date().toLocaleString()}`, margin, y);
      y = 126;

      addText('Workspace Summary', margin, 14, 'bold', [15, 23, 42]);
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
        addText(value, margin + 150, 10, 'normal', [15, 23, 42]);
        y += 20;
      });

      y += 18;
      addText('Forecast Sessions', margin, 14, 'bold', [15, 23, 42]);
      y += 24;

      if (!analyticsHistory.length) {
        addText('No analytics history has been saved yet.', margin, 10, 'normal', [71, 85, 105]);
      }

      analyticsHistory.forEach((item, index) => {
        ensureSpace(120);
        doc.setDrawColor(226, 232, 240);
        doc.setFillColor(248, 250, 252);
        doc.roundedRect(margin, y - 14, pageWidth - margin * 2, 100, 8, 8, 'FD');

        addText(`Session ${index + 1}`, margin + 16, 11, 'bold', [15, 23, 42]);
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

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <AnimatePresence>
        {saveDoneMessage && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[80] flex items-center justify-center bg-slate-950/55 backdrop-blur-sm px-6"
          >
            <motion.div
              initial={{ scale: 0.82, y: 18 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.95, y: -8 }}
              transition={{ type: 'spring', stiffness: 260, damping: 20 }}
              className="w-full max-w-sm rounded-3xl border border-emerald-400/30 bg-slate-900/95 p-8 text-center shadow-2xl shadow-emerald-500/10"
            >
              <motion.div
                initial={{ scale: 0, rotate: -18 }}
                animate={{ scale: 1, rotate: 0 }}
                transition={{ type: 'spring', stiffness: 320, damping: 16, delay: 0.08 }}
                className="mx-auto mb-5 flex h-20 w-20 items-center justify-center rounded-full bg-emerald-400 text-slate-950 shadow-lg shadow-emerald-400/25"
              >
                <CheckCircle size={42} strokeWidth={2.5} />
              </motion.div>
              <motion.h2
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.16 }}
                className="text-2xl font-black text-white"
              >
                {saveDoneMessage}
              </motion.h2>
              <motion.p
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.24 }}
                className="mt-2 text-sm text-slate-300"
              >
                Your workspace details are ready.
              </motion.p>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="bg-slate-900/60 backdrop-blur-xl border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-5 sm:py-6">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
            <div className="flex items-center gap-4">
              <div className="w-14 h-14 rounded-3xl bg-gradient-to-br from-cyan-500 via-indigo-500 to-sky-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
                <Sparkles className="text-white" size={24} />
              </div>
              <div>
                <h1 className="text-2xl sm:text-3xl font-black text-white">Analysis Workspace</h1>
                <p className="text-slate-400 text-sm">
                  Profile, forecasting history, and collaboration settings in one place.
                </p>
              </div>
            </div>

            <div className="flex flex-col items-stretch gap-3 lg:items-end">
              <button
                onClick={() => navigate('/dashboard')}
                className="inline-flex items-center justify-center gap-2 rounded-2xl bg-slate-800/80 hover:bg-slate-700/80 px-5 py-3 text-white transition-all"
              >
                <ArrowLeft size={16} />
                Back to Dashboard
              </button>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-3">
              <div className="rounded-2xl border border-white/10 bg-slate-900/70 px-4 py-3">
                <p className="text-[11px] uppercase tracking-widest text-slate-500">Runs</p>
                <p className="text-xl sm:text-2xl font-black text-white">{analyticsSummary.totalRuns}</p>
              </div>
              <div className="rounded-2xl border border-white/10 bg-slate-900/70 px-4 py-3">
                <p className="text-[11px] uppercase tracking-widest text-slate-500">Latest</p>
                <p className="text-xl sm:text-2xl font-black text-white">{Math.round(analyticsSummary.latestPrediction).toLocaleString()}</p>
              </div>
              <div className="rounded-2xl border border-white/10 bg-slate-900/70 px-4 py-3">
                <p className="text-[11px] uppercase tracking-widest text-slate-500">Avg Next Hour</p>
                <p className="text-xl sm:text-2xl font-black text-white">{Math.round(analyticsSummary.averageNextHour).toLocaleString()}</p>
              </div>
              <div className="rounded-2xl border border-white/10 bg-slate-900/70 px-4 py-3">
                <p className="text-[11px] uppercase tracking-widest text-slate-500">Fallback Rate</p>
                <p className="text-xl sm:text-2xl font-black text-white">{analyticsSummary.fallbackRate.toFixed(0)}%</p>
              </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-6 sm:py-8">
        <div className="mb-8 flex gap-2 overflow-x-auto pb-2 no-scrollbar sm:flex-wrap">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex shrink-0 items-center gap-2 sm:gap-3 px-4 sm:px-5 py-3 rounded-xl transition-all ${
                activeTab === tab.id
                  ? 'bg-cyan-500 text-slate-950 shadow-lg shadow-cyan-500/20'
                  : 'bg-slate-800/60 text-slate-300 hover:bg-slate-700/60'
              }`}
            >
              <tab.icon size={18} />
              <span className="font-semibold whitespace-nowrap">{tab.label}</span>
            </button>
          ))}
        </div>

        <AnimatePresence mode="wait">
          {activeTab === 'profile' && (
            <motion.div
              key="profile"
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -18 }}
              className="space-y-6"
            >
              {isNewUser && (
                <div className="bg-amber-500/10 border border-amber-500/30 rounded-2xl p-5">
                  <div className="flex items-center gap-3 mb-2">
                    <AlertTriangle className="text-amber-400" size={22} />
                    <h3 className="text-lg font-bold text-amber-300">Complete your analyst setup</h3>
                  </div>
                  <p className="text-slate-300 text-sm">
                    Fill out your profile and workspace details so your forecast history and collaboration link stay organized.
                  </p>
                </div>
              )}

              {inviteContext && (
                <div className="bg-cyan-500/10 border border-cyan-400/20 rounded-2xl p-5">
                  <div className="flex items-center gap-3 mb-2">
                    <Users className="text-cyan-300" size={22} />
                    <h3 className="text-lg font-bold text-cyan-200">Workspace invite detected</h3>
                  </div>
                  <p className="text-slate-300 text-sm">
                    You were invited to join <span className="font-semibold text-white">{inviteContext.workspaceName || 'a LoadIQ workspace'}</span>
                    {inviteContext.invitedBy ? ` by ${inviteContext.invitedBy}` : ''}. These fields are pre-filled below.
                  </p>
                </div>
              )}

              <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
                <div className="xl:col-span-2 rounded-3xl border border-white/10 bg-slate-900/65 p-4 sm:p-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                    <div>
                      <label className="block text-sm font-medium text-slate-300 mb-2">Full Name *</label>
                      <input
                        type="text"
                        value={formData.fullName}
                        onChange={(e) => handleInputChange('fullName', e.target.value)}
                        className="w-full px-4 py-3 bg-slate-800/70 border border-slate-700 rounded-xl text-white focus:border-cyan-400 focus:outline-none"
                        placeholder="Enter your full name"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-slate-300 mb-2">Email *</label>
                      <input
                        type="email"
                        value={formData.email}
                        onChange={(e) => handleInputChange('email', e.target.value)}
                        className="w-full px-4 py-3 bg-slate-800/70 border border-slate-700 rounded-xl text-white focus:border-cyan-400 focus:outline-none"
                        placeholder="Enter your email"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-slate-300 mb-2">Phone</label>
                      <input
                        type="tel"
                        value={formData.phone}
                        onChange={(e) => handleInputChange('phone', e.target.value)}
                        className="w-full px-4 py-3 bg-slate-800/70 border border-slate-700 rounded-xl text-white focus:border-cyan-400 focus:outline-none"
                        placeholder="Phone number"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-slate-300 mb-2">Location</label>
                      <input
                        type="text"
                        value={formData.location}
                        onChange={(e) => handleInputChange('location', e.target.value)}
                        className="w-full px-4 py-3 bg-slate-800/70 border border-slate-700 rounded-xl text-white focus:border-cyan-400 focus:outline-none"
                        placeholder="City, Country"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-slate-300 mb-2">Organization</label>
                      <input
                        type="text"
                        value={formData.organization}
                        onChange={(e) => handleInputChange('organization', e.target.value)}
                        className="w-full px-4 py-3 bg-slate-800/70 border border-slate-700 rounded-xl text-white focus:border-cyan-400 focus:outline-none"
                        placeholder="Company or institution"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-slate-300 mb-2">Workspace Name</label>
                      <input
                        type="text"
                        value={formData.workspaceName}
                        onChange={(e) => handleInputChange('workspaceName', e.target.value)}
                        className="w-full px-4 py-3 bg-slate-800/70 border border-slate-700 rounded-xl text-white focus:border-cyan-400 focus:outline-none"
                        placeholder="Grid Operations Team"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-slate-300 mb-2">Role</label>
                      <select
                        value={formData.role}
                        onChange={(e) => handleInputChange('role', e.target.value)}
                        className="w-full px-4 py-3 bg-slate-800/70 border border-slate-700 rounded-xl text-white focus:border-cyan-400 focus:outline-none"
                      >
                        <option value="">Select your role</option>
                        <option value="analyst">Energy Analyst</option>
                        <option value="engineer">Power Engineer</option>
                        <option value="manager">Grid Manager</option>
                        <option value="researcher">Researcher</option>
                        <option value="student">Student</option>
                        <option value="other">Other</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-slate-300 mb-2">Experience Level</label>
                      <select
                        value={formData.experience}
                        onChange={(e) => handleInputChange('experience', e.target.value)}
                        className="w-full px-4 py-3 bg-slate-800/70 border border-slate-700 rounded-xl text-white focus:border-cyan-400 focus:outline-none"
                      >
                        <option value="beginner">Beginner</option>
                        <option value="intermediate">Intermediate</option>
                        <option value="advanced">Advanced</option>
                        <option value="expert">Expert</option>
                      </select>
                    </div>
                  </div>
                </div>

                <div className="rounded-3xl border border-white/10 bg-slate-900/65 p-4 sm:p-6 space-y-5">
                  <div>
                    <p className="text-xs uppercase tracking-[0.2em] text-cyan-300 mb-2">Workspace Snapshot</p>
                    <h3 className="text-2xl font-black text-white">
                      {formData.workspaceName || formData.organization || 'Set up your workspace'}
                    </h3>
                    <p className="text-slate-400 text-sm mt-2">
                      Keep profile details, history, and collaboration context aligned for repeat analysis sessions.
                    </p>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <div className="rounded-2xl bg-slate-800/70 p-4 border border-white/5">
                      <Building2 className="text-cyan-300 mb-3" size={18} />
                      <p className="text-xs uppercase tracking-wider text-slate-500">Org</p>
                      <p className="text-white font-semibold mt-1 break-words">{formData.organization || 'Not set'}</p>
                    </div>
                    <div className="rounded-2xl bg-slate-800/70 p-4 border border-white/5">
                      <Activity className="text-cyan-300 mb-3" size={18} />
                      <p className="text-xs uppercase tracking-wider text-slate-500">Role</p>
                      <p className="text-white font-semibold mt-1 capitalize">{formData.role || 'Not set'}</p>
                    </div>
                  </div>
                  <button
                    onClick={saveProfile}
                    disabled={isLoading || !formData.fullName || !formData.email}
                    className="w-full flex items-center justify-center gap-3 px-6 py-4 bg-cyan-500 hover:bg-cyan-400 disabled:bg-slate-700 text-slate-950 font-bold rounded-2xl transition-all"
                  >
                    {isLoading ? (
                      <div className="w-5 h-5 border-2 border-slate-950/30 border-t-slate-950 rounded-full animate-spin" />
                    ) : (
                      <Save size={18} />
                    )}
                    {isNewUser ? 'Complete Workspace Setup' : 'Save Profile'}
                  </button>
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 'history' && (
            <motion.div
              key="history"
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -18 }}
              className="space-y-6"
            >
              <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                <div>
                  <h3 className="text-2xl font-black text-white">Prediction History</h3>
                  <p className="text-slate-400 text-sm">Stored model2 forecast sessions with horizon, source, and fallback context.</p>
                </div>
                <button
                  onClick={exportHistory}
                  className="inline-flex items-center gap-2 px-5 py-3 rounded-xl bg-slate-800/70 hover:bg-slate-700/70 text-white transition-all"
                >
                  <Download size={16} />
                  Export History
                </button>
              </div>

              {historyLoading ? (
                <div className="rounded-3xl border border-white/10 bg-slate-900/65 p-8 text-center">
                  <div className="w-10 h-10 border-2 border-cyan-400/30 border-t-cyan-400 rounded-full animate-spin mx-auto mb-4" />
                  <p className="text-slate-300">Loading forecast history...</p>
                </div>
              ) : analyticsHistory.length === 0 ? (
                <div className="rounded-3xl border border-white/10 bg-slate-900/65 p-10 text-center">
                  <BarChart3 className="mx-auto text-slate-500 mb-4" size={48} />
                  <p className="text-slate-300 font-medium">No analytics history yet</p>
                  <p className="text-slate-500 text-sm mt-2">Run forecasts in Load Predictor and they will appear here automatically.</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {analyticsHistory.map((item) => (
                    <div key={item.id} className="rounded-3xl border border-white/10 bg-slate-900/65 p-5">
                      <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
                        <div className="space-y-3">
                          <div className="flex items-center gap-3">
                            <div className="w-11 h-11 rounded-2xl bg-cyan-500/15 border border-cyan-400/20 flex items-center justify-center">
                              <TrendingUp className="text-cyan-300" size={18} />
                            </div>
                            <div>
                              <p className="text-white font-semibold">Load Forecast Session</p>
                              <p className="text-slate-400 text-sm">{new Date(item.timestamp).toLocaleString()}</p>
                            </div>
                          </div>
                          <div className="flex flex-wrap gap-2">
                            <span className="px-3 py-1 rounded-full bg-slate-800 text-slate-200 text-xs">
                              Horizon: {item.forecastHorizon}h
                            </span>
                            <span className="px-3 py-1 rounded-full bg-slate-800 text-slate-200 text-xs">
                              Source: {item.predictionSource}
                            </span>
                            <span className={`px-3 py-1 rounded-full text-xs ${
                              item.fallbackUsed ? 'bg-amber-500/15 text-amber-200' : 'bg-emerald-500/15 text-emerald-200'
                            }`}>
                              {item.fallbackUsed ? 'Fallback used' : 'Model output'}
                            </span>
                          </div>
                        </div>

                        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 min-w-0 lg:min-w-[420px]">
                          <div className="rounded-2xl bg-slate-800/70 p-4 border border-white/5">
                            <p className="text-xs uppercase tracking-wider text-slate-500">Next Hour</p>
                            <p className="text-white font-bold mt-2">{Math.round(item.nextHourPrediction).toLocaleString()} MW</p>
                          </div>
                          <div className="rounded-2xl bg-slate-800/70 p-4 border border-white/5">
                            <p className="text-xs uppercase tracking-wider text-slate-500">Latest Step</p>
                            <p className="text-white font-bold mt-2">{Math.round(item.latestPrediction).toLocaleString()} MW</p>
                          </div>
                          <div className="rounded-2xl bg-slate-800/70 p-4 border border-white/5">
                            <p className="text-xs uppercase tracking-wider text-slate-500">Peak</p>
                            <p className="text-white font-bold mt-2">{Math.round(item.statistics.peak || 0).toLocaleString()} MW</p>
                          </div>
                          <div className="rounded-2xl bg-slate-800/70 p-4 border border-white/5">
                            <p className="text-xs uppercase tracking-wider text-slate-500">Delta</p>
                            <p className={`font-bold mt-2 ${
                              item.statistics.delta > 0 ? 'text-rose-300' : item.statistics.delta < 0 ? 'text-emerald-300' : 'text-white'
                            }`}>
                              {Math.round(item.statistics.delta || 0).toLocaleString()} MW
                            </p>
                          </div>
                        </div>
                      </div>

                      {item.predictionSeries.length > 0 && (
                        <div className="mt-5 grid grid-cols-2 md:grid-cols-4 xl:grid-cols-6 gap-3">
                          {item.predictionSeries.slice(0, 6).map((entry) => (
                            <div key={`${item.id}-${entry.step}`} className="rounded-2xl border border-white/5 bg-slate-950/50 p-3">
                              <p className="text-[11px] uppercase tracking-wider text-slate-500">Step {entry.step}</p>
                              <p className="text-white font-semibold mt-2">{Math.round(entry.value).toLocaleString()} MW</p>
                            </div>
                          ))}
                        </div>
                      )}

                      {item.fallbackReason && (
                        <p className="mt-4 text-sm text-amber-200">
                          Fallback reason: {item.fallbackReason}
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </motion.div>
          )}

          {activeTab === 'workspace' && (
            <motion.div
              key="workspace"
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -18 }}
              className="grid grid-cols-1 xl:grid-cols-2 gap-6"
            >
              <div className="rounded-3xl border border-white/10 bg-slate-900/65 p-6 space-y-5">
                <div>
                  <p className="text-xs uppercase tracking-[0.2em] text-cyan-300 mb-2">Invite Teammates</p>
                  <h3 className="text-2xl font-black text-white">Share your analysis workspace</h3>
                  <p className="text-slate-400 text-sm mt-2">
                    Send a registration link that carries your workspace and organization context so new teammates can join the same analysis setup.
                  </p>
                </div>
                <div className="rounded-2xl bg-slate-950/50 border border-white/5 p-4">
                  <div className="flex items-start gap-3">
                    <Link2 className="text-cyan-300 mt-1" size={18} />
                    <p className="text-sm text-slate-200 break-all">{workspaceInviteLink}</p>
                  </div>
                </div>
                <div className="flex flex-wrap gap-3">
                  <button
                    onClick={copyInviteLink}
                    className="inline-flex items-center gap-2 px-5 py-3 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold transition-all"
                  >
                    <Copy size={16} />
                    Copy Link
                  </button>
                  <button
                    onClick={shareInviteLink}
                    className="inline-flex items-center gap-2 px-5 py-3 rounded-xl bg-slate-800 hover:bg-slate-700 text-white transition-all"
                  >
                    <Share2 size={16} />
                    Share Invite
                  </button>
                </div>
              </div>

              <div className="rounded-3xl border border-white/10 bg-slate-900/65 p-6 space-y-5">
                <div>
                  <p className="text-xs uppercase tracking-[0.2em] text-cyan-300 mb-2">Workspace State</p>
                  <h3 className="text-2xl font-black text-white">{formData.workspaceName || 'No workspace name yet'}</h3>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="rounded-2xl bg-slate-800/70 p-4 border border-white/5">
                    <Building2 className="text-cyan-300 mb-3" size={18} />
                    <p className="text-xs uppercase tracking-wider text-slate-500">Organization</p>
                    <p className="text-white font-semibold mt-2">{formData.organization || 'Not set'}</p>
                  </div>
                  <div className="rounded-2xl bg-slate-800/70 p-4 border border-white/5">
                    <Users className="text-cyan-300 mb-3" size={18} />
                    <p className="text-xs uppercase tracking-wider text-slate-500">Invited By</p>
                    <p className="text-white font-semibold mt-2">{inviteContext?.invitedBy || formData.fullName || 'You'}</p>
                  </div>
                  <div className="rounded-2xl bg-slate-800/70 p-4 border border-white/5">
                    <Clock className="text-cyan-300 mb-3" size={18} />
                    <p className="text-xs uppercase tracking-wider text-slate-500">Average Horizon</p>
                    <p className="text-white font-semibold mt-2">{analyticsSummary.averageHorizon.toFixed(1)} hours</p>
                  </div>
                  <div className="rounded-2xl bg-slate-800/70 p-4 border border-white/5">
                    <Zap className="text-cyan-300 mb-3" size={18} />
                    <p className="text-xs uppercase tracking-wider text-slate-500">Saved Runs</p>
                    <p className="text-white font-semibold mt-2">{analyticsSummary.totalRuns}</p>
                  </div>
                </div>
                <div className="rounded-2xl bg-cyan-500/10 border border-cyan-400/20 p-4">
                  <p className="text-sm text-slate-200">
                    Teammates opening the invite link land on registration with your workspace context preloaded, then continue into the shared analysis setup after sign-up.
                  </p>
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 'settings' && (
            <motion.div
              key="settings"
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -18 }}
              className="space-y-6"
            >
              <div className="rounded-3xl border border-white/10 bg-slate-900/65 p-6">
                <h3 className="text-xl font-bold text-white mb-4">Notification Settings</h3>
                <div className="flex items-center justify-between gap-6">
                  <div>
                    <p className="text-slate-200 font-medium">Browser Notifications</p>
                    <p className="text-slate-500 text-sm">Receive updates when long-running analytics complete.</p>
                  </div>
                  <button
                    onClick={toggleNotifications}
                    className={`w-14 h-7 rounded-full transition-all ${formData.notifications ? 'bg-cyan-500' : 'bg-slate-600'}`}
                  >
                    <div className={`w-6 h-6 bg-white rounded-full transition-all ${formData.notifications ? 'translate-x-7' : 'translate-x-1'}`} />
                  </button>
                </div>
              </div>

              <div className="rounded-3xl border border-white/10 bg-slate-900/65 p-6">
                <h3 className="text-xl font-bold text-white mb-4">Support</h3>
                <p className="text-slate-300 mb-4">
                  Report a workspace or analytics issue and keep the rest of the system running while we investigate.
                </p>
                <button
                  onClick={reportIssue}
                  className="inline-flex items-center gap-3 px-6 py-3 rounded-xl bg-rose-600 hover:bg-rose-500 text-white transition-all"
                >
                  <AlertTriangle size={18} />
                  Report Issue
                </button>
              </div>
            </motion.div>
          )}

          {activeTab === 'notifications' && (
            <motion.div
              key="notifications"
              initial={{ opacity: 0, y: 18 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -18 }}
              className="space-y-6"
            >
              <div className="rounded-3xl border border-white/10 bg-slate-900/65 p-6">
                <div className="flex items-center gap-3 mb-4">
                  <Bell className="text-cyan-300" size={22} />
                  <h3 className="text-xl font-bold text-white">Notification Center</h3>
                </div>
                <p className="text-slate-300 mb-4">
                  Forecast completion alerts use your browser permission state and your saved workspace preferences.
                </p>
                <div className="rounded-2xl bg-slate-950/50 p-4 border border-white/5">
                  <p className="text-sm text-slate-300">
                    Permission Status:{' '}
                    <span className="font-semibold text-white">
                      {'Notification' in window
                        ? Notification.permission === 'granted'
                          ? 'Granted'
                          : Notification.permission === 'denied'
                            ? 'Denied'
                            : 'Not requested'
                        : 'Not supported'}
                    </span>
                  </p>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <AnimatePresence>
        {notifications.map((notification) => (
          <motion.div
            key={notification.id}
            initial={{ opacity: 0, y: -30 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -30 }}
            className={`fixed top-4 right-4 z-50 p-4 rounded-2xl shadow-lg ${
              notification.type === 'success'
                ? 'bg-emerald-600 text-white'
                : notification.type === 'error'
                  ? 'bg-rose-600 text-white'
                  : 'bg-sky-600 text-white'
            }`}
          >
            <div className="flex items-center gap-3">
              {notification.type === 'success' ? (
                <CheckCircle size={18} />
              ) : notification.type === 'error' ? (
                <X size={18} />
              ) : (
                <Bell size={18} />
              )}
              <p className="font-medium">{notification.message}</p>
            </div>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
};

export default UserProfile;
