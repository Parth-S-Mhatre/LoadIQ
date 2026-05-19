import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { motion, AnimatePresence } from 'framer-motion';
import {
  User, Settings, History, Bell, Save, AlertTriangle,
  CheckCircle, X, Mail, Phone, MapPin, Calendar,
  TrendingUp, BarChart3, Clock, Zap
} from 'lucide-react';

const UserProfile = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [isNewUser, setIsNewUser] = useState(false);
  const [formData, setFormData] = useState({
    fullName: user?.displayName || '',
    email: user?.email || '',
    phone: '',
    location: '',
    organization: '',
    role: '',
    experience: '',
    notifications: true
  });
  const [analyticsHistory, setAnalyticsHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('profile');
  const [notifications, setNotifications] = useState([]);

  useEffect(() => {
    // Check if user is new (no profile data in localStorage)
    const userProfile = localStorage.getItem(`userProfile_${user?.uid}`);
    if (!userProfile) {
      setIsNewUser(true);
    } else {
      const savedData = JSON.parse(userProfile);
      setFormData(prev => ({ ...prev, ...savedData }));
    }

    // Load analytics history
    const history = JSON.parse(localStorage.getItem(`analyticsHistory_${user?.uid}`) || '[]');
    setAnalyticsHistory(history);

    // Request notification permission
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }
  }, [user]);

  const handleInputChange = (field, value) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const saveProfile = async () => {
    setIsLoading(true);
    try {
      localStorage.setItem(`userProfile_${user?.uid}`, JSON.stringify(formData));
      setIsNewUser(false);
      addNotification('Profile saved successfully!', 'success');
      if (isNewUser) {
        navigate('/dashboard');
      }
    } catch (error) {
      addNotification('Failed to save profile', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  const addNotification = (message, type = 'info') => {
    const id = Date.now();
    setNotifications(prev => [...prev, { id, message, type }]);
    setTimeout(() => setNotifications(prev => prev.filter(n => n.id !== id)), 5000);
  };

  const reportIssue = () => {
    // Simulate reporting issue
    addNotification('Issue reported successfully. Our team will contact you soon.', 'success');
  };

  const toggleNotifications = () => {
    setFormData(prev => ({ ...prev, notifications: !prev.notifications }));
  };

  // Browser notification when leaving app
  useEffect(() => {
    const handleBeforeUnload = () => {
      if (formData.notifications && 'Notification' in window && Notification.permission === 'granted') {
        new Notification('LoadIQ Analytics', {
          body: 'Your analytics session has been completed. Check back for results!',
          icon: '/favicon.ico'
        });
      }
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [formData.notifications]);

  const tabs = [
    { id: 'profile', label: 'Profile', icon: User },
    { id: 'settings', label: 'Settings', icon: Settings },
    { id: 'history', label: 'History', icon: History },
    { id: 'notifications', label: 'Notifications', icon: Bell }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      {/* Header */}
      <div className="bg-slate-900/50 backdrop-blur-xl border-b border-white/10">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                <User className="text-white" size={24} />
              </div>
              <div>
                <h1 className="text-2xl font-black text-white">User Profile</h1>
                <p className="text-slate-400 text-sm">Manage your LoadIQ account settings</p>
              </div>
            </div>
            <button
              onClick={() => navigate('/dashboard')}
              className="px-6 py-3 bg-slate-800 hover:bg-slate-700 text-white rounded-xl transition-all"
            >
              Back to Dashboard
            </button>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-8">
        {/* Tabs */}
        <div className="flex gap-2 mb-8">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-3 px-6 py-3 rounded-xl transition-all ${
                activeTab === tab.id
                  ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-500/30'
                  : 'bg-slate-800/50 text-slate-400 hover:bg-slate-700/50 hover:text-white'
              }`}
            >
              <tab.icon size={20} />
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <AnimatePresence mode="wait">
          {activeTab === 'profile' && (
            <motion.div
              key="profile"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              {isNewUser && (
                <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <AlertTriangle className="text-amber-500" size={24} />
                    <h3 className="text-lg font-bold text-amber-500">Complete Your Profile</h3>
                  </div>
                  <p className="text-slate-300">
                    Please fill in the mandatory fields below to continue using LoadIQ.
                  </p>
                </div>
              )}

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      Full Name *
                    </label>
                    <input
                      type="text"
                      value={formData.fullName}
                      onChange={(e) => handleInputChange('fullName', e.target.value)}
                      className="w-full px-4 py-3 bg-slate-800/50 border border-slate-600 rounded-xl text-white placeholder-slate-400 focus:border-indigo-500 focus:outline-none"
                      placeholder="Enter your full name"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      Email *
                    </label>
                    <input
                      type="email"
                      value={formData.email}
                      onChange={(e) => handleInputChange('email', e.target.value)}
                      className="w-full px-4 py-3 bg-slate-800/50 border border-slate-600 rounded-xl text-white placeholder-slate-400 focus:border-indigo-500 focus:outline-none"
                      placeholder="Enter your email"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      Phone
                    </label>
                    <input
                      type="tel"
                      value={formData.phone}
                      onChange={(e) => handleInputChange('phone', e.target.value)}
                      className="w-full px-4 py-3 bg-slate-800/50 border border-slate-600 rounded-xl text-white placeholder-slate-400 focus:border-indigo-500 focus:outline-none"
                      placeholder="Enter your phone number"
                    />
                  </div>
                </div>

                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      Location
                    </label>
                    <input
                      type="text"
                      value={formData.location}
                      onChange={(e) => handleInputChange('location', e.target.value)}
                      className="w-full px-4 py-3 bg-slate-800/50 border border-slate-600 rounded-xl text-white placeholder-slate-400 focus:border-indigo-500 focus:outline-none"
                      placeholder="City, Country"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      Organization
                    </label>
                    <input
                      type="text"
                      value={formData.organization}
                      onChange={(e) => handleInputChange('organization', e.target.value)}
                      className="w-full px-4 py-3 bg-slate-800/50 border border-slate-600 rounded-xl text-white placeholder-slate-400 focus:border-indigo-500 focus:outline-none"
                      placeholder="Company or organization"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      Role
                    </label>
                    <select
                      value={formData.role}
                      onChange={(e) => handleInputChange('role', e.target.value)}
                      className="w-full px-4 py-3 bg-slate-800/50 border border-slate-600 rounded-xl text-white focus:border-indigo-500 focus:outline-none"
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
                </div>
              </div>

              <div className="flex justify-end">
                <button
                  onClick={saveProfile}
                  disabled={isLoading || !formData.fullName || !formData.email}
                  className="flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 disabled:from-slate-600 disabled:to-slate-600 text-white font-bold rounded-xl transition-all shadow-lg shadow-indigo-500/30 disabled:shadow-none"
                >
                  {isLoading ? (
                    <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  ) : (
                    <Save size={20} />
                  )}
                  {isNewUser ? 'Complete Setup' : 'Save Profile'}
                </button>
              </div>
            </motion.div>
          )}

          {activeTab === 'settings' && (
            <motion.div
              key="settings"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              <div className="bg-slate-800/50 rounded-xl p-6">
                <h3 className="text-lg font-bold text-white mb-4">Notification Settings</h3>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-slate-300 font-medium">Browser Notifications</p>
                    <p className="text-slate-500 text-sm">Receive notifications when analytics complete</p>
                  </div>
                  <button
                    onClick={toggleNotifications}
                    className={`w-12 h-6 rounded-full transition-all ${
                      formData.notifications ? 'bg-indigo-600' : 'bg-slate-600'
                    }`}
                  >
                    <div className={`w-5 h-5 bg-white rounded-full transition-all ${
                      formData.notifications ? 'translate-x-6' : 'translate-x-1'
                    }`} />
                  </button>
                </div>
              </div>

              <div className="bg-slate-800/50 rounded-xl p-6">
                <h3 className="text-lg font-bold text-white mb-4">Troubleshooting</h3>
                <p className="text-slate-300 mb-4">
                  Having issues with LoadIQ? Report a problem and our team will help you resolve it.
                </p>
                <button
                  onClick={reportIssue}
                  className="flex items-center gap-3 px-6 py-3 bg-red-600 hover:bg-red-500 text-white rounded-xl transition-all"
                >
                  <AlertTriangle size={20} />
                  Report Issue
                </button>
              </div>
            </motion.div>
          )}

          {activeTab === 'history' && (
            <motion.div
              key="history"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-4"
            >
              <h3 className="text-lg font-bold text-white">Analytics History</h3>
              {analyticsHistory.length === 0 ? (
                <div className="bg-slate-800/50 rounded-xl p-8 text-center">
                  <BarChart3 className="mx-auto text-slate-500 mb-4" size={48} />
                  <p className="text-slate-400">No analytics history yet</p>
                  <p className="text-slate-500 text-sm">Run some predictions to see your history here</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {analyticsHistory.map((item, index) => (
                    <div key={index} className="bg-slate-800/50 rounded-xl p-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <TrendingUp className="text-green-500" size={20} />
                          <div>
                            <p className="text-white font-medium">Load Prediction</p>
                            <p className="text-slate-400 text-sm">
                              {new Date(item.timestamp).toLocaleString()}
                            </p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="text-white font-bold">{item.prediction} MW</p>
                          <p className="text-slate-400 text-sm">Predicted Load</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </motion.div>
          )}

          {activeTab === 'notifications' && (
            <motion.div
              key="notifications"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-4"
            >
              <h3 className="text-lg font-bold text-white">Notification Center</h3>
              <div className="bg-slate-800/50 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <Bell className="text-indigo-500" size={24} />
                  <h4 className="text-white font-medium">Analytics Completion</h4>
                </div>
                <p className="text-slate-300 mb-4">
                  You'll receive a browser notification when your analytics complete, even if you're not actively using the app.
                </p>
                <div className="bg-slate-900/50 rounded-lg p-4">
                  <p className="text-slate-400 text-sm">
                    <strong>Permission Status:</strong> {
                      'Notification' in window
                        ? Notification.permission === 'granted'
                          ? 'Granted'
                          : Notification.permission === 'denied'
                          ? 'Denied'
                          : 'Not requested'
                        : 'Not supported'
                    }
                  </p>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Notifications */}
      <AnimatePresence>
        {notifications.map((notification) => (
          <motion.div
            key={notification.id}
            initial={{ opacity: 0, y: -50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -50 }}
            className={`fixed top-4 right-4 z-50 p-4 rounded-xl shadow-lg ${
              notification.type === 'success'
                ? 'bg-green-600 text-white'
                : notification.type === 'error'
                ? 'bg-red-600 text-white'
                : 'bg-blue-600 text-white'
            }`}
          >
            <div className="flex items-center gap-3">
              {notification.type === 'success' ? (
                <CheckCircle size={20} />
              ) : notification.type === 'error' ? (
                <X size={20} />
              ) : (
                <Bell size={20} />
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