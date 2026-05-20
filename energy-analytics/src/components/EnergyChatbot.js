import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MessageSquare, X, Send, Cpu, User, Loader2 } from 'lucide-react';
import { ChatbotService } from '../services/ChatbotService';

const readStoredJson = (key) => {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : null;
  } catch (error) {
    return null;
  }
};

const EnergyChatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { id: 1, text: "Hello! I'm LoadIQ's advanced Energy AI. I can help analyze grid patterns across Germany, UK, USA, and India. How can I assist you today?", sender: 'ai' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    if (isOpen) {
      scrollToBottom();
    }
  }, [messages, isOpen]);

  const getAppContext = () => {
    const pendingInviteContext = readStoredJson('pendingInviteContext');
    const guestHistory = readStoredJson('analyticsHistory_guest') || [];
    const latestPredictionContext = readStoredJson('loadiq_latest_prediction_context');
    const dashboardOverview = readStoredJson('loadiq_overview_context');
    const advancedAnalytics = readStoredJson('loadiq_advanced_analytics_context');

    let latestPrediction = latestPredictionContext || guestHistory[0] || null;
    let workspaceProfile = null;

    for (let index = 0; index < localStorage.length; index += 1) {
      const key = localStorage.key(index);

      if (!key) {
        continue;
      }

      if (!latestPrediction && key.startsWith('analyticsHistory_') && key !== 'analyticsHistory_guest') {
        const userHistory = readStoredJson(key);
        if (Array.isArray(userHistory) && userHistory.length) {
          latestPrediction = userHistory[0];
        }
      }

      if (!workspaceProfile && key.startsWith('userProfile_')) {
        workspaceProfile = readStoredJson(key);
      }
    }

    return {
      workspaceName: workspaceProfile?.workspaceName || pendingInviteContext?.workspaceName || '',
      organization: workspaceProfile?.organization || pendingInviteContext?.organization || '',
      latestPrediction,
      recentHistory: Array.isArray(guestHistory) ? guestHistory.slice(0, 5) : [],
      overview: dashboardOverview,
      advancedAnalytics
    };
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { id: Date.now(), text: input, sender: 'user' };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

     try {
       const context = getAppContext();
       
       // Extract country from context or use a default
       const country = context.country || 'Germany';
       const reply = await ChatbotService.sendMessage(userMessage.text, country, context);
       setMessages(prev => [...prev, { id: Date.now() + 1, text: reply, sender: 'ai' }]);
     } catch (error) {
      setMessages(prev => [...prev, { id: Date.now() + 1, text: error.message || "Failed to connect to the AI service.", sender: 'ai' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 p-4 bg-indigo-600 hover:bg-indigo-500 text-white rounded-full shadow-lg shadow-indigo-600/30 transition-all hover:scale-105 z-50 flex items-center justify-center"
      >
        <MessageSquare size={24} />
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            className="fixed bottom-20 left-3 right-3 h-[70vh] max-h-[550px] md:left-auto md:right-6 md:bottom-24 w-auto md:w-[400px] bg-slate-900 border border-slate-700/50 rounded-2xl shadow-2xl flex flex-col z-50 overflow-hidden"
          >
            {/* Header */}
            <div className="bg-indigo-600 p-4 flex justify-between items-center text-white shadow-md">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-white/20 rounded-full flex items-center justify-center">
                  <Cpu size={18} />
                </div>
                <div>
                  <h3 className="font-bold text-sm">LoadIQ AI Assistant</h3>
                  <div className="flex items-center gap-1.5 mt-0.5">
                    <span className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse"></span>
                    <span className="text-[10px] text-indigo-100 tracking-wider">ONLINE</span>
                  </div>
                </div>
              </div>
              <button onClick={() => setIsOpen(false)} className="text-white/80 hover:text-white transition-colors p-1 rounded-lg hover:bg-white/10">
                <X size={20} />
              </button>
            </div>

            {/* Chat Area */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-slate-950/50">
              {messages.map((msg) => (
                <div key={msg.id} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[85%] rounded-2xl p-3 text-sm shadow-md ${msg.sender === 'user' ? 'bg-indigo-600 text-white rounded-br-sm' : 'bg-slate-800 text-slate-200 border border-slate-700 rounded-bl-sm'}`}>
                    <div className="flex items-center gap-2 mb-1 opacity-70">
                      {msg.sender === 'user' ? <User size={12} /> : <Cpu size={12} />}
                      <span className="text-[10px] uppercase font-bold">{msg.sender === 'user' ? 'You' : 'LoadIQ AI'}</span>
                    </div>
                    {/* Basic Markdown rendering for bolding text to keep it simple */}
                    <div className="leading-relaxed" dangerouslySetInnerHTML={{ __html: msg.text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br/>') }} />
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-slate-800 text-slate-200 border border-slate-700 rounded-2xl rounded-bl-sm p-3 text-sm shadow-md flex items-center gap-2">
                    <Loader2 size={16} className="animate-spin text-indigo-400" />
                    <span className="text-xs text-slate-400">Analyzing...</span>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="p-3 bg-slate-900 border-t border-slate-700/50">
              <div className="relative flex items-center">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                  placeholder="Ask about grids, patterns..."
                  className="w-full bg-slate-800 border border-slate-700 rounded-xl py-3 pl-4 pr-12 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition-all"
                />
                <button
                  onClick={handleSend}
                  disabled={!input.trim() || isLoading}
                  className="absolute right-2 p-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-slate-700 text-white rounded-lg transition-colors flex items-center justify-center"
                >
                  <Send size={16} />
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};

export default EnergyChatbot;
