import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { MessageSquare, X, Send, Cpu, User, Loader2 } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { ChatbotService } from '../services/ChatbotService';
import { EASE, StatusDot } from './ui';

const readStoredJson = (key) => {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : null;
  } catch (error) {
    return null;
  }
};

const buildInitialMessages = (isAuthenticated) => ([
  {
    id: 1,
    text: isAuthenticated
      ? "Hello! I'm LoadIQ's advanced Energy AI. I can help analyze grid patterns across Germany, UK, USA, and India. How can I assist you today?"
      : "Hello! I can answer general energy questions. Sign in to unlock your saved account history, dashboard context, and prediction results.",
    sender: 'ai'
  }
]);

const inlineMarkdownToNodes = (text) => {
  const segments = [];
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*|https?:\/\/[^\s]+|\n)/g;
  let lastIndex = 0;

  let match;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      segments.push(text.slice(lastIndex, match.index));
    }
    segments.push(match[0]);
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < text.length) {
    segments.push(text.slice(lastIndex));
  }

  return segments.filter(Boolean).map((segment, index) => {
    if (segment === '\n') {
      return <br key={`br-${index}`} />;
    }

    if (segment.startsWith('**') && segment.endsWith('**')) {
      return <strong key={`strong-${index}`}>{segment.slice(2, -2)}</strong>;
    }

    if (segment.startsWith('*') && segment.endsWith('*')) {
      return <em key={`em-${index}`}>{segment.slice(1, -1)}</em>;
    }

    if (segment.startsWith('`') && segment.endsWith('`')) {
      return (
        <code key={`code-${index}`} className="rounded bg-ink/10 px-1.5 py-0.5 font-mono text-[0.85em] text-current">
          {segment.slice(1, -1)}
        </code>
      );
    }

    if (/^https?:\/\/[^\s]+$/.test(segment)) {
      return (
        <a
          key={`link-${index}`}
          href={segment}
          target="_blank"
          rel="noreferrer"
          className="text-moss underline decoration-signal-400 decoration-2 underline-offset-2 hover:text-ink"
        >
          {segment}
        </a>
      );
    }

    return <React.Fragment key={`text-${index}`}>{segment}</React.Fragment>;
  });
};

const parseMessageBlocks = (rawText) => {
  const normalizedText = String(rawText || '').replace(/\r\n/g, '\n').trim();
  if (!normalizedText) {
    return [];
  }

  const lines = normalizedText.split('\n');
  const blocks = [];
  let paragraphBuffer = [];
  let listBuffer = null;
  let codeBuffer = null;
  let quoteBuffer = [];

  const flushParagraph = () => {
    if (paragraphBuffer.length) {
      blocks.push({ type: 'paragraph', text: paragraphBuffer.join(' ') });
      paragraphBuffer = [];
    }
  };

  const flushList = () => {
    if (listBuffer && listBuffer.items.length) {
      blocks.push({ type: 'list', ordered: listBuffer.ordered, items: listBuffer.items });
    }
    listBuffer = null;
  };

  const flushQuote = () => {
    if (quoteBuffer.length) {
      blocks.push({ type: 'quote', text: quoteBuffer.join(' ') });
      quoteBuffer = [];
    }
  };

  const flushCode = () => {
    if (codeBuffer) {
      blocks.push({ type: 'code', language: codeBuffer.language, text: codeBuffer.lines.join('\n') });
    }
    codeBuffer = null;
  };

  lines.forEach((line) => {
    const trimmed = line.trimEnd();
    const headingMatch = trimmed.match(/^(#{1,6})\s+(.*)$/);
    const unorderedMatch = trimmed.match(/^[-*•]\s+(.*)$/);
    const orderedMatch = trimmed.match(/^\d+[.)]\s+(.*)$/);
    const quoteMatch = trimmed.match(/^>\s?(.*)$/);
    const dividerMatch = trimmed.match(/^(\*{3,}|-{3,}|_{3,})$/);
    const codeFenceMatch = trimmed.match(/^```([\w-]*)\s*$/);

    if (codeFenceMatch) {
      if (codeBuffer) {
        flushCode();
      } else {
        flushParagraph();
        flushList();
        flushQuote();
        codeBuffer = { language: codeFenceMatch[1] || '', lines: [] };
      }
      return;
    }

    if (codeBuffer) {
      codeBuffer.lines.push(line);
      return;
    }

    if (!trimmed) {
      flushParagraph();
      flushList();
      flushQuote();
      return;
    }

    if (dividerMatch) {
      flushParagraph();
      flushList();
      flushQuote();
      return;
    }

    if (headingMatch) {
      flushParagraph();
      flushList();
      flushQuote();
      blocks.push({
        type: 'heading',
        level: headingMatch[1].length,
        text: headingMatch[2].replace(/\s*\*{2,}\s*/g, ' ').trim()
      });
      return;
    }

    if (unorderedMatch || orderedMatch) {
      flushParagraph();
      flushQuote();
      if (!listBuffer) {
        listBuffer = {
          ordered: Boolean(orderedMatch),
          items: []
        };
      }
      listBuffer.items.push((unorderedMatch || orderedMatch)[1]);
      return;
    }

    if (quoteMatch) {
      flushParagraph();
      flushList();
      quoteBuffer.push(quoteMatch[1]);
      return;
    }

    flushList();
    flushQuote();
    paragraphBuffer.push(trimmed);
  });

  flushParagraph();
  flushList();
  flushQuote();
  flushCode();

  return blocks;
};

const MessageContent = ({ text }) => {
  const blocks = parseMessageBlocks(text);

  if (!blocks.length) {
    return null;
  }

  return (
    <div className="space-y-2.5 leading-relaxed">
      {blocks.map((block, index) => {
        if (block.type === 'heading') {
          const headingSizes = {
            1: 'text-[0.95rem]',
            2: 'text-[0.9rem]',
            3: 'text-[0.875rem]'
          };

          return (
            <div key={`heading-${index}`} className={`font-semibold text-ink ${headingSizes[block.level] || 'text-[0.85rem]'}`}>
              {inlineMarkdownToNodes(block.text)}
            </div>
          );
        }

        if (block.type === 'list') {
          return (
            <div key={`list-${index}`} className="space-y-1.5">
              {block.items.map((item, itemIndex) => (
                <div key={`list-item-${index}-${itemIndex}`} className="flex gap-2">
                  <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-moss" />
                  <span>{inlineMarkdownToNodes(item)}</span>
                </div>
              ))}
            </div>
          );
        }

        if (block.type === 'quote') {
          return (
            <div key={`quote-${index}`} className="rounded-lg border border-ink/10 bg-paper-100 px-3 py-2 text-ink/60">
              <div className="border-l-2 border-signal-500 pl-3">{inlineMarkdownToNodes(block.text)}</div>
            </div>
          );
        }

        if (block.type === 'code') {
          return (
            <pre key={`code-${index}`} className="overflow-x-auto rounded-lg bg-ink px-3 py-2 text-[0.75rem] text-paper-100">
              <code className="font-mono">{block.text}</code>
            </pre>
          );
        }

        return (
          <p key={`paragraph-${index}`} className="whitespace-pre-wrap">
            {inlineMarkdownToNodes(block.text)}
          </p>
        );
      })}
    </div>
  );
};

const EnergyChatbot = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState(() => buildInitialMessages(false));
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const isAuthenticated = Boolean(user?.uid);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    if (isOpen) {
      scrollToBottom();
    }
  }, [messages, isOpen]);

  useEffect(() => {
    setMessages(buildInitialMessages(isAuthenticated));
    setInput('');
    setIsLoading(false);
  }, [isAuthenticated]);

  const getAppContext = (userId) => {
    if (!userId) {
      return null;
    }

    const pendingInviteContext = readStoredJson('pendingInviteContext');
    const userProfile = readStoredJson(`userProfile_${userId}`);
    const userHistory = readStoredJson(`analyticsHistory_${userId}`) || [];
    const latestPrediction = readStoredJson(`loadiq_latest_prediction_context_${userId}`) || userHistory[0] || null;
    const dashboardOverview = readStoredJson(`loadiq_overview_context_${userId}`);
    const advancedAnalytics = readStoredJson(`loadiq_advanced_analytics_context_${userId}`);

    return {
      workspaceName: userProfile?.workspaceName || pendingInviteContext?.workspaceName || '',
      organization: userProfile?.organization || pendingInviteContext?.organization || '',
      latestPrediction,
      recentHistory: Array.isArray(userHistory) ? userHistory.slice(0, 5) : [],
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
      const context = isAuthenticated ? getAppContext(user.uid) : null;
      const country = context?.country || 'Germany';
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
      {!isOpen && (
      <motion.button
        onClick={() => setIsOpen(true)}
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.35, ease: EASE }}
        whileHover={{ scale: 1.04 }}
        whileTap={{ scale: 0.96 }}
        aria-label="Open LoadIQ assistant"
        className="fixed bottom-24 right-4 z-50 flex items-center gap-2.5 rounded-full border border-white/10 bg-ink py-3 pl-3.5 pr-4 text-paper shadow-lift sm:bottom-6 sm:right-6"
      >
        <span className="relative flex h-7 w-7 items-center justify-center rounded-full bg-signal text-ink">
          <MessageSquare size={15} />
          <span className="absolute -right-0.5 -top-0.5 h-2 w-2 rounded-full border border-ink bg-signal-500" />
        </span>
        <span className="hidden text-[0.7rem] font-medium tracking-[0.02em] sm:block">Grid assistant</span>
      </motion.button>
      )}

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 24, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 18, scale: 0.97 }}
            transition={{ duration: 0.4, ease: EASE }}
            className="fixed inset-x-3 bottom-20 top-16 z-[60] flex h-auto max-h-[calc(100vh-5rem)] flex-col overflow-hidden rounded-2xl border border-ink/15 bg-paper shadow-lift sm:inset-auto sm:bottom-24 sm:left-auto sm:right-6 sm:h-[70vh] sm:max-h-[560px] sm:w-[400px]"
          >
            {/* Header */}
            <div className="flex items-center justify-between bg-ink px-4 pb-4 pt-[max(1rem,env(safe-area-inset-top))] text-paper">
              <div className="flex items-center gap-3">
                <span className="flex h-9 w-9 items-center justify-center rounded-xl border border-white/10 bg-white/[0.06] text-signal">
                  <Cpu size={17} />
                </span>
                <div>
                  <h3 className="text-[0.8rem] font-medium tracking-[-0.01em]">LoadIQ assistant</h3>
                  <div className="mt-1 flex items-center gap-1.5">
                    <StatusDot tone={isAuthenticated ? 'lime' : 'amber'} />
                    <span className="label-sm text-paper/45">
                      {isAuthenticated ? 'Context connected' : 'Sign in for history'}
                    </span>
                  </div>
                </div>
              </div>
              <button
                onClick={() => setIsOpen(false)}
                aria-label="Close assistant"
                className="flex h-8 w-8 items-center justify-center rounded-lg text-paper/55 transition-colors hover:bg-white/10 hover:text-paper"
              >
                <X size={17} />
              </button>
            </div>

            {/* Chat area */}
            <div className="flex-1 space-y-3 overflow-y-auto bg-paper p-3.5 custom-scrollbar sm:p-4">
              {!isAuthenticated && (
                <div className="rounded-xl border border-amber-600/25 bg-amber-50 p-3.5 text-[0.72rem]">
                  <p className="font-medium text-amber-900">Sign in for saved history and prediction context.</p>
                  <p className="mt-1 leading-relaxed text-amber-800/70">
                    You can still chat normally, but no account data is used until you log in.
                  </p>
                  <button
                    onClick={() => navigate('/login')}
                    className="mt-2.5 inline-flex items-center justify-center rounded-full bg-ink px-3.5 py-1.5 text-[0.62rem] font-semibold uppercase tracking-[0.14em] text-paper transition-colors hover:bg-ink-800"
                  >
                    Sign in
                  </button>
                </div>
              )}

              {messages.map((msg) => (
                <motion.div
                  key={msg.id}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, ease: EASE }}
                  className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[92%] rounded-xl p-3 text-[0.78rem] sm:max-w-[85%] ${
                      msg.sender === 'user'
                        ? 'rounded-br-sm bg-ink text-paper'
                        : 'rounded-bl-sm border border-ink/10 bg-paper-50 text-ink/75'
                    }`}
                  >
                    <div className={`mb-1.5 flex items-center gap-2 ${msg.sender === 'user' ? 'text-paper/45' : 'text-ink/35'}`}>
                      {msg.sender === 'user' ? <User size={11} /> : <Cpu size={11} />}
                      <span className="label-sm">{msg.sender === 'user' ? 'You' : 'LoadIQ AI'}</span>
                    </div>
                    <div className={msg.sender === 'user' ? '[&_strong]:text-paper [&_em]:text-paper' : ''}>
                      <MessageContent text={msg.text} />
                    </div>
                  </div>
                </motion.div>
              ))}

              {isLoading && (
                <div className="flex justify-start">
                  <div className="flex items-center gap-2 rounded-xl rounded-bl-sm border border-ink/10 bg-paper-50 px-3 py-2.5 text-[0.72rem] text-ink/45">
                    <Loader2 size={14} className="animate-spin text-moss" />
                    Analyzing grid context…
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="border-t border-ink/10 bg-paper-50 p-3 pb-[max(0.75rem,env(safe-area-inset-bottom))]">
              <div className="relative flex items-center">
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                  placeholder="Ask about grids, patterns…"
                  className="w-full rounded-xl border border-ink/15 bg-white py-2.5 pl-3.5 pr-11 text-[0.78rem] text-ink transition-all placeholder:text-ink/30 focus:border-ink focus:outline-none focus:ring-2 focus:ring-signal-300/60"
                />
                <button
                  onClick={handleSend}
                  disabled={!input.trim() || isLoading}
                  aria-label="Send message"
                  className="absolute right-1.5 flex h-8 w-8 items-center justify-center rounded-lg bg-ink text-paper transition-colors hover:bg-ink-800 disabled:bg-ink/20 active:scale-95"
                >
                  <Send size={14} />
                </button>
              </div>
              <p className="mt-2 px-1 text-[0.6rem] text-ink/30">
                Grounded in your saved workspace context. Verify before acting.
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};

export default EnergyChatbot;
