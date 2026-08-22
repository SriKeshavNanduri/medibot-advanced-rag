'use client';

import React, { useState } from 'react';
import { 
  Shield, Lock, User, LogIn, MessageSquare, FileText, 
  Send, ChevronDown, ChevronUp, LogOut, AlertCircle, Sparkles, Building2 
} from 'lucide-react';

interface Citation {
  source_document?: string;
  collection?: string;
  section_title?: string;
  chunk_type?: string;
  content?: string;
  page_content?: string;
  metadata?: {
    source_document?: string;
    collection?: string;
    section_title?: string;
  };
  score?: number;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
}

export default function MediAssistApp() {
  // Auth state
  const [token, setToken] = useState<string | null>(null);
  const [userRole, setUserRole] = useState<string | null>(null);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [authLoading, setAuthLoading] = useState(false);
  const [authError, setAuthError] = useState<string | null>(null);

  // Chat state
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputQuery, setInputQuery] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [expandedCitations, setExpandedCitations] = useState<{ [key: number]: boolean }>({});
  const [sessionId, setSessionId] = useState<string>('');

  // Handle Login
  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthLoading(true);
    setAuthError(null);

    try {
      const response = await fetch('http://localhost:8000/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || 'Invalid username or password');
      }

      const data = await response.json();
      const accessToken = data.access_token;
      
      // Decode JWT payload to extract user role securely
      const tokenPayload = JSON.parse(atob(accessToken.split('.')[1]));
      const role = tokenPayload.role || tokenPayload.sub || 'user';

      setToken(accessToken);
      setUserRole(role);
      // Generate a persistent session ID once per login session
      const now = new Date();
      const pad = (n: number) => n.toString().padStart(2, '0');
      const timestamp = now.getUTCFullYear().toString() +
        pad(now.getUTCMonth() + 1) +
        pad(now.getUTCDate()) +
        pad(now.getUTCHours()) +
        pad(now.getUTCMinutes()) +
        pad(now.getUTCSeconds());
      const randomHex = Math.random().toString(16).substring(2, 10);
      setSessionId(`session_${timestamp}_${randomHex}`);
    } catch (err: any) {
      setAuthError(err.message || 'Failed to authenticate');
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogout = () => {
    setToken(null);
    setUserRole(null);
    setUsername('');
    setPassword('');
    setMessages([]);
    setSessionId('');
  };

  // Handle Chat Submit
  const handleSendQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputQuery.trim() || chatLoading || !token) return;

    const userMsg = inputQuery.trim();
    setInputQuery('');
    setMessages((prev) => [...prev, { role: 'user', content: userMsg }]);
    setChatLoading(true);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({ question: userMsg, session_id: sessionId }),
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const data = await response.json();
      const answer = data.answer || data.response || 'No answer received.';
      // Backend returns sources under the key 'sources'
      const citations = data.sources || data.citations || data.source_nodes || [];

      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: answer, citations },
      ]);
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        { 
          role: 'assistant', 
          content: `Error connecting to RAG backend: ${err.message}. Please ensure FastAPI server is running on port 8000.` 
        },
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  const toggleCitation = (index: number) => {
    setExpandedCitations((prev) => ({ ...prev, [index]: !prev[index] }));
  };

  // ---------------------------------------------------------------------------
  // SCREEN 1: LOGIN SCREEN (Authentication Wall)
  // ---------------------------------------------------------------------------
  if (!token) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-950 to-indigo-950 flex items-center justify-center p-4">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_50%,#000_70%,transparent_100%)] opacity-20"></div>
        
        <div className="relative w-full max-w-md bg-slate-900/85 backdrop-blur-xl border border-slate-700/60 rounded-2xl shadow-2xl overflow-hidden p-8">
          <div className="flex flex-col items-center mb-8">
            <div className="w-16 h-16 bg-gradient-to-tr from-blue-600 to-indigo-500 rounded-2xl flex items-center justify-center shadow-lg shadow-blue-500/30 mb-4 ring-2 ring-blue-400/20">
              <Shield className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-2xl font-bold text-white tracking-tight">MediAssist RAG Portal</h1>
            <p className="text-sm text-slate-400 mt-1">Secure Enterprise Medical Knowledge Assistant</p>
          </div>

          {authError && (
            <div className="mb-6 bg-red-950/50 border border-red-500/50 text-red-200 px-4 py-3 rounded-xl flex items-center gap-3 text-sm">
              <AlertCircle className="w-5 h-5 text-red-400 shrink-0" />
              <span>{authError}</span>
            </div>
          )}

          <form onSubmit={handleLogin} className="space-y-5">
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-2">
                Username
              </label>
              <div className="relative">
                <span className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-500">
                  <User className="w-4 h-4" />
                </span>
                <input
                  type="text"
                  required
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="e.g. admin, doctor_john"
                  className="w-full bg-slate-950/60 border border-slate-700 rounded-xl pl-10 pr-4 py-3 text-white placeholder-slate-500 text-sm focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all"
                />
              </div>
            </div>

            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-2">
                Password
              </label>
              <div className="relative">
                <span className="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-500">
                  <Lock className="w-4 h-4" />
                </span>
                <input
                  type="password"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full bg-slate-950/60 border border-slate-700 rounded-xl pl-10 pr-4 py-3 text-white placeholder-slate-500 text-sm focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={authLoading}
              className="w-full mt-2 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-medium py-3 px-4 rounded-xl shadow-lg shadow-blue-600/25 flex items-center justify-center gap-2 transition-all transform active:scale-[0.99] disabled:opacity-50 cursor-pointer"
            >
              {authLoading ? (
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              ) : (
                <>
                  <LogIn className="w-4 h-4" />
                  <span>Authenticate & Access</span>
                </>
              )}
            </button>
          </form>

          <div className="mt-8 pt-6 border-t border-slate-800 text-center">
            <p className="text-xs text-slate-500">
              Protected by FastAPI OAuth2 RBAC. Role is derived securely from access token claims.
            </p>
          </div>
        </div>
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // SCREEN 2: WORKING CHAT SCREEN WITH CITATIONS & ROLE BADGE
  // ---------------------------------------------------------------------------
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      {/* Top Header */}
      <header className="h-16 border-b border-slate-800 bg-slate-900/60 backdrop-blur-md px-6 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-tr from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center shadow-md shadow-blue-500/20">
            <Shield className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="font-bold text-base text-white leading-tight">MediAssist Enterprise RAG</h2>
            <p className="text-xs text-slate-400">Connected to FastAPI Backend</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 bg-slate-800/80 border border-slate-700/80 px-3.5 py-1.5 rounded-full">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span className="text-xs text-slate-300 font-medium">Role:</span>
            <span className="text-xs font-bold text-blue-400 uppercase tracking-wide bg-blue-950/80 px-2 py-0.5 rounded-md border border-blue-800/50">
              {userRole}
            </span>
          </div>

          <button
            onClick={handleLogout}
            className="flex items-center gap-2 bg-slate-800 hover:bg-red-950/60 hover:border-red-500/50 text-slate-300 hover:text-red-200 text-xs font-medium px-3 py-2 rounded-xl border border-slate-700 transition-all cursor-pointer"
            title="Logout"
          >
            <LogOut className="w-4 h-4" />
            <span className="hidden sm:inline">Sign Out</span>
          </button>
        </div>
      </header>

      {/* Main Chat Container */}
      <main className="flex-1 max-w-4xl w-full mx-auto p-4 sm:p-6 flex flex-col overflow-hidden">
        {messages.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center text-center p-8">
            <div className="w-20 h-20 bg-blue-900/20 border border-blue-500/30 rounded-3xl flex items-center justify-center text-blue-400 mb-6 shadow-inner">
              <Sparkles className="w-10 h-10" />
            </div>
            <h3 className="text-xl font-bold text-white mb-2">Welcome to MediAssist RAG</h3>
            <p className="text-sm text-slate-400 max-w-md mb-8">
              Ask questions regarding hospital guidelines, billing claims, clinical notes, nursing procedures, or equipment manuals. Responses are backed by verified document citations.
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-lg text-left">
              {[
                "What are the claim submission guidelines?",
                "What is the patient confidentiality protocol?",
                "Explain nursing handover procedures.",
                "What equipment calibration standards apply?"
              ].map((suggestion, idx) => (
                <button
                  key={idx}
                  onClick={() => setInputQuery(suggestion)}
                  className="bg-slate-900 hover:bg-slate-800/80 border border-slate-800 hover:border-blue-500/40 p-3.5 rounded-xl text-xs text-slate-300 transition-all text-left flex items-start gap-2.5 group cursor-pointer"
                >
                  <MessageSquare className="w-4 h-4 text-blue-400 shrink-0 mt-0.5 group-hover:scale-110 transition-transform" />
                  <span>{suggestion}</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="flex-1 overflow-y-auto space-y-6 pr-2 mb-4">
            {messages.map((msg, index) => (
              <div
                key={index}
                className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}
              >
                <div
                  className={`max-w-[85%] sm:max-w-[75%] rounded-2xl px-4 py-3.5 text-sm shadow-md leading-relaxed ${
                    msg.role === 'user'
                      ? 'bg-blue-600 text-white rounded-br-none'
                      : 'bg-slate-900 border border-slate-800 text-slate-100 rounded-bl-none'
                  }`}
                >
                  <div className="whitespace-pre-wrap">{msg.content}</div>
                </div>

                {/* Citations Display for Assistant Messages */}
                {msg.role === 'assistant' && msg.citations && msg.citations.length > 0 && (
                  <div className="mt-3 max-w-[85%] sm:max-w-[75%] w-full bg-slate-900/70 border border-slate-800/80 rounded-xl p-3">
                    <button
                      onClick={() => toggleCitation(index)}
                      className="flex items-center justify-between w-full text-xs font-semibold text-blue-400 hover:text-blue-300 transition-colors cursor-pointer"
                    >
                      <div className="flex items-center gap-2">
                        <FileText className="w-4 h-4 text-blue-400" />
                        <span>Sources & Citations ({msg.citations.length})</span>
                      </div>
                      {expandedCitations[index] ? (
                        <ChevronUp className="w-4 h-4" />
                      ) : (
                        <ChevronDown className="w-4 h-4" />
                      )}
                    </button>

                    {expandedCitations[index] && (
                      <div className="mt-3 space-y-2.5 pt-2 border-t border-slate-800">
                        {msg.citations.map((cite, cIdx) => (
                          <div
                            key={cIdx}
                            className="bg-slate-950/80 border border-slate-800/60 rounded-lg p-2.5 text-xs space-y-1.5"
                          >
                            <div className="flex items-center justify-between">
                              <span className="font-semibold text-slate-200 flex items-center gap-1.5">
                                <Building2 className="w-3.5 h-3.5 text-indigo-400" />
                                {cite.source_document || cite.metadata?.source_document || 'Document'}
                              </span>
                              <span className="bg-slate-800 text-slate-300 px-2 py-0.5 rounded text-[10px] uppercase font-mono">
                                {cite.collection || cite.metadata?.collection || 'general'}
                              </span>
                            </div>
                            {(cite.section_title || cite.metadata?.section_title) && (
                              <p className="text-slate-400 font-medium">
                                Section: <span className="text-slate-300">{cite.section_title || cite.metadata?.section_title}</span>
                              </p>
                            )}
                            {(cite.content || cite.page_content || cite.snippet) && (
                              <p className="text-slate-400 italic bg-slate-900/50 p-2 rounded border border-slate-800/40 line-clamp-3">
                                "{cite.content || cite.page_content || cite.snippet}"
                              </p>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}

            {chatLoading && (
              <div className="flex items-start">
                <div className="bg-slate-900 border border-slate-800 rounded-2xl rounded-bl-none px-4 py-3.5 text-sm text-slate-400 flex items-center gap-3">
                  <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                  <span>Searching medical knowledge base and generating response...</span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Bottom Fixed Query Input Box */}
        <div className="mt-auto pt-2">
          <form onSubmit={handleSendQuery} className="relative flex items-center">
            <input
              type="text"
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              placeholder="Ask a medical policy, clinical guideline, or billing question..."
              disabled={chatLoading}
              className="w-full bg-slate-900 border border-slate-700/80 rounded-2xl pl-4 pr-14 py-4 text-white placeholder-slate-500 text-sm focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 shadow-xl transition-all disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={chatLoading || !inputQuery.trim()}
              className="absolute right-2.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-40 text-white p-2.5 rounded-xl transition-all shadow-md shadow-blue-600/30 flex items-center justify-center cursor-pointer"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
          <p className="text-[11px] text-slate-500 text-center mt-2">
            MediAssist RAG AI Assistant • Secure RBAC Enforced
          </p>
        </div>
      </main>
    </div>
  );
}
