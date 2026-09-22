import { useEffect, useState, useCallback } from "react";
import { supabase } from "./supabaseClient";
import SubjectSelect from "./SubjectSelect";
import ChatWorkspace from "./ChatWorkspace";
import { API_URL } from "./subjects";
import "./App.css";

function subjectFromHash() {
  const h = window.location.hash.replace(/^#/, "");
  return h || null;
}

function App() {
  const [subject, setSubject] = useState(() => subjectFromHash());
  const [session, setSession] = useState(null);
  const [loadingSession, setLoadingSession] = useState(true);
  const [sessions, setSessions] = useState([]);
  const [loadingSessions, setLoadingSessions] = useState(true);

  useEffect(() => {
    const onPopState = () => setSubject(subjectFromHash());
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const refreshSessions = useCallback(async () => {
    if (!session?.user?.id) return;
    try {
      const { data } = await supabase.auth.getSession();
      const token = data.session?.access_token;
      const res = await fetch(`${API_URL}/chats/${session.user.id}`, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      if (res.ok) {
        const d = await res.json();
        setSessions(d.sessions || []);
      }
    } catch {
      // ignore
    } finally {
      setLoadingSessions(false);
    }
  }, [session?.user?.id]);

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      setLoadingSession(false);
    });

    const { data: listener } = supabase.auth.onAuthStateChange(
      (_event, session) => {
        setSession(session);
      }
    );

    return () => listener.subscription.unsubscribe();
  }, []);

  useEffect(() => {
    if (session?.user?.id) {
      refreshSessions();
    } else {
      setSessions([]);
      setLoadingSessions(false);
    }
  }, [session?.user?.id, refreshSessions]);

  const openSubject = (s) => {
    setSubject(s);
    window.history.pushState({ subject: s }, "", `#${s}`);
  };

  const goHome = () => {
    setSubject(null);
    window.history.pushState({ subject: null }, "", "#");
  };

  const handleGoogleLogin = () => {
    supabase.auth.signInWithOAuth({ provider: "google" });
  };

  const handleLogout = () => {
    supabase.auth.signOut();
  };

  if (loadingSession) {
    return <div className="rag-app">Loading...</div>;
  }

  if (!session) {
    return (
      <div className="login-page">
        <div className="login-card">
          <div className="login-logo">💊</div>
          <h1>Pharmacy RAG</h1>
          <p>To access please login through Google.</p>
          <button className="google-login-btn" onClick={handleGoogleLogin}>
            <svg width="18" height="18" viewBox="0 0 48 48">
              <path fill="#FFC107" d="M43.6 20.5H42V20H24v8h11.3c-1.6 4.6-6 8-11.3 8-6.6 0-12-5.4-12-12s5.4-12 12-12c3.1 0 5.8 1.1 8 3l5.7-5.7C34.5 6 29.5 4 24 4 12.9 4 4 12.9 4 24s8.9 20 20 20 20-8.9 20-20c0-1.2-.1-2.4-.4-3.5z" />
              <path fill="#FF3D00" d="M6.3 14.7l6.6 4.8C14.6 15.9 18.9 13 24 13c3.1 0 5.8 1.1 8 3l5.7-5.7C34.5 6 29.5 4 24 4 16.3 4 9.6 8.3 6.3 14.7z" />
              <path fill="#4CAF50" d="M24 44c5.4 0 10.3-1.8 14.1-5l-6.5-5.3C29.5 35.5 26.9 36 24 36c-5.3 0-9.7-3.4-11.3-8.1l-6.6 5.1C9.5 39.6 16.2 44 24 44z" />
              <path fill="#1976D2" d="M43.6 20.5H42V20H24v8h11.3c-.8 2.2-2.2 4.1-4 5.5l6.5 5.3C41.6 35.1 44 30 44 24c0-1.2-.1-2.4-.4-3.5z" />
            </svg>
            Sign in with Google
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="rag-app">
      <button onClick={handleLogout} className="signout-btn">
        Sign out
      </button>
      {subject ? (
        <ChatWorkspace
          subject={subject}
          onBack={goHome}
          userId={session.user.id}
          cachedSessions={sessions}
          onRefreshSessions={refreshSessions}
        />
      ) : (
        <SubjectSelect
          onSelect={openSubject}
          userId={session.user.id}
          sessions={sessions}
          loading={loadingSessions}
        />
      )}
    </div>
  );
}

export default App;