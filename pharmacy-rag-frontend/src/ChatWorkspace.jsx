import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { v4 as uuidv4 } from "uuid";
import { SUBJECT_COLORS, API_URL, USER_ID } from "./subjects";

function SourceCard({ src }) {
  return (
    <div className={`source ${src.content_type}`}>
      <div className="source-header">
        <span className="source-book">{(src.book_name || src.doc_id).replace(/_/g, " ")}</span>
        <span className={`source-tag ${src.content_type}`}>{src.content_type}</span>
        <span className="source-page">Page {src.pages.join(", ")}</span>
      </div>
      <p className="source-text">{src.text.slice(0, 200)}...</p>
      {src.formula_image_path && (
        <img src={`${API_URL}/formulas/${src.formula_image_path.split("/").pop()}`} alt="formula" />
      )}
      {src.table_image_path && (
        <img src={`${API_URL}/tables/${src.table_image_path.split("/").pop()}`} alt="table" />
      )}
      {src.image_path && (
        <img src={`${API_URL}/images/${src.image_path.split("/").pop()}`} alt="figure" />
      )}
    </div>
  );
}

function IntentBadge({ intent }) {
  if (intent === "COURSE_QUESTION") {
    return <span className="intent-badge searched"><span className="dot" />Searched course materials</span>;
  }
  return <span className="intent-badge chat"><span className="dot" />General chat</span>;
}

function ChatWorkspace({ subject, onBack }) {
  const color = SUBJECT_COLORS[subject];

  const [threads, setThreads] = useState([]);
  const [threadId, setThreadId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState("");
  const [error, setError] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const bottomRef = useRef(null);

  const refreshThreads = (preserveThreadId) => {
    fetch(`${API_URL}/chats/${USER_ID}`)
      .then((r) => r.json())
      .then((d) => {
        const forSubject = (d.sessions || []).filter((s) => s.subject === subject);
        setThreads((prev) => {
          const stillPending = prev.filter(
            (s) => s._pending && !forSubject.some((ss) => ss.thread_id === s.thread_id)
          );
          return [...stillPending, ...forSubject];
        });
        return forSubject;
      })
      .catch(() => []);
  };

  // On opening a subject: load its threads, then jump into the most recent
  // one if it has any, or start a fresh thread if it doesn't.
  useEffect(() => {
    let cancelled = false;
    fetch(`${API_URL}/chats/${USER_ID}`)
      .then((r) => r.json())
      .then((d) => {
        if (cancelled) return;
        const forSubject = (d.sessions || []).filter((s) => s.subject === subject);
        setThreads(forSubject);
        if (forSubject.length > 0) {
          selectThread(forSubject[0]);
        } else {
          startNewThread();
        }
      })
      .catch(() => {
        if (!cancelled) startNewThread();
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [subject]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const startNewThread = () => {
    const existingDraft = threads.find((t) => t._pending);
    if (existingDraft) {
      setThreadId(existingDraft.thread_id);
      setMessages([]);
      setQuery("");
      setError(null);
      return;
    }
    const newId = uuidv4();
    setThreadId(newId);
    setMessages([]);
    setQuery("");
    setError(null);
    setThreads((prev) => [{ thread_id: newId, title: "New chat", subject, _pending: true }, ...prev]);
  };

  const selectThread = async (thread) => {
    setThreadId(thread.thread_id);
    setQuery("");
    setError(null);
    try {
      const res = await fetch(`${API_URL}/chats/${USER_ID}/${thread.thread_id}/messages`);
      const data = await res.json();
      const loaded = (data.messages || []).map((m) => ({
        id: m.id || uuidv4(),
        role: m.role,
        content: m.content,
        intent: m.intent,
        is_grounded: m.is_grounded,
        retry_count: m.retry_count,
        sources: m.sources || [],
      }));
      setMessages(loaded);
    } catch {
      setMessages([]);
    }
  };

  const deleteThread = async (thread, e) => {
    e.stopPropagation();
    if (!window.confirm("Delete this thread? This can't be undone.")) return;
    try {
      await fetch(`${API_URL}/chats/${USER_ID}/${thread.thread_id}`, { method: "DELETE" });
    } catch {
      setError("Couldn't delete the thread — check the backend and try again.");
      return;
    }
    const remaining = threads.filter((t) => t.thread_id !== thread.thread_id);
    setThreads(remaining);
    if (thread.thread_id === threadId) {
      if (remaining.length > 0) {
        selectThread(remaining[0]);
      } else {
        startNewThread();
      }
    }
  };

  const handleSearch = async () => {
    if (!query.trim() || loading) return;
    const userText = query;
    setQuery("");
    setMessages((prev) => [...prev, { id: uuidv4(), role: "user", content: userText }]);
    setLoading(true);
    setLoadingStage("Thinking...");
    setError(null);

    try {
      const classifyRes = await fetch(`${API_URL}/classify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userText, subject }),
      });
      if (classifyRes.ok) {
        const { intent } = await classifyRes.json();
        if (intent === "COURSE_QUESTION") {
          setLoadingStage(`Searching ${subject.replace(/_/g, " ")} books...`);
        }
      }

      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: userText, subject, thread_id: threadId, user_id: USER_ID }),
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        {
          id: uuidv4(),
          role: "assistant",
          content: data.answer,
          intent: data.intent,
          is_grounded: data.is_grounded,
          retry_count: data.retry_count,
          sources: data.sources || [],
        },
      ]);
      refreshThreads();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
      setLoadingStage("");
    }
  };

  return (
    <div
      className={`workspace-page ${sidebarOpen ? "sidebar-open" : ""}`}
      style={{ "--subject": color.accent, "--subject-soft": color.soft }}
    >
      <button
        className="sidebar-toggle"
        onClick={() => setSidebarOpen((o) => !o)}
        aria-label={sidebarOpen ? "Collapse threads" : "Expand threads"}
        aria-expanded={sidebarOpen}
      >
        {sidebarOpen ? "‹" : "☰"}
      </button>

      <div className={`sidebar ${sidebarOpen ? "open" : "closed"}`}>
        <div className="sidebar-header">
          <button className="back-btn" onClick={onBack}>← Subjects</button>
        </div>
        <div className="sidebar-subject">
          <span className="subject-card-dot" />
          {subject.replace(/_/g, " ")}
        </div>
        <button className="new-chat-btn" onClick={startNewThread}>+ New thread</button>
        <div className="sessions-list">
          {threads.map((t) => (
            <div
              key={t.thread_id}
              className={`session-item ${t.thread_id === threadId ? "active" : ""} ${t._pending ? "pending" : ""}`}
              onClick={() => selectThread(t)}
            >
              <div className="session-title">{t.title || "Untitled"}</div>
              <button
                className="delete-thread-btn"
                onClick={(e) => deleteThread(t, e)}
                aria-label="Delete thread"
                title="Delete thread"
              >
                ✕
              </button>
            </div>
          ))}
          {threads.length === 0 && <p className="no-sessions">No threads yet — ask something to start one.</p>}
        </div>
      </div>

      <div className="main">
        <div className="workspace-header">
          <h1>{subject.replace(/_/g, " ")}</h1>
        </div>

        <div className="results-container">
          {messages.map((m) =>
            m.role === "user" ? (
              <div key={m.id} className="message user">
                <div className="message-bubble">{m.content}</div>
              </div>
            ) : (
              <div key={m.id} className="message assistant">
                <div className="result-panel">
                  <IntentBadge intent={m.intent} />
                  <div className="answer-section">
                    <div className="answer-text">
                      <ReactMarkdown>{m.content}</ReactMarkdown>
                    </div>
                    {m.intent === "COURSE_QUESTION" && (
                      <p className="grounded">
                        {m.is_grounded ? (
                          <span className="seal-yes">✓ Grounded in course materials</span>
                        ) : (
                          <span className="seal-no">✗ Not grounded</span>
                        )}
                        {m.retry_count > 0 && (
                          <span> · {m.retry_count} {m.retry_count === 1 ? "retry" : "retries"}</span>
                        )}
                      </p>
                    )}
                  </div>
                  {m.sources?.length > 0 && (
                    <details className="sources-section">
                      <summary>📚 Sources ({m.sources.length})</summary>
                      <div className="sources-list">
                        {m.sources.map((src, i) => <SourceCard key={i} src={src} />)}
                      </div>
                    </details>
                  )}
                </div>
              </div>
            )
          )}
          {loading && <div className="loading"><div className="spinner" /><p>{loadingStage}</p></div>}
          {error && <div className="error">Error: {error}</div>}
          <div ref={bottomRef} />
        </div>

        <div className="composer">
          <div className="search-box">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              placeholder={`Ask a ${subject.replace(/_/g, " ")} question...`}
              disabled={loading}
            />
            <button onClick={handleSearch} disabled={loading}>
              {loading ? "Sending..." : "Send"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ChatWorkspace;
