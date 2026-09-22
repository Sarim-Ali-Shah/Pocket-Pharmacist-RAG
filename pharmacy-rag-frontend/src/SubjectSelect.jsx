import { useMemo } from "react";
import { SUBJECTS, SUBJECT_COLORS } from "./subjects";

function SubjectSelect({ onSelect, sessions = [], loading = false }) {
  const counts = useMemo(() => {
    const next = {};
    (sessions || []).forEach((s) => {
      next[s.subject] = (next[s.subject] || 0) + 1;
    });
    return next;
  }, [sessions]);

  return (
    <div className="subject-select-page">
      <div className="subject-select-header">
        <h1>Pharmacy RAG</h1>
        <p>Pick a subject to open its threads.</p>
      </div>
      <div className="subject-grid">
        {SUBJECTS.map((s) => {
          const color = SUBJECT_COLORS[s];
          const count = counts[s] || 0;
          return (
            <button
              key={s}
              className="subject-card"
              style={{ "--tab-color": color.accent, "--tab-soft": color.soft }}
              onClick={() => onSelect(s)}
            >
              <span className="subject-card-dot" />
              <span className="subject-card-name">{s.replace(/_/g, " ")}</span>
              <span className="subject-card-count">
                {loading && sessions.length === 0
                  ? "Loading..."
                  : count > 0
                  ? `${count} thread${count === 1 ? "" : "s"}`
                  : "No threads yet"}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export default SubjectSelect;