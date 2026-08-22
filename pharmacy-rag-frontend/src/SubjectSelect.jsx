import { useEffect, useState } from "react";
import { SUBJECTS, SUBJECT_COLORS, API_URL, USER_ID } from "./subjects";

function SubjectSelect({ onSelect }) {
  const [counts, setCounts] = useState({});

  useEffect(() => {
    fetch(`${API_URL}/chats/${USER_ID}`)
      .then((r) => r.json())
      .then((d) => {
        const next = {};
        (d.sessions || []).forEach((s) => {
          next[s.subject] = (next[s.subject] || 0) + 1;
        });
        setCounts(next);
      })
      .catch(() => {});
  }, []);

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
                {count > 0 ? `${count} thread${count === 1 ? "" : "s"}` : "No threads yet"}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

export default SubjectSelect;
