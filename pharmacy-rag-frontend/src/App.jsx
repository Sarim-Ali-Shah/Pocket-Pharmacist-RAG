import { useEffect, useState } from "react";
import SubjectSelect from "./SubjectSelect";
import ChatWorkspace from "./ChatWorkspace";
import "./App.css";

function subjectFromHash() {
  const h = window.location.hash.replace(/^#/, "");
  return h || null;
}

function App() {
  const [subject, setSubject] = useState(() => subjectFromHash());

  useEffect(() => {
    const onPopState = () => setSubject(subjectFromHash());
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const openSubject = (s) => {
    setSubject(s);
    window.history.pushState({ subject: s }, "", `#${s}`);
  };

  const goHome = () => {
    setSubject(null);
    window.history.pushState({ subject: null }, "", "#");
  };

  return (
    <div className="rag-app">
      {subject ? (
        <ChatWorkspace subject={subject} onBack={goHome} />
      ) : (
        <SubjectSelect onSelect={openSubject} />
      )}
    </div>
  );
}

export default App;
