export const SUBJECTS = [
  "biopharmaceutics",
  "clinical_pharmacy",
  "industrial_pharmacy",
  "hospital_pharmacy",
  "pqm",
  "anatomy",
  "biochemistry",
  "medical_physiology",
  "organic_chemistry",
];

// Same palette as before — one hue per subject, echoed across both pages
// (subject cards, thread sidebar, message accents, focus rings).
export const SUBJECT_COLORS = {
  biopharmaceutics: { accent: "#146b57", soft: "#dcefe9" },
  clinical_pharmacy: { accent: "#b83a26", soft: "#fbe4de" },
  industrial_pharmacy: { accent: "#b3791f", soft: "#f8ecd6" },
  hospital_pharmacy: { accent: "#275c92", soft: "#dfe9f3" },
  pqm: { accent: "#5c4590", soft: "#eae4f5" },
  anatomy: { accent: "#a83e6c", soft: "#f5e1eb" },
  biochemistry: { accent: "#5c7a29", soft: "#e8efd9" },
  medical_physiology: { accent: "#46538f", soft: "#e2e4f2" },
  organic_chemistry: { accent: "#8a5a2b", soft: "#f0e4d3" },
};

export const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";