import { useState, useEffect } from "react";

async function call(path, { token, json, form } = {}) {
  const r = await fetch("/api" + path, {
    method: json || form ? "POST" : "GET",
    headers: { ...(token && { Authorization: `Bearer ${token}` }), ...(json && { "Content-Type": "application/json" }) },
    body: form || (json && JSON.stringify(json)),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Request failed");
  return data;
}

const List = ({ items }) => <ul>{items.map((x, i) => <li key={i}>{x}</li>)}</ul>;
const Trend = ({ scores }) => scores.length < 2 ? null : (
  <svg viewBox="0 0 200 50" width="100%" height="60" role="img" aria-label={`Score trend: ${scores.join(", ")}`}>
    <polyline fill="none" stroke="var(--accent)" strokeWidth="2"
      points={scores.map((v, i) => `${(i * 190) / (scores.length - 1) + 5},${45 - (v / 100) * 40}`).join(" ")} />
  </svg>
);
const Bar = ({ v }) => <div className="bar"><i style={{ width: `${v}%` }} /></div>;

export default function App() {
  const [token, setToken] = useState(localStorage.getItem("t") || "");
  const [form, setForm] = useState({ email: "", password: "" });
  const [resume, setResume] = useState(null);
  const [iv, setIv] = useState(null);
  const [answer, setAnswer] = useState("");
  const [last, setLast] = useState(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [history, setHistory] = useState([]);
  useEffect(() => { if (token) call("/interviews", { token }).then(setHistory).catch(() => {}); }, [token, iv?.status]);

  const run = async (fn) => { setBusy(true); setErr(""); try { await fn(); } catch (e) { setErr(e.message); } setBusy(false); };
  const auth = (kind) => run(async () => {
    const { token } = await call(`/auth/${kind}`, { json: form });
    localStorage.setItem("t", token); setToken(token);
  });
  const upload = (e) => run(async () => {
    const fd = new FormData(); fd.append("file", e.target.files[0]);
    setResume(await call("/resumes", { token, form: fd }));
  });
  const start = () => run(async () => setIv(await call("/interviews", { token, json: { resume_id: resume.id, num_questions: 5 } })));
  const submit = () => run(async () => {
    const out = await call(`/interviews/${iv.id}/answer`, { token, json: { answer } });
    setLast(out.last_evaluation); setAnswer(""); setIv(out);
  });
  const reset = () => { setIv(null); setResume(null); setLast(null); };

  return (
    <main>
      <h1>Interview Coach</h1>
      <p>Upload your resume, get questioned on it, and see exactly where your answers fall short.</p>
      {err && <p className="err" role="alert">{err}</p>}

      {!token && (
        <div className="box">
          <label>Email<input value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} /></label>
          <label>Password (8+ characters)<input type="password" value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} /></label>
          <button disabled={busy} onClick={() => auth("login")}>Log in</button>{" "}
          <button className="link" disabled={busy} onClick={() => auth("register")}>Create account</button>
        </div>
      )}

      {token && !iv && (
        <div className="box">
          <label>Resume (PDF or text)<input type="file" accept=".pdf,.txt,.md" onChange={upload} /></label>
          {resume && (<>
            <h2>What an interviewer will notice</h2>
            <strong>Strengths</strong><List items={resume.analysis.strengths} />
            <strong>Weak spots they will probe</strong><List items={resume.analysis.weaknesses} />
            <button disabled={busy} onClick={start}>{busy ? "Preparing questions…" : "Start mock interview"}</button>
          </>)}
        </div>
      )}

      {token && !iv && history.length > 0 && (
        <div className="box">
          <h2>Your progress</h2>
          <Trend scores={history.filter((h) => h.overall != null).map((h) => h.overall).reverse()} />
          {history.map((h) => (
            <div key={h.id} style={{ margin: "14px 0" }}>
              <button className="link" onClick={() => run(async () => setIv(await call(`/interviews/${h.id}`, { token })))}>
                {new Date(h.created_at).toLocaleDateString()}: {h.status === "complete" ? `${h.overall}/100` : "in progress"}
              </button>
              {h.category_scores && Object.entries(h.category_scores).map(([c, v]) => (
                <div key={c}><small>{c}: {v}</small><Bar v={v} /></div>
              ))}
            </div>
          ))}
        </div>
      )}

      {iv && iv.status === "active" && (
        <>
          {last && (
            <div className="box">
              <strong>Last answer: {last.overall}/100</strong>
              <p>{last.feedback}</p><strong>A strong answer covers</strong><List items={last.ideal_points} />
            </div>
          )}
          <div className="box">
            <strong>Question {iv.answered + 1} of {iv.total} ({iv.current.category})</strong>
            <p>{iv.current.question}</p>
            <textarea value={answer} onChange={(e) => setAnswer(e.target.value)} aria-label="Your answer" />
            <button disabled={busy || !answer.trim()} onClick={submit}>{busy ? "Scoring…" : "Submit answer"}</button>
          </div>
        </>
      )}

      {iv && iv.status === "complete" && (
        <div className="box">
          <div className="score">{iv.report.overall}/100</div>
          <strong>{iv.report.readiness}</strong>
          <p>{iv.report.summary}</p>
          {Object.entries(iv.report.category_scores).map(([c, v]) => (
            <div key={c}><small>{c}: {v}</small><Bar v={v} /></div>
          ))}
          <h2>Strengths</h2><List items={iv.report.top_strengths} />
          <h2>Improve</h2><List items={iv.report.improvements} />
          <h2>Study plan</h2><List items={iv.report.study_plan} />
          <button onClick={reset}>Practice again</button>
        </div>
      )}

      {token && <button className="link" onClick={() => { localStorage.removeItem("t"); setToken(""); reset(); }}>Log out</button>}
    </main>
  );
}
