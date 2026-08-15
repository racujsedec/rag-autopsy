import { useState } from 'react'
import './App.css'

const sampleQuestion = {
  question_id: 'q031',
  question:
    'What service-desk changes did Vector Systems introduce, and what was the result?',
  answerable: true,
}

function App() {
  const [generate, setGenerate] = useState(true)
  const [hasRun, setHasRun] = useState(false)

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <span className="brand-mark">RA</span>
          <span className="brand-name">RAG Autopsy</span>
        </div>
        <span className="status-pill">Frontend ready</span>
      </header>

      <section className="hero-section">
        <p className="eyebrow">RAG FAILURE DIAGNOSTICS</p>
        <h1>Find where your RAG pipeline failed.</h1>
        <p className="hero-copy">
          Inspect retrieval, generation, citations, and evidence support
          through one diagnostic workflow.
        </p>
      </section>

      <section className="workspace">
        <div className="control-panel card">
          <div className="section-heading">
            <div>
              <p className="section-label">INPUT</p>
              <h2>Run an autopsy</h2>
            </div>
          </div>

          <label className="field-label" htmlFor="question">
            Benchmark question
          </label>

          <select id="question" value={sampleQuestion.question_id} disabled>
            <option value={sampleQuestion.question_id}>
              {sampleQuestion.question_id} — {sampleQuestion.question}
            </option>
          </select>

          <div className="question-preview">
            <span className="question-id">{sampleQuestion.question_id}</span>
            <p>{sampleQuestion.question}</p>
            <span className="answerable-badge">Answerable</span>
          </div>

          <div className="setting-row">
            <div>
              <strong>Generate grounded answer</strong>
              <p>Run generation and citation diagnostics.</p>
            </div>

            <label className="switch">
              <input
                type="checkbox"
                checked={generate}
                onChange={(event) => setGenerate(event.target.checked)}
              />
              <span className="slider"></span>
            </label>
          </div>

          <div className="setting-row">
            <div>
              <strong>Top K</strong>
              <p>Number of retrieved chunks to inspect.</p>
            </div>
            <span className="top-k">3</span>
          </div>

          <button
            className="run-button"
            type="button"
            onClick={() => setHasRun(true)}
          >
            Run Autopsy
          </button>
        </div>

        <div className="results-panel card">
          <div className="section-heading">
            <div>
              <p className="section-label">DIAGNOSTICS</p>
              <h2>Autopsy result</h2>
            </div>
          </div>

          {!hasRun ? (
            <div className="empty-state">
              <div className="empty-icon">◎</div>
              <h3>No autopsy has been run yet</h3>
              <p>
                Select a benchmark question and run the pipeline to inspect
                its diagnosis.
              </p>
            </div>
          ) : (
            <div className="sample-result">
              <div className="diagnosis-row">
                <span>Primary diagnosis</span>
                <strong className="diagnosis-badge">RANKING_FAILURE</strong>
              </div>

              <p className="diagnosis-copy">
                Relevant evidence was retrieved but ranked below the first
                position.
              </p>

              <div className="metric-grid">
                <div>
                  <span>Retrieval</span>
                  <strong>Ranking failure</strong>
                </div>
                <div>
                  <span>Generation</span>
                  <strong>{generate ? 'Enabled' : 'Skipped'}</strong>
                </div>
                <div>
                  <span>Citations</span>
                  <strong>{generate ? 'Valid' : '—'}</strong>
                </div>
                <div>
                  <span>Top K</span>
                  <strong>3</strong>
                </div>
              </div>

              <p className="connection-note">
                This is the UI shell. The next step connects these results to
                the real FastAPI backend.
              </p>
            </div>
          )}
        </div>
      </section>

      <section className="pipeline">
        <span>Question</span>
        <b>→</b>
        <span>pgvector retrieval</span>
        <b>→</b>
        <span>Generation</span>
        <b>→</b>
        <span>Citation diagnostics</span>
        <b>→</b>
        <span>Verdict</span>
      </section>
    </main>
  )
}

export default App
