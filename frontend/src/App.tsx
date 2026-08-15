import { useEffect, useState } from 'react'
import './App.css'

type BenchmarkQuestion = {
  question_id: string
  question: string
  answerable: boolean
}

type RetrievedChunk = {
  rank: number
  chunk_id: string
  score: number
}

type RetrievalAutopsyResponse = {
  question_id: string
  question: string
  diagnosis: string
  relevant_chunk_ids: string[]
  retrieved_chunks: RetrievedChunk[]
}

type FullAutopsyResponse = {
  question_id: string
  question: string
  primary_diagnosis: string
  primary_explanation: string
  retrieval_diagnosis: string
  generation: {
    answer: string
    cited_chunk_ids: string[]
    invalid_citation_ids: string[]
  }
  citation_validity: string
  citation_support: string
  citation_coverage: string
  citation_coverage_score: number
  retrieved_chunks: RetrievedChunk[]
}

type AutopsyResponse =
  | RetrievalAutopsyResponse
  | FullAutopsyResponse

function isFullAutopsy(
  result: AutopsyResponse,
): result is FullAutopsyResponse {
  return 'primary_diagnosis' in result
}

function App() {
  const [questions, setQuestions] = useState<BenchmarkQuestion[]>([])
  const [selectedId, setSelectedId] = useState('q031')
  const [generate, setGenerate] = useState(false)
  const [loadingQuestions, setLoadingQuestions] = useState(true)
  const [questionError, setQuestionError] = useState('')

  const [result, setResult] = useState<AutopsyResponse | null>(null)
  const [running, setRunning] = useState(false)
  const [autopsyError, setAutopsyError] = useState('')

  useEffect(() => {
    async function loadQuestions() {
      try {
        const response = await fetch('/api/questions')

        if (!response.ok) {
          throw new Error(
            'Failed to load benchmark questions.',
          )
        }

        const data =
          (await response.json()) as BenchmarkQuestion[]

        setQuestions(data)

        if (
          !data.some(
            (item) => item.question_id === 'q031',
          )
        ) {
          setSelectedId(
            data[0]?.question_id ?? '',
          )
        }
      } catch {
        setQuestionError(
          'Could not connect to the RAG Autopsy backend.',
        )
      } finally {
        setLoadingQuestions(false)
      }
    }

    void loadQuestions()
  }, [])

  const selectedQuestion = questions.find(
    (item) => item.question_id === selectedId,
  )

  async function runAutopsy() {
    if (!selectedQuestion) {
      return
    }

    setRunning(true)
    setAutopsyError('')
    setResult(null)

    try {
      const response = await fetch('/api/autopsy', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question_id: selectedId,
          generate,
          top_k: 3,
        }),
      })

      if (!response.ok) {
        const body = (await response.json()) as {
          detail?: string
        }

        throw new Error(
          body.detail ??
            `Autopsy request failed (${response.status}).`,
        )
      }

      const data =
        (await response.json()) as AutopsyResponse

      setResult(data)
    } catch (error) {
      setAutopsyError(
        error instanceof Error
          ? error.message
          : 'Autopsy request failed.',
      )
    } finally {
      setRunning(false)
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <span className="brand-mark">RA</span>
          <span className="brand-name">RAG Autopsy</span>
        </div>

        <span className="status-pill">
          {questionError
            ? 'Backend unavailable'
            : loadingQuestions
              ? 'Connecting...'
              : 'Backend connected'}
        </span>
      </header>

      <section className="hero-section">
        <p className="eyebrow">
          RAG FAILURE DIAGNOSTICS
        </p>

        <h1>Find where your RAG pipeline failed.</h1>

        <p className="hero-copy">
          Inspect retrieval, generation, citations, and evidence
          support through one diagnostic workflow.
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

          <label
            className="field-label"
            htmlFor="question"
          >
            Benchmark question
          </label>

          <select
            id="question"
            value={selectedId}
            disabled={
              loadingQuestions ||
              Boolean(questionError) ||
              running
            }
            onChange={(event) => {
              setSelectedId(event.target.value)
              setResult(null)
              setAutopsyError('')
            }}
          >
            {loadingQuestions ? (
              <option>Loading questions...</option>
            ) : (
              questions.map((item) => (
                <option
                  key={item.question_id}
                  value={item.question_id}
                >
                  {item.question_id} — {item.question}
                </option>
              ))
            )}
          </select>

          {questionError ? (
            <div className="question-preview">
              <p>{questionError}</p>
            </div>
          ) : selectedQuestion ? (
            <div className="question-preview">
              <span className="question-id">
                {selectedQuestion.question_id}
              </span>

              <p>{selectedQuestion.question}</p>

              <span className="answerable-badge">
                {selectedQuestion.answerable
                  ? 'Answerable'
                  : 'Unanswerable'}
              </span>
            </div>
          ) : null}

          <div className="setting-row">
            <div>
              <strong>
                Generate grounded answer
              </strong>
              <p>
                Run generation and citation diagnostics.
              </p>
            </div>

            <label className="switch">
              <input
                type="checkbox"
                checked={generate}
                disabled={running}
                onChange={(event) => {
                  setGenerate(event.target.checked)
                  setResult(null)
                  setAutopsyError('')
                }}
              />
              <span className="slider"></span>
            </label>
          </div>

          <div className="setting-row">
            <div>
              <strong>Top K</strong>
              <p>
                Number of retrieved chunks to inspect.
              </p>
            </div>

            <span className="top-k">3</span>
          </div>

          <button
            className="run-button"
            type="button"
            disabled={!selectedQuestion || running}
            onClick={() => void runAutopsy()}
          >
            {running
              ? 'Running Autopsy...'
              : 'Run Autopsy'}
          </button>
        </div>

        <div className="results-panel card">
          <div className="section-heading">
            <div>
              <p className="section-label">
                DIAGNOSTICS
              </p>
              <h2>Autopsy result</h2>
            </div>
          </div>

          {running ? (
            <div className="empty-state">
              <div className="empty-icon">◎</div>
              <h3>Running autopsy...</h3>
              <p>
                Retrieving evidence and diagnosing the
                RAG pipeline.
              </p>
            </div>
          ) : autopsyError ? (
            <div className="empty-state">
              <div className="empty-icon">!</div>
              <h3>Autopsy failed</h3>
              <p>{autopsyError}</p>
            </div>
          ) : !result ? (
            <div className="empty-state">
              <div className="empty-icon">◎</div>
              <h3>No autopsy has been run yet</h3>
              <p>
                Select a benchmark question and run the
                pipeline to inspect its diagnosis.
              </p>
            </div>
          ) : isFullAutopsy(result) ? (
            <div className="sample-result">
              <div className="diagnosis-row">
                <span>Primary diagnosis</span>

                <strong className="diagnosis-badge">
                  {result.primary_diagnosis}
                </strong>
              </div>

              <p className="diagnosis-copy">
                {result.primary_explanation}
              </p>

              <div className="metric-grid">
                <div>
                  <span>Retrieval</span>
                  <strong>
                    {result.retrieval_diagnosis}
                  </strong>
                </div>

                <div>
                  <span>Citation validity</span>
                  <strong>
                    {result.citation_validity}
                  </strong>
                </div>

                <div>
                  <span>Citation support</span>
                  <strong>
                    {result.citation_support}
                  </strong>
                </div>

                <div>
                  <span>Citation coverage</span>
                  <strong>
                    {Math.round(
                      result.citation_coverage_score * 100,
                    )}
                    %
                  </strong>
                </div>
              </div>

              <div className="question-preview">
                <span className="question-id">
                  GENERATED ANSWER
                </span>
                <p>{result.generation.answer}</p>
              </div>

              <div className="question-preview">
                <span className="question-id">
                  RETRIEVED CHUNKS
                </span>

                {result.retrieved_chunks.map(
                  (chunk) => (
                    <p key={chunk.chunk_id}>
                      #{chunk.rank} {chunk.chunk_id}
                      {' — '}
                      {chunk.score.toFixed(4)}
                    </p>
                  ),
                )}
              </div>
            </div>
          ) : (
            <div className="sample-result">
              <div className="diagnosis-row">
                <span>Retrieval diagnosis</span>

                <strong className="diagnosis-badge">
                  {result.diagnosis}
                </strong>
              </div>

              <p className="diagnosis-copy">
                {result.question}
              </p>

              <div className="metric-grid">
                <div>
                  <span>Retrieved chunks</span>
                  <strong>
                    {result.retrieved_chunks.length}
                  </strong>
                </div>

                <div>
                  <span>Relevant chunks</span>
                  <strong>
                    {result.relevant_chunk_ids.length}
                  </strong>
                </div>

                <div>
                  <span>Generation</span>
                  <strong>Skipped</strong>
                </div>

                <div>
                  <span>Top K</span>
                  <strong>3</strong>
                </div>
              </div>

              <div className="question-preview">
                <span className="question-id">
                  RETRIEVED CHUNKS
                </span>

                {result.retrieved_chunks.map(
                  (chunk) => (
                    <p key={chunk.chunk_id}>
                      #{chunk.rank} {chunk.chunk_id}
                      {' — '}
                      {chunk.score.toFixed(4)}
                    </p>
                  ),
                )}
              </div>
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
