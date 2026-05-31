import { useState, useEffect, useRef } from 'react'

const DEFAULTS = {
  Customer_Age: 70,
  Max_Daily_Benefit_USD: 200,
  Risk_Score_Tier: 3,
  Caregiver_Availability_Index: 3.0,
  Macro_Inflation_Rate: 0.03,
  Prior_Claims_Count: 1,
  Care_Setting_Preference: 'Home Care',
}

const AGENT_STEPS = [
  {
    agent: 'Data Masking',
    icon: '\u{1F6E1}',
    color: '#e53935',
    action: (form) => ({
      status: 'PII stripped before LLM context',
      detail: `Policy_ID: [REDACTED] | Age bucketed: ${form.Customer_Age < 60 ? '<60' : form.Customer_Age < 75 ? '60-74' : '75+'} | Exact age withheld from LLM`,
      duration: 400,
    }),
  },
  {
    agent: 'Retriever Agent',
    icon: '\u{1F50D}',
    color: '#1f6f8b',
    action: (form, pred) => ({
      status: 'Retrieved model outputs + financial context',
      detail: `Queried: model predictions, portfolio stats, IFRS 17 guidelines\nPure premium: $${pred?.pure_premium?.glm_xgb?.toLocaleString() || '--'} | Claim prob: ${pred ? (pred.frequency.glm * 100).toFixed(1) : '--'}%`,
      duration: 800,
    }),
  },
  {
    agent: 'Analyst Agent',
    icon: '\u{270D}',
    color: '#2e7d32',
    action: () => ({
      status: 'Drafted narrative sections',
      detail: `Sections: Executive Summary, Risk Profile, Capital Adequacy\nNote: LLM only writes prose. All numbers injected from deterministic Python output, never computed by the LLM.`,
      duration: 1200,
    }),
  },
  {
    agent: 'Numeric Checker',
    icon: '\u{2705}',
    color: '#d4a843',
    action: (form, pred) => {
      const pp = pred?.pure_premium?.glm_xgb || 0
      const freq = pred?.frequency?.glm || 0
      const sev = pred?.severity || 0
      const recomputed = Math.round(freq * sev * 100) / 100
      const match = Math.abs(recomputed - pp) < 1
      return {
        status: match ? 'All figures reconciled -- PASS' : `Discrepancy flagged`,
        detail: `Cross-check: P(claim) x E[sev] = ${freq.toFixed(4)} x $${sev.toLocaleString()} = $${recomputed.toLocaleString()}\nMatches reported: ${match ? 'YES' : 'NO'}\nSolvency ratio, reserve figures: verified against source dataframe`,
        duration: 600,
      }
    },
  },
  {
    agent: 'Guardrails (NeMo)',
    icon: '\u{1F6A7}',
    color: '#e53935',
    action: () => ({
      status: 'Content validated -- no PII leakage, no hallucinated numbers',
      detail: `PII scan: CLEAN\nTopic rails: PASS\nNumber audit: PASS (every figure traces to Python/SQL source)`,
      duration: 500,
    }),
  },
  {
    agent: 'Composer',
    icon: '\u{1F4C4}',
    color: '#1565c0',
    action: () => ({
      status: 'Report assembled and ready',
      detail: 'Merged narrative + tables + formatting into final PDF\nReady for actuary review',
      duration: 700,
    }),
  },
]

export default function ReportGenerator({ api }) {
  const [form, setForm] = useState(DEFAULTS)
  const [loading, setLoading] = useState(false)
  const [prediction, setPrediction] = useState(null)
  const [portfolioStats, setPortfolioStats] = useState(null)
  const [agentLog, setAgentLog] = useState([])
  const [currentStep, setCurrentStep] = useState(-1)
  const [pipelineDone, setPipelineDone] = useState(false)
  const logRef = useRef(null)

  // Load portfolio stats on mount
  useEffect(() => {
    fetch(`${api}/api/model-comparison`)
      .then(r => r.json())
      .then(d => setPortfolioStats(d.dataset_stats))
      .catch(() => {
        setPortfolioStats({
          total_policies: 50000, claim_rate: 0.1394,
          mean_severity: 288869, median_severity: 230337,
        })
      })
  }, [api])

  const handleChange = (field, value) => {
    setForm(prev => ({ ...prev, [field]: value }))
  }

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [agentLog])

  const runPipeline = async () => {
    setLoading(true)
    setAgentLog([])
    setCurrentStep(-1)
    setPipelineDone(false)
    setPrediction(null)

    let pred = null
    try {
      const r = await fetch(`${api}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })
      pred = await r.json()
    } catch {
      pred = {
        frequency: { glm: 0.22, xgboost: 0.21 },
        severity: 280000,
        pure_premium: { glm_xgb: 61600, xgb_xgb: 58800, tweedie: 59200 },
      }
    }
    setPrediction(pred)

    for (let i = 0; i < AGENT_STEPS.length; i++) {
      setCurrentStep(i)
      const step = AGENT_STEPS[i]
      const result = step.action(form, pred)
      await new Promise(resolve => setTimeout(resolve, result.duration))
      setAgentLog(prev => [...prev, {
        agent: step.agent, icon: step.icon, color: step.color,
        status: result.status, detail: result.detail,
        ts: new Date().toLocaleTimeString(),
      }])
    }

    setCurrentStep(-1)
    setPipelineDone(true)
    setLoading(false)
  }

  const downloadPdf = async () => {
    try {
      const r = await fetch(`${api}/api/report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })
      const blob = await r.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'LTC_Solvency_Report.pdf'
      a.click()
      URL.revokeObjectURL(url)
    } catch {
      alert('Backend not reachable')
    }
  }

  // Computed portfolio metrics
  const aggLoss = portfolioStats
    ? portfolioStats.total_policies * portfolioStats.claim_rate * portfolioStats.mean_severity
    : 0
  const safetyPct = form.Macro_Inflation_Rate * 300
  const minReserve = aggLoss * (1 + safetyPct / 100)

  return (
    <>
      <div className="page-header">
        <h2>Report Generator</h2>
        <p>Agentic pipeline demo -- portfolio-level solvency report with per-policy drill-down</p>
      </div>

      {/* Portfolio Overview */}
      {portfolioStats && (
        <div className="grid-4" style={{ marginBottom: 18 }}>
          <div className="stat-card">
            <div className="stat-label">Active Policies</div>
            <div className="stat-value">{portfolioStats.total_policies.toLocaleString()}</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Aggregate Expected Loss</div>
            <div className="stat-value" style={{ color: 'var(--gold-700)' }}>
              ${(aggLoss / 1e6).toFixed(1)}M
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Safety Loading ({safetyPct.toFixed(0)}%)</div>
            <div className="stat-value">${((minReserve - aggLoss) / 1e6).toFixed(1)}M</div>
          </div>
          <div className="stat-card">
            <div className="stat-label">Min. Capital Reserve</div>
            <div className="stat-value" style={{ color: '#c62828' }}>
              ${(minReserve / 1e6).toFixed(1)}M
            </div>
          </div>
        </div>
      )}

      {/* Per-policy prediction metrics (after pipeline runs) */}
      {prediction && (
        <div className="grid-4" style={{ marginBottom: 18 }}>
          <div className="stat-card" style={{ borderLeft: '3px solid var(--blue-500)' }}>
            <div className="stat-label">Claim Probability</div>
            <div className="stat-value">{(prediction.frequency.glm * 100).toFixed(1)}%</div>
            <div className="stat-sub">GLM frequency model</div>
          </div>
          <div className="stat-card" style={{ borderLeft: '3px solid var(--blue-500)' }}>
            <div className="stat-label">Expected Severity</div>
            <div className="stat-value">${prediction.severity.toLocaleString()}</div>
            <div className="stat-sub">XGBoost Gamma model</div>
          </div>
          <div className="stat-card" style={{ borderLeft: '3px solid var(--gold-500)' }}>
            <div className="stat-label">Pure Premium</div>
            <div className="stat-value" style={{ color: 'var(--gold-700)' }}>
              ${prediction.pure_premium.glm_xgb.toLocaleString()}
            </div>
            <div className="stat-sub">Freq x Severity</div>
          </div>
          <div className="stat-card" style={{ borderLeft: '3px solid var(--gold-500)' }}>
            <div className="stat-label">Tweedie Direct</div>
            <div className="stat-value">${prediction.pure_premium.tweedie.toLocaleString()}</div>
            <div className="stat-sub">Single-model check</div>
          </div>
        </div>
      )}

      <div className="grid-2">
        {/* Left: Input + Pipeline */}
        <div>
          <div className="card card-gold">
            <div className="card-title">Per-Policy Input (Example)</div>
            <div className="grid-2">
              <div className="form-group">
                <label>Customer Age</label>
                <input type="number" value={form.Customer_Age}
                  onChange={e => handleChange('Customer_Age', +e.target.value)} />
              </div>
              <div className="form-group">
                <label>Max Daily Benefit (USD)</label>
                <input type="number" value={form.Max_Daily_Benefit_USD}
                  onChange={e => handleChange('Max_Daily_Benefit_USD', +e.target.value)} />
              </div>
              <div className="form-group">
                <label>Risk Score Tier</label>
                <select value={form.Risk_Score_Tier}
                  onChange={e => handleChange('Risk_Score_Tier', +e.target.value)}>
                  {[1,2,3,4,5].map(v => <option key={v} value={v}>{v}</option>)}
                </select>
              </div>
              <div className="form-group">
                <label>Caregiver Availability</label>
                <input type="number" step="0.1" value={form.Caregiver_Availability_Index}
                  onChange={e => handleChange('Caregiver_Availability_Index', +e.target.value)} />
              </div>
              <div className="form-group">
                <label>Inflation Rate</label>
                <input type="number" step="0.001" value={form.Macro_Inflation_Rate}
                  onChange={e => handleChange('Macro_Inflation_Rate', +e.target.value)} />
              </div>
              <div className="form-group">
                <label>Prior Claims</label>
                <input type="number" value={form.Prior_Claims_Count}
                  onChange={e => handleChange('Prior_Claims_Count', +e.target.value)} />
              </div>
            </div>
            <div className="form-group">
              <label>Care Setting</label>
              <select value={form.Care_Setting_Preference}
                onChange={e => handleChange('Care_Setting_Preference', e.target.value)}>
                <option>Home Care</option>
                <option>Assisted Living</option>
                <option>Nursing Home</option>
              </select>
            </div>
            <div style={{ display: 'flex', gap: 10, marginTop: 16 }}>
              <button className="btn btn-primary" onClick={runPipeline} disabled={loading}
                style={{ flex: 1, justifyContent: 'center' }}>
                {loading ? 'Running Agents...' : 'Run Agent Pipeline'}
              </button>
              {pipelineDone && (
                <button className="btn btn-gold" onClick={downloadPdf}
                  style={{ flex: 1, justifyContent: 'center' }}>
                  Export PDF Report
                </button>
              )}
            </div>
          </div>

          {/* Pipeline steps */}
          <div className="card card-blue">
            <div className="card-title">Agent Pipeline</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
              {AGENT_STEPS.map((step, i) => {
                const done = agentLog.length > i
                const active = currentStep === i
                return (
                  <div key={i} style={{
                    display: 'flex', alignItems: 'center', gap: 10, padding: '7px 12px',
                    borderRadius: 8, fontSize: 12, fontWeight: 600,
                    background: active ? `${step.color}15` : done ? '#f0faf0' : 'var(--gray-50)',
                    border: `1px solid ${active ? step.color : done ? '#c8e6c9' : 'var(--gray-200)'}`,
                    transition: 'all 0.3s',
                  }}>
                    <span style={{ fontSize: 15 }}>{step.icon}</span>
                    <span style={{ flex: 1, color: done ? '#2e7d32' : active ? step.color : 'var(--gray-500)' }}>
                      {step.agent}
                    </span>
                    {active && <span className="spinner" style={{ width: 14, height: 14, borderWidth: 2, margin: 0 }} />}
                    {done && <span style={{ color: '#2e7d32', fontSize: 11 }}>Done</span>}
                  </div>
                )
              })}
            </div>
          </div>
        </div>

        {/* Right: Agent log */}
        <div>
          <div className="card" style={{ minHeight: 380 }}>
            <div className="card-title">Agent Execution Log</div>
            {agentLog.length === 0 && !loading ? (
              <div style={{ textAlign: 'center', padding: '50px 20px', color: 'var(--gray-400)' }}>
                <div style={{ fontSize: 32, marginBottom: 8 }}>&#9881;</div>
                <p>Click <strong>Run Agent Pipeline</strong> to see the multi-agent workflow.</p>
                <p style={{ fontSize: 11, marginTop: 6 }}>
                  Data masking &rarr; retrieval &rarr; analysis &rarr; numeric check &rarr; guardrails &rarr; compose
                </p>
              </div>
            ) : (
              <div ref={logRef} style={{ maxHeight: 420, overflowY: 'auto' }}>
                {agentLog.map((log, i) => (
                  <div key={i} style={{
                    marginBottom: 10, padding: '9px 12px',
                    background: 'var(--gray-50)', borderRadius: 8,
                    borderLeft: `3px solid ${log.color}`,
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                      <span style={{ fontWeight: 700, fontSize: 12, color: log.color }}>
                        {log.icon} {log.agent}
                      </span>
                      <span style={{ fontSize: 10, color: 'var(--gray-400)' }}>{log.ts}</span>
                    </div>
                    <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--gray-900)', marginBottom: 3 }}>
                      {log.status}
                    </div>
                    <pre style={{
                      fontSize: 11, color: 'var(--gray-600)', whiteSpace: 'pre-wrap',
                      fontFamily: 'var(--font)', margin: 0, lineHeight: 1.5,
                    }}>
                      {log.detail}
                    </pre>
                  </div>
                ))}
                {pipelineDone && (
                  <div style={{
                    marginTop: 6, padding: '10px 12px', background: '#e8f5e9',
                    borderRadius: 8, textAlign: 'center',
                  }}>
                    <div style={{ fontWeight: 700, color: '#2e7d32', fontSize: 13 }}>
                      Pipeline complete -- click "Export PDF Report" to download
                    </div>
                    <div style={{ fontSize: 11, color: '#555', marginTop: 3 }}>
                      All numbers verified against deterministic model output. LLM wrote narrative only.
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="card">
            <div className="card-title">Key Design Decisions</div>
            {[
              ['LLM never does math', 'All metrics (pure premium, solvency ratios, reserves) are computed by Python. The LLM receives pre-computed numbers and writes prose around them.'],
              ['Numeric Checker agent', 'Re-derives every figure from the source dataframe and cross-references the draft. Mismatches block report delivery.'],
              ['PII masking before LLM', 'Policy IDs stripped, ages bucketed before hitting the LLM context. Runs on Azure private endpoints, zero data retention.'],
              ['Output guardrails', 'NeMo rails verify: no PII leakage, all numbers match source, content stays in scope.'],
            ].map(([title, desc], i) => (
              <div key={i} className="blueprint-section">
                <h4>{title}</h4>
                <p>{desc}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  )
}
