import { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell
} from 'recharts'

const DEFAULTS = {
  Customer_Age: 70,
  Max_Daily_Benefit_USD: 200,
  Risk_Score_Tier: 3,
  Caregiver_Availability_Index: 3.0,
  Macro_Inflation_Rate: 0.03,
  Prior_Claims_Count: 1,
  Care_Setting_Preference: 'Home Care',
}

export default function PredictionEngine({ api }) {
  const [form, setForm] = useState(DEFAULTS)
  const [result, setResult] = useState(null)
  const [samples, setSamples] = useState([])
  const [comparison, setComparison] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetch(`${api}/api/samples`).then(r => r.json()).then(setSamples).catch(() => {})
    fetch(`${api}/api/model-comparison`).then(r => r.json()).then(setComparison).catch(() => {})
  }, [api])

  const handleChange = (field, value) => {
    setForm(prev => ({ ...prev, [field]: value }))
  }

  const loadSample = (s) => {
    setForm({
      Customer_Age: s.Customer_Age,
      Max_Daily_Benefit_USD: s.Max_Daily_Benefit_USD,
      Risk_Score_Tier: s.Risk_Score_Tier,
      Caregiver_Availability_Index: s.Caregiver_Availability_Index,
      Macro_Inflation_Rate: s.Macro_Inflation_Rate,
      Prior_Claims_Count: s.Prior_Claims_Count,
      Care_Setting_Preference: s.Care_Setting_Preference,
    })
  }

  const predict = async () => {
    setLoading(true)
    try {
      const r = await fetch(`${api}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })
      setResult(await r.json())
    } catch (e) {
      alert('Backend not reachable. Start it with: python backend/main.py')
    }
    setLoading(false)
  }

  // Build feature importance chart data
  const importanceData = result ? result.feature_importance.features.map((f, i) => ({
    name: f.replace('_', ' ').replace('Setting ', ''),
    GLM: +(result.feature_importance.glm_global[i] * 100).toFixed(1),
    XGBoost: +(result.feature_importance.xgb_global[i] * 100).toFixed(1),
  })).sort((a, b) => b.XGBoost - a.XGBoost) : []

  return (
    <>
      <div className="page-header">
        <h2>Prediction Engine</h2>
        <p>Input policyholder data to estimate expected loss (pure premium)</p>
      </div>

      <div className="grid-2">
        {/* Left: Input Form */}
        <div>
          <div className="card card-blue">
            <div className="card-title">Policyholder Input</div>

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
                <label>Risk Score Tier (1-5)</label>
                <select value={form.Risk_Score_Tier}
                  onChange={e => handleChange('Risk_Score_Tier', +e.target.value)}>
                  {[1,2,3,4,5].map(v => <option key={v} value={v}>{v}</option>)}
                </select>
              </div>
              <div className="form-group">
                <label>Caregiver Availability Index</label>
                <input type="number" step="0.1" value={form.Caregiver_Availability_Index}
                  onChange={e => handleChange('Caregiver_Availability_Index', +e.target.value)} />
              </div>
              <div className="form-group">
                <label>Macro Inflation Rate</label>
                <input type="number" step="0.001" value={form.Macro_Inflation_Rate}
                  onChange={e => handleChange('Macro_Inflation_Rate', +e.target.value)} />
              </div>
              <div className="form-group">
                <label>Prior Claims Count</label>
                <input type="number" value={form.Prior_Claims_Count}
                  onChange={e => handleChange('Prior_Claims_Count', +e.target.value)} />
              </div>
            </div>

            <div className="form-group">
              <label>Care Setting Preference</label>
              <select value={form.Care_Setting_Preference}
                onChange={e => handleChange('Care_Setting_Preference', e.target.value)}>
                <option>Home Care</option>
                <option>Assisted Living</option>
                <option>Nursing Home</option>
              </select>
            </div>

            <div style={{ display: 'flex', gap: 10, marginTop: 16 }}>
              <button className="btn btn-primary" onClick={predict} disabled={loading}>
                {loading ? 'Predicting...' : 'Run Prediction'}
              </button>
              <button className="btn btn-outline" onClick={() => setForm(DEFAULTS)}>
                Reset
              </button>
            </div>
          </div>

          {/* Sample data picker */}
          <div className="card">
            <div className="card-title">Load From Dataset</div>
            <div className="table-wrap" style={{ maxHeight: 220, overflowY: 'auto' }}>
              <table>
                <thead>
                  <tr>
                    <th>ID</th><th>Age</th><th>Risk</th><th>Claim</th><th></th>
                  </tr>
                </thead>
                <tbody>
                  {samples.map((s, i) => (
                    <tr key={i}>
                      <td>{s.Policy_ID}</td>
                      <td>{s.Customer_Age}</td>
                      <td>{s.Risk_Score_Tier}</td>
                      <td><span className={`badge ${s.Claim_Occurred ? 'badge-gold' : 'badge-blue'}`}>
                        {s.Claim_Occurred ? 'Yes' : 'No'}
                      </span></td>
                      <td>
                        <button className="btn btn-outline" style={{ padding: '4px 12px', fontSize: 11 }}
                          onClick={() => loadSample(s)}>Load</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Right: Results */}
        <div>
          {result ? (
            <>
              {/* Pure premium results */}
              <div className="card card-gold">
                <div className="card-title">Prediction Results</div>
                <div className="grid-3" style={{ marginBottom: 16 }}>
                  <div className="stat-card">
                    <div className="stat-label">Claim Probability (GLM)</div>
                    <div className="stat-value">{(result.frequency.glm * 100).toFixed(1)}%</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Expected Severity</div>
                    <div className="stat-value">${result.severity.toLocaleString()}</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Pure Premium</div>
                    <div className="stat-value" style={{ color: 'var(--gold-700)' }}>
                      ${result.pure_premium.glm_xgb.toLocaleString()}
                    </div>
                  </div>
                </div>

                <div className="result-row">
                  <span className="result-label">Frequency (GLM)</span>
                  <span className="result-value">{(result.frequency.glm * 100).toFixed(2)}%</span>
                </div>
                <div className="result-row">
                  <span className="result-label">Frequency (XGBoost)</span>
                  <span className="result-value">{(result.frequency.xgboost * 100).toFixed(2)}%</span>
                </div>
                <div className="result-row">
                  <span className="result-label">Severity (XGBoost Gamma)</span>
                  <span className="result-value">${result.severity.toLocaleString()}</span>
                </div>
                <div className="result-row">
                  <span className="result-label">Pure Premium (GLM + XGB)</span>
                  <span className="result-value" style={{ color: 'var(--gold-700)' }}>
                    ${result.pure_premium.glm_xgb.toLocaleString()}
                  </span>
                </div>
                <div className="result-row">
                  <span className="result-label">Pure Premium (XGB + XGB)</span>
                  <span className="result-value">${result.pure_premium.xgb_xgb.toLocaleString()}</span>
                </div>
                <div className="result-row">
                  <span className="result-label">Pure Premium (Tweedie Direct)</span>
                  <span className="result-value">${result.pure_premium.tweedie.toLocaleString()}</span>
                </div>
              </div>

              {/* Feature importance chart */}
              <div className="card card-blue">
                <div className="card-title">Feature Importance (%)</div>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={importanceData} layout="vertical" margin={{ left: 10 }}>
                    <XAxis type="number" tick={{ fontSize: 11 }} />
                    <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 11 }} />
                    <Tooltip formatter={(v) => `${v}%`} />
                    <Bar dataKey="GLM" fill="#1f6f8b" name="GLM" barSize={10} />
                    <Bar dataKey="XGBoost" fill="#d4a843" name="XGBoost" barSize={10} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </>
          ) : (
            <div className="card" style={{ textAlign: 'center', padding: 60 }}>
              <div style={{ fontSize: 40, marginBottom: 12 }}>&#9881;</div>
              <p style={{ color: 'var(--gray-500)' }}>
                Enter policyholder details and click <strong>Run Prediction</strong> to see results.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Model Comparison Tables */}
      {comparison && (
        <div style={{ marginTop: 12 }}>
          <div className="card">
            <div className="card-title">Frequency Model Comparison (Grid-Searched)</div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr><th>Model</th><th>AUC</th><th>Gini</th><th>Predicted Freq</th><th>Actual Freq</th></tr>
                </thead>
                <tbody>
                  {comparison.frequency.map((r, i) => (
                    <tr key={i}>
                      <td><strong>{r.Model}</strong></td>
                      <td>{r.AUC}</td>
                      <td style={{ fontWeight: 700, color: 'var(--blue-900)' }}>{r.Gini}</td>
                      <td>{(r.Pred_Freq * 100).toFixed(2)}%</td>
                      <td>{(r.Actual_Freq * 100).toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card">
            <div className="card-title">Severity Model Comparison (Grid-Searched)</div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr><th>Model</th><th>RMSE</th><th>MAE</th><th>Mean Actual</th></tr>
                </thead>
                <tbody>
                  {comparison.severity.map((r, i) => (
                    <tr key={i}>
                      <td><strong>{r.Model}</strong></td>
                      <td>${Number(r.RMSE).toLocaleString()}</td>
                      <td style={{ fontWeight: 700, color: 'var(--blue-900)' }}>
                        ${Number(r.MAE).toLocaleString()}
                      </td>
                      <td>${Number(r.Mean_Actual).toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
