import { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend
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

const MODEL_INFO = {
  frequency: [
    {
      name: 'Logistic GLM',
      desc: 'Generalized Linear Model with logit link. Fully interpretable coefficients for regulatory filings. Baseline model.',
      params: 'Solver: lbfgs | Penalty: L2 | C: grid-searched',
      strengths: 'Interpretable, stable, auditable',
    },
    {
      name: 'XGBoost',
      desc: 'Gradient boosted trees with binary logistic objective. Captures non-linear interactions between risk factors.',
      params: 'max_depth: 4 | n_estimators: 200 | learning_rate: 0.05 | 3-fold CV',
      strengths: 'Handles interactions, robust to outliers',
    },
    {
      name: 'LightGBM',
      desc: 'Histogram-based gradient boosting. Faster training with leaf-wise growth strategy.',
      params: 'num_leaves: 31 | n_estimators: 200 | learning_rate: 0.05 | 3-fold CV',
      strengths: 'Fast training, memory efficient',
    },
    {
      name: 'CatBoost',
      desc: 'Ordered boosting with symmetric trees. Native handling of categorical features without encoding.',
      params: 'depth: 6 | iterations: 200 | learning_rate: 0.05 | 3-fold CV',
      strengths: 'Handles categoricals natively, reduces overfitting',
    },
    {
      name: 'Random Forest',
      desc: 'Bagged ensemble of decision trees. Provides variance reduction through averaging many decorrelated trees.',
      params: 'n_estimators: 200 | max_depth: grid-searched | 3-fold CV',
      strengths: 'Low variance, feature importance via permutation',
    },
  ],
  severity: [
    {
      name: 'XGBoost (Gamma)',
      desc: 'Gradient boosted trees with Gamma objective (reg:gamma). Suited for positive, right-skewed continuous targets like claim costs.',
      params: 'max_depth: 5 | n_estimators: 300 | learning_rate: 0.03 | 3-fold CV',
      strengths: 'Handles heavy right tail, positive predictions',
    },
    {
      name: 'LightGBM (Gamma)',
      desc: 'Histogram-based boosting with Gamma deviance loss. Fast severity estimation with natural handling of skewed distributions.',
      params: 'num_leaves: 31 | n_estimators: 300 | objective: gamma | 3-fold CV',
      strengths: 'Fast, handles skewed severity',
    },
    {
      name: 'CatBoost (RMSE)',
      desc: 'Ordered boosting for severity. Uses RMSE objective with positive value constraint for claim amount prediction.',
      params: 'depth: 6 | iterations: 300 | learning_rate: 0.03 | 3-fold CV',
      strengths: 'Robust to noise, ordered boosting reduces leakage',
    },
    {
      name: 'Random Forest',
      desc: 'Bagged ensemble for severity. Provides stable severity estimates with built-in uncertainty quantification.',
      params: 'n_estimators: 200 | max_depth: grid-searched | 3-fold CV',
      strengths: 'Stable predictions, confidence intervals via tree variance',
    },
  ],
}

export default function PredictionEngine({ api }) {
  const [form, setForm] = useState(DEFAULTS)
  const [result, setResult] = useState(null)
  const [samples, setSamples] = useState([])
  const [comparison, setComparison] = useState(null)
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('predict') // predict | models

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
        <p>LTC Pure Premium estimation using frequency-severity decomposition</p>
      </div>

      {/* Tab Toggle */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        <button className={`btn ${activeTab === 'predict' ? 'btn-primary' : 'btn-outline'}`}
          onClick={() => setActiveTab('predict')}>
          Run Predictions
        </button>
        <button className={`btn ${activeTab === 'models' ? 'btn-primary' : 'btn-outline'}`}
          onClick={() => setActiveTab('models')}>
          Models & Methodology
        </button>
      </div>

      {activeTab === 'models' && (
        <>
          {/* Methodology Overview */}
          <div className="card card-blue" style={{ marginBottom: 20 }}>
            <div className="card-title">Modeling Approach</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
              <div style={{ padding: '14px', background: 'rgba(218, 238, 245, 0.4)', borderRadius: 12, textAlign: 'center' }}>
                <div style={{ fontSize: 28, marginBottom: 6 }}>&#9881;</div>
                <h4 style={{ fontSize: 13, color: 'var(--blue-900)', marginBottom: 4 }}>Stage 1: Frequency</h4>
                <p style={{ fontSize: 11, color: 'var(--gray-600)', lineHeight: 1.5 }}>
                  P(Claim) - Binary classification estimating claim probability per policyholder
                </p>
              </div>
              <div style={{ padding: '14px', background: 'rgba(250, 243, 224, 0.4)', borderRadius: 12, textAlign: 'center' }}>
                <div style={{ fontSize: 28, marginBottom: 6 }}>&#128200;</div>
                <h4 style={{ fontSize: 13, color: 'var(--blue-900)', marginBottom: 4 }}>Stage 2: Severity</h4>
                <p style={{ fontSize: 11, color: 'var(--gray-600)', lineHeight: 1.5 }}>
                  E[Cost|Claim] - Regression on claimants only, predicting expected payout amount
                </p>
              </div>
              <div style={{ padding: '14px', background: 'rgba(232, 245, 233, 0.4)', borderRadius: 12, textAlign: 'center' }}>
                <div style={{ fontSize: 28, marginBottom: 6 }}>&#9733;</div>
                <h4 style={{ fontSize: 13, color: 'var(--blue-900)', marginBottom: 4 }}>Pure Premium</h4>
                <p style={{ fontSize: 11, color: 'var(--gray-600)', lineHeight: 1.5 }}>
                  P(Claim) x E[Cost|Claim] = Expected Loss per policy. Also validated with Tweedie GLM.
                </p>
              </div>
            </div>
          </div>

          {/* Frequency Models */}
          <div className="card">
            <div className="card-title">Frequency Models (Claim Probability)</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              {MODEL_INFO.frequency.map((m, i) => (
                <div key={i} className="model-info-card">
                  <h4>{m.name}</h4>
                  <p>{m.desc}</p>
                  <p style={{ marginTop: 6, fontSize: 10, color: 'var(--gray-500)' }}>
                    <strong>Hyperparameters:</strong> {m.params}
                  </p>
                  <span className="metric">{m.strengths}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Severity Models */}
          <div className="card">
            <div className="card-title">Severity Models (Claim Amount)</div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              {MODEL_INFO.severity.map((m, i) => (
                <div key={i} className="model-info-card">
                  <h4>{m.name}</h4>
                  <p>{m.desc}</p>
                  <p style={{ marginTop: 6, fontSize: 10, color: 'var(--gray-500)' }}>
                    <strong>Hyperparameters:</strong> {m.params}
                  </p>
                  <span className="metric">{m.strengths}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Comparison Tables */}
          {comparison && (
            <>
              <div className="card card-gold">
                <div className="card-title">Frequency Model Comparison (Grid-Searched)</div>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr><th>Model</th><th>AUC</th><th>Gini</th><th>Predicted Freq</th><th>Actual Freq</th><th>Calibration</th></tr>
                    </thead>
                    <tbody>
                      {comparison.frequency.map((r, i) => (
                        <tr key={i}>
                          <td><strong>{r.Model}</strong></td>
                          <td>{r.AUC}</td>
                          <td style={{ fontWeight: 700, color: 'var(--blue-900)' }}>{r.Gini}</td>
                          <td>{(r.Pred_Freq * 100).toFixed(2)}%</td>
                          <td>{(r.Actual_Freq * 100).toFixed(2)}%</td>
                          <td style={{ color: Math.abs(r.Pred_Freq - r.Actual_Freq) < 0.005 ? '#2e7d32' : '#e65100' }}>
                            {(r.Pred_Freq / r.Actual_Freq).toFixed(3)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="card card-gold">
                <div className="card-title">Severity Model Comparison (Grid-Searched)</div>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr><th>Model</th><th>RMSE</th><th>MAE</th><th>Mean Actual</th><th>Ratio</th></tr>
                    </thead>
                    <tbody>
                      {comparison.severity.map((r, i) => (
                        <tr key={i}>
                          <td><strong>{r.Model}</strong></td>
                          <td>${Number(r.RMSE).toLocaleString()}</td>
                          <td style={{ fontWeight: 700, color: 'var(--blue-900)' }}>${Number(r.MAE).toLocaleString()}</td>
                          <td>${Number(r.Mean_Actual).toLocaleString()}</td>
                          <td>{(Number(r.MAE) / Number(r.Mean_Actual)).toFixed(3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}

          {/* Feature Info */}
          <div className="card">
            <div className="card-title">Feature Engineering</div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr><th>Feature</th><th>Type</th><th>Description</th><th>Expected Effect</th></tr>
                </thead>
                <tbody>
                  <tr><td><strong>Customer_Age</strong></td><td>Numeric</td><td>Policyholder age at valuation</td><td>Positive - older age increases claim likelihood</td></tr>
                  <tr><td><strong>Max_Daily_Benefit_USD</strong></td><td>Numeric</td><td>Maximum daily benefit cap</td><td>Positive - higher benefit correlates with higher severity</td></tr>
                  <tr><td><strong>Risk_Score_Tier</strong></td><td>Ordinal (1-5)</td><td>Underwriting risk classification</td><td>Positive - higher tier means higher risk</td></tr>
                  <tr><td><strong>Caregiver_Availability_Index</strong></td><td>Numeric</td><td>Regional caregiver supply score</td><td>Negative - low availability pushes to expensive care settings</td></tr>
                  <tr><td><strong>Macro_Inflation_Rate</strong></td><td>Numeric</td><td>Current macroeconomic inflation</td><td>Positive - drives up care costs</td></tr>
                  <tr><td><strong>Prior_Claims_Count</strong></td><td>Count</td><td>Number of prior claims filed</td><td>Positive - prior claims indicate future risk</td></tr>
                  <tr><td><strong>Care_Setting_Preference</strong></td><td>Categorical</td><td>Preferred care setting (one-hot encoded)</td><td>Nursing Home most expensive, Home Care least</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {activeTab === 'predict' && (
        <>
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
                        <div className="stat-label">Claim Probability</div>
                        <div className="stat-value">{(result.frequency.glm * 100).toFixed(1)}%</div>
                        <div className="stat-sub">GLM frequency</div>
                      </div>
                      <div className="stat-card">
                        <div className="stat-label">Expected Severity</div>
                        <div className="stat-value">${result.severity.toLocaleString()}</div>
                        <div className="stat-sub">XGBoost Gamma</div>
                      </div>
                      <div className="stat-card">
                        <div className="stat-label">Pure Premium</div>
                        <div className="stat-value" style={{ color: 'var(--gold-700)' }}>
                          ${result.pure_premium.glm_xgb.toLocaleString()}
                        </div>
                        <div className="stat-sub">Freq x Severity</div>
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
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        <Bar dataKey="GLM" fill="#1f6f8b" name="GLM (coefficients)" barSize={10} radius={[0, 4, 4, 0]} />
                        <Bar dataKey="XGBoost" fill="#d4a843" name="XGBoost (gain)" barSize={10} radius={[0, 4, 4, 0]} />
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
                  <p style={{ color: 'var(--gray-400)', fontSize: 12, marginTop: 8 }}>
                    Models: GLM + XGBoost (frequency) | XGBoost Gamma (severity) | Tweedie (benchmark)
                  </p>
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </>
  )
}
