export default function Architecture() {
  return (
    <>
      <div className="page-header">
        <h2>GenAI Solution Architecture</h2>
        <p>Workflow design for automated actuarial valuation and solvency reporting</p>
      </div>

      <div className="grid-2">
        {/* Left: Architecture Diagram */}
        <div>
          <div className="card card-blue">
            <div className="card-title">System Architecture</div>

            {/* Layer 1 */}
            <div className="arch-layer">
              <div className="arch-layer-header">Layer 1 &middot; Data Sources</div>
              <div className="arch-layer-body">
                <div className="arch-box arch-box-blue">Part 1 Model<sub>Pure premium output</sub></div>
                <div className="arch-box arch-box-blue">Financials<sub>GL, reserves, SQL</sub></div>
                <div className="arch-box arch-box-blue">Macro Data<sub>Rates, inflation</sub></div>
                <div className="arch-box arch-box-blue">Regulations<sub>IFRS 17, RBC docs</sub></div>
              </div>
            </div>
            <div className="arch-connector">&#x25BC;</div>

            {/* Layer 2 */}
            <div className="arch-layer">
              <div className="arch-layer-header">Layer 2 &middot; Ingestion & API &middot; Azure</div>
              <div className="arch-layer-body">
                <div className="arch-box arch-box-gold">FastAPI Endpoint<sub>App Service / AKS</sub></div>
                <div className="arch-box arch-box-gold">Data Factory<sub>ETL into Blob / ADLS</sub></div>
                <div className="arch-box arch-box-gold">Azure AI Search<sub>Vector index (RAG)</sub></div>
              </div>
            </div>
            <div className="arch-connector">&#x25BC;</div>

            {/* Layer 3 */}
            <div className="arch-layer">
              <div className="arch-layer-header">Layer 3 &middot; Orchestration &middot; LangGraph Agents</div>
              <div className="arch-layer-body" style={{ background: '#f0f0ff' }}>
                <div className="arch-box arch-box-blue">Retriever<sub>Grounds figures</sub></div>
                <div className="arch-box arch-box-blue">Analyst Agent<sub>Drafts narrative</sub></div>
                <div className="arch-box arch-box-gold">Numeric Checker<sub>Reconciles totals</sub></div>
                <div className="arch-box arch-box-blue">Composer<sub>Builds report</sub></div>
              </div>
            </div>
            <div className="arch-connector">&#x25BC;</div>

            {/* Layer 4 */}
            <div className="arch-layer">
              <div className="arch-layer-header">Layer 4 &middot; Guardrails & Evaluation</div>
              <div className="arch-layer-body">
                <div className="arch-box arch-box-gold">NeMo Guardrails<sub>Topic, PII, output rails</sub></div>
                <div className="arch-box arch-box-gold">RAGAS Evaluation<sub>Faithfulness, context recall</sub></div>
              </div>
            </div>
            <div className="arch-connector">&#x25BC;</div>

            {/* Layer 5 */}
            <div className="arch-layer">
              <div className="arch-layer-header">Layer 5 &middot; Human-in-the-Loop & Delivery</div>
              <div className="arch-layer-body">
                <div className="arch-box arch-box-green">Actuary Review Console<sub>Approve, annotate, track</sub></div>
                <div className="arch-box arch-box-green">Approved Report<sub>To executive leadership</sub></div>
              </div>
            </div>

            {/* Cross-cutting */}
            <div style={{
              marginTop: 16, padding: '10px 14px', background: '#e8f5e9',
              borderRadius: 8, border: '1px solid #66bb6a', fontSize: 12, textAlign: 'center',
              color: '#2e7d32', fontWeight: 600
            }}>
              Cross-cutting &middot; LangSmith Tracing &middot; Azure Key Vault &middot; Entra ID RBAC &middot; Audit Log
              <div style={{ fontWeight: 400, fontSize: 10, color: '#555', marginTop: 3 }}>
                Observability, security, and governance span every layer
              </div>
            </div>
          </div>
        </div>

        {/* Right: Technical Blueprint */}
        <div>
          <div className="card card-gold">
            <div className="card-title">Technical Blueprint</div>

            <div className="blueprint-section">
              <h4>Layer 1 &mdash; Data Sources</h4>
              <p>
                The Part 1 model outputs (pure premium predictions, frequency/severity scores,
                feature importances) are stored as structured JSON/Parquet artifacts. Financial
                data is pulled from the General Ledger and reserve systems via SQL connectors.
                Macroeconomic data (inflation rates, interest rates) is ingested from external
                feeds. Regulatory documents (IFRS 17, RBC guidelines) are chunked and embedded
                for RAG retrieval.
              </p>
            </div>

            <div className="blueprint-section">
              <h4>Layer 2 &mdash; Ingestion & API (Azure)</h4>
              <p>
                <strong>FastAPI on Azure App Service / AKS</strong> serves as the prediction
                API gateway, accepting policyholder inputs and returning model scores. <strong>Azure
                Data Factory</strong> orchestrates ETL pipelines that move raw data into Azure
                Blob / ADLS in standardized schemas. <strong>Azure AI Search</strong> maintains
                a vector index over regulatory documents and historical reports, enabling
                grounded retrieval for the LLM agents.
              </p>
            </div>

            <div className="blueprint-section">
              <h4>Layer 3 &mdash; LangGraph Multi-Agent Orchestration</h4>
              <p>
                A <strong>LangGraph</strong> state machine coordinates four specialized agents
                running on <strong>Azure OpenAI</strong>:
              </p>
              <p style={{ marginTop: 6 }}>
                <strong>Retriever</strong> &mdash; Queries the vector index and SQL sources to
                pull the exact financial figures, model outputs, and regulatory citations needed
                for each report section. Grounds every claim in source data.
              </p>
              <p style={{ marginTop: 4 }}>
                <strong>Analyst Agent</strong> &mdash; Drafts narrative commentary (executive
                summary, risk assessment, capital adequacy analysis) using retrieved data as
                context. Uses structured prompts aligned with actuarial reporting standards.
              </p>
              <p style={{ marginTop: 4 }}>
                <strong>Numeric Checker</strong> &mdash; Cross-validates every number in the
                draft against source data. Flags mismatches, rounding errors, and inconsistencies
                before the report moves forward. This agent is critical for financial data accuracy.
              </p>
              <p style={{ marginTop: 4 }}>
                <strong>Composer</strong> &mdash; Assembles the final PDF/HTML report with proper
                formatting, tables, charts, and section structure. Handles template rendering and
                branding compliance.
              </p>
            </div>

            <div className="blueprint-section">
              <h4>Layer 4 &mdash; Guardrails & Evaluation</h4>
              <p>
                <strong>NeMo Guardrails</strong> enforces topic boundaries (no off-topic
                responses), PII detection and redaction, and output format rails (ensuring
                the LLM doesn't hallucinate financial figures). <strong>RAGAS evaluation</strong>
                scores each generated report on faithfulness (are claims supported by retrieved
                context?) and context recall (did the retriever find all relevant documents?).
                Reports below threshold are flagged for manual review.
              </p>
            </div>

            <div className="blueprint-section">
              <h4>Layer 5 &mdash; Human-in-the-Loop & Delivery</h4>
              <p>
                The <strong>Actuary Review Console</strong> presents the generated report with
                inline annotations showing source citations, confidence scores, and flagged
                discrepancies. Actuaries can approve, edit, or reject sections. Approved reports
                are versioned and delivered to executive leadership via secure channels.
                All review actions are captured in an audit log for regulatory traceability.
              </p>
            </div>

            <div className="blueprint-section">
              <h4>Cross-Cutting &mdash; Observability & Security</h4>
              <p>
                <strong>LangSmith</strong> traces every LLM call with inputs, outputs, latency,
                and token usage for debugging and cost monitoring. <strong>Azure Key Vault</strong>
                manages API keys and secrets. <strong>Entra ID (Azure AD) RBAC</strong> controls
                who can run reports, approve outputs, and access financial data. A comprehensive
                <strong> audit log</strong> tracks every action from data ingestion through final
                report delivery, satisfying regulatory compliance requirements.
              </p>
            </div>
          </div>

          {/* Hallucination Prevention */}
          <div className="card" style={{ borderLeft: '4px solid #e53935' }}>
            <div className="card-title" style={{ color: '#c62828' }}>Solving the Hallucination Problem</div>
            <div className="blueprint-section" style={{ borderLeftColor: '#e53935' }}>
              <h4>The LLM never does math</h4>
              <p>
                All quantitative metrics — pure premiums, solvency ratios, reserve figures,
                loss ratios — are calculated <strong>deterministically by Python code</strong> (Part 1 model).
                The LLM receives pre-computed numbers as structured context and only writes
                narrative prose around them. It never performs arithmetic.
              </p>
            </div>
            <div className="blueprint-section" style={{ borderLeftColor: '#e53935' }}>
              <h4>Numeric Checker agent (output validation)</h4>
              <p>
                A dedicated agent re-derives every figure from the source DataFrame/SQL
                and compares against the drafted report using Pydantic output parsers.
                If any number doesn't match the deterministic source, the report is
                blocked and flagged for human review. This catches any case where the
                LLM might paraphrase a number incorrectly (e.g. rounding, unit errors).
              </p>
            </div>
            <div className="blueprint-section" style={{ borderLeftColor: '#e53935' }}>
              <h4>NeMo Guardrails output layer</h4>
              <p>
                NeMo output rails intercept the final generated text and cross-reference
                every number against the original Python output. Topic rails ensure the
                LLM stays within actuarial scope and doesn't generate off-topic commentary.
                PII rails scan for any leaked identifiers before delivery.
              </p>
            </div>
          </div>

          {/* DLP / Data Privacy */}
          <div className="card" style={{ borderLeft: '4px solid var(--gold-500)' }}>
            <div className="card-title">Data Privacy & DLP Strategy</div>
            <div className="blueprint-section">
              <h4>Pre-processing data masking</h4>
              <p>
                Before any data enters the LLM context window, a preprocessing node strips
                all <strong>Policy_IDs</strong>, masks exact <strong>Customer_Age</strong> into
                buckets (e.g. "60-74"), and removes any other PII/PHI fields. The LLM
                never sees raw identifiers — only anonymized, bucketed attributes.
              </p>
            </div>
            <div className="blueprint-section">
              <h4>Private cloud deployment</h4>
              <p>
                The entire pipeline runs on <strong>Azure OpenAI private endpoints</strong> within
                the corporate VNet — data never leaves the corporate network. Azure OpenAI
                provides <strong>zero data retention</strong> guarantees (no training on customer data,
                no prompt logging outside the tenant). Alternatively deployable on AWS Bedrock
                or on-prem GPU inference for maximum data sovereignty.
              </p>
            </div>
            <div className="blueprint-section">
              <h4>Encryption & access control</h4>
              <p>
                Data at rest: Azure Storage SSE (AES-256). Data in transit: TLS 1.3.
                Access control: Entra ID RBAC with least-privilege roles — separate permissions
                for model inference, report generation, and report approval. All access logged
                to immutable audit trail for regulatory review.
              </p>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
