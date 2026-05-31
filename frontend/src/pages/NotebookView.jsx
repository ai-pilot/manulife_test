export default function NotebookView() {
  return (
    <>
      <div className="page-header">
        <h2>Notebook Walkthrough</h2>
        <p>Interactive analysis notebook showing EDA, model training, and evaluation</p>
      </div>
      <iframe
        className="notebook-frame"
        src="/notebook_walkthrough.html"
        title="LTC Modeling Notebook"
      />
    </>
  )
}
