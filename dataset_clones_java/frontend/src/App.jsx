import { useState } from "react";
import axios from "axios";
import "./App.css";

function ShieldIcon() {
  return (
    <svg width="36" height="36" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M18 3L6 8v10c0 7.18 5.15 13.9 12 15.5C24.85 31.9 30 25.18 30 18V8L18 3z" fill="url(#shieldGrad)" />
      <path d="M13 18l3.5 3.5L23 14" stroke="white" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
      <defs>
        <linearGradient id="shieldGrad" x1="6" y1="3" x2="30" y2="33.5" gradientUnits="userSpaceOnUse">
          <stop stopColor="#34d399" />
          <stop offset="1" stopColor="#059669" />
        </linearGradient>
      </defs>
    </svg>
  );
}

function FileIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M11 2H5a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V8l-6-6z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
      <path d="M11 2v6h6" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  );
}

function FolderIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M2 6a2 2 0 012-2h4l2 2h6a2 2 0 012 2v7a2 2 0 01-2 2H4a2 2 0 01-2-2V6z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
    </svg>
  );
}

function ConfidenceBadge({ confidence }) {
  const pct = (confidence * 100).toFixed(1);
  const level = confidence >= 0.85 ? "high" : confidence >= 0.5 ? "medium" : "low";
  return (
    <span className={`confidence-badge confidence-${level}`}>
      {pct}%
    </span>
  );
}

function PredictionBadge({ prediction }) {
  const isClone = prediction && prediction !== "T0";
  const displayPrediction = prediction && prediction.endsWith("T3") ? "T3" : prediction;
  return (
    <span className={`prediction-badge ${isClone ? "badge-clone" : "badge-original"}`}>
      {isClone ? `⚠ Plagio detectado (${displayPrediction})` : "✓ Original"}
    </span>
  );
}

export default function App() {
  const [file1, setFile1] = useState(null);
  const [file2, setFile2] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const [folderFiles, setFolderFiles] = useState([]);
  const [folderResults, setFolderResults] = useState([]);
  const [folderLoading, setFolderLoading] = useState(false);

  const [activeTab, setActiveTab] = useState("pair");

  const handleCompare = async () => {
    if (!file1 || !file2) {
      alert("Selecciona ambos archivos .java");
      return;
    }
    const formData = new FormData();
    formData.append("file1", file1);
    formData.append("file2", file2);
    setLoading(true);
    setResult(null);
    try {
      const response = await axios.post("http://127.0.0.1:8000/predict", formData);
      setResult(response.data);
    } catch (error) {
      console.error(error);
      alert("Error llamando a la API. Verifica que el servidor esté corriendo.");
    } finally {
      setLoading(false);
    }
  };

  const handleFolderCompare = async () => {
    if (folderFiles.length < 2) {
      alert("Selecciona al menos 2 archivos Java");
      return;
    }
    const formData = new FormData();
    for (const file of folderFiles) {
      formData.append("files", file);
    }
    setFolderLoading(true);
    setFolderResults([]);
    try {
      const response = await axios.post("http://127.0.0.1:8000/predict-folder", formData);
      setFolderResults(response.data);
    } catch (error) {
      console.error(error);
      alert("Error procesando carpeta. Verifica que el servidor esté corriendo.");
    } finally {
      setFolderLoading(false);
    }
  };

  const cloneCount = folderResults.filter(r =>
    r.prediction && r.prediction !== "T0"
  ).length;

  return (
    <div className="app-root">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <ShieldIcon />
          <div className="logo-text">
            <span className="logo-title">CloneGuard</span>
            <span className="logo-sub">Detector de Plagio Java</span>
          </div>
        </div>

        <nav className="sidebar-nav">
          <button
            className={`nav-item ${activeTab === "pair" ? "nav-active" : ""}`}
            onClick={() => setActiveTab("pair")}
          >
            <FileIcon />
            <span>Comparar par</span>
          </button>
          <button
            className={`nav-item ${activeTab === "folder" ? "nav-active" : ""}`}
            onClick={() => setActiveTab("folder")}
          >
            <FolderIcon />
            <span>Analizar carpeta</span>
          </button>
        </nav>

        <div className="sidebar-footer">
          <p>Detección basada en<br />análisis de código fuente</p>
        </div>
      </aside>

      {/* Main content */}
      <main className="main-content">

        {/* Header */}
        <header className="page-header">
          <div className="header-inner">
            {activeTab === "pair" ? (
              <>
                <h1 className="page-title">Comparar dos archivos</h1>
                <p className="page-desc">Sube dos archivos <code>.java</code> para analizar si comparten código.</p>
              </>
            ) : (
              <>
                <h1 className="page-title">Analizar carpeta completa</h1>
                <p className="page-desc">Sube múltiples archivos <code>.java</code> y detecta todos los pares sospechosos.</p>
              </>
            )}
          </div>
        </header>

        {/* Tab: Par */}
        {activeTab === "pair" && (
          <section className="section">
            <div className="card">
              <h2 className="card-title">Archivos a comparar</h2>
              <div className="file-grid">
                <div className="file-slot">
                  <label className="file-label">Archivo 1</label>
                  <label className={`file-drop ${file1 ? "file-drop--active" : ""}`}>
                    <input
                      type="file"
                      accept=".java"
                      className="file-input-hidden"
                      onChange={(e) => setFile1(e.target.files[0])}
                    />
                    <span className="drop-icon">
                      <FileIcon />
                    </span>
                    <span className="drop-text">
                      {file1 ? file1.name : "Haz clic o arrastra un .java"}
                    </span>
                  </label>
                </div>

                <div className="file-divider">
                  <span>VS</span>
                </div>

                <div className="file-slot">
                  <label className="file-label">Archivo 2</label>
                  <label className={`file-drop ${file2 ? "file-drop--active" : ""}`}>
                    <input
                      type="file"
                      accept=".java"
                      className="file-input-hidden"
                      onChange={(e) => setFile2(e.target.files[0])}
                    />
                    <span className="drop-icon">
                      <FileIcon />
                    </span>
                    <span className="drop-text">
                      {file2 ? file2.name : "Haz clic o arrastra un .java"}
                    </span>
                  </label>
                </div>
              </div>

              <div className="card-footer">
                <button
                  className="btn-primary"
                  onClick={handleCompare}
                  disabled={loading}
                >
                  {loading ? (
                    <><span className="spinner" /> Analizando...</>
                  ) : (
                    "Comparar archivos"
                  )}
                </button>
              </div>
            </div>

            {result && (
              <div className={`result-card ${(result.prediction && result.prediction !== "T0") ? "result-card--alert" : "result-card--clear"}`}>
                <div className="result-header">
                  <h2 className="result-title">Resultado del análisis</h2>
                </div>
                <div className="result-body">
                  <div className="result-row">
                    <span className="result-label">Predicción</span>
                    <PredictionBadge prediction={result.prediction} />
                  </div>
                  <div className="result-row">
                    <span className="result-label">Confianza del modelo</span>
                    <div className="confidence-wrap">
                      <ConfidenceBadge confidence={result.confidence} />
                      <div className="confidence-bar-track">
                        <div
                          className="confidence-bar-fill"
                          style={{ width: `${(result.confidence * 100).toFixed(1)}%` }}
                        />
                      </div>
                    </div>
                  </div>
                  <div className="result-row">
                    <span className="result-label">Archivos comparados</span>
                    <span className="result-files">
                      {file1?.name} <span className="vs-small">vs</span> {file2?.name}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </section>
        )}

        {/* Tab: Carpeta */}
        {activeTab === "folder" && (
          <section className="section">
            <div className="card">
              <h2 className="card-title">Seleccionar archivos</h2>
              <p className="card-hint">Selecciona una carpeta entera o múltiples archivos <code>.java</code>. El sistema generará todas las comparaciones posibles.</p>

              <label className={`file-drop file-drop--wide ${folderFiles.length > 0 ? "file-drop--active" : ""}`}>
                <input
                  type="file"
                  multiple
                  className="file-input-hidden"
                  onChange={(e) => setFolderFiles(Array.from(e.target.files))}
                  webkitdirectory=""
                />
                <span className="drop-icon">
                  <FolderIcon />
                </span>
                <span className="drop-text">
                  {folderFiles.length > 0
                    ? `${folderFiles.length} archivo${folderFiles.length !== 1 ? "s" : ""} seleccionado${folderFiles.length !== 1 ? "s" : ""}`
                    : "Haz clic para seleccionar una carpeta"}
                </span>
              </label>

              {folderFiles.length > 0 && (
                <div className="file-list">
                  <p className="file-list-title">Archivos cargados:</p>
                  <ul className="file-chips">
                    {folderFiles.map((file, i) => (
                      <li key={i} className="file-chip">
                        <FileIcon /> {file.name}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              <div className="card-footer">
                <button
                  className="btn-primary"
                  onClick={handleFolderCompare}
                  disabled={folderLoading}
                >
                  {folderLoading ? (
                    <><span className="spinner" /> Procesando...</>
                  ) : (
                    "Analizar carpeta"
                  )}
                </button>
              </div>
            </div>

            {folderResults.length > 0 && (
              <div className="results-section">
                <div className="results-summary">
                  <div className="summary-stat">
                    <span className="stat-number">{folderResults.length}</span>
                    <span className="stat-label">Comparaciones</span>
                  </div>
                  <div className="summary-stat summary-stat--alert">
                    <span className="stat-number">{cloneCount}</span>
                    <span className="stat-label">Posibles plagios</span>
                  </div>
                  <div className="summary-stat summary-stat--clear">
                    <span className="stat-number">{folderResults.length - cloneCount}</span>
                    <span className="stat-label">Originales</span>
                  </div>
                </div>

                <div className="table-card">
                  <h2 className="card-title">Resultados detallados</h2>
                  <div className="table-wrap">
                    <table className="results-table">
                      <thead>
                        <tr>
                          <th>#</th>
                          <th>Archivo 1</th>
                          <th>Archivo 2</th>
                          <th>Predicción</th>
                          <th>Confianza</th>
                        </tr>
                      </thead>
                      <tbody>
                        {folderResults.map((row, index) => {
                          const isClone = row.prediction && row.prediction !== "T0";
                          return (
                            <tr key={index} className={isClone ? "row-alert" : ""}>
                              <td className="td-num">{index + 1}</td>
                              <td className="td-file">{row.file1}</td>
                              <td className="td-file">{row.file2}</td>
                              <td>
                                <PredictionBadge prediction={row.prediction} />
                              </td>
                              <td>
                                <ConfidenceBadge confidence={row.confidence} />
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  );
}