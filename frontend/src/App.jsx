import { useState, useRef } from 'react'
import './App.css'

const CLASS_COLORS = {
  cardboard: '#d97706',
  glass:     '#0891b2',
  metal:     '#6b7280',
  paper:     '#3b82f6',
  plastic:   '#f59e0b',
  trash:     '#ef4444',
}

const CLASS_INFO = {
  cardboard: 'Flatten and recycle in the paper/cardboard bin',
  glass:     'Rinse and place in the glass recycling bin',
  metal:     'Rinse cans and recycle in the metals bin',
  paper:     'Recycle in the paper bin',
  plastic:   'Check local guidelines for plastic type',
  trash:     'Dispose in general waste',
}

function App() {
  const [image, setImage] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef(null)

  function handleFile(file) {
    if (!file) return
    setImage(file)
    setPreview(URL.createObjectURL(file))
    setResult(null)
    setError(null)
  }

  function reset() {
    setImage(null)
    setPreview(null)
    setResult(null)
    setError(null)
  }

  function handleDrop(e) {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }

  function handleDragOver(e) {
    e.preventDefault()
    setDragging(true)
  }

  async function classify() {
    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('image', image)

    try {
      const res = await fetch('https://adamlabbate-waste-classifier.hf.space/predict', {
        method: 'POST',
        body: formData,
      })
      const data = await res.json()
      if (data.error) {
        setError(data.error)
      } else {
        setResult(data.class)
      }
    } catch {
      setError('Could not reach the server.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <div className="header">
        <h1>Waste Classifier</h1>
        <p className="subtitle">Drop in a photo to find out how to sort your waste</p>
      </div>

      <div
        className={`upload-zone${dragging ? ' dragging' : ''}${preview ? ' has-preview' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={() => setDragging(false)}
        onClick={() => !preview && inputRef.current.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          onChange={e => handleFile(e.target.files[0])}
          hidden
        />
        {preview ? (
          <img src={preview} alt="Selected" className="preview" />
        ) : (
          <div className="upload-prompt">
            <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="17 8 12 3 7 8"/>
              <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
            <p className="upload-hint">Drop an image here</p>
            <p className="upload-sub">or <span className="upload-link">click to browse</span></p>
          </div>
        )}
      </div>

      {image && !result && (
        <div className="actions">
          {loading ? (
            <div className="spinner" />
          ) : (
            <>
              <button className="classify-btn" onClick={classify}>Classify</button>
              <button className="reset-btn" onClick={reset}>Clear</button>
            </>
          )}
        </div>
      )}

      {error && <p className="error">{error}</p>}

      {result && (
        <div className="result" style={{ borderColor: CLASS_COLORS[result] }}>
          <span className="result-label">Identified as</span>
          <span className="result-value" style={{ color: CLASS_COLORS[result] }}>
            {result}
          </span>
          <span className="result-info">{CLASS_INFO[result]}</span>
          <button className="reset-btn" onClick={reset}>Try another image</button>
        </div>
      )}
    </div>
  )
}

export default App
