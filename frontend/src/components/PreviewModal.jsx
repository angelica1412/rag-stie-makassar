import React from 'react';
import './PreviewModal.css';

function PreviewModal({ file, onClose }) {
  if (!file) return null;

  const isPdf = file.filename.toLowerCase().endsWith('.pdf');
  const previewUrl = 'http://localhost:8000/preview/' + file.filename;
  const downloadUrl = 'http://localhost:8000/download/' + file.filename;
  const fileExt = file.filename.split('.').pop().toUpperCase();

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={function(e) { e.stopPropagation(); }}>

        <div className="modal-header">
          <h3>{'📄 ' + file.name}</h3>
          <div className="modal-actions">
            <a href={downloadUrl} download={file.filename} className="btn-download">
              Download
            </a>
            <button className="btn-close" onClick={onClose}>X</button>
          </div>
        </div>

        <div className="modal-body">
          {isPdf ? (
            <iframe
              src={previewUrl}
              title={file.name}
              width="100%"
              height="100%"
              style={{ border: 'none' }}
            />
          ) : (
            <div className="no-preview">
              <p>Preview tidak tersedia untuk format {fileExt}</p>
              <p>Silakan download file untuk membukanya</p>
              <a href={downloadUrl} download={file.filename} className="btn-download-large">
                Download {fileExt}
              </a>
            </div>
          )}
        </div>

      </div>
    </div>
  );
}

export default PreviewModal;