/**
 * TrustLens AI - Client Application Logic
 * Orchestrates multi-modal document upload, live webcam biometric capture,
 * forensic ELA heatmap rendering, Verhoeff checksum inspection, and AI Copilot interaction.
 */

// Global State
let selectedDocFile = null;
let selectedSelfieFile = null;
let currentStream = null;
let lastScreeningData = null;

// DOM Elements
const docDropzone = document.getElementById('doc-dropzone');
const docFileInput = document.getElementById('doc-file-input');
const docPreviewWrap = document.getElementById('doc-preview-wrap');
const docPreviewImg = document.getElementById('doc-preview-img');
const docPreviewName = document.getElementById('doc-preview-name');
const docPreviewSize = document.getElementById('doc-preview-size');
const docDropContent = document.getElementById('doc-dropzone-content');
const btnRemoveDoc = document.getElementById('btn-remove-doc');

const selfieDropzone = document.getElementById('selfie-dropzone');
const selfieFileInput = document.getElementById('selfie-file-input');
const selfiePreviewWrap = document.getElementById('selfie-preview-wrap');
const selfiePreviewImg = document.getElementById('selfie-preview-img');
const selfiePreviewName = document.getElementById('selfie-preview-name');
const selfieDropContent = document.getElementById('selfie-dropzone-content');
const btnRemoveSelfie = document.getElementById('btn-remove-selfie');

const btnOpenCamera = document.getElementById('btn-open-camera');
const webcamDialog = document.getElementById('webcam-dialog');
const webcamVideo = document.getElementById('webcam-video');
const webcamCanvas = document.getElementById('webcam-canvas');
const btnCloseWebcam = document.getElementById('btn-close-webcam');
const btnCaptureShutter = document.getElementById('btn-capture-shutter');

const btnRunScreening = document.getElementById('btn-run-screening');
const resultsEmptyState = document.getElementById('results-empty-state');
const resultsLoadingState = document.getElementById('results-loading-state');
const resultsDashboard = document.getElementById('results-dashboard');

const btnExportReport = document.getElementById('btn-export-report');
const toastShelf = document.getElementById('toast-shelf');

// Copilot Elements
const copilotDrawer = document.getElementById('copilot-drawer');
const btnOpenCopilotNav = document.getElementById('btn-open-copilot-nav');
const btnCloseCopilot = document.getElementById('btn-close-copilot');
const copilotForm = document.getElementById('copilot-form');
const copilotInput = document.getElementById('copilot-input');
const copilotMessages = document.getElementById('copilot-messages');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  setupFileUploads();
  setupWebcam();
  setupTabs();
  setupCopilot();
  setupPresets();
  setupExport();
});

/* ==========================================================================
   File Upload & Drag-and-Drop
   ========================================================================== */

function setupFileUploads() {
  // Document Dropzone
  docDropzone.addEventListener('click', (e) => {
    if (e.target !== btnRemoveDoc && !btnRemoveDoc.contains(e.target)) {
      docFileInput.click();
    }
  });

  docFileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleDocFile(e.target.files[0]);
    }
  });

  ['dragenter', 'dragover'].forEach(eventName => {
    docDropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      docDropzone.classList.add('drag-active');
    }, false);
  });

  ['dragleave', 'drop'].forEach(eventName => {
    docDropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      docDropzone.classList.remove('drag-active');
    }, false);
  });

  docDropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleDocFile(e.dataTransfer.files[0]);
    }
  });

  btnRemoveDoc.addEventListener('click', (e) => {
    e.stopPropagation();
    selectedDocFile = null;
    docFileInput.value = '';
    docPreviewWrap.classList.add('hidden');
    docDropContent.classList.remove('hidden');
  });

  // Selfie Dropzone
  selfieDropzone.addEventListener('click', (e) => {
    if (e.target !== btnRemoveSelfie && !btnRemoveSelfie.contains(e.target)) {
      selfieFileInput.click();
    }
  });

  selfieFileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleSelfieFile(e.target.files[0]);
    }
  });

  btnRemoveSelfie.addEventListener('click', (e) => {
    e.stopPropagation();
    selectedSelfieFile = null;
    selfieFileInput.value = '';
    selfiePreviewWrap.classList.add('hidden');
    selfieDropContent.classList.remove('hidden');
  });

  btnRunScreening.addEventListener('click', executeScreening);
}

function handleDocFile(file) {
  selectedDocFile = file;
  docPreviewName.textContent = file.name;
  docPreviewSize.textContent = formatBytes(file.size);

  const reader = new FileReader();
  reader.onload = (e) => {
    docPreviewImg.src = e.target.result;
    docDropContent.classList.add('hidden');
    docPreviewWrap.classList.remove('hidden');
  };
  reader.readAsDataURL(file);
  showToast(`Document loaded: ${file.name}`, 'info');
}

function handleSelfieFile(file) {
  selectedSelfieFile = file;
  selfiePreviewName.textContent = file.name;

  const reader = new FileReader();
  reader.onload = (e) => {
    selfiePreviewImg.src = e.target.result;
    selfieDropContent.classList.add('hidden');
    selfiePreviewWrap.classList.remove('hidden');
  };
  reader.readAsDataURL(file);
  showToast('Selfie uploaded for biometric comparison', 'info');
}

/* ==========================================================================
   Live Webcam Selfie Capture
   ========================================================================== */

function setupWebcam() {
  btnOpenCamera.addEventListener('click', async () => {
    try {
      webcamDialog.showModal();
      currentStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }
      });
      webcamVideo.srcObject = currentStream;
    } catch (err) {
      showToast('Camera access denied or unavailable: ' + err.message, 'error');
      webcamDialog.close();
    }
  });

  const stopCamera = () => {
    if (currentStream) {
      currentStream.getTracks().forEach(track => track.stop());
      currentStream = null;
    }
    webcamDialog.close();
  };

  btnCloseWebcam.addEventListener('click', stopCamera);

  btnCaptureShutter.addEventListener('click', () => {
    if (!currentStream) return;

    webcamCanvas.width = webcamVideo.videoWidth || 640;
    webcamCanvas.height = webcamVideo.videoHeight || 480;
    const ctx = webcamCanvas.getContext('2d');

    // Flip canvas horizontally to match mirrored video preview
    ctx.translate(webcamCanvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);

    webcamCanvas.toBlob((blob) => {
      const capturedFile = new File([blob], 'live_webcam_selfie.jpg', { type: 'image/jpeg' });
      handleSelfieFile(capturedFile);
      stopCamera();
      showToast('Live selfie captured successfully!', 'success');
    }, 'image/jpeg', 0.95);
  });
}

/* ==========================================================================
   Quick Demo Presets
   ========================================================================== */

function setupPresets() {
  const presets = [
    { btnId: 'preset-aadhaar', file: 'sample_valid_aadhaar.png', type: 'AADHAAR' },
    { btnId: 'preset-pan', file: 'sample_tampered_pan.png', type: 'PAN' },
    { btnId: 'preset-passport', file: 'sample_passport.png', type: 'PASSPORT' }
  ];

  presets.forEach(p => {
    const btn = document.getElementById(p.btnId);
    if (!btn) return;
    btn.addEventListener('click', async () => {
      try {
        showToast(`Loading demo preset: ${p.file}...`, 'info');
        const res = await fetch(`/api/samples/${p.file}`);
        if (!res.ok) throw new Error('Could not load preset file');
        const blob = await res.blob();
        const file = new File([blob], p.file, { type: blob.type });

        handleDocFile(file);
        document.getElementById('doc-type-select').value = p.type;

        // Auto load matching selfie if authentic aadhaar or passport
        if (p.type === 'AADHAAR' || p.type === 'PASSPORT') {
          const sRes = await fetch('/api/samples/sample_selfie.png');
          if (sRes.ok) {
            const sBlob = await sRes.blob();
            const sFile = new File([sBlob], 'sample_selfie.png', { type: 'image/png' });
            handleSelfieFile(sFile);
          }
        }
      } catch (err) {
        showToast('Error loading preset: ' + err.message, 'error');
      }
    });
  });
}

/* ==========================================================================
   Screening Execution & Pipeline
   ========================================================================== */

async function executeScreening() {
  if (!selectedDocFile) {
    showToast('Please select or drop an identity document image first.', 'error');
    docDropzone.classList.add('drag-active');
    setTimeout(() => docDropzone.classList.remove('drag-active'), 800);
    return;
  }

  // Switch UI to loading state
  resultsEmptyState.classList.add('hidden');
  resultsDashboard.classList.add('hidden');
  resultsLoadingState.classList.remove('hidden');

  // Staged loading step indicator animation
  const steps = [
    document.getElementById('step-ocr'),
    document.getElementById('step-val'),
    document.getElementById('step-tamper'),
    document.getElementById('step-face'),
    document.getElementById('step-risk')
  ];

  let currentStepIdx = 0;
  const stepInterval = setInterval(() => {
    if (currentStepIdx < steps.length - 1) {
      steps[currentStepIdx].classList.remove('step-active');
      steps[currentStepIdx].classList.add('step-done');
      steps[currentStepIdx].innerHTML = `<i class="fa-solid fa-check"></i> ${steps[currentStepIdx].textContent.replace('...', ' complete')}`;
      currentStepIdx++;
      steps[currentStepIdx].classList.add('step-active');
      steps[currentStepIdx].innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> ${steps[currentStepIdx].textContent}`;
    }
  }, 450);

  const formData = new FormData();
  formData.append('document', selectedDocFile);
  if (selectedSelfieFile) {
    formData.append('selfie', selectedSelfieFile);
  }
  formData.append('doc_type', document.getElementById('doc-type-select').value);

  const claimedName = document.getElementById('claimed-name-input').value.trim();
  const claimedDob = document.getElementById('claimed-dob-input').value.trim();
  if (claimedName) formData.append('claimed_name', claimedName);
  if (claimedDob) formData.append('claimed_dob', claimedDob);

  try {
    const response = await fetch('/api/screen', {
      method: 'POST',
      body: formData
    });

    clearInterval(stepInterval);

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || 'Screening failed on server');
    }

    const data = await response.json();
    lastScreeningData = data;
    btnExportReport.removeAttribute('disabled');

    // Render results
    setTimeout(() => {
      renderDashboard(data);
      resultsLoadingState.classList.add('hidden');
      resultsDashboard.classList.remove('hidden');
      showToast(`Document Screening Finished in ${data.processing_time_sec}s`, 'success');
    }, 600);

  } catch (err) {
    clearInterval(stepInterval);
    resultsLoadingState.classList.add('hidden');
    resultsEmptyState.classList.remove('hidden');
    showToast(`Screening error: ${err.message}`, 'error');
  }
}

/* ==========================================================================
   Render Dashboard Results
   ========================================================================== */

function renderDashboard(data) {
  const { ocr, validation, tamper, face, risk, processing_time_sec } = data;

  // 1. Risk Header Banner & Gauge
  const scoreVal = Math.round(risk.risk_score);
  animateScoreGauge(scoreVal, risk.badge_color);

  const tierBadge = document.getElementById('risk-tier-badge');
  tierBadge.textContent = risk.action === 'APPROVE' ? 'APPROVED - AUTHENTIC' :
                          risk.action === 'MANUAL_REVIEW' ? 'MANUAL REVIEW REQUIRED' : 'REJECTED - FRAUD DETECTED';
  tierBadge.style.backgroundColor = `${risk.badge_color}25`;
  tierBadge.style.color = risk.badge_color;
  tierBadge.style.borderColor = `${risk.badge_color}60`;

  document.getElementById('risk-decision-title').textContent = risk.decision_title;
  document.getElementById('risk-decision-recommendation').textContent = risk.recommendation;
  document.getElementById('metric-doc-type').textContent = ocr.document_type || 'ID DOCUMENT';
  document.getElementById('metric-proc-time').textContent = `${processing_time_sec}s`;
  document.getElementById('metric-ocr-engine').textContent = ocr.extraction_engine || 'AI Engine';

  // 2. Risk Flags
  const flagsContainer = document.getElementById('flags-list-container');
  flagsContainer.innerHTML = '';
  if (!risk.flags || risk.flags.length === 0) {
    flagsContainer.innerHTML = `
      <div class="flag-alert flag-info">
        <i class="fa-solid fa-circle-check"></i>
        <span>No security risks or tampering flags identified. Document passed all integrity thresholds.</span>
      </div>
    `;
  } else {
    risk.flags.forEach(f => {
      const flagDiv = document.createElement('div');
      const cls = f.severity === 'CRITICAL' ? 'flag-critical' : f.severity === 'WARNING' ? 'flag-warning' : 'flag-info';
      const icon = f.severity === 'CRITICAL' ? 'fa-triangle-exclamation' : 'fa-circle-exclamation';
      flagDiv.className = `flag-alert ${cls}`;
      flagDiv.innerHTML = `
        <i class="fa-solid ${icon}"></i>
        <div><strong>[${f.category}]</strong> ${f.message}</div>
      `;
      flagsContainer.appendChild(flagDiv);
    });
  }

  // 3. Mathematical & Structural Validation Matrix
  const checksTbody = document.getElementById('checks-table-body');
  checksTbody.innerHTML = '';
  if (validation.checks) {
    validation.checks.forEach(c => {
      const row = document.createElement('tr');
      const badgeCls = c.passed ? 'check-pass' : 'check-fail';
      const badgeIcon = c.passed ? 'fa-check' : 'fa-xmark';
      const statusText = c.passed ? 'PASSED' : 'FAILED';
      row.innerHTML = `
        <td><strong>${c.field}</strong></td>
        <td><code>${c.rule}</code></td>
        <td><span class="check-status-badge ${badgeCls}"><i class="fa-solid ${badgeIcon}"></i> ${statusText}</span></td>
        <td>${c.details}</td>
      `;
      checksTbody.appendChild(row);
    });
  }

  // 4. Mini Pillar Cards
  const b = risk.breakdown || {};
  setMiniScore('score-ocr-risk', 'bar-ocr-risk', b.ocr_risk || 0);
  setMiniScore('score-val-risk', 'bar-val-risk', b.validation_risk || 0);
  setMiniScore('score-tamper-risk', 'bar-tamper-risk', b.tamper_risk || 0);
  setMiniScore('score-bio-risk', 'bar-bio-risk', b.biometric_risk || 0);

  // 5. Tab 2: ELA Tamper Comparison
  if (docPreviewImg.src) {
    document.getElementById('tamper-view-orig').src = docPreviewImg.src;
  }
  const heatmapImg = document.getElementById('tamper-view-heatmap');
  if (tamper.heatmap_base64) {
    heatmapImg.src = tamper.heatmap_base64;
  }

  document.getElementById('t-stat-composite').textContent = `${tamper.tamper_score}/100`;
  document.getElementById('t-stat-ela').textContent = `${tamper.ela_score}/100`;
  document.getElementById('t-stat-noise').textContent = `${tamper.noise_score}/100`;
  document.getElementById('t-stat-edge').textContent = `${tamper.edge_score}/100`;

  const findingsBox = document.getElementById('tamper-findings-box');
  findingsBox.innerHTML = '<strong>Forensic Observations:</strong><br>';
  if (tamper.findings) {
    tamper.findings.forEach(f => {
      findingsBox.innerHTML += `• ${f}<br>`;
    });
  }

  // ELA Toggle Buttons
  const btnJet = document.getElementById('btn-show-jet');
  const btnDiff = document.getElementById('btn-show-diff');
  btnJet.onclick = () => {
    btnJet.classList.add('active');
    btnDiff.classList.remove('active');
    if (tamper.heatmap_base64) heatmapImg.src = tamper.heatmap_base64;
  };
  btnDiff.onclick = () => {
    btnDiff.classList.add('active');
    btnJet.classList.remove('active');
    if (tamper.ela_image_base64) heatmapImg.src = tamper.ela_image_base64;
  };

  // 6. Tab 3: Biometric Comparison
  const bioIdCrop = document.getElementById('bio-id-crop');
  const bioSelfieCrop = document.getElementById('bio-selfie-crop');
  const bioMatchPct = document.getElementById('bio-match-pct');
  const bioVerdictBadge = document.getElementById('bio-verdict-badge');
  const bioRing = document.getElementById('bio-ring');

  if (face.doc_face_crop_base64) {
    bioIdCrop.src = face.doc_face_crop_base64;
    document.getElementById('bio-id-meta').textContent = 'ID Photo Isolated';
  } else {
    bioIdCrop.src = '';
    document.getElementById('bio-id-meta').textContent = 'No face detected on ID';
  }

  if (face.selfie_face_crop_base64) {
    bioSelfieCrop.src = face.selfie_face_crop_base64;
    document.getElementById('bio-selfie-meta').textContent = 'Selfie Face Isolated';
  } else {
    bioSelfieCrop.src = '';
    document.getElementById('bio-selfie-meta').textContent = face.selfie_provided ? 'Face not found in selfie' : 'No selfie submitted';
  }

  if (face.selfie_provided && face.doc_face_found && face.selfie_face_found) {
    bioMatchPct.textContent = `${Math.round(face.match_score)}%`;
    bioVerdictBadge.textContent = face.is_match ? 'VERIFIED MATCH' : 'BIOMETRIC MISMATCH';
    const bioColor = face.is_match ? '#10b981' : '#ef4444';
    bioRing.style.borderColor = bioColor;
    bioRing.style.boxShadow = `0 0 20px ${bioColor}50`;
    bioVerdictBadge.style.backgroundColor = `${bioColor}25`;
    bioVerdictBadge.style.color = bioColor;
  } else {
    bioMatchPct.textContent = '--';
    bioVerdictBadge.textContent = 'OPTIONAL PENDING';
    bioRing.style.borderColor = '#64748b';
  }

  // Liveness notes
  const livenessNotesList = document.getElementById('liveness-notes-list');
  livenessNotesList.innerHTML = '';
  document.getElementById('liveness-badge').textContent = `Liveness Score: ${face.liveness_score}%`;
  if (face.liveness_notes && face.liveness_notes.length > 0) {
    face.liveness_notes.forEach(note => {
      livenessNotesList.innerHTML += `<li><i class="fa-solid fa-angle-right"></i> ${note}</li>`;
    });
  } else {
    livenessNotesList.innerHTML = '<li><i class="fa-solid fa-check"></i> Standard dynamic range and focus verified.</li>';
  }

  // 7. Tab 4: Extracted OCR Fields
  const ocrGrid = document.getElementById('ocr-fields-grid');
  ocrGrid.innerHTML = '';
  document.getElementById('ocr-confidence-tag').textContent = `Confidence: ${Math.round((ocr.ocr_confidence || 0.8) * 100)}%`;

  const displayFields = [
    { label: 'Document Type', val: ocr.document_type },
    { label: 'Document Number', val: ocr.document_number },
    { label: 'Full Legal Name', val: ocr.full_name },
    { label: 'Date of Birth (DOB)', val: ocr.dob },
    { label: 'Gender', val: ocr.gender },
    { label: 'Father / Spouse Name', val: ocr.father_name },
    { label: 'Issuing Authority', val: ocr.issuing_authority },
    { label: 'Expiry Date', val: ocr.expiry_date }
  ];

  displayFields.forEach(f => {
    if (f.val) {
      const card = document.createElement('div');
      card.className = 'ocr-field-card';
      card.innerHTML = `
        <span class="ocr-field-lbl">${f.label}</span>
        <div class="ocr-field-val">${f.val}</div>
        <button class="ocr-copy-btn" title="Copy value" onclick="copyText('${escapeQuotes(f.val)}')">
          <i class="fa-regular fa-copy"></i>
        </button>
      `;
      ocrGrid.appendChild(card);
    }
  });

  document.getElementById('raw-ocr-text').textContent = ocr.raw_text || 'No raw text buffer available.';
}

function setMiniScore(scoreId, barId, val) {
  const elScore = document.getElementById(scoreId);
  const elBar = document.getElementById(barId);
  if (elScore && elBar) {
    elScore.textContent = `${Math.round(val)}%`;
    elBar.style.width = `${Math.min(100, Math.max(0, val))}%`;
    elBar.style.backgroundColor = val > 60 ? 'var(--accent-rose)' : val > 30 ? 'var(--accent-amber)' : 'var(--accent-emerald)';
  }
}

function animateScoreGauge(score, color) {
  const gaugeBar = document.getElementById('risk-gauge-bar');
  const scoreNum = document.getElementById('gauge-score-value');

  // Full circle circumference for r=66 is 2 * PI * 66 = 414.69
  const circumference = 414.7;
  const offset = circumference - (score / 100) * circumference;

  gaugeBar.style.stroke = color;
  gaugeBar.style.strokeDashoffset = offset;

  // Counter animation
  let current = 0;
  const duration = 1000;
  const stepTime = 20;
  const steps = duration / stepTime;
  const increment = score / steps;

  const timer = setInterval(() => {
    current += increment;
    if (current >= score) {
      scoreNum.textContent = score;
      clearInterval(timer);
    } else {
      scoreNum.textContent = Math.round(current);
    }
  }, stepTime);
}

/* ==========================================================================
   Tabs Management
   ========================================================================== */

function setupTabs() {
  const tabBtns = document.querySelectorAll('.results-tabs .tab-btn');
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetId = btn.getAttribute('data-target');
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add('active');
    });
  });
}

/* ==========================================================================
   AI Copilot Chat Interface
   ========================================================================== */

function setupCopilot() {
  btnOpenCopilotNav.addEventListener('click', () => {
    copilotDrawer.classList.toggle('open');
  });

  btnCloseCopilot.addEventListener('click', () => {
    copilotDrawer.classList.remove('open');
  });

  // Suggested prompt chips
  document.querySelectorAll('.copilot-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const prompt = chip.getAttribute('data-prompt');
      sendCopilotMessage(prompt);
    });
  });

  copilotForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const query = copilotInput.value.trim();
    if (query) {
      sendCopilotMessage(query);
      copilotInput.value = '';
    }
  });
}

async function sendCopilotMessage(userText) {
  // Ensure drawer is open
  copilotDrawer.classList.add('open');

  // Append user bubble
  appendChatBubble(userText, 'user');

  // Append temporary thinking indicator
  const botBubble = appendChatBubble('<i class="fa-solid fa-spinner fa-spin"></i> Consulting TrustLens AI engine...', 'bot', true);

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: userText,
        screening_context: lastScreeningData
      })
    });

    const data = await res.json();
    botBubble.querySelector('.bubble-content').innerHTML = formatMarkdown(data.reply);
  } catch (err) {
    botBubble.querySelector('.bubble-content').innerHTML = `Sorry, I encountered an error: ${err.message}`;
  }
}

function appendChatBubble(text, sender, isHtml = false) {
  const bubble = document.createElement('div');
  bubble.className = `chat-bubble ${sender}-bubble`;

  if (sender === 'bot') {
    bubble.innerHTML = `
      <div class="bubble-header"><i class="fa-solid fa-robot"></i> TrustLens Copilot</div>
      <div class="bubble-content">${isHtml ? text : formatMarkdown(text)}</div>
    `;
  } else {
    bubble.textContent = text;
  }

  copilotMessages.appendChild(bubble);
  copilotMessages.scrollTop = copilotMessages.scrollHeight;
  return bubble;
}

function formatMarkdown(text) {
  if (!text) return '';
  // Basic markdown conversion
  let html = text
    .replace(/^### (.*$)/gim, '<h4 style="color:#00f2fe;margin:4px 0;">$1</h4>')
    .replace(/^## (.*$)/gim, '<h3 style="color:#00f2fe;margin:6px 0;">$1</h3>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`(.*?)`/g, '<code style="background:rgba(255,255,255,0.1);padding:2px 4px;border-radius:3px;">$1</code>')
    .replace(/\n\n/g, '<br><br>')
    .replace(/\n- /g, '<br>• ');
  return html;
}

/* ==========================================================================
   Export Audit Dossier
   ========================================================================== */

function setupExport() {
  btnExportReport.addEventListener('click', () => {
    if (!lastScreeningData) return;
    const jsonStr = JSON.stringify(lastScreeningData, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `TrustLens_Audit_${lastScreeningData.ocr.document_number || 'Report'}_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast('Audit dossier exported as JSON', 'success');
  });
}

/* ==========================================================================
   Utilities
   ========================================================================== */

function formatBytes(bytes, decimals = 1) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

function escapeQuotes(str) {
  if (!str) return '';
  return str.replace(/'/g, "\\'");
}

function copyText(val) {
  navigator.clipboard.writeText(val).then(() => {
    showToast(`Copied: "${val}"`, 'info');
  });
}

function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  const icon = type === 'success' ? 'fa-circle-check' : type === 'error' ? 'fa-circle-exclamation' : 'fa-bell';
  toast.innerHTML = `<i class="fa-solid ${icon}"></i> <span>${message}</span>`;
  toastShelf.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}
