const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const snapBtn = document.getElementById('snap');
const startBtn = document.getElementById('start');
const preview = document.getElementById('preview');
const selfieInput = document.getElementById('selfie_data');

let stream = null;

startBtn?.addEventListener('click', async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
  } catch (e) {
    alert('Could not start camera: ' + e);
  }
});

snapBtn?.addEventListener('click', () => {
  if (!stream) { alert('Start the camera first'); return; }
  const ctx = canvas.getContext('2d');
  canvas.width = video.videoWidth || 480;
  canvas.height = video.videoHeight || 360;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL('image/jpeg', 0.95);
  selfieInput.value = dataUrl;
  preview.src = dataUrl;
});


// ===== Decision Modal =====
(function(){
  const data = document.getElementById('result-data');
  if (!data) return;

  const result = (data.dataset.result || '').trim();
  const live = (data.dataset.live || '').trim();
  if (!result) return; // nothing to show

  const modal = document.getElementById('decision-modal');
  const title = document.getElementById('modal-title');
  const text  = document.getElementById('modal-text');
  const close = document.getElementById('modal-close');
  const okBtn = document.getElementById('modal-ok');

  // Set content + color
  title.textContent = (result === 'Verified') ? '✅ Verified' :
                      (result === 'No match') ? '❌ No match' :
                      'ℹ️ Verification';
  text.textContent = `Decision: ${result} • Liveness: ${live}`;

  modal.classList.remove('hidden');
  modal.classList.add(result === 'Verified' ? 'modal-ok' : 'modal-bad');

  function hide() { modal.classList.add('hidden'); }
  close.addEventListener('click', hide);
  okBtn.addEventListener('click', hide);
  modal.addEventListener('click', (e)=>{ if(e.target === modal) hide(); });
  window.addEventListener('keydown', (e)=>{ if(e.key === 'Escape') hide(); });
});

// ===== Decision Modal (with admin manual verify on failure) =====
(function(){
  const data = document.getElementById('result-data');
  if (!data) return;

  const result = (data.dataset.result || '').trim();
  if (!result) return;

  const live = (data.dataset.live || '').trim();
  const n = (data.dataset.ocrName || '').trim();
  const d = (data.dataset.ocrDob || '').trim();
  const a = (data.dataset.ocrAddr || '').trim();

  const modal = document.getElementById('decision-modal');
  const title = document.getElementById('modal-title');
  const text  = document.getElementById('modal-text');
  const extra = document.getElementById('modal-extra');
  const close = document.getElementById('modal-close');
  const okBtn = document.getElementById('modal-ok');

  const success = (result === 'Verified successfully');

  title.textContent = success ? '✅ Verified successfully' : "❗ couldn't verify face";
  text.textContent  = `Liveness: ${live}`;

  // Always show OCR fields
  const rows = [];
  if (n) rows.push(`<li><strong>Name:</strong> ${n}</li>`);
  if (d) rows.push(`<li><strong>Date of Birth:</strong> ${d}</li>`);
  if (a) rows.push(`<li><strong>Address:</strong> ${a}</li>`);
  extra.innerHTML = rows.join('');

  // Admin manual verify controls (only on failure)
  let adminWrap = null;
  if (!success) {
    adminWrap = document.createElement('div');
    adminWrap.className = 'mt';
    adminWrap.innerHTML = `
      <div class="admin-box">
        <label>Admin password
          <input type="password" id="admin-pass" placeholder="••••••••••" />
        </label>
        <button class="primary" id="admin-verify">Mark as Verified</button>
        <span id="admin-msg" class="muted"></span>
      </div>`;
    extra.parentElement.appendChild(adminWrap);

    const passEl = adminWrap.querySelector('#admin-pass');
    const btnEl  = adminWrap.querySelector('#admin-verify');
    const msgEl  = adminWrap.querySelector('#admin-msg');

    btnEl.addEventListener('click', async () => {
      msgEl.textContent = 'Verifying...';
      btnEl.disabled = true;
      try {
        const res = await fetch('/admin/verify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ password: passEl.value, name: n, address: a })
        });
        const j = await res.json();
        if (!j.ok) {
          msgEl.textContent = j.error || 'Failed';
          btnEl.disabled = false;
          return;
        }
        // Success: flip the modal to success state
        title.textContent = '✅ Verified successfully';
        modal.classList.add('modal-ok');
        modal.classList.remove('modal-bad');
        msgEl.textContent = 'Saved.';
        adminWrap.remove();
      } catch (e) {
        msgEl.textContent = 'Network error';
        btnEl.disabled = false;
      }
    });
  }

  modal.classList.remove('hidden');
  modal.classList.toggle('modal-ok', success);
  modal.classList.toggle('modal-bad', !success);

  function hide(){ modal.classList.add('hidden'); }
  close.addEventListener('click', hide);
  okBtn.addEventListener('click', hide);
  modal.addEventListener('click', (e)=>{ if(e.target === modal) hide(); });
  window.addEventListener('keydown', (e)=>{ if(e.key === 'Escape') hide(); });
})();