// Videos load when they approach the viewport and play only while visible.
const videos = document.querySelectorAll('video.lazy');
const io = new IntersectionObserver((entries) => {
  for (const e of entries) {
    const v = e.target;
    if (e.isIntersecting) {
      if (!v.src && v.dataset.src) v.src = v.dataset.src;
      if (v.muted) v.play().catch(() => {});
    } else if (!v.paused) {
      v.pause();
    }
  }
}, { rootMargin: '200px 0px' });
videos.forEach((v) => io.observe(v));

// Head-to-head player: one OmniContact failure mode at a time, auto-advancing through its clips.
const dataEl = document.getElementById('h2h-data');
if (dataEl && document.getElementById('h2h-video')) {
  const data = JSON.parse(dataEl.textContent);
  const video = document.getElementById('h2h-video');
  const tabs = [...document.querySelectorAll('.h2h-tabs [data-mode]')];
  const $ = (id) => document.getElementById(id);
  let mode = 'fell';
  let index = 0;
  let visible = false;

  const show = (autoplay) => {
    const clips = data.modes[mode];
    const c = clips[index];
    video.poster = c.poster;
    video.src = c.src;
    $('h2h-title').innerHTML = '';
    $('h2h-title').append(c.title);
    if (c.tag) {
      const tag = document.createElement('span');
      tag.className = 'tag';
      tag.textContent = `ours: ${c.tag}`;
      $('h2h-title').append(' ', tag);
    }
    $('h2h-ours').textContent = `Ours: ✓ ${c.ours}`;
    $('h2h-oc').textContent = `OmniContact: ✗ ${c.oc}`;
    $('h2h-speed').textContent = c.speed !== 1 ? `${c.speed}× speed` : '';
    $('h2h-caption').textContent = c.caption;
    $('h2h-count').textContent = `${index + 1} / ${clips.length}`;
    $('h2h-prev').disabled = $('h2h-next').disabled = clips.length < 2;
    tabs.forEach((t) => {
      const on = t.dataset.mode === mode;
      t.classList.toggle('on', on);
      t.setAttribute('aria-selected', on);
    });
    document.querySelectorAll('.bar [data-mode]').forEach((s) => s.classList.toggle('dim', s.dataset.mode !== mode && s.dataset.mode !== 'success'));
    if (autoplay) video.play().catch(() => {});
  };

  tabs.forEach((t) => t.addEventListener('click', () => { mode = t.dataset.mode; index = 0; show(true); }));
  $('h2h-next').addEventListener('click', () => { index = (index + 1) % data.modes[mode].length; show(true); });
  $('h2h-prev').addEventListener('click', () => { const n = data.modes[mode].length; index = (index + n - 1) % n; show(true); });
  // At the end of a clip, play the next one of the same mode (or replay a single clip).
  video.addEventListener('ended', () => { index = (index + 1) % data.modes[mode].length; show(true); });
  video.addEventListener('click', () => (video.paused ? video.play() : video.pause()));

  // Outcome bars.
  const NAMES = { success: 'Success', fell: 'Fell', never: 'Never lifted', holding: 'Still holding at the end', drop: 'Dropped / set down off the goal', other: 'Other' };
  const bar = (el, rows) => {
    const total = rows.reduce((s, [, n]) => s + n, 0);
    for (const [key, n] of rows) {
      if (!n) continue;
      const seg = document.createElement(key in data.modes ? 'button' : 'span');
      seg.className = `seg-${key}`;
      seg.dataset.mode = key;
      seg.style.flexGrow = n;
      seg.title = `${NAMES[key]}: ${n} of ${total}`;
      if (n / total > 0.045) seg.textContent = key === 'success' ? `${(100 * n / total).toFixed(1)}%` : n;
      if (key in data.modes) seg.addEventListener('click', () => { mode = key; index = 0; show(true); });
      el.append(seg);
    }
  };
  bar($('bar-oc'), data.outcomes.oc);
  if ($('bar-boxonly') && data.outcomes.boxonly) bar($('bar-boxonly'), data.outcomes.boxonly);
  bar($('bar-ours'), data.outcomes.ours);
  $('bar-legend').innerHTML = Object.entries(NAMES)
    .map(([k, v]) => `<span><i class="seg-${k}"></i>${v}</span>`).join('');

  show(false);
  new IntersectionObserver((entries) => {
    visible = entries.some((e) => e.isIntersecting);
    if (visible) video.play().catch(() => {});
    else video.pause();
  }, { threshold: 0.25 }).observe(video);
}

// Outcome bars above the comparison gallery.
if (dataEl && !document.getElementById('h2h-video')) {
  const data = JSON.parse(dataEl.textContent);
  const NAMES = { success: 'Success', fell: 'Fell', never: 'Never lifted', holding: 'Still holding at the end', drop: 'Dropped / set down off the goal', other: 'Other' };
  const bar = (el, rows) => {
    if (!el || !rows) return;
    const total = rows.reduce((s, [, n]) => s + n, 0);
    for (const [key, n] of rows) {
      if (!n) continue;
      const seg = document.createElement('span');
      seg.className = `seg-${key}`;
      seg.style.flexGrow = n;
      seg.title = `${NAMES[key]}: ${n} of ${total}`;
      if (n / total > 0.045) seg.textContent = key === 'success' ? `${(100 * n / total).toFixed(1)}%` : n;
      el.append(seg);
    }
  };
  bar(document.getElementById('bar-oc'), data.outcomes.oc);
  bar(document.getElementById('bar-boxonly'), data.outcomes.boxonly);
  bar(document.getElementById('bar-ours'), data.outcomes.ours);
  document.getElementById('bar-legend').innerHTML = Object.entries(NAMES)
    .map(([k, v]) => `<span><i class="seg-${k}"></i>${v}</span>`).join('');
}

// Comparison carousel: arrows scroll by one clip and hide at the ends.
for (const car of document.querySelectorAll('.carousel')) {
  const row = car.querySelector('.gallery');
  const prev = car.querySelector('.car-btn.prev');
  const next = car.querySelector('.car-btn.next');
  const step = () => (row.firstElementChild ? row.firstElementChild.getBoundingClientRect().width + 18 : row.clientWidth);
  const count = car.querySelector('.car-count');
  const sync = () => {
    prev.disabled = row.scrollLeft <= 2;
    next.disabled = row.scrollLeft + row.clientWidth >= row.scrollWidth - 2;
    if (count) count.textContent = `${Math.round(row.scrollLeft / step()) + 1} / ${row.children.length}`;
  };
  prev.addEventListener('click', () => row.scrollBy({ left: -step(), behavior: 'smooth' }));
  next.addEventListener('click', () => row.scrollBy({ left: step(), behavior: 'smooth' }));
  row.addEventListener('scroll', sync, { passive: true });
  window.addEventListener('resize', sync);
  sync();
}

// Enlarge any video in an overlay (Esc or a click outside closes it).
const openLightbox = (video, caption) => {
  const box = document.createElement('div');
  box.className = 'lightbox';
  box.setAttribute('role', 'dialog');
  const v = document.createElement('video');
  v.src = video.currentSrc || video.src || video.dataset.src;
  v.muted = true; v.loop = true; v.playsInline = true; v.controls = true; v.autoplay = true;
  if (video.currentTime) v.currentTime = video.currentTime;
  const close = document.createElement('button');
  close.className = 'close'; close.textContent = '×'; close.setAttribute('aria-label', 'Close');
  box.append(close, v);
  if (caption) { const p = document.createElement('p'); p.textContent = caption; box.append(p); }
  const done = () => { box.remove(); document.removeEventListener('keydown', onKey); if (!video.paused) video.play().catch(() => {}); };
  const onKey = (e) => { if (e.key === 'Escape') done(); };
  box.addEventListener('click', (e) => { if (e.target === box) done(); });
  close.addEventListener('click', done);
  document.addEventListener('keydown', onKey);
  video.pause();
  document.body.append(box);
  v.play().catch(() => {});
};
for (const fig of document.querySelectorAll('figure')) {
  const video = fig.querySelector('video');
  if (!video) continue;
  const b = document.createElement('button');
  b.className = 'expand'; b.textContent = '⤢'; b.title = 'Enlarge'; b.setAttribute('aria-label', 'Enlarge video');
  const cap = fig.querySelector('figcaption, .clip-title');
  const open = () => openLightbox(video, cap ? cap.textContent.trim() : '');
  b.addEventListener('click', open);
  const wrap = document.createElement('div');  // the button sits on the video's corner, not the caption's
  wrap.className = 'vwrap';
  video.replaceWith(wrap);
  wrap.append(video, b);
  video.style.cursor = 'zoom-in';
  video.addEventListener('click', open);
}
