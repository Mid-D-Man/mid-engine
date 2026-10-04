const rows = document.querySelectorAll('.tree-row');
const inspectorTitle = document.querySelector('#inspectorTitle');
const playButton = document.querySelector('#playButton');
const simulationLabel = document.querySelector('.timeline-label');

rows.forEach((row) => {
  row.addEventListener('click', () => {
    rows.forEach((item) => item.classList.remove('selected'));
    row.classList.add('selected');
    inspectorTitle.textContent = row.dataset.node || 'Selection';
  });
});

document.querySelectorAll('.tool').forEach((tool) => {
  tool.addEventListener('click', () => {
    document.querySelectorAll('.tool').forEach((item) => item.classList.remove('active'));
    tool.classList.add('active');
  });
});

document.querySelectorAll('.rail-button[data-panel]').forEach((button) => {
  button.addEventListener('click', () => {
    document.querySelectorAll('.rail-button').forEach((item) => item.classList.remove('active'));
    button.classList.add('active');
  });
});

let playing = false;
playButton.addEventListener('click', () => {
  playing = !playing;
  playButton.textContent = playing ? 'Ⅱ' : '▶';
  simulationLabel.innerHTML = playing ? 'SIMULATION <span class="pulse"></span> RUNNING' : 'SIMULATION <span class="pulse"></span>';
  simulationLabel.style.color = playing ? 'var(--green)' : 'var(--dim)';
});

document.querySelectorAll('.toggle').forEach((toggle) => {
  toggle.addEventListener('click', () => toggle.classList.toggle('on'));
});

document.querySelector('.scene-search input').addEventListener('input', (event) => {
  const query = event.target.value.toLowerCase();
  rows.forEach((row) => {
    row.hidden = query.length > 0 && !row.textContent.toLowerCase().includes(query);
  });
});
