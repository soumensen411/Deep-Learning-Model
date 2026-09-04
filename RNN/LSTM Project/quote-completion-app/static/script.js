const seedInput = document.getElementById('seed-input');
const nWordsInput = document.getElementById('n-words');
const suggestBtn = document.getElementById('suggest-btn');
const generateBtn = document.getElementById('generate-btn');
const copyBtn = document.getElementById('copy-btn');
const statusEl = document.getElementById('status');

const suggestionsSection = document.getElementById('suggestions-section');
const suggestionsList = document.getElementById('suggestions-list');
const resultSection = document.getElementById('result-section');
const resultText = document.getElementById('result-text');

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle('error', isError);
}

function setBusy(busy) {
  suggestBtn.disabled = busy;
  generateBtn.disabled = busy;
}

function currentText() {
  return seedInput.value.trim();
}

async function postJSON(url, payload) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || `Request failed (${response.status})`);
  }
  return data;
}

async function handleSuggest() {
  const text = currentText();
  if (!text) {
    setStatus('Write a few words first.', true);
    return;
  }

  setBusy(true);
  setStatus('Thinking...');
  resultSection.classList.add('hidden');

  try {
    const data = await postJSON('/api/predict', { text, k: 6 });
    renderSuggestions(data.suggestions || []);
    setStatus('');
  } catch (err) {
    setStatus(err.message, true);
    suggestionsSection.classList.add('hidden');
  } finally {
    setBusy(false);
  }
}

async function handleGenerate() {
  const text = currentText();
  if (!text) {
    setStatus('Write a few words first.', true);
    return;
  }

  const nWords = Math.max(1, Math.min(30, parseInt(nWordsInput.value, 10) || 5));

  setBusy(true);
  setStatus('Writing...');
  suggestionsSection.classList.add('hidden');

  try {
    const data = await postJSON('/api/generate', { text, n_words: nWords });
    resultText.textContent = data.completed;
    resultSection.classList.remove('hidden');
    setStatus('');
  } catch (err) {
    setStatus(err.message, true);
    resultSection.classList.add('hidden');
  } finally {
    setBusy(false);
  }
}

function renderSuggestions(suggestions) {
  suggestionsList.innerHTML = '';

  if (suggestions.length === 0) {
    suggestionsSection.classList.add('hidden');
    setStatus('No suggestions for that text.');
    return;
  }

  suggestions.forEach(({ word, probability }) => {
    const li = document.createElement('li');
    li.tabIndex = 0;

    const wordSpan = document.createElement('span');
    wordSpan.textContent = word;

    const probSpan = document.createElement('span');
    probSpan.className = 'prob';
    probSpan.textContent = `${Math.round(probability * 100)}%`;

    li.appendChild(wordSpan);
    li.appendChild(probSpan);

    li.addEventListener('click', () => appendWord(word));
    li.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        appendWord(word);
      }
    });

    suggestionsList.appendChild(li);
  });

  suggestionsSection.classList.remove('hidden');
}

function appendWord(word) {
  const text = currentText();
  seedInput.value = text ? `${text} ${word}` : word;
  seedInput.focus();
  seedInput.setSelectionRange(seedInput.value.length, seedInput.value.length);
  handleSuggest();
}

async function handleCopy() {
  try {
    await navigator.clipboard.writeText(resultText.textContent);
    const original = copyBtn.textContent;
    copyBtn.textContent = 'copied';
    setTimeout(() => { copyBtn.textContent = original; }, 1200);
  } catch {
    setStatus('Could not copy — select the text manually.', true);
  }
}

suggestBtn.addEventListener('click', handleSuggest);
generateBtn.addEventListener('click', handleGenerate);
copyBtn.addEventListener('click', handleCopy);

seedInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleSuggest();
  }
});