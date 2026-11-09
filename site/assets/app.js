(() => {
  'use strict';

  const cards = Array.from(document.querySelectorAll('.evidence-card'));
  const filterButtons = Array.from(document.querySelectorAll('[data-filter]'));
  const searchInput = document.getElementById('evidence-search');
  const resultLine = document.getElementById('results-line');
  const emptyResult = document.getElementById('empty-result');

  if (!cards.length || !filterButtons.length || !searchInput || !resultLine || !emptyResult) return;

  let activeFilter = 'all';
  const normalise = (value) => value.toLocaleLowerCase('en-GB').trim();

  function updateResults() {
    const query = normalise(searchInput.value);
    let visible = 0;
    for (const card of cards) {
      const categoryMatches = activeFilter === 'all' || card.dataset.category === activeFilter;
      const text = `${card.dataset.search || ''} ${card.textContent || ''}`;
      const searchMatches = !query || normalise(text).includes(query);
      const show = categoryMatches && searchMatches;
      card.hidden = !show;
      if (show) visible++;
    }
    emptyResult.hidden = visible !== 0;
    resultLine.textContent = `Showing ${visible} of ${cards.length} source-linked ${cards.length === 1 ? 'note' : 'notes'}`;
  }

  filterButtons.forEach((button) => {
    button.addEventListener('click', () => {
      activeFilter = button.dataset.filter || 'all';
      filterButtons.forEach((other) => {
        const active = other === button;
        other.classList.toggle('active', active);
        other.setAttribute('aria-pressed', String(active));
      });
      updateResults();
    });
  });

  searchInput.addEventListener('input', updateResults);
  updateResults();
})();
