'use strict';
for (const widget of document.querySelectorAll('[data-tabs]')) {
  const tabs = [...widget.querySelectorAll(':scope > .tabs > button')];
  const panels = [...widget.querySelectorAll(':scope > [role="tabpanel"]')];
  const select = (index, focus = false) => {
    tabs.forEach((tab, i) => { tab.setAttribute('aria-selected', String(i === index)); tab.tabIndex = i === index ? 0 : -1; });
    panels.forEach((panel, i) => { panel.hidden = i !== index; });
    if (focus) tabs[index].focus();
  };
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => select(index));
    tab.addEventListener('keydown', (event) => {
      const next = {ArrowRight: (index + 1) % tabs.length, ArrowLeft: (index + tabs.length - 1) % tabs.length, Home: 0, End: tabs.length - 1}[event.key];
      if (next === undefined) return;
      event.preventDefault(); select(next, true);
    });
  });
  select(0);
  widget.dataset.ready = 'true';
}
for (const explorer of document.querySelectorAll('[data-filter]')) {
  const buttons = [...explorer.querySelectorAll('button[data-category]')];
  const rows = [...explorer.querySelectorAll('tbody tr')];
  const result = explorer.querySelector('[data-count]');
  explorer.dataset.ready = 'true';
  for (const button of buttons) button.addEventListener('click', () => {
    const key = button.dataset.category;
    for (const other of buttons) other.setAttribute('aria-pressed', String(other === button));
    for (const row of rows) row.hidden = key !== 'all' && !row.dataset.categories.split(' ').includes(key);
    const count = rows.filter(row => !row.hidden).length;
    result.textContent = `${count} of ${rows.length} examples shown`;
  });
}
