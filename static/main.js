document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('url-form');
  const input = document.getElementById('url');
  if (form && input) {
    form.addEventListener('submit', (event) => {
      if (!input.value.trim()) {
        event.preventDefault();
        input.focus();
      }
    });
  }
});
