// A failed dependency must leave a useful page instead of an empty screen.
import('./app.js').catch(() => {
  const root = document.getElementById('app');
  if (!root) return;
  root.innerHTML = `<main id="main-content" class="page" tabindex="-1">
    <section class="empty not-found" role="alert">
      <h1>MacFit couldn’t finish loading.</h1>
      <p>Check your connection, then reload this page. Your saved work stays in this browser.</p>
      <a class="button primary" href="">Reload MacFit</a>
    </section>
  </main>`;
});
