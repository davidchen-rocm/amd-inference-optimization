// Keep keyboard and screen-reader users with the page they just opened.
// Wait for async views without stealing focus after another user action.
export function focusAfterNavigation(root) {
  const main = root.querySelector('main');
  if (!main) return () => {};
  let observer = null, timer = null, stopped = false;
  const cancel = () => {
    stopped = true;
    observer?.disconnect();
    clearTimeout(timer);
    document.removeEventListener('pointerdown', cancel, true);
    document.removeEventListener('keydown', cancel, true);
  };
  const focusHeading = () => {
    if (stopped || !main.isConnected) return false;
    const heading = main.querySelector('h1');
    if (!heading) return false;
    heading.setAttribute('tabindex', '-1');
    heading.focus({preventScroll: true});
    cancel();
    return true;
  };
  if (!focusHeading()) {
    main.focus({preventScroll: true});
    observer = new MutationObserver(focusHeading);
    observer.observe(main, {childList: true, subtree: true});
    document.addEventListener('pointerdown', cancel, true);
    document.addEventListener('keydown', cancel, true);
    timer = setTimeout(cancel, 30000);
  }
  return cancel;
}
