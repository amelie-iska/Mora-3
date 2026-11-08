(() => {
  const root = document.documentElement;
  const key = "mora3-theme";
  const colors = {dark: "#121c1b", light: "#f4f2eb"};
  let theme = "dark";
  try {
    if (localStorage.getItem(key) === "light") theme = "light";
  } catch (_) {}
  root.dataset.theme = theme;

  document.addEventListener("DOMContentLoaded", () => {
    const toggle = document.getElementById("theme-toggle");
    if (!toggle) return;
    const themeColor = document.querySelector('meta[name="theme-color"]');
    function apply() {
      root.dataset.theme = theme;
      themeColor.content = colors[theme];
      toggle.textContent = theme === "dark" ? "Light mode" : "Dark mode";
    }
    toggle.addEventListener("click", () => {
      theme = theme === "dark" ? "light" : "dark";
      try { localStorage.setItem(key, theme); } catch (_) {}
      apply();
    });
    apply();
    toggle.hidden = false;
  });
})();
