import { Activity, ChevronRight, CircleDot, Code2, Moon, Sun } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, NavLink, Outlet } from "react-router";

type Theme = "dark" | "light";

function initialTheme(): Theme {
  const stored = window.localStorage.getItem("pickleball-vision-theme");
  if (stored === "dark" || stored === "light") return stored;
  return typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-color-scheme: light)").matches
    ? "light"
    : "dark";
}

export function AppShell() {
  const [theme, setTheme] = useState<Theme>(initialTheme);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem("pickleball-vision-theme", theme);
  }, [theme]);

  return (
    <div className="app-shell">
      <header className="site-header">
        <Link to="/matches" className="brand" aria-label="Pickleball Vision matches">
          <span className="brand-mark" aria-hidden="true">
            <CircleDot />
          </span>
          <span>
            <strong>Pickleball Vision</strong>
            <small>Match intelligence</small>
          </span>
        </Link>
        <nav className="main-nav" aria-label="Primary navigation">
          <NavLink to="/matches">
            <Activity aria-hidden="true" />
            Matches
          </NavLink>
          <button
            type="button"
            className="theme-toggle"
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
            title={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
            onClick={() => setTheme((current) => (current === "dark" ? "light" : "dark"))}
          >
            {theme === "dark" ? <Sun aria-hidden="true" /> : <Moon aria-hidden="true" />}
          </button>
        </nav>
      </header>
      <main>
        <Outlet />
      </main>
      <footer className="site-footer">
        <span>Pickleball Vision match analysis</span>
        <a href="https://github.com/John5675/rallymetry" target="_blank" rel="noreferrer">
          <Code2 aria-hidden="true" /> Repository <ChevronRight aria-hidden="true" />
        </a>
      </footer>
    </div>
  );
}
