import { Activity, ChevronRight, CircleDot, Code2 } from "lucide-react";
import { Link, NavLink, Outlet } from "react-router";

export function AppShell() {
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
