import { ArrowLeft } from "lucide-react";
import { Link } from "react-router";

export function NotFoundPage() {
  return (
    <div className="page not-found">
      <span className="eyebrow">404</span>
      <h1>This court is out of bounds.</h1>
      <p>The dashboard route you requested does not exist.</p>
      <Link className="button button--primary" to="/matches">
        <ArrowLeft aria-hidden="true" /> Return to matches
      </Link>
    </div>
  );
}
