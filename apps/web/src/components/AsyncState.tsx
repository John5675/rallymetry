import { AlertCircle, LoaderCircle } from "lucide-react";
import type { PropsWithChildren, ReactNode } from "react";

interface AsyncStateProps extends PropsWithChildren {
  loading: boolean;
  error: Error | null;
  empty?: boolean;
  emptyTitle?: string;
  emptyMessage?: string;
  loadingMessage?: string;
  action?: ReactNode;
}

export function AsyncState({
  loading,
  error,
  empty = false,
  emptyTitle = "Nothing here yet",
  emptyMessage = "Data will appear here when it is available.",
  loadingMessage = "Loading match data…",
  action,
  children,
}: AsyncStateProps) {
  if (loading) {
    return (
      <div className="state-panel" role="status">
        <LoaderCircle className="spin" aria-hidden="true" />
        <p>{loadingMessage}</p>
      </div>
    );
  }
  if (error !== null) {
    return (
      <div className="state-panel state-panel--error" role="alert">
        <AlertCircle aria-hidden="true" />
        <div>
          <strong>Unable to load this view</strong>
          <p>{error.message}</p>
        </div>
        {action}
      </div>
    );
  }
  if (empty) {
    return (
      <div className="state-panel">
        <div>
          <strong>{emptyTitle}</strong>
          <p>{emptyMessage}</p>
        </div>
        {action}
      </div>
    );
  }
  return children;
}
