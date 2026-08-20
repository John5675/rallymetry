import { useEffect, useState } from "react";

interface AsyncData<T> {
  data: T | null;
  error: Error | null;
  loading: boolean;
}

export function useAsyncData<T>(loader: (signal: AbortSignal) => Promise<T>): AsyncData<T> {
  const [state, setState] = useState<AsyncData<T>>({ data: null, error: null, loading: true });

  useEffect(() => {
    const controller = new AbortController();
    // A changed loader represents a different resource and must not display stale data.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setState({ data: null, error: null, loading: true });
    void loader(controller.signal)
      .then((data) => {
        if (!controller.signal.aborted) {
          setState({ data, error: null, loading: false });
        }
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setState({
            data: null,
            error: error instanceof Error ? error : new Error("An unexpected error occurred."),
            loading: false,
          });
        }
      });
    return () => controller.abort();
  }, [loader]);

  return state;
}
