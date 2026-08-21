import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { ProcessingProgress } from "./ProcessingProgress";

describe("ProcessingProgress", () => {
  afterEach(cleanup);

  it("exposes percentage progress and renders a transform-only fill", () => {
    render(<ProcessingProgress value={0.42} label="Player analysis progress" />);

    const progressbar = screen.getByRole("progressbar", {
      name: "Player analysis progress",
    });
    expect(progressbar).toHaveAttribute("aria-valuenow", "42");
    expect(progressbar.firstElementChild).toHaveStyle({ transform: "scaleX(0.42)" });
  });

  it("clamps invalid and out-of-range values", () => {
    const { rerender } = render(<ProcessingProgress value={-0.5} />);
    let progressbar = screen.getByRole("progressbar", { name: "Analysis progress" });
    expect(progressbar).toHaveAttribute("aria-valuenow", "0");
    expect(progressbar.firstElementChild).toHaveStyle({ transform: "scaleX(0)" });

    rerender(<ProcessingProgress value={1.5} />);
    progressbar = screen.getByRole("progressbar", { name: "Analysis progress" });
    expect(progressbar).toHaveAttribute("aria-valuenow", "100");
    expect(progressbar.firstElementChild).toHaveStyle({ transform: "scaleX(1)" });

    rerender(<ProcessingProgress value={Number.NaN} />);
    progressbar = screen.getByRole("progressbar", { name: "Analysis progress" });
    expect(progressbar).toHaveAttribute("aria-valuenow", "0");
  });
});
