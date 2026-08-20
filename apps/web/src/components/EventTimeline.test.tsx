import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { EventTimeline } from "./EventTimeline";

describe("EventTimeline", () => {
  it("seeks to a structured event and allows event kinds to be hidden", async () => {
    const user = userEvent.setup();
    const onSeek = vi.fn();
    render(
      <EventTimeline
        durationSeconds={20}
        onSeek={onSeek}
        events={[
          {
            id: "shot-1",
            kind: "shot",
            label: "shot-1",
            timestampSeconds: 7.25,
            confidence: 0.8,
          },
        ]}
      />,
    );

    await user.click(screen.getByRole("button", { name: /seek to shot-1/i }));
    expect(onSeek).toHaveBeenCalledWith(7.25);

    await user.click(screen.getByRole("button", { name: "Shot" }));
    expect(screen.queryByRole("button", { name: /seek to shot-1/i })).not.toBeInTheDocument();
  });
});
