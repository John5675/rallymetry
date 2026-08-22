import themeCss from "./watermelon-theme.css?raw";
import { describe, expect, it } from "vitest";

describe("Watermelon visual theme", () => {
  it("keeps the media source badge content-sized in the video overlay", () => {
    const mediaSourceRules = themeCss.match(/\.media-source-label\s*{[^}]*}/g);
    const overlayRule = mediaSourceRules?.at(-1);

    expect(overlayRule).toBeDefined();
    expect(overlayRule).toContain("top: auto");
    expect(overlayRule).toContain("bottom: 0.75rem");
    expect(overlayRule).toContain("width: max-content");
    expect(overlayRule).toContain("height: fit-content");
  });
});
