import { Clapperboard, Film, LockKeyhole, Play } from "lucide-react";
import { forwardRef, useImperativeHandle, useRef } from "react";

import type { Artifact } from "../api/types";

export interface SeekableMediaHandle {
  seek(timestampSeconds: number): void;
}

interface VideoPanelProps {
  youtubeVideoId: string | null;
  videoArtifact: Artifact | null;
  hasPrivateVideo: boolean;
}

function validYouTubeId(value: string | null): value is string {
  return value !== null && /^[A-Za-z0-9_-]{6,20}$/.test(value);
}

export const VideoPanel = forwardRef<SeekableMediaHandle, VideoPanelProps>(function VideoPanel(
  { youtubeVideoId, videoArtifact, hasPrivateVideo },
  ref,
) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const youtubeRef = useRef<HTMLIFrameElement>(null);

  useImperativeHandle(ref, () => ({
    seek(timestampSeconds: number) {
      const safeTimestamp = Math.max(0, timestampSeconds);
      if (videoRef.current !== null) {
        videoRef.current.currentTime = safeTimestamp;
        void videoRef.current.play().catch(() => undefined);
      } else if (youtubeRef.current?.contentWindow !== null) {
        youtubeRef.current?.contentWindow?.postMessage(
          JSON.stringify({ event: "command", func: "seekTo", args: [safeTimestamp, true] }),
          "https://www.youtube.com",
        );
      }
    },
  }));

  if (videoArtifact !== null && videoArtifact.url !== null) {
    return (
      <div className="video-frame">
        <video ref={videoRef} controls preload="metadata" poster="">
          <source src={videoArtifact.url} type={videoArtifact.contentType} />
          Your browser cannot play this annotated video.
        </video>
        <span className="media-source-label">
          <Film aria-hidden="true" /> Annotated analysis video
        </span>
      </div>
    );
  }

  if (validYouTubeId(youtubeVideoId)) {
    const source = `https://www.youtube.com/embed/${encodeURIComponent(youtubeVideoId)}?enablejsapi=1&rel=0&origin=${encodeURIComponent(window.location.origin)}`;
    return (
      <div className="video-frame video-frame--youtube">
        <iframe
          ref={youtubeRef}
          src={source}
          title="Unlisted YouTube match video"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
        />
        <span className="media-source-label">
          <Clapperboard aria-hidden="true" /> Unlisted YouTube source
        </span>
      </div>
    );
  }

  return (
    <div className="video-placeholder">
      {hasPrivateVideo ? <LockKeyhole aria-hidden="true" /> : <Play aria-hidden="true" />}
      <strong>{hasPrivateVideo ? "Analysis video is private" : "No playable video available"}</strong>
      <p>
        {hasPrivateVideo
          ? "The API returned a private artifact. The browser will not request it without a public view URL."
          : "Add a YouTube video ID or a public annotated video artifact to enable playback."}
      </p>
    </div>
  );
});
