// Return the exposure whose presentation interval contains this time.
export function replayIndexAt(times, time, tolerance = 0) {
  let low = 0;
  let high = times.length - 1;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (times[middle] <= time + tolerance) low = middle;
    else high = middle - 1;
  }
  return low;
}

const HAS_FRAME = 2; // HTMLMediaElement.HAVE_CURRENT_DATA
export const MEDIA_TIME_TOLERANCE = .000001;

function matchesExposure(times, mediaTime, target) {
  return times?.length &&
    replayIndexAt(times, mediaTime, MEDIA_TIME_TOLERANCE) === replayIndexAt(times, target, MEDIA_TIME_TOLERANCE);
}

export class VideoExposureClock {
  #video;
  #onFrame;
  #source = '';
  #generation = 0;
  #callback = null;
  #mediaTime = null;
  #previousTime = null;
  #seekTime = null;
  #seekTimeline = null;

  constructor(video, onFrame) {
    this.#video = video;
    this.#onFrame = onFrame;
    this.supported = typeof video.requestVideoFrameCallback === 'function';
  }

  get time() {
    const video = this.#video;
    if (!this.#source || video.currentSrc !== this.#source) return null;
    if (this.supported) return this.#mediaTime;
    return video.seeking || video.readyState < HAS_FRAME ? null : video.currentTime;
  }

  setSource(url = '') {
    this.#source = url;
    this.#mediaTime = null;
    this.#previousTime = null;
    this.#seekTime = null;
    this.#seekTimeline = null;
    this.#watch();
  }

  beginSeek(times) {
    this.#previousTime = this.#mediaTime ?? this.#previousTime;
    this.#mediaTime = null;
    this.#seekTimeline = times;
    this.#watch();
  }

  finishSeek(times) {
    if (this.#video.seeking) return;
    // The target can be presented before seeked fires. A paused seek within
    // the same exposure can also reuse the old picture without a callback.
    const target = this.#video.currentTime;
    this.#mediaTime ??= times?.length ? [this.#seekTime, this.#previousTime]
      .find(time => time !== null && matchesExposure(times, time, target)) ?? null : this.#seekTime;
    this.#seekTime = null;
    this.#previousTime = null;
    this.#seekTimeline = null;
  }

  #watch() {
    const generation = ++this.#generation;
    const video = this.#video;
    if (this.#callback !== null) video.cancelVideoFrameCallback?.(this.#callback);
    this.#callback = null;
    if (!this.supported || !this.#source) return;
    const receive = (_, {mediaTime}) => {
      if (this.#generation !== generation) return;
      this.#callback = video.requestVideoFrameCallback(receive);
      if (video.currentSrc !== this.#source || !Number.isFinite(mediaTime)) return;
      if (video.seeking) {
        this.#seekTime = mediaTime;
        if (!matchesExposure(this.#seekTimeline, mediaTime, video.currentTime)) return;
      }
      this.#mediaTime = mediaTime;
      this.#onFrame();
    };
    this.#callback = video.requestVideoFrameCallback(receive);
  }
}
