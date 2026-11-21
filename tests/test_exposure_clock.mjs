import assert from 'node:assert/strict';
import {test} from 'node:test';
import {MEDIA_TIME_TOLERANCE, replayIndexAt, VideoExposureClock} from '../web/exposure-clock.mjs';

const FRONT = 'https://demo.local/front.mp4';
const LEFT = 'https://demo.local/left.mp4';

class VideoStub {
  currentSrc = FRONT;
  currentTime = 0;
  readyState = 4;
  seeking = false;
  callbacks = new Map();
  lastId = 0;

  requestVideoFrameCallback(callback) {
    this.callbacks.set(++this.lastId, callback);
    return this.lastId;
  }

  cancelVideoFrameCallback(id) {
    this.callbacks.delete(id);
  }

  present(mediaTime) {
    const [[id, callback]] = this.callbacks;
    this.callbacks.delete(id);
    callback(0, {mediaTime});
  }
}

test('recorded exposure indices stay inside the presentation interval', () => {
  const times = [0, .1, .23, .34, .48];
  assert.equal(replayIndexAt(times, .2295, MEDIA_TIME_TOLERANCE), 1);
  assert.equal(replayIndexAt(times, .2299999, MEDIA_TIME_TOLERANCE), 2);
  assert.equal(replayIndexAt(times, .31, MEDIA_TIME_TOLERANCE), 2);
  assert.equal(replayIndexAt(times, .5, MEDIA_TIME_TOLERANCE), 4);
});

test('use the presented timestamp while the transport clock moves or skips frames', () => {
  const video = new VideoStub();
  const captured = [];
  const clock = new VideoExposureClock(video, () => captured.push(clock.time));
  clock.setSource(FRONT);
  assert.equal(clock.time, null);

  video.currentTime = .14;
  video.present(.1);
  video.currentTime = .225;
  assert.equal(clock.time, .1);

  video.currentTime = .295;
  video.present(.3);
  assert.deepEqual(captured, [.1, .3]);
  assert.equal(video.callbacks.size, 1);
});

test('discard old camera and canceled seek callbacks, then latch the rewind', () => {
  const video = new VideoStub();
  const captured = [];
  const clock = new VideoExposureClock(video, () => captured.push(clock.time));
  clock.setSource(FRONT);
  video.present(1);
  const oldCamera = video.callbacks.get(video.lastId);
  clock.setSource(LEFT);
  video.present(1.1);
  assert.equal(clock.time, null);
  video.currentSrc = LEFT;
  oldCamera(0, {mediaTime: 1.2});
  assert.equal(clock.time, null);
  video.present(1);

  const beforeSeek = video.callbacks.get(video.lastId);
  clock.beginSeek();
  video.seeking = true;
  video.currentTime = .14;
  video.present(1);
  assert.equal(clock.time, null);
  video.seeking = false;
  clock.finishSeek([0, .1, .2, 1]);
  beforeSeek(0, {mediaTime: 1.2});
  assert.equal(clock.time, null);
  video.present(.1);
  assert.deepEqual(captured, [1, 1, .1]);
  clock.setSource();
  assert.equal(video.callbacks.size, 0);
  assert.equal(clock.time, null);
});

test('same-exposure seek can settle even when a paused browser reuses its frame', () => {
  const video = new VideoStub();
  const clock = new VideoExposureClock(video, () => {});
  clock.setSource(FRONT);
  video.present(.1);
  clock.beginSeek();
  clock.beginSeek();
  video.currentTime = .17;
  clock.finishSeek([0, .1, .2]);
  assert.equal(clock.time, .1);
});

test('latches the decoded exposure when its only callback precedes seeked', () => {
  const video = new VideoStub();
  const clock = new VideoExposureClock(video, () => {});
  clock.setSource(FRONT);
  clock.beginSeek();
  video.seeking = true;
  video.currentTime = .27;
  video.present(.2);
  clock.beginSeek();
  assert.equal(clock.time, null);
  video.seeking = false;
  clock.finishSeek([0, .1, .2, .3]);
  assert.equal(clock.time, .2);
});

test('a decoded seek target updates the rider before seeked', () => {
  const video = new VideoStub();
  const captured = [];
  const clock = new VideoExposureClock(video, () => captured.push(clock.time));
  clock.setSource(FRONT);
  clock.beginSeek([0, .1, .2, .3]);
  video.seeking = true;
  video.currentTime = .27;
  video.present(0);
  assert.equal(clock.time, null);
  video.present(.2);
  assert.deepEqual(captured, [.2]);
  assert.equal(clock.time, .2);
});

test('browsers without video frame callbacks use transport only with current media data', () => {
  const video = {currentSrc: FRONT, currentTime: .174, readyState: 1, seeking: false};
  const clock = new VideoExposureClock(video, () => assert.fail('unexpected callback'));
  clock.setSource(FRONT);
  assert.equal(clock.supported, false);
  assert.equal(clock.time, null);
  video.readyState = 2;
  assert.equal(replayIndexAt([0, .1, .2], clock.time), 1);
  video.seeking = true;
  assert.equal(clock.time, null);
  video.seeking = false;
  clock.setSource(LEFT);
  assert.equal(clock.time, null);
  video.currentSrc = LEFT;
  assert.equal(clock.time, .174);
});
