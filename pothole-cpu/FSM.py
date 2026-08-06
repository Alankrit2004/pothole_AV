import queue
import threading


class FrameStreamManager:
    """Pub/sub broadcaster for live video frames, same pattern as SSEManager
    but carrying raw JPEG bytes instead of JSON text. One queue per
    connected viewer; whoever is processing the video calls broadcast()
    with each new annotated frame, and every subscribed viewer's MJPEG
    stream picks it up."""

    def __init__(self, maxsize=2):
        self._clients = {}
        self._lock = threading.Lock()
        # Small maxsize on purpose: for a LIVE feed we only care about the
        # newest frame. If a viewer's connection is slow, we'd rather drop
        # old frames than let them pile up and make the stream lag behind.
        self._maxsize = maxsize

    def subscribe(self, video_id):
        q = queue.Queue(maxsize=self._maxsize)
        with self._lock:
            self._clients.setdefault(video_id, []).append(q)
        return q

    def unsubscribe(self, video_id, q):
        with self._lock:
            if video_id in self._clients:
                self._clients[video_id] = [c for c in self._clients[video_id] if c is not q]
                if not self._clients[video_id]:
                    del self._clients[video_id]

    def broadcast(self, video_id, jpeg_bytes):
        with self._lock:
            clients = list(self._clients.get(video_id, []))

        for q in clients:
            # Drop the oldest queued frame if this viewer is falling behind,
            # then push the newest one -- always prefer freshness over
            # completeness for a live stream.
            if q.full():
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
            try:
                q.put_nowait(jpeg_bytes)
            except queue.Full:
                pass

    def has_subscribers(self, video_id):
        with self._lock:
            return bool(self._clients.get(video_id))

    def generate_mjpeg(self, video_id, timeout=30):
        """Generator yielding a multipart/x-mixed-replace stream. Plug this
        straight into a Flask Response with the matching mimetype."""
        q = self.subscribe(video_id)
        try:
            while True:
                try:
                    frame_bytes = q.get(timeout=timeout)
                except queue.Empty:
                    # No new frame in `timeout` seconds -- likely the job
                    # finished/stalled. End the stream rather than hang forever.
                    break

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n\r\n"
                    + frame_bytes + b"\r\n"
                )
        except GeneratorExit:
            pass
        finally:
            self.unsubscribe(video_id, q)


frame_manager = FrameStreamManager()