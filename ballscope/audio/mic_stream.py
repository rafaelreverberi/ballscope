import queue
import sys
import site

MIC_SAMPLE_RATE = 48000
MIC_CHANNELS = 2
MIC_BLOCKSIZE = 480
MIC_DEVICE_HINTS = ["comica", "vm10", "usb audio"]


def _wav_header(sample_rate: int, channels: int, bits_per_sample: int) -> bytes:
    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    block_align = channels * (bits_per_sample // 8)
    riff_size = 0xFFFFFFFF
    data_size = 0xFFFFFFFF
    return (
        b"RIFF" +
        riff_size.to_bytes(4, "little") +
        b"WAVE" +
        b"fmt " +
        (16).to_bytes(4, "little") +
        (1).to_bytes(2, "little") +
        channels.to_bytes(2, "little") +
        sample_rate.to_bytes(4, "little") +
        byte_rate.to_bytes(4, "little") +
        block_align.to_bytes(2, "little") +
        bits_per_sample.to_bytes(2, "little") +
        b"data" +
        data_size.to_bytes(4, "little")
    )


def _import_sounddevice():
    try:
        import sounddevice as sd  # type: ignore
        return sd
    except Exception:
        try:
            user_site = site.getusersitepackages()
            if user_site and user_site not in sys.path:
                sys.path.append(user_site)
            import sounddevice as sd  # type: ignore
            return sd
        except Exception:
            return None


def _resolve_audio_device(sd, device_hint: str):
    if isinstance(device_hint, int):
        return device_hint
    if isinstance(device_hint, str) and device_hint.isdigit():
        return int(device_hint)
    if isinstance(device_hint, str) and device_hint.lower() != "auto":
        needle = device_hint.lower()
        for idx, dev in enumerate(sd.query_devices()):
            if dev.get("max_input_channels", 0) > 0 and needle in dev.get("name", "").lower():
                return idx
        return device_hint

    for idx, dev in enumerate(sd.query_devices()):
        name = dev.get("name", "").lower()
        if dev.get("max_input_channels", 0) > 0 and any(h in name for h in MIC_DEVICE_HINTS):
            return idx
    return None


def audio_stream(device_hint: str = "auto"):
    sd = _import_sounddevice()
    if sd is None:
        return

    device = _resolve_audio_device(sd, device_hint)

    q: "queue.Queue[bytes]" = queue.Queue(maxsize=8)

    def callback(indata, frames, time_info, status):
        if status:
            return
        try:
            q.put_nowait(bytes(indata))
        except queue.Full:
            pass

    stream = sd.RawInputStream(
        samplerate=MIC_SAMPLE_RATE,
        channels=MIC_CHANNELS,
        dtype="int16",
        blocksize=MIC_BLOCKSIZE,
        device=device,
        latency="low",
        callback=callback,
    )

    stream.start()
    yield _wav_header(MIC_SAMPLE_RATE, MIC_CHANNELS, 16)

    try:
        while True:
            chunk = q.get()
            yield chunk
    except GeneratorExit:
        pass
    finally:
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
