from sawti.segmenter import StubSegmenter
from sawti.sources import StubAudioSource
from sawti.types import AudioChunk


def test_stub_segmenter_yields_chunks():
    src = StubAudioSource(n_frames=4, samples_per_frame=16000)
    seg = StubSegmenter(chunk_frames=2, sample_rate=16000)
    chunks = list(seg.process(src.iter_frames()))
    assert len(chunks) == 2
    assert all(isinstance(c, AudioChunk) for c in chunks)
    assert chunks[0].id == "c0"
    assert chunks[1].start_time == chunks[0].end_time
