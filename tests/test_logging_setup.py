import io
import json

from sawti.logging_setup import configure_logging, get_logger


def test_logger_emits_jsonl():
    buf = io.StringIO()
    configure_logging(stream=buf, force=True)  # order-independent: reset first
    log = get_logger("test")
    log.info("chunk", chunk_id="c0", confidence=0.7)
    line = buf.getvalue().strip()
    record = json.loads(line)
    assert record["event"] == "chunk"
    assert record["chunk_id"] == "c0"
    assert record["confidence"] == 0.7
