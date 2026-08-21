import pytest

from hf_space import app
from neurinferno import MessageResult, Segment


def sample_result() -> MessageResult:
    return MessageResult(
        hex="000102",
        n_bytes=3,
        cuts=[True, False],
        scores=[0.95, 0.1],
        segments=[Segment(0, 1, "00"), Segment(1, 3, "0102")],
        original_n_bytes=3,
    )


def test_examples_are_valid_batches() -> None:
    assert len(app.parse_messages(app.EXAMPLE_ARP).messages) == 8
    assert len(app.parse_messages(app.EXAMPLE_IGMP).messages) == 8


def test_space_rejects_a_partially_invalid_batch() -> None:
    with pytest.raises(ValueError, match="Line 2"):
        app.parse_messages("0001\nnot-hex")


def test_space_rejects_too_many_messages() -> None:
    text = "\n".join(["0001"] * (app.MAX_MSGS + 1))
    with pytest.raises(ValueError, match="maximum"):
        app.parse_messages(text)


def test_status_html_escapes_user_visible_errors() -> None:
    rendered = app.status_html("bad <script>alert(1)</script>", "error")
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_result_views_include_offsets_and_machine_readable_data() -> None:
    result = sample_result()

    byte_map = app.render_byte_map([result])
    payload = app.result_payload([result], 0.75)

    assert "[1:3]" in byte_map
    assert 'aria-label="offset 2, value 02"' in byte_map
    assert payload["messages"][0]["scores"] == [0.95, 0.1]
