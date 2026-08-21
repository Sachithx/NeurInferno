from types import SimpleNamespace

import pytest
import torch

from neurinferno import FieldBoundaryModel, parse_hex
from neurinferno import inference as inference_module


class FixedBoundaryModel:
    """Minimal model that emits deterministic logits for API tests."""

    def __init__(self, selected_gaps: tuple[int, ...] = ()):
        self.selected_gaps = selected_gaps

    def __call__(self, inputs: torch.Tensor) -> SimpleNamespace:
        logits = torch.full((*inputs.shape, 1), -10.0)
        for gap in self.selected_gaps:
            if gap < inputs.shape[-1]:
                logits[:, :, gap, 0] = 10.0
        return SimpleNamespace(boundary_logits=logits)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0001aaff", bytes.fromhex("0001aaff")),
        ("0x00 0x01 0xaa 0xff", bytes.fromhex("0001aaff")),
        ("00:01-aa_ff", bytes.fromhex("0001aaff")),
    ],
)
def test_parse_hex_accepts_documented_formats(value: str, expected: bytes) -> None:
    assert parse_hex(value) == expected


@pytest.mark.parametrize("value", ["", "0", "00zz", "00,01"])
def test_parse_hex_rejects_invalid_input(value: str) -> None:
    with pytest.raises(ValueError):
        parse_hex(value)


def test_infer_builds_segments_from_boundary_scores() -> None:
    model = FieldBoundaryModel(FixedBoundaryModel((0, 2)))

    result = model.infer(["00010203"], threshold=0.75)[0]

    assert result.cuts == [True, False, True]
    assert [(segment.start, segment.end, segment.hex) for segment in result.segments] == [
        (0, 1, "00"),
        (1, 3, "0102"),
        (3, 4, "03"),
    ]
    assert result.original_n_bytes == 4
    assert result.truncated is False


def test_infer_reports_length_truncation_consistently() -> None:
    model = FieldBoundaryModel(FixedBoundaryModel(), max_len=2)

    result = model.infer(["00010203"])[0]

    assert result.hex == "0001"
    assert result.n_bytes == 2
    assert result.original_n_bytes == 4
    assert result.truncated is True
    assert result.segments[0].hex == "0001"


def test_infer_rejects_implicit_batch_truncation() -> None:
    model = FieldBoundaryModel(FixedBoundaryModel())

    with pytest.raises(ValueError, match="maximum is 1"):
        model.infer(["00", "01"], max_msgs=1)


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_infer_rejects_invalid_threshold(threshold: float) -> None:
    model = FieldBoundaryModel(FixedBoundaryModel())

    with pytest.raises(ValueError, match="threshold"):
        model.infer(["0001"], threshold=threshold)


def test_checkpoint_loader_uses_restricted_deserialization(monkeypatch, tmp_path) -> None:
    load_arguments: dict = {}

    def fake_load(path, **kwargs):
        load_arguments.update(kwargs)
        return {"state_dict": {}}

    class DummyFullModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load_state_dict(self, state_dict, strict):
            assert state_dict == {}
            assert strict is False

        def eval(self):
            return self

        def to(self, device):
            return self

    monkeypatch.setattr(inference_module.torch, "load", fake_load)
    monkeypatch.setattr(inference_module, "FullModel", DummyFullModel)

    inference_module.load_full_model(tmp_path / "model.ckpt")

    assert load_arguments["map_location"] == "cpu"
    assert load_arguments["weights_only"] is True
