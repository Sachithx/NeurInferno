import json

from neurinferno import MessageResult, Segment, cli


class StubInferenceModel:
    def infer(self, messages, threshold):
        assert messages == [bytes.fromhex("0001")]
        assert threshold == 0.75
        return [
            MessageResult(
                hex="0001",
                n_bytes=2,
                cuts=[True],
                scores=[0.99],
                segments=[Segment(0, 1, "00"), Segment(1, 2, "01")],
                original_n_bytes=2,
            )
        ]


def test_cli_emits_json(monkeypatch, tmp_path, capsys) -> None:
    input_path = tmp_path / "messages.hex"
    input_path.write_text("0001\n", encoding="utf-8")
    monkeypatch.setattr(
        cli.FieldBoundaryModel,
        "from_pretrained",
        lambda **kwargs: StubInferenceModel(),
    )

    exit_code = cli.main(["infer", str(input_path), "--format", "json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload[0]["segments"][1] == {"start": 1, "end": 2, "hex": "01"}


def test_cli_reports_line_number_for_bad_input(tmp_path, capsys) -> None:
    input_path = tmp_path / "messages.hex"
    input_path.write_text("0001\ninvalid!\n", encoding="utf-8")

    exit_code = cli.main(["infer", str(input_path)])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "line 2" in captured.err


def test_cli_validates_threshold_before_loading_model(capsys) -> None:
    try:
        cli.main(["infer", "--threshold", "1.5"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("argparse should reject an invalid threshold")
    assert "between 0 and 1" in capsys.readouterr().err
