import json

import pytest

import src.r2_2_native_finish_calls as finish_calls


def _source(tmp_path, stages):
    source = tmp_path / "native_S1"
    source.mkdir()
    for name in ("sorted.bam", "sorted.bam.bai", "flagstat.txt"):
        (source / name).write_text(name)
    meta = {"sample_id": "S1", "state": "running", "stage": "methRatio", "min_depth": 4,
            "expected_filtered_records": 10, "validated_sorted_records": 10,
            "commands": [{"stage": s, "exit_code": 0} for s in stages] + [{"stage": "methRatio"}]}
    (source / "recovery_metadata.json").write_text(json.dumps(meta))
    return source


def _fake_calls(bam, out_dir, *args):
    assert bam.is_symlink() and args[4] == 4  # min_depth from the source record
    for target in ("methRatio", "CHALM"):
        (out_dir / f"{target}.txt").write_text(f"chr\tpos\n{target}\n")
    return {"mode": "per_contig"}


def test_finish_links_inputs_and_marks_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(finish_calls, "call_by_contig", _fake_calls)
    source = _source(tmp_path, finish_calls.REQUIRED_STAGES)
    out = tmp_path / "new" / "native_S1"
    meta = finish_calls.finish(source, out, tmp_path, tmp_path, tmp_path, "python", "samtools", 2)
    assert meta["state"] == finish_calls.COMPLETE_STATE and meta["sample_id"] == "S1"
    assert (out / "sorted.bam").resolve() == (source / "sorted.bam").resolve()
    assert set(meta["call_sha256"]) == {"methRatio", "CHALM"}
    assert json.loads((source / "recovery_metadata.json").read_text())["state"] == "running"  # source untouched


def test_finish_refuses_unvalidated_bam(tmp_path, monkeypatch):
    monkeypatch.setattr(finish_calls, "call_by_contig", _fake_calls)
    source = _source(tmp_path, ["historical_sort", "sorted_quickcheck"])
    with pytest.raises(RuntimeError, match="not complete"):
        finish_calls.finish(source, tmp_path / "out", tmp_path, tmp_path, tmp_path, "python", "samtools", 2)
