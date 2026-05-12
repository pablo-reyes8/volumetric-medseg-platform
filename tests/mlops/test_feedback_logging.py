from src.mlops.feedback_logging import append_feedback_record, build_feedback_record, load_feedback_records, summarize_feedback


def test_feedback_jsonl_roundtrip_and_summary(tmp_path):
    path = tmp_path / "feedback.jsonl"
    append_feedback_record(
        path,
        build_feedback_record(
            request_id="req-1",
            model_version="v0.2.0",
            accepted=False,
            quality_score=2,
            requires_reannotation=True,
        ),
    )
    records = load_feedback_records(path)
    summary = summarize_feedback(records)

    assert records[0]["request_id"] == "req-1"
    assert summary["acceptance_rate"] == 0.0
    assert summary["reannotation_rate"] == 1.0
