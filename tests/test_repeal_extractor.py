# mypy: ignore-errors
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.authority.writer import propose_lifecycle_repeal
from app.ingestion import repeal_extractor as re


def conn_resolves(row=("old-work",)):
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = row
    return conn


def test_resolved_repeal_match():
    row = re.classify_repeal(
        "३०. खारेजी र बचाउ : (१) लेखापरीक्षण ऐन, २०४८ खारेज गरिएको छ ।",
        conn_resolves(),
    )
    assert row.resolution_status == "auto_extracted"
    assert row.repealed_work_id == "old-work"
    assert row.repealed_title_ne == "लेखापरीक्षण ऐन २०४८"
    assert "खारेज गरिएको छ" in row.raw_clause_text


def test_unresolved_repeal_match():
    row = re.classify_repeal("पुरानो ऐन, १९९९ खारेज गरिएको छ ।", conn_resolves(None))
    assert row.resolution_status == "repealed_work_not_in_corpus"
    assert row.repealed_work_id is None
    assert row.raw_clause_text


def test_no_match_sentinel():
    row = re.classify_repeal("कुनै खारेजी दफा छैन ।", conn_resolves(None))
    assert row.resolution_status == "no_repeal_clause"
    assert row.raw_clause_text == ""


def test_savings_and_partial_repeal_do_not_match():
    savings = "(२) लेखापरीक्षण ऐन, २०४८ बमोजिम भए गरेका काम यसै ऐन बमोजिम मानिनेछ ।"
    partial = "सवारी तथा यातायात व्यवस्था ऐन, २०४९ को दफा १६८ को प्रतिबन्धात्मक वाक्यांश खारेज गरिएको छ ।"
    assert (
        re.classify_repeal(savings, conn_resolves(None)).resolution_status
        == "no_repeal_clause"
    )
    assert (
        re.classify_repeal(partial, conn_resolves(None)).resolution_status
        == "no_repeal_clause"
    )


def test_resolved_extraction_calls_writer(monkeypatch):
    calls = []
    monkeypatch.setattr(
        re, "propose_lifecycle_repeal", lambda *a, **kw: calls.append((a, kw))
    )
    re.extract_repeal_proposals(
        law=SimpleNamespace(uri="/new"),
        content="पुरानो ऐन, २०४८ खारेज गरिएको छ ।",
        source_pub_id="s",
        conn=conn_resolves(),
    )
    assert calls[0][0][1:] == ("old-work", "s")
    assert calls[0][1]["repealing_work_uri"] == "/new"


def test_corpus_count_regression():
    docs = matches = 0
    with open("laws.jsonl", encoding="utf-8") as fh:
        for line in fh:
            content = json.loads(line).get("content") or ""
            found = list(re._REPEAL_RE.finditer(content))
            docs += bool(found)
            matches += len(found)
    assert (docs, matches) == (157, 157)


def test_writer_fans_out_dedups_and_leaves_date_empty():
    conn = MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.side_effect = [None, ("/old",), None, (1,)]
    cur.fetchall.return_value = [("/old/dafa/1",), ("/old/dafa/2",)]

    propose_lifecycle_repeal(
        conn,
        "old-work",
        "00000000-0000-0000-0000-000000000001",
        repealing_work_uri="/new",
        raw_clause_text="raw",
    )

    inserts = [
        c
        for c in cur.execute.call_args_list
        if "INSERT INTO lifecycle_effect" in c.args[0]
    ]
    assert len(inserts) == 1
    assert inserts[0].args[1][0] == "/old/dafa/1"
    assert inserts[0].args[1][1] == "empty"
    assert inserts[0].args[1][2] is None
    assert inserts[0].args[1][3] == "repealing_work_commencement:/new"
    assert inserts[0].args[1][5] == "raw"
