from __future__ import annotations

from app.authority.parser import parse_law


def test_parse_law_splits_and_strips_amend() -> None:
    law = parse_law(
        {
            "name": "नमुना_ऐन_२०८०",
            "english_name": "Demo Act",
            "document_type": "act",
            "content": "प्रमाणीकरण र प्रकाशन मिति\n२०८०।१२।१६\n\n**१. नाम:** <amend>पुरानो</amend> लामो पाठ यहाँ छ।\n\nदफा २. अर्को लामो पाठ यहाँ छ।",
            "_id": "demo-id",
        }
    )
    assert law.uri == "/np/act/2080/demo-id"
    assert law.enactment_ad is not None
    assert [c.number for c in law.components] == ["0", "1", "2"]
    assert "<amend>" not in law.components[1].text_ne
