from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path

from app.authority.parser import parse_law

ROOT = Path(__file__).resolve().parents[1]


def _law_record(name: str) -> dict[str, object]:
    with (ROOT / "laws.jsonl").open(encoding="utf-8") as laws:
        for line in laws:
            record = json.loads(line)
            if record.get("name") == name:
                return dict(record)
    raise AssertionError(f"law fixture not found: {name}")


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


def test_inline_dafa_reference_is_not_a_header() -> None:
    law = parse_law(
        {
            "name": "नमुना_ऐन_२०८०",
            "document_type": "act",
            "content": "दफा २. यो वास्तविक दफा हो। यस ऐनको दफा ३ बमोजिम गरिएको सन्दर्भ मात्र हो। पर्याप्त लामो पाठ।",
            "_id": "demo-id",
        }
    )

    assert [c.number for c in law.components] == ["2"]
    assert "दफा ३ बमोजिम" in law.components[0].text_ne


def test_schedule_items_are_not_dafa() -> None:
    law = parse_law(
        {
            "name": "नमुना_ऐन_२०८०",
            "document_type": "act",
            "_id": "demo-id",
            "content": "**१. मूल दफा:** मुख्य दफाको पर्याप्त लामो पाठ यहाँ छ।\n\n**अनुसूची-१**\n\n(दफा १ सँग सम्बन्धित)\n\n**१. सूची:** अनुसूची भित्रको पर्याप्त लामो पाठ यहाँ छ।",
        }
    )

    assert [(c.component_type, c.number) for c in law.components] == [
        ("dafa", "1"),
        ("anushuchi", "1"),
        ("anushuchi", "1.1"),
    ]
    assert len({c.uri for c in law.components}) == len(law.components)


def test_compound_dafa_number_kept_in_uri() -> None:
    law = parse_law(_law_record("आयुर्वेद_चिकित्सा_परिषद्_ऐन_२०४५"))
    dafa_numbers = [c.number for c in law.components if c.component_type == "dafa"]

    assert "2.1" in dafa_numbers
    assert "2.9" in dafa_numbers
    assert any(c.uri.endswith("/dafa/2.1") for c in law.components)
    assert not any(c.uri.endswith("/dafa/2/occurrence/2") for c in law.components)


def test_schedule_header_spacing_variants_are_boundaries() -> None:
    law = parse_law(
        {
            "name": "नमुना_नियम_२०८०",
            "document_type": "regulation",
            "_id": "demo-rule",
            "content": "**१. मूल नियम:** मुख्य नियमको पर्याप्त लामो पाठ यहाँ छ।\n\n**अनुसूची - ३**\n\n**१. फाराम:** पहिलो अनुसूची पाठ पर्याप्त लामो छ।\n\n**अनुसूची ४ (ख)**\n\n**१. अर्को:** दोस्रो अनुसूची पाठ पर्याप्त लामो छ।",
        }
    )

    assert [(c.component_type, c.number) for c in law.components] == [
        ("dafa", "1"),
        ("anushuchi", "3.1"),
        ("anushuchi", "4.1"),
    ]


def test_repeated_source_numbers_get_occurrence_uri() -> None:
    law = parse_law(
        {
            "name": "नमुना_ऐन_२०८०",
            "document_type": "act",
            "_id": "demo-id",
            "content": "**४१. पहिलो:** पर्याप्त लामो पाठ यहाँ छ।\n\n**४१. दोस्रो:** फरक पर्याप्त लामो पाठ यहाँ छ।",
        }
    )

    assert [c.number for c in law.components] == ["41", "41"]
    assert [c.uri for c in law.components] == [
        "/np/act/2080/demo-id/dafa/41",
        "/np/act/2080/demo-id/dafa/41/occurrence/2",
    ]


def test_source_sha256_hashes_nfc_content() -> None:
    content = "दफा १. गैर-NFC अक्षर क़ सहित पर्याप्त लामो पाठ।"
    law = parse_law(
        {
            "name": "नमुना_ऐन_२०८०",
            "document_type": "act",
            "content": content,
            "_id": "demo-id",
        }
    )

    assert content != unicodedata.normalize("NFC", content)
    assert (
        law.source_sha256
        == hashlib.sha256(
            unicodedata.normalize("NFC", content).encode("utf-8")
        ).hexdigest()
    )
