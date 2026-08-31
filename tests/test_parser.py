from __future__ import annotations

import hashlib
import unicodedata

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
    law = parse_law(
        {
            "name": "नमुना_ऐन_२०८०",
            "document_type": "act",
            "_id": "demo-id",
            "content": "**२.१. पहिलो:** पर्याप्त लामो पाठ यहाँ छ।\n\n**२.२. दोस्रो:** अर्को पर्याप्त लामो पाठ यहाँ छ।",
        }
    )

    assert [c.number for c in law.components] == ["2.1", "2.2"]
    assert [c.uri for c in law.components] == [
        "/np/act/2080/demo-id/dafa/2.1",
        "/np/act/2080/demo-id/dafa/2.2",
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
