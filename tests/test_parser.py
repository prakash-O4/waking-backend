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
