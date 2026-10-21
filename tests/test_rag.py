import pytest

from copilot.rag import get_index


@pytest.mark.parametrize(
    "query,section",
    [
        ("what's the fee for overseas card transactions", "Overseas card transactions"),
        ("is a replacement card free if it was stolen", "Card replacement"),
        ("what is the fall below fee", "Fall below fee"),
        ("can I dispute a transaction from 3 months ago", "Dispute window"),
    ],
)
def test_retrieves_expected_section(query, section):
    assert get_index().search(query, k=1)[0]["source"].endswith(section)
