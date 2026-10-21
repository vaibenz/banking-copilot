import pytest

from copilot.agent import Copilot
from copilot.bank_api import reset_store
from copilot.planner import RulePlanner


@pytest.fixture(autouse=True)
def clean_store():
    reset_store()
    yield
    reset_store()


@pytest.fixture
def copilot():
    return Copilot(planner=RulePlanner())
