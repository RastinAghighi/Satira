import pytest


@pytest.fixture
def sample_text():
    return "This is a sample text for testing satire detection."


@pytest.fixture
def api_base_url():
    return "http://localhost:8000"
