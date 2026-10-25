import sys
from pathlib import Path

# Ensure backend directory is in sys.path for direct test execution
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_read_root() -> None:
    """Test GET / returns HTTP 200 and expected service information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "CipherMind API"
    assert "description" in data
    assert data["status"] == "running"


def test_health_check() -> None:
    """Test GET /api/health returns HTTP 200 and healthy status JSON."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "CipherMind API"
