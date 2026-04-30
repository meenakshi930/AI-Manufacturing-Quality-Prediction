import sys
import os

# 🔥 ensure backend path works in CI
sys.path.append(os.path.abspath("backend"))

from backend.src.api.main import app


def test_health_endpoint():
    client = app.test_client()
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"
