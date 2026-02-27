from fastapi.testclient import TestClient

from tsfit.main import create_app


def test_health_endpoint() -> None:
    app = create_app()
    client = TestClient(app)

    r = client.get('/health')
    assert r.status_code == 200, r.text
    assert r.json() == {'status': 'ok'}
