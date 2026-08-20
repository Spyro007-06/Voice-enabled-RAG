"""Quick live API verification script across all Phase 1-5 endpoints."""

import asyncio
from httpx import AsyncClient, ASGITransport
from app.main import app


async def test_live_api():
    print("Testing live FastAPI endpoints...")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Root
        r = await client.get("/")
        print(f"GET / -> Status {r.status_code}: {r.json()}")
        assert r.status_code == 200

        # 2. Health
        r = await client.get("/health")
        print(f"GET /health -> Status {r.status_code}: {r.json()}")
        assert r.status_code == 200

        # 3. OpenAPI Docs
        r = await client.get("/openapi.json")
        print(f"GET /openapi.json -> Status {r.status_code}, routes found: {list(r.json()['paths'].keys())}")
        assert r.status_code == 200

        # 4. POST /api/retrieve
        r = await client.post(
            "/api/retrieve",
            json={
                "query": "भारत की राजधानी क्या है?",
                "top_k": 3,
                "fusion_method": "rrf",
            },
        )
        print(f"POST /api/retrieve -> Status {r.status_code}, returned {len(r.json()['results'])} results")
        assert r.status_code == 200

        # 5. POST /api/rerank
        r = await client.post(
            "/api/rerank",
            json={
                "query": "भारत की राजधानी क्या है?",
                "top_k": 3,
                "candidate_k": 10,
                "retrieval_mode": "rrf_hybrid",
            },
        )
        res_data = r.json()
        print(f"POST /api/rerank -> Status {r.status_code}")
        print(f"  - Selected chunks: {len(res_data['results'])}")
        print(f"  - Latency breakdown: {res_data['latency_ms']}")
        print(f"  - Context stats: {res_data['context_stats']}")
        assert r.status_code == 200
        assert len(res_data["results"]) == 3

    print("\n=======================================================")
    print("ALL 5 LIVE ENDPOINTS VERIFIED SUCCESSFULLY AND PASSING!")
    print("=======================================================")


if __name__ == "__main__":
    asyncio.run(test_live_api())
