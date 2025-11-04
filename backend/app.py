"""
Minimal Flask application providing stub endpoints for the report dashboard UI.
"""

from flask import Flask, jsonify
from flask_cors import CORS


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    @app.get("/health")
    def health() -> dict:
        return jsonify({"status": "ok"})

    @app.get("/api/report/<int:report_id>")
    def get_report(report_id: int):
        report = {
            "report_id": report_id,
            "title": "World - Hunger Report",
            "summary": (
                "The 2025 World Hunger and Food Chain Disruptions report highlights how climate shocks, conflict-driven "
                "displacement, and fragile logistics networks are converging to keep 735 million people in chronic food "
                "insecurity. It contrasts regions with resilient storage and cold-chain investments against those relying on "
                "volatile grain imports, and underscores that rapid response funds and nutrition-focused safety nets remain "
                "under-capitalised."
            ),
            "scores": {
                "overall": 0.78,
                "accuracy": 0.74,
                "credibility": 0.81,
                "validity": 0.69,
            },
            "internal_sources": [
                "2025 World Hunger.pdf",
                "World Hunger & Food Chain Disruptions.pdf",
                "Disruptions in the Food Supply Chain.pdf",
            ],
            "recommended_sources": [
                "Global Food Resilience Index 2025",
                "Nutrition Equity Observatory Brief",
                "AgriSupply Chain Stability Outlook",
                "Climate Resilient Harvests 2024",
                "Urban Food Access Benchmark 2025",
                "FAO Logistics Pulse - June 2025",
            ],
            "chat_suggestions": [
                {"author": "System", "text": "Ask about regions with the highest data uncertainty."},
                {"author": "User", "text": "How can I corroborate the year-over-year trends?"},
            ],
        }
        return jsonify(report)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True, port=5001)
