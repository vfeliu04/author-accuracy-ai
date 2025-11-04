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
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed ligula erat, molestie vel tortor, "
                "aliquet imperdiet mi. Donec posuere interdum mi vitae fermentum."
            ),
            "scores": {
                "overall": 0.78,
                "accuracy": 0.74,
                "credibility": 0.81,
                "validity": 0.69,
            },
            "internal_sources": [
                "Source One",
                "Source Two",
                "Source Three",
            ],
            "recommended_sources": [
                "Source One",
                "Source Two",
                "Source Three",
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
