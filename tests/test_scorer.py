"""
Scorer unit tests.

Tests for the scoring/inference logic.
"""

from inference_api.schemas import PredictRequest
from inference_api.scorer import score


class TestScorer:
    """Tests for score() function."""

    def test_score_returns_response(self):
        """Score should return PredictResponse."""
        request = PredictRequest(
            sport="NBA",
            stat_type="points",
            line=25.5,
        )
        result = score(request)
        assert hasattr(result, "probability")
        assert hasattr(result, "confidence")
        assert hasattr(result, "recommendation")

    def test_probability_bounded(self):
        """Probability should be between 0 and 1."""
        request = PredictRequest(
            sport="NBA",
            stat_type="points",
            line=25.5,
            glicko_mu=2000.0,  # Very high
        )
        result = score(request)
        assert 0.0 <= result.probability <= 1.0

    def test_high_glicko_increases_probability(self):
        """Higher Glicko rating should increase probability."""
        low_glicko = PredictRequest(
            sport="NBA",
            stat_type="points",
            line=25.5,
            glicko_mu=1400.0,
            last_5_avg=25.0,
            last_10_avg=25.0,
        )
        high_glicko = PredictRequest(
            sport="NBA",
            stat_type="points",
            line=25.5,
            glicko_mu=1700.0,
            last_5_avg=25.0,
            last_10_avg=25.0,
        )
        low_result = score(low_glicko)
        high_result = score(high_glicko)
        assert high_result.probability > low_result.probability

    def test_recent_form_affects_probability(self):
        """Better recent form should increase probability."""
        low_form = PredictRequest(
            sport="NBA",
            stat_type="points",
            line=25.5,
            last_5_avg=20.0,
            last_10_avg=20.0,
        )
        high_form = PredictRequest(
            sport="NBA",
            stat_type="points",
            line=25.5,
            last_5_avg=30.0,
            last_10_avg=30.0,
        )
        low_result = score(low_form)
        high_result = score(high_form)
        assert high_result.probability > low_result.probability

    def test_home_advantage(self):
        """Home games should have slightly higher probability."""
        away = PredictRequest(
            sport="NBA",
            stat_type="points",
            line=25.5,
            is_home=False,
        )
        home = PredictRequest(
            sport="NBA",
            stat_type="points",
            line=25.5,
            is_home=True,
        )
        away_result = score(away)
        home_result = score(home)
        assert home_result.probability >= away_result.probability

    def test_confidence_based_on_phi(self):
        """Confidence should be based on Glicko phi."""
        low_phi = PredictRequest(
            sport="NBA",
            stat_type="points",
            line=25.5,
            glicko_phi=100.0,
        )
        high_phi = PredictRequest(
            sport="NBA",
            stat_type="points",
            line=25.5,
            glicko_phi=250.0,
        )
        low_result = score(low_phi)
        high_result = score(high_phi)
        assert low_result.confidence == "high"
        assert high_result.confidence == "low"

    def test_recommendation_over(self):
        """High probability should recommend OVER."""
        request = PredictRequest(
            sport="NBA",
            stat_type="points",
            line=20.0,  # Low line
            glicko_mu=1700.0,
            last_5_avg=30.0,
            last_10_avg=30.0,
        )
        result = score(request)
        assert result.recommendation == "OVER"

    def test_recommendation_under(self):
        """Low probability should recommend UNDER."""
        request = PredictRequest(
            sport="NBA",
            stat_type="points",
            line=35.0,  # High line
            glicko_mu=1300.0,
            last_5_avg=20.0,
            last_10_avg=20.0,
        )
        result = score(request)
        assert result.recommendation == "UNDER"

    def test_model_version_included(self):
        """Response should include model version."""
        request = PredictRequest(
            sport="NBA",
            stat_type="points",
            line=25.5,
        )
        result = score(request)
        assert result.model_version is not None
        assert len(result.model_version) > 0
