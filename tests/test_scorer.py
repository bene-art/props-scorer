"""Scorer unit tests — runs against the heuristic fallback (no model file in CI)."""
from inference_api.schemas import PredictRequest
from inference_api.scorer import score


def make(**kw) -> PredictRequest:
    return PredictRequest(**{"stat_type":"points","line":25.0,
                              "rolling_5_avg":26.0,"rolling_10_avg":25.0, **kw})


class TestScorer:
    def test_returns_response(self):
        r = score(make())
        assert all(hasattr(r, f) for f in ["probability","confidence","recommendation","source"])

    def test_probability_bounded(self):
        r = score(make(line=1.0, rolling_5_avg=100.0, rolling_10_avg=100.0))
        assert 0.0 <= r.probability <= 1.0

    def test_source_field(self):
        assert score(make()).source in ["model","heuristic"]

    def test_higher_rolling_avg_increases_probability(self):
        low  = score(make(line=25.0, rolling_5_avg=20.0, rolling_10_avg=20.0))
        high = score(make(line=25.0, rolling_5_avg=30.0, rolling_10_avg=30.0))
        assert high.probability > low.probability

    def test_lower_line_increases_probability(self):
        hi = score(make(line=35.0, rolling_5_avg=25.0, rolling_10_avg=25.0))
        lo = score(make(line=15.0, rolling_5_avg=25.0, rolling_10_avg=25.0))
        assert lo.probability > hi.probability

    def test_home_advantage(self):
        assert score(make(is_home=True)).probability >= score(make(is_home=False)).probability

    def test_tougher_defense_lowers_probability(self):
        assert score(make(opp_def_rtg=105.0)).probability >= score(make(opp_def_rtg=115.0)).probability

    def test_confidence_from_distance(self):
        # Way above avg -> high P(over) -> high confidence
        strong = score(make(line=10.0, rolling_5_avg=30.0, rolling_10_avg=30.0))
        assert strong.confidence == "high" and strong.recommendation == "OVER"
        # Coin flip -> low confidence
        coin = score(make(line=25.5, rolling_5_avg=25.5, rolling_10_avg=25.5))
        assert coin.confidence == "low" and coin.recommendation == "NO_EDGE"

    def test_recommendation_over(self):
        assert score(make(line=10.0, rolling_5_avg=30.0, rolling_10_avg=30.0)).recommendation == "OVER"

    def test_recommendation_under(self):
        assert score(make(line=50.0, rolling_5_avg=25.0, rolling_10_avg=25.0)).recommendation == "UNDER"

    def test_recommendation_no_edge(self):
        assert score(make(line=25.0, rolling_5_avg=25.0, rolling_10_avg=25.0)).recommendation == "NO_EDGE"

    def test_model_version_present(self):
        assert score(make()).model_version
