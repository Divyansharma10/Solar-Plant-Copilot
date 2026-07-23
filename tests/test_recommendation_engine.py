import unittest

import pandas as pd

from recommendation_engine import evaluate_recommendations


class RecommendationEngineTests(unittest.TestCase):
    def setUp(self):
        self.window = pd.DataFrame({
            "time": pd.date_range("2026-01-01", periods=8, freq="h"),
            "global_tilted_irradiance": [500] * 8,
            "ac_power_kw": [2500] * 8
        })
        self.normal = {
            "missing_values": 0,
            "stuck_sensors": [],
            "model_performance_pct": 100,
            "rolling_baseline_pct": 100,
            "irradiance_vs_baseline_pct": 100,
            "cause": "Normal variation",
            "curtailment_detected": False,
            "clipping_detected": False
        }

    def evaluate(self, **changes):
        return evaluate_recommendations(
            {**self.normal, **changes}, self.window
        )

    def test_normal_operation_requires_no_action(self):
        self.assertEqual(self.evaluate()["primary_rule_id"], "OPS-000")

    def test_weather_reduction_does_not_dispatch_maintenance(self):
        decision = self.evaluate(
            cause="Weather",
            rolling_baseline_pct=82,
            irradiance_vs_baseline_pct=80
        )
        self.assertEqual(decision["primary_rule_id"], "WX-001")
        self.assertIn("do not dispatch", decision["action"])

    def test_curtailment_is_immediate(self):
        decision = self.evaluate(
            curtailment_detected=True,
            model_performance_pct=60
        )
        self.assertEqual(decision["primary_rule_id"], "OPS-001")
        self.assertEqual(decision["urgency"], "Immediate")

    def test_missing_data_has_priority(self):
        decision = self.evaluate(
            missing_values=8,
            curtailment_detected=True,
            model_performance_pct=60
        )
        self.assertEqual(decision["primary_rule_id"], "DQ-001")

    def test_sustained_low_output_triggers_grid_rule(self):
        low_window = self.window.assign(ac_power_kw=1200)
        decision = evaluate_recommendations(self.normal, low_window)
        self.assertEqual(decision["primary_rule_id"], "GRID-001")


if __name__ == "__main__":
    unittest.main()
