import json
import unittest
from pathlib import Path

from banking77 import (
    ASSISTED_SUPPORT_INTENTS,
    INVESTIGATION_INTENTS,
    SECURITY_INTENTS,
    BankingClassifier,
    action_for_intent,
    percentile,
)


class PolicyTests(unittest.TestCase):
    def test_security_intent_routes_to_immediate_action(self):
        self.assertEqual(action_for_intent("lost_or_stolen_card"), "secure_account_now")

    def test_investigation_intent_routes_to_investigation(self):
        self.assertEqual(
            action_for_intent("pending_card_payment"), "investigate_transaction"
        )

    def test_assisted_intent_routes_to_support(self):
        self.assertEqual(action_for_intent("pin_blocked"), "assisted_support")

    def test_unmapped_intent_routes_to_self_service(self):
        self.assertEqual(action_for_intent("card_arrival"), "self_service")

    def test_policy_groups_do_not_overlap(self):
        groups = [SECURITY_INTENTS, INVESTIGATION_INTENTS, ASSISTED_SUPPORT_INTENTS]
        for index, group in enumerate(groups):
            for other in groups[index + 1 :]:
                self.assertTrue(group.isdisjoint(other))

    def test_every_policy_intent_exists_in_adapter_metadata(self):
        metadata_path = Path(__file__).parent / "adapter-banking77" / "banking77_metadata.json"
        labels = set(json.loads(metadata_path.read_text())["label_names"])
        configured = SECURITY_INTENTS | INVESTIGATION_INTENTS | ASSISTED_SUPPORT_INTENTS
        self.assertTrue(configured <= labels)


class PercentileTests(unittest.TestCase):
    def test_percentile_interpolates_sorted_values(self):
        self.assertEqual(percentile([10.0, 20.0, 30.0], 0.5), 20.0)

    def test_percentile_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            percentile([], 0.5)

    def test_percentile_rejects_invalid_percentage(self):
        with self.assertRaises(ValueError):
            percentile([1.0], 1.1)


class PredictionInputTests(unittest.TestCase):
    def setUp(self):
        self.classifier = BankingClassifier.__new__(BankingClassifier)

    def test_rejects_invalid_threshold(self):
        with self.assertRaises(ValueError):
            self.classifier.predict("hello", threshold=1.1)

    def test_rejects_nonpositive_top_k(self):
        with self.assertRaises(ValueError):
            self.classifier.predict("hello", top_k=0)


if __name__ == "__main__":
    unittest.main()
