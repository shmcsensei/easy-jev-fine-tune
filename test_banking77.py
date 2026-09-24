import unittest

from banking77 import action_for_intent, percentile


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


class PercentileTests(unittest.TestCase):
    def test_percentile_interpolates_sorted_values(self):
        self.assertEqual(percentile([10.0, 20.0, 30.0], 0.5), 20.0)


if __name__ == "__main__":
    unittest.main()

