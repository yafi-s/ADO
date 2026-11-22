import unittest

from crosswalk.stationary_intent import FEATURES, StationaryIntent


class StationaryIntentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=StationaryIntent()

    def test_source_vehicle_motion_changes_the_fitted_preference(self):
        examples=[
            ([35.,8.,.08,.05,.05,0.,1.,30.],"follow"),
            ([7.,.1,0.,.1,-.1,2.,9.,30.],"follow"),
            ([7.,.1,0.,.1,-.1,22.,29.,30.],"defer_to_existing_choice"),
            ([18.,6.,1.7,1.7,0.,0.,4.,30.],"defer_to_existing_choice"),
            ([18.,6.,2.,.8,2.,0.,2.,30.],"follow"),
        ]
        for row,expected in examples:
            with self.subTest(row=row):
                action,frequency=self.model.predict(dict(zip(FEATURES,row)))
                self.assertEqual(action,expected)
                self.assertGreaterEqual(frequency,0.)
                self.assertLessEqual(frequency,1.)
                self.assertEqual((action,frequency),self.model.predict(row))

    def test_incomplete_or_nonfinite_measurements_cannot_issue_a_choice(self):
        for row in ([0.]*7,[float('nan')]*8,[float('inf')]*8):
            with self.assertRaises(ValueError):
                self.model.predict(row)
