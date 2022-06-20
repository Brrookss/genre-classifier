"""Test cases for genre classifier."""

import unittest

from dataset.preprocess.encode import integer_encode_mapping
from dataset.preprocess.encode import is_sorted


class TestDatasetPreprocess(unittest.TestCase):

    def testIntegerEncodeMapping_CorrectMapping(self):
        input = ["a", "b", "c", "d", "e"]
        expected = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
        self.assertEqual(integer_encode_mapping(input), expected)

    def testIsSorted_InputSorted(self):
        input = [0, 1, 2, 3, 4, 5]
        self.assertTrue(is_sorted(input))

    def testIsSorted_InputNotSorted(self):
        input = [5, 4, 3, 2, 1, 0]
        self.assertFalse(is_sorted(input))

    def testIsSorted_InputPartiallyNotSorted(self):
        input = [0, 2, 1, 3, 4, 5]
        self.assertFalse(is_sorted(input))

    def testIsSorted_InputEmpty(self):
        input = []
        self.assertTrue(is_sorted(input))

    def testIsSorted_InputDuplicateSorted(self):
        input = [0, 0, 1, 2, 3, 4, 5]
        self.assertTrue(is_sorted(input))


if __name__ == "__main__":
    unittest.main()
