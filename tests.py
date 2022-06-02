"""Test cases for genre classifier."""

import unittest


class TestCase(unittest.TestCase):

    def testStringMatches_HelloWorld(self):
        input = "Hello World"
        expected = "Hello World"
        self.assertEqual(input, expected)


if __name__ == "__main__":
    unittest.main()
