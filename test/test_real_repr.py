import unittest
from nn_smt2 import *

class TestRealRepr(unittest.TestCase):

    def test_real_repr(self):
        self.assertEqual(real_repr(1), '1.0')
        self.assertEqual(real_repr(1.5), '1.5')
        self.assertEqual(real_repr(1e+15), '1000000000000000.0')
        self.assertEqual(real_repr(1e+16), '10000000000000000.0')
        self.assertEqual(real_repr(1e+22), '10000000000000000000000.0')
        self.assertEqual(real_repr(1e-4), '0.0001')
        self.assertEqual(real_repr(1e-5), '0.00001')
        self.assertEqual(real_repr(1e-20), '0.00000000000000000001')
        self.assertEqual(real_repr(1.5e+20), '150000000000000000000.0')
        self.assertEqual(real_repr(1.5e-20), '0.000000000000000000015')
        self.assertEqual(real_repr(1.2345678901234568e+17), '123456789012345680.0')
        self.assertEqual(real_repr(1.2345678901234568e+16), '12345678901234568.0')
        self.assertEqual(real_repr(1.2345678901234568e+15), '1234567890123456.8')
        self.assertEqual(real_repr(1.2345678901234569e+14), '123456789012345.69')
        self.assertEqual(real_repr(1.2345678901234567e+1), '12.345678901234567')
        self.assertEqual(real_repr(1.2345678901234567e+0), '1.2345678901234567')
        self.assertEqual(real_repr(1.2345678901234569e-1), '0.12345678901234569')
        self.assertEqual(real_repr(1.2345678901234568e-2), '0.012345678901234568')
        self.assertEqual(real_repr(1.2345678901234567e-3), '0.0012345678901234567')
        self.assertEqual(real_repr(1.2345678901234567e-4), '0.00012345678901234567')
        self.assertEqual(real_repr(1.2345678901234568e-5), '0.000012345678901234568')

if __name__ == '__main__':
    unittest.main()
