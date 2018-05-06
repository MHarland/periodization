import unittest
#from pytriqs.utility import mpi #initializes MPI, required on some clusters

from test_selfenergyperiodization import TestLatticeSelfenergy
from test_greensfunctionperiodization import TestLatticeGreensfunction
from test_cumulantperiodization import TestLatticeCumulant
from test_dualfermionperiodization import TestDFPeriodization


suite = unittest.TestSuite()
suite.addTest(TestLatticeSelfenergy("test_LatticeSelfenergy_full"))
suite.addTest(TestLatticeGreensfunction("test_LatticeGreensfunction_full"))
suite.addTest(TestLatticeCumulant("test_LatticeCumulant_full"))
#suite.addTest(TestDFPeriodization("test_LatticeSelfenergy_full"))
unittest.TextTestRunner(verbosity = 2).run(suite)
