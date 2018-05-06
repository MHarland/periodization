import matplotlib, sys
matplotlib.use("Agg")

from ClusterDMFT.cdmft import CDmft
from ClusterDMFT.lattice.superlattices import BilayerGraphene as Sl
from ClusterDMFT.periodization.selfenergy_periodization import SEPeriodization

sl = Sl()
n_kpts = int(sys.argv[1])

for fname in sys.argv[2:]:
    x = CDmft(archive = fname)
    lat = SEPeriodization(sl.get_cartesian_clusterlatticevectors(),
                          sl.get_clusterlatticebasis(),
                          sl.get_hopping(),
                          n_kpts,
                          sym_path = [[0,0],[.5,0],[2./3.,-1./3.],[0,0]],
                          sym_path_lbls = ["$\Gamma$","$M$","$K$","$\Gamma$"])
    lat.set_all(x.load('sigma_c_iw'), sl.get_periodization())
    lat.export_results(fname[:-3]+"_sep.pdf")
