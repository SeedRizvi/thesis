import dsgp4
import torch
import matplotlib.pyplot as plt

#we load all TLEs:
tles = dsgp4.tle.load("../example.tle")
#we only extract the first one:
my_tle = tles[0]

#we always have to initialize the TLE before we can use it. If that does not, it can be directly initialized during propagation (with a small performance penalty):
dsgp4.initialize_tle(my_tle)

#I propagate for 1 day:
n_days = 1
tsinces = torch.linspace(0,n_days*24*60,10000)
state_teme=dsgp4.propagate(my_tle,tsinces)

dsgp4.plot_orbit(state_teme,
                 color='lightcoral',
                 label=f'SATCAT n°: {my_tle.satellite_catalog_number}')

plt.show()