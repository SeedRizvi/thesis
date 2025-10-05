import dsgp4
import torch
import matplotlib.pyplot as plt

#we load all TLEs:
tles = dsgp4.tle.load("example_bulk.tle")
#we first need to prepare the data, the API requires that there are as many TLEs as times. Let us assume we want to
#propagate each of the 
tles_=[]
for tle in tles:
    tles_+=[tle]*10000
tsinces = torch.cat([torch.linspace(0,24*60,10000)]*len(tles))
#first let's initialize them:
_,tle_batch=dsgp4.initialize_tle(tles_)

#we propagate the batch of 3,000 TLEs for 1 day:
states_teme=dsgp4.propagate_batch(tle_batch,tsinces)

#Let's plot the first orbit:
ax=dsgp4.plot_orbit(states_teme[:10000],
                    color='lightcoral',
                    label=f'SATCAT n°:{tles[0].satellite_catalog_number}')
ax=dsgp4.plot_orbit(states_teme[10000:20000],
                    ax=ax, 
                    color='darkkhaki', 
                    label=f'SATCAT n°:{tles[1].satellite_catalog_number}')
ax=dsgp4.plot_orbit(states_teme[20000:],
                    ax=ax, 
                    color='lightseagreen', 
                    label=f'SATCAT n°:{tles[2].satellite_catalog_number}')

plt.show()
