import os, sys, time
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
sys.path.insert(0,'/home/z5363026/thesis'); os.chdir('/home/z5363026/thesis')
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from fgo_pipeline import load_config_parameters
from truth_cache import load_truth
from tier0_diag import InstrumentedFGO, build_seed
ARC=1/3600; C="configs/config_geo_one_rev_deltaRIC0.5.yml"
truth,times,dt,dvr,dve,mst,ts = load_truth(C,"deltaRIC0.5")
cp,gs = load_config_parameters(C)
p={'q_pos_ric':np.array(cp['process_noise_pos'],float),
   'q_vel_ric':np.array(cp['process_noise_vel'],float),'use_range':False,
   'measurement_noise_deg':2.0*ARC,'range_noise_m':cp['range_noise_m'],
   'initial_pos_error':cp['initial_pos_error'],'initial_vel_error':cp['initial_vel_error'],
   'dv_initial_error':cp['dv_initial_error'],'t_star_initial_error':cp['t_star_initial_error'],
   'epsilon':cp['epsilon'],'max_iterations':50}
fgo,_,_ = build_seed(1,truth,times,dt,gs,p,dve,mst,ts,InstrumentedFGO)

def t(f,n=3):
    f(); t0=time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter()-t0)/n

t_L = t(fgo.create_L)
t_y = t(fgo.create_y)
# cost of the F_man_mat block alone
t0=time.perf_counter()
for _ in range(3):
    for i in range(1,fgo.N):
        fgo.F_man_mat(fgo.states[i-1],(i-1)*fgo.dt)
t_fman=(time.perf_counter()-t0)/3
t0=time.perf_counter()
for _ in range(3):
    for i in range(1,fgo.N):
        fgo.F_mat(fgo.states[i-1],(i-1)*fgo.dt)
t_fmat=(time.perf_counter()-t0)/3
print(f"N={fgo.N}  stations={fgo.n_stations}  meas/station={fgo.meas_per_station}")
print(f"create_L            {t_L:7.3f} s")
print(f"  of which F_man_mat{t_fman:7.3f} s  ({100*t_fman/t_L:.1f}% of create_L)")
print(f"  of which F_mat    {t_fmat:7.3f} s  ({100*t_fmat/t_L:.1f}% of create_L)")
print(f"create_y            {t_y:7.3f} s")
print()
for nls in (1,4,8):
    it = t_L + t_y*(1+nls)
    sav = t_fman
    print(f"iteration with {nls} line-search trials: {it:6.3f} s   "
          f"analytic F_man saves {sav:.3f} s = {100*sav/it:4.1f}%")
