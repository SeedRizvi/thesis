from propagator import OrbitPropagator, plot_orbit_3d

prop = OrbitPropagator("orbDetHOUSE")
csv = prop.propagate("yamls/config_orb.yml", output_file="results.csv")
plot_orbit_3d(csv, output_file="orbit.png")