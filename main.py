from propagator import OrbitPropagator, plot_orbit_3d

prop = OrbitPropagator("orbDetHOUSE")
csv = prop.propagate("configs/config_geo_realistic.yml", output_file="out/geo_truth.csv")
plot_orbit_3d(csv, output_file="plots/orbit.png")