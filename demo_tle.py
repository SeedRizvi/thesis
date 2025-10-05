import dsgp4

#let us assume we have the following two lines for the TLE, plus the first line that indicates the satellite name:
tle_lines = []
tle_lines.append('0 TIMATION 1')
tle_lines.append('1  2847U 67053E   24063.46171465  .00000366  00000-0  27411-3 0  9994')
tle_lines.append('2  2847  69.9643 216.8651 0003597  77.7866 282.3646 14.02285835897007')

#let us construct the TLE object
tle=dsgp4.tle.TLE(tle_lines)
print(tle)

#let's print all TLE elements:
print("TLE elements:")
print(f"Satellite catalog number: {tle.satellite_catalog_number}")
print(f"Classification: {tle.classification}")
print(f"International designator: {tle.international_designator}")
print(f"Epoch year: {tle.epoch_year}")
print(f"Epoch day: {tle.epoch_days}")
print(f"First time derivative of the mean motion: {tle._ndot}")
print(f"Second time derivative of the mean motion: {tle._nddot}")
print(f"BSTAR drag term: {tle._bstar}")
print(f"Element set number: {tle.element_number}")
print(f"Inclination [rad]: {tle._inclo}")
print(f"Right ascension of the ascending node [rad]: {tle._nodeo}")
print(f"Eccentricity [-]: {tle._ecco}")
print(f"Argument of perigee [rad]: {tle._argpo}")
print(f"Right ascension of ascending node [rad]: {tle._nodeo}")
print(f"Mean anomaly [rad]: {tle._mo}")
print(f"Mean motion [rad/min]: {tle._no_kozai}")

#let's first define the Earth radius according to WSG-84:
r_earth=dsgp4.util.get_gravity_constants('wgs-84')[2].numpy()*1e3

#let's extract the semi-major axis:
print(f"Semi-major axis [km]: {tle.semi_major_axis.numpy()*1e-3}")

#let's extract the TLE apogee & perigee altitudes:
print(f"Apogee radius [km]: {tle.apogee_alt(r_earth).numpy()*1e-3}")
print(f"Perigee radius [km]: {tle.perigee_alt(r_earth).numpy()*1e-3}")

