import re

def parse_gravity_file(filepath='./auxdata/GGM03S.txt'):
    print("="*70)
    print("GRAVITY FILE CONSTANTS")
    print("="*70)
    print(f"Parsing: {filepath}\n")
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        gm = None
        re_earth = None
        j2 = None
        
        for i, line in enumerate(lines[:50]):
            if 'GM' in line or 'earth_gravity_constant' in line.lower():
                match = re.search(r'([\d.]+[eE][+-]?\d+)', line)
                if match:
                    gm = float(match.group(1))
                    print(f"GM constant: {gm:.12e}")
                    print(f"  (Line {i+1}: {line.strip()})")
            
            if 'radius' in line.lower() or 're' in line:
                match = re.search(r'([\d.]+[eE][+-]?\d+)', line)
                if match and not gm:
                    re_earth = float(match.group(1))
                    print(f"\nEarth radius: {re_earth:.6f}")
                    print(f"  (Line {i+1}: {line.strip()})")
        
        for i, line in enumerate(lines[:500]):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    if parts[0] == '2' and parts[1] == '0':
                        j2 = -float(parts[2])
                        print(f"\nJ2 coefficient (C20): {j2:.12e}")
                        print(f"  (Line {i+1}: {line.strip()})")
                        print(f"  Note: C20 = -J2, so J2 = {-j2:.12e}")
                        break
                except:
                    pass
        
        print("\n" + "="*70)
        print("COMPARISON WITH FGO CODE")
        print("="*70)
        
        fgo_gm = 3.986004418e14
        fgo_j2 = 1.082626683e-3
        fgo_re = 6378137.0
        
        print(f"\nGM:")
        print(f"  FGO:  {fgo_gm:.12e}")
        if gm:
            print(f"  File: {gm:.12e}")
            diff_gm = abs(gm - fgo_gm) / fgo_gm * 100
            print(f"  Difference: {diff_gm:.6f}%")
            if diff_gm > 0.001:
                print(f"  ⚠ WARNING: GM values differ significantly!")
        
        print(f"\nJ2:")
        print(f"  FGO:  {fgo_j2:.12e}")
        if j2:
            print(f"  File: {-j2:.12e}")
            diff_j2 = abs(-j2 - fgo_j2) / fgo_j2 * 100
            print(f"  Difference: {diff_j2:.6f}%")
            if diff_j2 > 0.01:
                print(f"  ⚠ WARNING: J2 values differ significantly!")
        
        print(f"\nEarth Radius:")
        print(f"  FGO:  {fgo_re:.6f} m")
        if re_earth:
            print(f"  File: {re_earth:.6f} m")
            diff_re = abs(re_earth - fgo_re) / fgo_re * 100
            print(f"  Difference: {diff_re:.6f}%")
            if diff_re > 0.001:
                print(f"  ⚠ WARNING: Radius values differ significantly!")
        
        print("\n" + "="*70)
        print("RECOMMENDATION")
        print("="*70)
        
        if gm and abs(gm - fgo_gm) / fgo_gm > 0.001:
            print("\n❌ GM constants differ significantly!")
            print(f"   Update Orbit_FGO.py line 48:")
            print(f"   self.GE = {gm:.12e}")
        
        if j2 and abs(-j2 - fgo_j2) / fgo_j2 > 0.01:
            print("\n❌ J2 coefficients differ significantly!")
            print(f"   Update Orbit_FGO.py line 49:")
            print(f"   self.J2 = {-j2:.12e}")
        
        if re_earth and abs(re_earth - fgo_re) / fgo_re > 0.001:
            print("\n❌ Earth radius values differ significantly!")
            print(f"   Update Orbit_FGO.py line 50:")
            print(f"   self.R_earth = {re_earth:.6f}")
        
        if (not gm or abs(gm - fgo_gm) / fgo_gm < 0.001) and \
           (not j2 or abs(-j2 - fgo_j2) / fgo_j2 < 0.01) and \
           (not re_earth or abs(re_earth - fgo_re) / fgo_re < 0.001):
            print("\n✓ Constants match well!")
            print("  Dynamics mismatch must be from something else.")
        
        print("="*70)
        
    except FileNotFoundError:
        print(f"ERROR: Could not find {filepath}")
        print("Make sure you're running from the thesis project root directory")
        print("and that auxdata/GGM03S.txt exists")
    except Exception as e:
        print(f"ERROR parsing file: {e}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        parse_gravity_file(sys.argv[1])
    else:
        parse_gravity_file()
