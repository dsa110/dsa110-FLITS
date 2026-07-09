# Two-screen consistency (beta campaign, DSA band)

product = tau(1.4 GHz)[s] x dnu_d(1.4 GHz)[Hz] = C1/(2*pi): one screen gives 0.159 (thin) ... 1.0 (extended); accepted range [0.1, 2]; product >> 2 => the resolved delta_nu_d samples a NEARER screen than the scattering one (wilhelm pattern).

| burst | beta | tau_1GHz [ms] | tau_1.4GHz [ms] | dnu_d(1.4 GHz) [MHz] | tau.dnu | verdict |
|---|---|---|---|---|---|---|
| freya | 3.722 | 0.1194 | 0.02787 | 0.0934 +/- 0.016 | 2.6 | different_screens |
| casey | 3.990 | 0.01859 | 0.004839 | 12.5 +/- 0.4 | 60.5 | different_screens |
| chromatica | 3.990 | 0.2202 | 0.05733 | 1.64 +/- 0.06 | 93.9 | different_screens |
| wilhelm | 3.979 | 0.2693 | 0.06961 | 0.129 +/- 0.016 | 8.98 | different_screens |
| hamilton | 3.978 | 0.02447 | 0.006322 | 0.26 +/- 0.037 | 1.64 | same_screen |
| mahi | 3.785 | 0.2193 | 0.05265 | 1.66 +/- 0.3 | 87.2 | different_screens |
| zach | 3.990 | 0.1864 | 0.04852 | 0.492 +/- 0.054 | 23.9 | different_screens |
| oran | 3.987 | 0.8428 | 0.2194 | 0.34 +/- 0.065 | 74.6 | different_screens |
| isha | 3.841 | 0.3138 | 0.07707 | 0.506 +/- 0.25 | 39 | different_screens |
| johndoeII | 3.936 | 2.219 | 0.5649 | 0.497 +/- 0.074 | 281 | different_screens |
| whitney_fine | 3.968 | 1.182 | 0.3044 | 20.6 +/- 2.9 | 6.26e+03 | different_screens |
| phineas | 3.228 | 0.4694 | 0.08004 | 7.79 +/- 1.1 | 624 | different_screens |

CHIME-band delta_nu_d is not in stored_fits (needs a fresh ACF pass; 4 bursts have CHIME ACF pkls).
