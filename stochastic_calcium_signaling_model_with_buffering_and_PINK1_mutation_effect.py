# HOW TO ACCESS DATA (python command line or python script)
# f = h5py.File(ip3_X.XX_accoa_X.XXeXXX_N_XXX_lambda_XXX.hdf5, "r", libver='latest', swmr=True)
# f['t']         -> time points
# f['y']         -> concentrations of all species at each time point
# f['flux']      -> fluxes of all processes at each time point
# f['DG']        -> \Delta_r G of all processes at each time point
# f['EPR']       -> EPR (T*\sigma) of all processes at each time point
# f['ncW']       -> Nonconservative work for "r1out", "r1in" and "r2out"
#
# t.id.refresh(), y.id.refresh()
# tt, cac, atpc = t[()], y[6,:], y[4,:]
# -----------------------------------------------------------------------------
# INDICES
#       SPECIES:                    |   PROCESSES
#       0:adpc          10:isoc     |   0:aco           10:mdh
#       1:adpm          11:mal      |   1:ant           11:ncx
#       2:akg           12:nadhm    |   2:cs            12:ox
#       3:asp           13:nadm     |   3:erout         13:sdh
#       4:atpc          14:oaa      |   4:f1            14:serca
#       5:atpm          15:psi      |   5:fh            15:sl
#       6:cac           16:scoa     |   6:hl            16:uni
#       7:cam           17:suc      |   7:hyd           17:IP3R
#       8:cit                       |   8:idh           18:leak
#       9:fum                       |   9:kgdh          
# -----------------------------------------------------------------------------



import sys, os
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import h5py

# Parameters entered in command line
ip3 = float(sys.argv[1])                # in uM                
accoa = float(sys.argv[2])              # in mM
#run = sys.argv[3]                       # '' 'uncoupled/' 'HB/' 'large/' 'HB_low_SERCA/' 'low_SERCA/' = subfolder where input files are stored
vserca = float(sys.argv[3])
N = float(sys.argv[4])
lmda = float(sys.argv[5])
#mutation_scale = float(sys.argv[6])

ic = 'no'
ss = 'no'


def fluxes(v):
        v = v[:18]  # Ignore CaB (buffer) for Gibbs energy calculations
        # Fluxes are all returned in mM/s
        # For convenience of notation
        [adpc, adpm, akg, asp, atpc, atpm, cac, cam, cit, fum, isoc, mal, nadhm, nadm, oaa, psi, scoa, suc]=v
        caer = coef1 - cac * coef2 - cam * coef3

        # Aconitase
        Jaco = kfaco * (cit -isoc/keaco)
        
        # Exchange of mitochondrial ATP for cytosolic ADP (antiporter)
        Jant = vant * (1. - alphac * atpc * adpm * np.exp(-psi / RTF) / (alpham * adpc * atpm)) / ((1. + alphac * atpc * np.exp(-f * psi / RTF)/ adpc) * (1. + adpm / (alpham * atpm)))
        
        # Citrate synthase (CS)
        Jcs = Vmaxcs * accoa / (accoa + kmaccoa + (kmoaa* accoa /oaa)*(1.+accoa/kiaccoa) + (ksaccoa * kmoaa)/oaa)        # Dudycha
        
        # Ca2+ release from ER via IP3Rs + passive leak
        Jerout = (vip3 * ( ip3 ** 2. / (ip3 ** 2. + kip3 ** 2.)) * (cac ** 2. / (cac ** 2. + kcaa ** 2. )) * (kcai1 ** 4. / (kcai2 ** 4. + cac ** 4.)) + vleak) * (caer - cac) / uMmM
        
        Jip3R = (vip3 * ( ip3 ** 2. / (ip3 ** 2. + kip3 ** 2.)) * (cac ** 2. / (cac ** 2. + kcaa ** 2. )) * (kcai1 ** 4. / (kcai2 ** 4. + cac ** 4.))) * (caer - cac) / uMmM
        Jleak = vleak * (caer - cac) / uMmM
        
        # Phosphorylation of mitochondrial ADP via F1F0 ATPase
        deltaMuH = psi - 2.303 * RTF * DpH
        VDf1 = np.exp(3. * psi / RTF)
        Af1 = kf1 * atpm / (adpm * pim)
        Jf1 = - rhof1 * (Pac1 * Af1 - pa * VDf1 + pc2 * Af1 * VDf1) / ((1. + p1 * Af1) * VDf1B + (p2 + p3 * Af1) * VDf1)
        
        # Fumarate hydratase
        Jfh = kffh * (fum - mal/kefh)
        
        # Proton leak from cytosol to mitochondria
        Jhl = gH * deltaMuH
        
        # Isocitrate dehydrogenase
        Jidh = Vmaxidh / ( 1. + hm/kh1 + kh2/hm + (kmisoc/isoc) ** ni /((1. + adpm/kaadp) * (1. + cam/kaca)) + (kmnad/nadm)*(1. + nadhm/kinadh) + (kmisoc/isoc) ** ni *(kmnad/nadm)*(1. + nadhm/kinadh) /((1. + adpm/kaadp) * (1. + cam/kaca)))
        
        # Alpha-ketoglutarate dehydrogenase
        Jkgdh = Vmaxkgdh / (1. + (kmakg / akg) /((1. + mg/kdmg) * (1. + cam/kdca)) + (kmnadkgdh/nadm) ** nakg /((1. + mg/kdmg) * (1. + cam/kdca)))
        
        # Malate dehydrogenase
        fha = 0.351195
        fhi = 0.999346
        Jmdh = Vmdh * (mal * nadm - oaa * nadhm / Keqmdh) / ((1. + mal/kmmal) * (1. + nadm/kmnadmdh) + (1. + oaa/kmoaamdh)*(1. + nadhm/kmnadhmdh) - 1.)
        
        # Ca2+ release from mitochondria via NCX (Cortassa 2003)
        Jncx = vncx * (cam / cac) * np.exp(b * (psi - psiStar) / RTF) / ((1. + kca / cam) * (1. + kna / nac) ** n)
        
        # Oxidation of NADH by respiration
        VDres = np.exp(6. * g * psi / RTF)
        Ares = kres * (nadhm / nadm) ** 0.5
        Jo = 0.5 * rhores * (Rac1 * Ares - ra * VDres + rc2 * Ares * VDres ) / ((1. + r1 * Ares) * VDresB + (r2 + r3 * Ares) * VDres)
        
        # Hydrolysis of cytosolic ATP
        Jhyd = khyd * atpc / (katpcH + atpc)
        
        # Succinate dehydrogenase
        Jsdh = Vmaxsdh / (1. + (kmsuc/suc) * (1. + oaa/kioaasdh) * (1. + fum/kifum))
        
        # SERCA pumps
        Jserca = vserca * (cac ** 2. / (kserca ** 2. + cac ** 2.)) * (atpc / (atpc + katpc))
        
        # Succinyl Coa Lyase
        Jsl = kfsl * (scoa * adpm * pim*1.e3 - suc * atpm * coa / kesl)
        
        # Ca2+ uptake to mitochondria via MCU
        VDuni = 2. * (psi - psiStar) / RTF
        trc = cac / ktrans
  # Avoid division by zero or negative powers
        cac = max(v[6], 1e-6)   # Ca²⁺ in cytosol
        cam = max(v[7], 1e-6)   # Ca²⁺ in mitochondria
        mwc = trc * (1. + trc) ** 3. / ((1. + trc) ** 4. + L / (1. + cac / kact) ** ma)
        #Juni = vuni * VDuni * (mwc - cam * np.exp(-VDuni)) / (1. - np.exp(-VDuni))
        Juni=vuni * VDuni * mwc / (1. - np.exp(-VDuni))
       
        return Jaco, Jant, Jcs, Jerout, Jf1, Jfh, Jhl, Jhyd, Jidh, Jkgdh, Jmdh, Jncx, Jo, Jsdh, Jserca, Jsl, Juni, Jip3R, Jleak;
        
def mmk(s, v, r):
    # s = dt
    # v = vector with chemical species
    # r = random number for the iteration * sqrt(dt)
   
        Jaco, Jant, Jcs, Jerout, Jf1, Jfh, Jhl, Jhyd, Jidh, Jkgdh, Jmdh, Jncx, Jo, Jsdh, Jserca, Jsl, Juni, Jip3R, Jleak = fluxes(v)
          
        dv = np.zeros((len(v), 1))

    # Original species updates
        dv[0] = s * (-delta * Jant + Jhyd + 0.5 * Jserca)                  # adpc
        dv[1] = s * (Jant - Jf1 - Jsl)                                     # adpm
        dv[2] = s * (Jidh - Jkgdh)                                         # akg
        dv[3] = 0.                                                         # asp
        dv[4] = -dv[0]                                                     # atpc
        dv[5] = -dv[1]                                                     # atpm

    # Cytosolic Ca2+ before buffering
        #dv[6] = s * uMmM * fc * (-Jserca + Jerout + delta * (Jncx - Juni)) + r * np.sqrt(uMmM * fc * Jip3R)

        dv[7] = uMmM * fm * s * (Juni - Jncx)                              # cam
        dv[8] = s * (Jcs - Jaco)                                           # cit
        dv[9] = s * (Jsdh - Jfh)                                           # fum
        dv[10] = s * (Jaco - Jidh)                                         # isoc
        dv[11] = s * (Jfh - Jmdh)                                          # mal
        dv[12] = s * (-Jo + Jidh + Jkgdh + Jmdh)                           # nadhm
        dv[13] = -dv[12]                                                  # nadm
        dv[14] = s * (Jmdh - Jcs)                                          # oaa
        dv[15] = s * (10. * Jo - 3. * Jf1 - Jant - Jhl - Jncx - 2. * Juni) / cmito  # psi
        dv[16] = s * (Jkgdh - Jsl)                                         # scoa
        dv[17] = s * (Jsl - Jsdh)                                          # suc

    # --- BUFFERING of Ca2+ in cytosol ---
        CaB = v[18]                                 # Bound buffer (uM)
        F_free = max(F_total - CaB, 1e-6)          # Free buffer (uM), avoid negatives
        Jbuff = kon * v[6] * F_free - koff * CaB    # Binding flux (uM/s)
        cac = max(v[6], 1e-6)   # Ca²⁺ in cytosol
        cam = max(v[7], 1e-6)   # Ca²⁺ in mitochondria
        Jip3R_clamped = max(Jip3R, 0.0)
        dv[6] = s * uMmM * fc * (-Jserca + Jerout + delta * (Jncx - Juni)) + r * np.sqrt(uMmM * fc * Jip3R_clamped)
        dv[6] += -s * Jbuff                         # Subtract from free Ca2+
        dv[18] = s * Jbuff                          # Add to bound buffer

        return dv  # (in uM/s or mM/s)
def deltaG(v):
        # Concentrations appearing in \Delta G are renormalised with respect to the same volume of reference as the corresponding flux.
        v = v[:18]  # Ignore CaB (buffer) for flux calculations
        # For convenience of notation
        [adpc, adpm, akg, asp, atpc, atpm, cac, cam, cit, fum, isoc, mal, nadhm, nadm, oaa, psi, scoa, suc]=v
        caer = coef1 - cac * coef2 - cam * coef3
        
        safe_ratio = max(adpc**0.5 * (1.e-3 * pic)**0.5 * caer / (atpc**0.5 * cac), 1e-12)
        DGserca = 0.5*DGhydc0 + RT * np.log(safe_ratio)        
        DGerout = RT * np.log(cac / caer)
        DGuni = RT * np.log(cam / cac) - 2. * F * psi
        DGncx = RT * np.log((cac * nam ** 3.) / (cam * nac ** 3.)) - F * psi
        atpc4m = 0.05 * atpc
        atpm4m = 0.05 * atpm
        adpc3m = 0.45 * adpc
        adpm3m = 0.45 * 0.8 * adpm
        DGant = RT * np.log(atpc4m * adpm3m / (atpm4m * adpc3m)) - F * psi
        DGf1 = - DGhydm0 + RT * np.log(hm ** 3. * atpm / (adpm * pim * hc ** 3.))- 3. * F * psi
        DGhl = RT * np.log(hm / hc) - F * psi
        DGO = DGO0 + RT * np.log(hc ** 10. * nadm / (hm ** 10. * nadhm * o2 ** 0.5)) + 10. * F * psi
        # Real vectorial equation: nadh + 11 Hm + 0.5 O2 = Nad + 10 Hc + H2O
        DGhyd = DGhydc0 + RT * np.log(1.e-3 * adpc * pic / atpc)
        DGcs = DGcs0 + RT * np.log(cit * coa / (oaa * accoa))
        DGaco = DGaco0 + RT * np.log(isoc / cit)
        DGidh = DGidh0 + RT * np.log(1.e-3 * akg * co2 * nadhm / (isoc * nadm))
        DGkgdh = DGkgdh0 + RT * np.log(scoa * nadhm * co2 / (akg * nadm * coa))
        DGsl = DGsl0 + RT * np.log(suc * coa * atpm / (scoa * adpm * pim * 1.e3))
        DGsdh = DGsdh0 + RT * np.log(fum * coqh2 / (suc * coq))
        DGfh = DGfh0 + RT * np.log(mal / fum)
        DGmdh = DGmdh0 + RT * np.log(oaa * nadhm / (nadm * mal))
        DGip3R = DGerout
        DGleak = DGerout
        
        return [DGaco, DGant, DGcs, DGerout, DGf1, DGfh, DGhl, DGhyd, DGidh, DGkgdh, DGmdh, DGncx, DGO, DGsdh, DGserca, DGsl, DGuni, DGip3R, DGleak]

# KINETIC PARAMETERS
# ------------------
# Conversions (nmol/(ml*min) = uM/min)
uMmM = 1000.            # (uM/mM)

# Conservation relations
amtot = 15.                     # = atpm + adpm (mM)
atot = 3.                       # = atpc + adpc (mM)
ckint = 1.                      # = cit + isoc + akg + scoa + suc + fum + mal + oaa (mM) -- Cortassa 2003
nadtot = 0.8                    # = nadm + nadhm (mM)
ctot = 1500.                    # = cac/fc + alpha*caer/fe + delta*cam/fm (uM) -- Wacquier 2016
F_total = 20.0       # uM, total buffer concentration (adjustable)
kon = 100            # 1/uM/s, on-rate of buffer binding Ca2+
koff = 20.0 
CaB0 = 0.0  # uM, initial bound buffer (assume all buffer is free initially)

# Cell parameters
cmito = 1.812e-3        # mitochondrial membrane capacitance (mM/mV)
fc = 0.01               # fraction of free Ca2+ in cytosol -- MK 1998a, Fall-Keizer 2001
fe = 0.01               # fraction of free Ca2+ in ER -- Fall-Keizer 2001
fm = 0.0003             # fraction of free Ca2+ in mitochondria -- MK 1997
alpha = 0.10            # volumic ratio between ER and cytosol (Ve / Vc) -- Wacquier 2016
delta = 0.15            # volumic ratio between mito and cytosol (Vm / Vc) -- Siess et al. 1976, Lund & Wiggins 1987
coef1 = fe * ctot / alpha
coef2 = fe / (alpha * fc)
coef3 = delta * fe / (alpha * fm)

# Other fixed concentrations and cell parameters
co2 = 21.4                      # total CO2 concentration (mM) -- Beard 2007
coa = 0.02                      # concentration of coenzyme A (mM) -- Cortassa 2003
coq = 0.97                      # concentration of coenzyme q10 oxidized (mM) -- Beard 2007
coqh2 = 0.38                    # concentration of coenzyme q10 reduced (mM) -- Beard 2007
DpH = -0.8                      # = pHc - pHm -- Casey 2010
F = 96.485                      # Faraday constant in kC / mol
o2 = 2.6e-5                     # O2 concentration (M) -- Beard 2005
pic = 1.                        # inorganic phosphate concentration in cytosol (mM) -- Bevington et al. 1986
RT = 2577.34                    # = R*T, R=8.314 J/(mol*K), T=310 K
RTF = 26.7123                   # = R*T/F (1/mV), R=8.314 J/(mol*K), T=310 K, F=96485 C/mol -- MK 1997

# --- LEAK and IP3Rs ---#
vip3 = 30.                      # 15. max release rate of Ca2+ through IP3R (1/s)
vleak = 0.15                    # 0.15 rate of Ca2+ release by leakage from ER (1/s)
kcai1 = 1.4                     # Inhibition constant 1 of IP3R for Ca2+ (uM) -- Moeien thesis Table 3.1 : 1.3)
kcai2 = kcai1                   # Inhibition constant 1 of IP3R for Ca2+ (uM) -- Moeien thesis Table 3.1 : 1.3)
kcaa = 0.70                     # Activation constant of IP3R for Ca2+ (uM) -- Moien 2017 : 0.9
kip3 = 1.00                     # Activation constant of IP3R for IP3 (uM) -- Wacquier 2016

# --- SERCA ---#
#vserca = 0.12                  # max SERCA rate (mM/s)
kserca = 0.35                   # Km of SERCA pumps for Ca2+ (uM) -- Wacquier 2016, Lytton 1992
katpc = 0.05                    # Km of Serca for ATPc (mM) -- Lytton 1992, Scofano 1979: 0.05 (mM) (Moien 2017: 0.06)

# --- MCU ---#
#vuni = 0.300                    # max uniporter rate at psi=psiStar (mM/s)
psiStar = 91.                   # psi offset for Ca2+ transport (mV) -- MK 1997 Table 5
L = 110.                        # Keq for uniporter conformations -- MK 1998a Table 1
ma = 2.8                        # uniporter activation cooperativity -- MK 1997 Table 5
ktrans = 19.                    # Kd for uniporter translocated Ca2+ (uM) -- MK 1998a Table 1
kact = 0.38                     # Kd for uniporter activating Ca2+ (uM) -- MK 1997 Table 5
f = 0.5                         # fraction effective psi -- MK 1997 Table 5, Cortassa 2003

# --- NCX ---#
#vncx = 2.e-3                    # max NCX rate (mM/s)
n = 3.                          # nunmber of Na+ binding to NCX (electroneutral/electrogenic: n=2/3) -- MK 1997 Table 5
nac = 10.                       # cytosolic Na+ concentration (mM) -- Cortassa 2003
nam = 5.                        # mitochondrial Na+ concentration (mM) -- Donoso et al. 1992
kca = 0.375                     # Km (Ca2+) for NCX (uM) -- Bertram 2006, Cortassa 2003
b = 0.5                         # NCX dependence on psi (electroneutral/electrogenic: b=0/0.5) -- MK 1997 Table 5
kna = 9.4                       # Km (Na+) for NCX (mM) -- MK 1997 Table 5

# --- ATP hydrolysis ---#     
khyd = 9.e-2                    # Basal hydrolysis rate of ATPc (1/s)
katpcH = 1.00                   # Michaelis-Menten constant for ATP hydrolysis (mM) -- Wacquier 2016

# --- ATPase ---#
#rhof1 = 0.23                    # concentration of ATPase pumps (mM)
psiB = 50.                      # total phase boundary potentials (mV) -- MK 1997 Tables 2 and 3
p1 = 1.346e-8                   # MK 1997
p2 = 7.739e-7                   # MK 1997
p3 = 6.65e-15                   # MK 1997
pa = 1.656e-5                   # (1/s) -- MK 1997
pb = 3.373e-7                   # (1/s) -- MK 1997
pc1 = 9.651e-14                 # (1/s) -- MK 1997
pc2 = 4.845e-19                 # (1/s) -- MK 1997
pim = 0.020                     # inorganic phosphate concentration in mitochondrial matrix (M) -- MK 1997
kf1 = 1.71e6                    # Equilibrium constant of ATP hydrolysis -- Cortassa 2003, 2006, 2011

VDf1B = np.exp(3. * psiB / RTF)
Pa1 = pa * 10 ** (3.*DpH)
Pac1 = (Pa1 + pc1 * VDf1B) 

# --- Respiration ---#
#rhores = 1.00                  # respiration-driven H+ pump concentration (mM)
r1 = 2.077e-18                  # MK 1997
r2 = 1.728e-9                   # MK 1997
r3 = 1.059e-26                  # MK 1997
ra = 6.394e-10                  # (1/s) -- MK 1997
rb = 1.762e-13                  # (1/s) -- MK 1997
rc1 = 2.656e-19                 # (1/s) -- MK 1997
rc2 = 8.632e-27                 # (1/s) -- MK 1997
g = 0.85                        # fitting factor for voltage -- MK 1997 Table 2
kres = 1.35e18                  # MK 1997 Table 2

VDresB = np.exp(6. * psiB / RTF)
Ra1 = ra * 10 ** (6.*DpH)
Rac1 = (Ra1 + rc1 * VDresB)

# --- Antiporter ---#
vant = 4.                       # (mM/s)
alphac = 0.11111                # = 0.05 / 0.45 -- Cortassa 2003, MK 1997
alpham = 0.1388889              # = 0.05 / (0.45 * 0.8) -- MK 1997

# --- Proton leak ---#
gH = 1.e-5                      # (mM/(mV * s)) -- Cortassa 2003
hc = 6.31e-5                    # cytosolic proton concentration (mM) -- Casey 2010
hm = 1.e-5                      # mitochondrial proton concentration (mM) -- Casey 2010

# --- TCA cycle --- #
# Citrate synthase
Vmaxcs = 104.                   # Vmax of citrate synthase (mM/s)
kmaccoa = 1.2614e-2             # Km of CS for AcCoa (mM) -- Cortassa 2003, Dudycha 2000: 1.26e-2 mM; BRENDA: 5-10 uM, Shepherd-Garland 1969: 16 uM
kiaccoa = 3.7068e-2             # Ki of CS for AcCoa (mM) -- Cortassa 2003
ksaccoa = 8.0749e-2             # other Km of CS for AcCoA (mM) -- Dudycha 2000
kmoaa = 5.e-3                   # Km of CS for OAA (mM) -- Cortassa 2003, Dudycha 2000: 6.2981e-4, Shepherd-Garland: 0.002, Kurz...Srere: 0.0059 mM, Matsuoka & Srere 1973+Berndt 2015 = 0.0050 mM

# 2. Aconitase
keaco = 0.067                   # equilibrium constant of aconitase -- equilibrator pH=8, I=0.12M, pMg=3.4, Berndt 2015, Garret & Grisham
kfaco = 12.5                    # forward rate constant of aconitase (1/s) -- Cortassa 2003

# 3. Isocitrate dehydrogenase
Vmaxidh = 1.7767                # Vmax of isocitrate dehydrogenase (mM/s) -- Cortassa 2003 (push conditions)
kmisoc = 1.52                   # Km of IDH for isocitrate (mM) -- Cortassa 2003
kmnad = 0.923                   # Km of IDH for NAD (mM) -- Cortassa 2003
kinadh = 0.19                   # Inhibition constant of IDH for NADHm (mM) -- Cortassa 2003
kaadp = 6.2e-2                  # Activation constant of IDH for ADPm (mM) -- Cortassa 2003
kaca = 1.41                     # Activation constant of IDH for mitochondrial Ca2+ (uM) -- Cortassa 2003
kh1 = 8.1e-5                    # Ionization constant of IDH (mM) -- Dudycha 2000, Cortassa 2003
kh2 = 5.98e-5                   # Ionisation constant of IDH (mM) -- Dudycha 2000, Cortassa 2003
ni = 2.                         # Cooperativity of Isoc in IDH -- Cortassa 2011: 2. (lacking in Cortassa 2003)

# 4. Alpha-ketoglutarate dehydrogenase
Vmaxkgdh = 2.5                  # Vmax of aKG dehydrogenase (mM/s) -- Cortassa 2003
kmakg = 1.94                    # Km of KGDH for aKG (mM) -- Cortassa 2003
kmnadkgdh = 0.0387              # Km of KGDH for NAD (mM)
nakg = 1.2                      # Hill coefficient of KGDH for aKG -- Cortassa 2003
kdca = 1.27                     # activation constant of KGDH for mitochondrial Ca2+ (uM) -- Cortassa 2003
kdmg = 0.0308                   # activation constant of KGDH for Mg2+ (mM) -- Cortassa 2003
mg = 0.4                        # mitochondrial Mg2+ concentration (mM) -- Cortassa 2003

# 5. Succinyl-CoA Lyase (Succinyl-CoA Synthetase)
kesl = 0.724                    # equilibrium constant of succinyl coa lyase -- equilibrator pH=8, I=0.12M, pMg=3.4 (reac with ADP/ATP: 0.724, with GDP/GTP: 2.152)
kfsl = 0.127                    # forward rate constant of succinyl coa lyase (1/s) -- Cortassa 2003 (0.127)

# 6. Succinate dehydrogenase
Vmaxsdh = 0.5                   # Vmax of succinate dehydrogenase (mM/s) -- Cortassa 2003
kmsuc = 3.0e-2                  # Km of SDH for succinate (mM) -- Cortassa 2003
kioaasdh = 0.15                 # Inhibition constant of SDH for OAA (mM) -- Cortassa 2003
kifum = 1.3                     # Inhibition constant of SDH for fumarate (mM) -- Cortassa 2003

# 7. Fumarate hydratase (fumarase)
kefh = 3.942                    # equilibrium constant of fumarate hydratase -- equilibrator pH=8, I=0.12M, pMg=3.4
kffh = 8.3                      # forward rate constant of fumarate hydratase (1/s) -- Cortassa 2003 (0.83)

# 8. Malate dehydrogenase
Vmdh = 128.             # mM/s
Keqmdh = 2.756e-5       # equilibrator pH=8, I=0.12M, pMg=3.4
kmmal = 0.145           # Km of MDH for malate (mM) -- Berndt 2015
kmnadmdh = 0.06         # Km of MDH for NAD (mM) -- Berndt 2015 (0.06)
kmoaamdh = 0.017        # Inhibition constant of MDH for OAA (mM) -- Berndt 2015
kmnadhmdh = 0.044       # Km of MDH for NADH (mM) -- Berndt 2015

# ------------------------------------------------------------------------
# Standard Gibbs energies of reactions at ionic strength=0.12 M (Robinson 2006), pMg=3.4, pHm=8.0 and pHc=7.2
DGhydc0 = -28300.
DGhydm0 = -32200. 
DGO0 = -225300.
DGcs0 = -41200.
DGaco0 = 6700.
DGidh0 = 5100.
DGkgdh0 = -27600.
DGsl0 = 800.                    # with ADP/ATP: 800 J/mol, with GDP/GTP: -1900 J/mol
DGsdh0 = -24200.
DGfh0 = -3400.
DGmdh0 = 24200.
DGtca0 = -59600.       # = DGcs0 + DGaco0 + DGidh0 + DGkgdh0 + DGsl0 + DGsdh0 + DGfh0 + DGmdh0

# ------------------------------------------------------------------------
# Initial conditions
if ic == 'yes':
        # Internal IC
        t0 = tf
        [adpc0, adpm0, akg0, asp0, atpc0, atpm0, cac0, cam0, cit0, fum0, isoc0, mal0, nadhm0, nadm0, oaa0, psi0, scoa0, suc0] = yf
        
        # External IC
        # t0 = np.load(filename, allow_pickle=True).item().t[-1]
        # [adpm0, atpm0, psi0, nadhm0, cit0, isoc0, akg0, scoa0, suc0, fum0, mal0, oaa0, asp0, cam0] = np.load('test.npy', allow_pickle=True).item().y[:,-1]

elif ic == 'no':
        '''
        [adpc0, adpm0, akg0, asp0, atpc0, atpm0, cac0, cam0, cit0, fum0, isoc0, mal0, nadhm0, nadm0, oaa0, psi0, scoa0, suc0] = 0.01 * np.load('/media/vvoorslu/LaCie/20211216/HB/ip3_0.088916_accoa_2.00e-03_test.npy', allow_pickle=True).item().y[:,-1]
        t0 = 0.
        adpc0 = atot*0.95
        adpm0 = amtot*0.95
        nadhm0 = nadtot*0.95
        cit0 = ckint - isoc0 - akg0 - scoa0 - suc0 - fum0 - mal0 - oaa0
        if cit0 + isoc0 + akg0 + scoa0 + suc0 + fum0 + mal0 + oaa0 > 1. :
                print('cit=%.6f' % (cit0))
                mal0 = mal0/2.
                cit0 = ckint - isoc0 - akg0 - scoa0 - suc0 - fum0 - mal0 - oaa0
        
        atpc0 = atot - adpc0
        atpm0 = amtot - adpm0
        nadm0 = nadtot - nadhm0
        print("atot=%.6f amtot=%.6f nadtot=%.6f ckint=%.6f\n" % (atpc0+adpc0, adpm0+atpm0, nadm0+nadhm0, cit0 + isoc0 + akg0 + scoa0 + suc0 + fum0 + mal0 + oaa0))
        '''
        t0 = 0.
        adpc0 = atot*0.20               # mM
        adpm0 = amtot*0.50              # mM
        akg0 = ckint*0.01               # mM
        asp0 = 0.                       # mM
        atpc0 = atot - adpc0            # mM
        atpm0 = amtot - adpm0           # mM
        cac0 = 0.20                     # uM
        cam0 = 0.10                     # uM
        fum0 = ckint*0.01               # mM
        isoc0 = ckint*0.01              # mM
        mal0 = ckint*0.01               # mM
        nadhm0 = 0.125*nadtot           # mM
        nadm0 = nadtot - nadhm0         # mM
        oaa0 = ckint*1.e-3              # mM
        #psi0 = 160.                     # mV
        scoa0 = ckint*0.01              # mM
        suc0 = ckint*0.01               # mM
        cit0 = ckint - isoc0 - akg0 - scoa0 - suc0 - fum0 - mal0 - oaa0 # mM
        
else:
        print("Error")


		
# --- RUN SIMULATION FOR A RANGE OF MUTATION SCALES --- #
for mutation_scale in [round(x * 0.1, 2) for x in range(5, 10, 2)]:    # redefine mutation-dependent parameters
    psi0 = 160. * mutation_scale
    vuni = 0.300 * mutation_scale
    vncx = 2.e-3 * (2. - mutation_scale)
    rhores = 1.00 * mutation_scale
    rhof1 = 0.23 * mutation_scale

    # reset initial conditions
    y0 = [atot*0.20, amtot*0.50, ckint*0.01, 0.0, atot*0.80, amtot*0.50,
          0.20, 0.10, ckint*0.91, ckint*0.01, ckint*0.01, ckint*0.01,
          0.125*nadtot, nadtot - 0.125*nadtot, ckint*1.e-3, psi0,
          ckint*0.01, ckint*0.01, CaB0]
    t0 = 0.0
    dt = 5.e-5
    total = 12000.00
    tdisc = 0.00
    NT = 1 + int(round(total / dt))
    Ndisc = int(round(tdisc / dt))
    dtprint = 1.e-2
    NTprint = 1 + int(round((total - tdisc) / dtprint))
    DNprint = int(round((NT - Ndisc - 1) / (NTprint - 1)))

    j = 0
    tnew, ynew = t0, y0
    sqrtdt = np.sqrt(dt)
    sqrtdtoN = sqrtdt / N

    filename = f'ip3_{ip3:.6f}_accoa_{accoa:.2e}_N_{int(N):03d}_lambda_{lmda:.2f}_mut_{mutation_scale:.2f}.hdf5'
    if os.path.isfile(filename):
        os.remove(filename)

    with h5py.File(filename, "a", libver='latest') as output:
        dsett = output.create_dataset('t', ((NTprint,)), chunks=True, dtype='float')
        dsety = output.create_dataset('y', ((19, NTprint)), chunks=True, dtype='float')
        dsetj = output.create_dataset('flux', ((19, NTprint)), chunks=True, dtype='float')
        dsetG = output.create_dataset('DG', ((19, NTprint)), chunks=True, dtype='float')
        dsetS = output.create_dataset('EPR', ((19, NTprint)), chunks=True, dtype='float')
        dsetW = output.create_dataset('ncW', ((3, NTprint)), chunks=True, dtype='float')
        output.swmr_mode = True

        flx = np.zeros((1, 19))
        dG = np.zeros((1, 19))

        for i in range(NT):
            told, yold = tnew, ynew
            if i >= Ndisc and (i - Ndisc) % DNprint == 0:
                dsett[j] = told
                dsety[:, j] = yold
                flx, dG = fluxes(yold), deltaG(yold)
                dsetj[:, j] = flx
                dsetG[:, j] = dG
                flx = np.transpose(flx)
                dG = np.transpose(dG)
                dsetS[:, j] = -flx * dG
                dsetW[0, j] = (dG[1]+dG[4]-3.*dG[6]) * (-0.5*flx[14] - flx[7]) / delta
                dsetW[1, j] = (dG[1] + (10.*dG[4] + 3.*dG[12] + dG[2] + dG[0]+ dG[8] + dG[9] + dG[15] + dG[13] + dG[5] + dG[10]) / 11. - dG[1]+dG[4]-3.*dG[6]) * (-0.5*flx[14] - flx[7]) / delta
                dsetW[2, j] = (- dG[4] + 3.*dG[12] + dG[2] + dG[0] + dG[8] + dG[9] + dG[15] + dG[13] + dG[5] + dG[10]) / 33. * (3.*flx[4] - 10.*flx[12])
                j += 1
                for dset in [dsett, dsety, dsetj, dsetG, dsetS, dsetW]:
                    dset.flush()

            tnew = told + dt
            ynew = yold + mmk(dt, yold, lmda * sqrtdtoN * np.random.normal(0., 1., 1))[:, 0]
