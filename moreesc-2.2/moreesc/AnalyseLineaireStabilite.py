#!/usr/bin/python
# -*- coding: utf-8 -*-
#Copyright (C) 2009 Fabricio Silva

"""
:mod:`AnalyseLineaireStabilite` -- Linear Stability Analysis
============================================================
.. moduleauthor:: Fabricio Silva <silva@lma.cnrs-mrs.fr.invalid>

This module is intended for the computation of the oscillation threshold
(the instability threshold of the static regime in fact) [Silva:2008]_.
Two ways are provided to calculate the eigenfrequencies

* a first one implemented in :func:`AnalyseLineaireStabilite.analyse_stab_poly` 
  makes extensive use of the modal decomposition of the input impedance of the
  resonator. But it is considered unsafe due to the troubles of numerical
  representation of high-order polynomials. Their accuracy is only guaranteed
  for a small number of acoustic modes in the resonator (:math:`N<=5`).

* a second one, provided in :func:`AnalyseLineaireStabilite.analyse_stab_jac`,
  computes the eigenvalues of the jacobian matrix of the system.
"""

import warnings
import numpy as np
import numpy.lib.polynomial as pol
from . import ModesGraph
import Solveur_dim as solv
import matplotlib.pyplot as plt
import scipy.linalg as la
debug = False

__all__ = ['analyse_stab', 'recherche_seuil','trace_diagramme_stabilite','sortList']

# La routine ROOTS connait des problèmes de coefficients trop grands.
# On fait un changement de variable X = s*1e-4.
# Tous les coefficients des polynomes sont affectés
coef = 10.**(-5.4)
coef = 10.**(-0)

fctt = lambda x: ((np.abs(x)).min(), (np.abs(x)).max())


def analyse_stab_poly(Config, GrilleGamma=None):
    """
    Evaluates the eigenfrequencies of the coupled system for a grid of values of 
    the (dimensionless) mouth pressure :math:`\gamma`. The computation involves
    the expressions of the polynomial numerator and denominator of the input 
    impedance, which are considered unsafe when considering more than 10 modes.
    
    Parameters
    ----------
    Config : :class:`Solveur_dim.SimulationTemporelle`
      The configuration to investigate
    GrilleGamma : array_like, optional.
      A set of mouth pressure values on which stability is evaluated. Default is a regular mesh of 100 values between 0 and 1.
    
    Returns
    -------
    TableSol : 2d-array
      An array of the complex eigenfrequencies. The column :math:`n` of ``TableSol`` lists the eigenfrequencies computed for the :math:`n` value of :math:`\gamma` listed in ``GrilleGamma``.
    GrilleGamma : arrays and the values of :math:`\gamma`.
    
    """
    if hasattr(Config, 'solveur')==False:
        raise IOError("Config doit être une instance de la classe "\
            +"solv.SimulationTemporelle.")
    if GrilleGamma==None:
        eps = 1e-2
        GrilleGamma = np.linspace(eps, 1-eps, 100)
        
    Anche = solv.meca.ReedDynamics(Config.wr, Config.qr, Config.k, Config.W)
    
    Tuyau = solv.acous.ResonateurModesComplexes(poles=Config.sn, \
        residus=Config.Cn, dimensionne=True, Zc=Config.Zc)
    ZeNum, ZeDen = Tuyau._evalNumDenCoef(coef=coef)
    if not (np.all(np.isfinite(ZeNum)) and np.all(np.isfinite(ZeDen)) ):
        raise ValueError("Le calcul polynomial de Ze a des problèmes d'Overflow. \n"\
            + "Prendre moins de modes est une solution")
    
    if Config.profil_ouverture.pts.shape[1]>1:
        warnings.warn(u"Profil d'ouverture non constant : le maximum est retenu.")
        h0 = Config.profil_ouverture.pts[1,:].max()
    else:
        h0 = Config.profil_ouverture.pts[1,0]
    zetaZc = Anche.W*np.sqrt(2.*h0/(solv.acous.rho*Anche.Kr))
    
    N = Config.nbinc
    TableSol = np.empty((N, len(GrilleGamma)), dtype=complex)*np.nan
    
    for idx, gamma in enumerate(GrilleGamma[::-1]):
        try:
            YaNum = (Anche.num*2*gamma-Anche.den*(1-gamma))*zetaZc
        except:
            print(Anche.num)
            print(gamma)
            print(Anche.den)
            print(zetaZc)
            raise ValueError
        YaDen = Anche.den*2*np.sqrt(gamma)
        YaNum = pol.poly1d(YaNum.c*(coef**(np.arange(len(YaNum.c)))))
        YaDen = pol.poly1d(YaDen.c*(coef**(np.arange(len(YaDen.c)))))
        PolyNum = ZeNum*YaNum*coef**(-ZeNum.order-YaNum.order)
        PolyDen = ZeDen*YaDen*coef**(-ZeDen.order-YaDen.order)            
        print((PolyNum-PolyDen).c)
        try:
            sol = pol.roots(PolyNum-PolyDen)
        except np.linalg.LinAlgError:
            print(fctt(ZeNum), fctt(ZeDen))
            #print(fctt(YaNum), fctt(YaDen))
            raise np.linalg.LinAlgError("Le calcul des racines de l'équation "\
                +"caractéristique a des problèmes d'Overflow.\n" \
                +"Prendre moins de modes est une solution.")
        if idx==0:
            TableSol[:, -idx-1] = sortList(sol, None)
        else:
            TableSol[:, -idx-1] = sortList(sol, TableSol[:, -idx])
        
    TableSol = TableSol/coef
    GrilleGamma = GrilleGamma
    return TableSol, GrilleGamma

    
def analyse_stab_jac(Config, GrilleGamma=None, h0=None):
    """
    Evaluates the eigenfrequencies of the coupled system for a grid of values of 
    the (dimensionless) mouth pressure :math:`\gamma`. It uses the jacobian
    of the system provided with the system definition.
    
    Parameters
    ----------
    Config : :class:`Solveur_dim.SimulationTemporelle`
      The configuration to investigate
    GrilleGamma : array_like, optional.
      A set of mouth pressure values on which stability is evaluated. Default is a regular mesh of 100 values between 0 and 1.
    
    Returns
    -------
    TableSol : 2d-array
      An array of the complex eigenfrequencies. The column :math:`n` of ``TableSol`` lists the eigenfrequencies computed for the :math:`n` value of :math:`\gamma` listed in ``GrilleGamma``.
    GrilleGamma : arrays and the values of :math:`\gamma`.
    
    """
    if hasattr(Config, 'solveur')==False:
        raise IOError("Config doit être une instance de la classe "\
            +"solv.SimulationTemporelle.")
    if GrilleGamma==None:
        eps = 1e-2
        GrilleGamma = np.linspace(eps, 1-eps, 100)
    
    if h0==None:
        warnings.warn(u"h0 non fourni : la valeur minimale du profil d'ouverture est retenue.")
        h0 = Config.profil_ouverture.pts[1,:].min()
    else:
        h0 = float(h0)
        
    N = len(Config.sn)        
    TableSol = np.empty((2*N+2, len(GrilleGamma)), dtype=complex)*np.nan
    
    for idx, gamma in enumerate(GrilleGamma[::-1]):
        PM = Config.k*h0
        Pm = gamma*PM
        try:
            peq = Config._regime_statique(Pm=Pm, h0=h0)
        except ValueError:
            print("%.3f : pb regime statique" % gamma)
            TableSol[:, -idx-1] = np.nan
        else:
            heq = h0+(peq-Pm)/Config.k
            ueq = peq/Config.Z0.real
            tmp = -Config.Cn*ueq/Config.sn
            X0 = np.zeros(2*N+2, dtype=float)
            X0[0:-2:2] = tmp.real
            X0[1:-2:2] = tmp.imag
            X0[-2] = heq
            Jac = solv.systeme.jacobien(X0,0., Config.sn, Config.Cn, \
                solv.Profil().controle(instants=0., valeurs=Pm).pts, \
                solv.Profil().controle(instants=0., valeurs=h0).pts)
                
            sol = la.eigvals(Jac)
            if idx==0 or np.all(np.isnan(TableSol[:, -idx])):
                TableSol[:, -idx-1] = sortList(sol, None)
            else:
                TableSol[:, -idx-1] = sortList(sol, TableSol[:, -idx])
        
    TableSol = TableSol/coef
    GrilleGamma = GrilleGamma
    return TableSol, GrilleGamma

analyse_stab = analyse_stab_jac
    
def recherche_seuil(Config, precision=1e-6):
    """
    Estimates the oscillation threshold of a given configuration, within a specified precision on :math:`\gamma`.
    
    
    Parameters
    ----------
    Config : :class:`Solveur_dim.SimulationTemporelle`
      The configuration to investigate
    precision : float, optional
      The precision in the oscillation threshold estimation.
    
    Returns
    -------
    Gamma_seuil : float
      The threshold dimensionless pressure
    eig_seuil: complex
      The eigenfrequency which destabilises the static regime. It must be nearly purely imaginary, with a tiny real part.
    """
    eps = 1e-4
    Gamma_min, Gamma_max = eps, 1-eps
    SolComplexe = 0.
    compteur = 0
    while Gamma_max-Gamma_min>precision:
        Gamma_tmp = np.linspace(Gamma_min, Gamma_max, 10)
        TableSol, Gamma = analyse_stab(Config, GrilleGamma=Gamma_tmp)
        degre_stab = TableSol.real.max(axis=0)
        idx = np.nonzero(np.logical_and(degre_stab[1:]>0, degre_stab[:-1]<0))[0]
        if (idx.size>0):
            Gamma_min, Gamma_max = Gamma_tmp[idx[0]], Gamma_tmp[idx[0]+1]
            if debug: print(compteur, idx, Gamma_min, Gamma_max)
        compteur += 1
        if compteur>-2*np.log10(precision):
            warnings.warn('Dichotomie problématique pour la recherche de seuil.')
            return np.nan, 1.j*np.nan
    Gamma_seuil = .5*(Gamma_min+Gamma_max)
    TableSol, Gamma = analyse_stab(Config, GrilleGamma=np.array([Gamma_seuil]))
    idx = np.argmax(TableSol[TableSol.imag>0].real)
    return Gamma_seuil, TableSol[TableSol.imag>0][idx]
    
            
def trace_diagramme_stabilite(GrilleGamma, Sol, strict=False):
    """
    Plots a stability diagram, i.e., graphs the evolution (with the mouth pressure) of the complex eigenfrequencies.
    
    Parameters
    ----------
    GrilleGamma : array-like
      A list of values of :math:`\gamma`.
    Sol : 2d array-like
      The eigenfrequencies for each :math:`\gamma`. It is the results of :func:`analyse_stab`.
    
    Returns
    -------
    A ``ModesGraph.ModesGraph`` instance.
    """
    Dmax = np.clip(1.1*Sol.real.max(), 0, 100.)
    Dmin = -5*Dmax
    Dmax = 1.1*Sol.real.max()
    Dmin = 1.1*Sol.real.min()
    Tmax = 1.1*Sol.imag.max()
    Tmin = 0*Sol.imag.min()
    a = ModesGraph.ModesGraph( Glim= (GrilleGamma.min(), GrilleGamma.max()), \
        Tlim=(Tmin,Tmax), Dlim=(Dmin, Dmax))
    iTest = len(GrilleGamma)-1
    Lls = ['--','-.', ':']
    for n in np.arange(Sol.shape[0]):
        tmp = Sol[n,:]
        mask1 = np.logical_and(tmp.imag<a.Tlim[-1], tmp.imag>a.Tlim[0])
        mask2 = np.logical_and(tmp.real<a.Dlim[-1], tmp.real>min(0., a.Dlim[0]))
        if strict:
            mask = np.logical_and(mask1, np.logical_and(mask2, tmp.imag>0))
        else:
            mask = np.logical_or(mask1, np.logical_and(mask2, tmp.imag>0))
        if np.any(mask):
            idx = np.mod(len(a.SubFreq.get_lines()), len(Lls))
            if tmp.imag[-1]<0:
                LT, LD = a.addtoModesGraph(GrilleGamma, tmp, \
                    ls='-.', label='', scaley=True)
            else:
                LT, LD = a.addtoModesGraph(GrilleGamma, tmp, \
                    ls='-', label='', scaley=True)

    a.SubDamp.set_ylabel(r'$Re(j\omega/\omega_r)$',a.Font)
    a.SubFreq.set_ylabel(r'$Im(j\omega/\omega_r)$',a.Font)
    for ax in [a.SubDamp, a.SubFreq]:
        ax.grid(False)
    return a

def sortList(sol, ref=None):
    u"Fonction de tri des racines de l'équation caractéristique."
    if not(isinstance(sol, np.ndarray)):
        raise TypeError(u"Sol must be an array.")
    solordonnee = np.empty_like(sol)*np.nan
    if not(isinstance(ref, np.ndarray)):
        # Premier appel, sans référence : on isole la solution la plus 
        # amortie (le plus souvent le canard), et on trie le reste par
        # fréquence croissante.
        idx = (np.isnan(sol)==False)
        solordonnee[:idx.sum()] = np.sort_complex(sol[idx])
        tmp = solordonnee[2:]
        tmp2 = tmp[np.isnan(tmp)==False]
        solordonnee[2:2+len(tmp2)] = tmp2[tmp2.imag.argsort()]
    else:
        if sol.shape != ref.shape:
            print(u"Size trouble in sorting solutions.")
            print(sol.shape, ref.shape)
            raise ValueError
        # Pour les appels suivants, on associe les solutions courantes
        # à celles de la référence (la solution précédente) par ordre
        # de proximité.
        solcp = sol.copy()
        for ind, vref in enumerate(ref):
            if np.isnan(vref)==False:
                idx = np.nanargmin(np.abs(vref-solcp))
                solordonnee[ind] = sol[idx]
                solcp[idx] = np.nan
                flag = False
            else:
                flag = True
                break
        if flag:
            idx = (np.isnan(solcp)==False)
            solordonnee[ind:ind+len(solcp[idx])] = np.sort_complex(solcp[idx])
            
#        MatSol = sol*np.ones(len(ref))[:,np.newaxis]
#        MatRef = ref[:, np.newaxis]*np.ones(len(sol))
#        D = abs(MatSol-MatRef)
#        for nn in np.arange(NSol):
#            if np.any(np.isnan(D)==False):
#                Idx = np.unravel_index(np.nanargmin(D), (NSol, NSol))
#                solordonnee[Idx[0]] = sol[Idx[1]]
#                D[:,Idx[1]] = np.nan
#                D[Idx[0],:] = np.nan
    return solordonnee



if __name__=='__main__':
    D = solv.meca.ReedDynamics(wr=2*np.pi*1000., qr=0.3, kr=2e7, W=1.5e-2)
    Ze = solv.acous.ResonateurAnalytique(L=.16, r=7e-3, \
        rayonne=False, nbmodes=10, D=D)
    
    PBouche = solv.Profil().standard(tfin=3.0, val1=1.e1, val2=1.e3)
    Hauteur = solv.Profil().controle(instants=(3.0,), valeurs=(3e-4,), pentes=(0,))
    
    Simu = solv.SimulationTemporelle(D=D, Z=Ze, pbouche=PBouche, h0=Hauteur, Fe=44100.)
    Simu.grandeurs_controle()
    if False: # Teste Diagramme de stabilite
        TableSol, GrilleGamma = analyse_stab(Simu)
        Fig = trace_diagramme_stabilite(GrilleGamma, TableSol/D.wr)
    if True:
        gamma = recherche_seuil(Simu, precision=1e-6)
        TableSol, GrilleGamma = analyse_stab(Simu)
        Fig = trace_diagramme_stabilite(GrilleGamma, TableSol/D.wr)
    plt.show()
