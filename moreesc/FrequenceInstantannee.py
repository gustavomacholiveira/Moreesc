#!/usr/bin/python
# -*- coding: utf-8 -*-

#Copyright (C) 2008 Fabricio Silva


import numpy as np
from scipy import signal
from pylab import figure, subplot, setp, show, plot

class FrequenceInstantannee:
    """
    A FrequenceInstantannee instance allows the evaluation and the
    representation of the instantaneous frequency of a sound object.
    """
    def __init__(self, data, Fe=44100.0):
        self.Fe = Fe
        if isinstance(data, str):
            try:
                # Chargement du fichier : wav?
                try:
                    import scikits.audiolab as au
                except ImportError:
                    raise ImportError("Module audiolab non disponible.")
                WavObj = au.sndfile(data, 'read')
                print(u"%s est un fichier audio." %data)
                self.Fe = WavObj.get_samplerate()
                self.signal = WavObj.read_frames(WavObj.get_nframes())
                if WavObj.get_channels()>1:
                    print("Fichier non mono : seule la première piste est traitée")
                    self.signal = self.signal[:,0]
                WavObj.close()
            except IOError:
                try:
                    import scipy.io as io
                    # Chargement du fichier : mat binaire?
                    WavObj = io.loadmat(data, None, 1)
                    print("%s est un fichier Mat binaire." %data)
                except IOError:
                    raise IOError("Ce n'est pas un .mat ni un .wav!")
                if WavObj.has_key('son'):
                    self.signal = WavObj['son']
                else:
                    raise IOError("Problème avec le fichier %s" %data)
                if WavObj.has_key('Fe'):
                    self.Fe = WavObj['Fe']
                    
        if isinstance(data, np.ndarray):
            self.signal = data
            if data.ndim>1 :
                print(u"Signal non mono : seule le premier canal est traité")
                self.signal = self.signal[:,0]
        self.temps = np.arange(len(self.signal))/(self.Fe*1.0)

    def eval_zero_crossing(self, Fest=170, p=2, Filtering=False):
        """ Estimates instantaneous frequency from zero-crossing detection
            and evaluation of the duration of p periods.
            Dominique Rodrigues, 2006 (Matlab)
            Fabricio Silva, 2007 (Python).
        """
        self.Fest = Fest
        if Filtering:
            # Filtrage du signal par un passe-bande résonant
            Q = 5.0
            w0n = 2*pi*Fest/self.Fe
            BFil = (w0n/(2*Q))*np.array([1,0,-1])
            AFil = np.array([1+w0n/(2*Q), w0n**2-2, 1-w0n/(2*Q)])
            self.signal = signal.lfilter(BFil, AFil, self.signal)
            # Réponse fréquentielle du IIR
            #w, H = signal.freqz(BFil, AFil, None, plot)
            ## Réponse impulsionnelle du IIR
            #X = np.zeros(2^14, dtype=float)
            #X[0] = 1
            #plot(X, 'b+')
            #plot(signal.lfilter(BFil/np.max(abs(H)), AFil, X), 'ro')
            ##Signal avant et après filtrage
            #Temps = np.arange(len(Signal), dtype=float)/Fe
            #Temps0 = np.arange(len(Signal0), dtype=float)/Fe
            #plot(Temps, Signal,'r', Temps0, Signal0, 'b')

        # Reperage des passages par zero
        self.signal[np.where(self.signal==0)] += 1e-10
        ChgFlag = np.not_equal(np.sign(self.signal[:-1]),np.sign(self.signal[1:]))
        Idx0 = []
        for k in np.where(ChgFlag)[0]:
            estk = k+self.signal[k]/(self.signal[k]-self.signal[k+1])
            Idx0.append(estk)
        Instants = self.temps[Idx0]

        # Estimation de la fréquence instantannée Frequence[i] à partir
        # des p périodes suivant Instants[i]
        self.insteval = (Instants[p:]+Instants[:-p])/2.0
        self.freqinst = p/(2.0*(Instants[p:]-Instants[:-p]))
        return self.insteval, self.freqinst

    def eval_analytical_signal(self, Fest=170):
        """ Estimates instantaneous frequency from analytical signal
            (of fundamental component of signal).
            Fabricio Silva, 2007 (Python) from Kronland ideas.
        """
        self.Fest = Fest
        N = len(self.signal)*1.0
        CoefBP = 1.25
        freq = np.fft.fftfreq(N, d=1./self.Fe)
        s_fft = np.fft.fft(self.signal)
        BP = np.logical_and(freq>self.Fest/CoefBP, freq<self.Fest*CoefBP)[0]
        s_fft[BP] *= 2*signal.signaltools.hanning(len(BP))
        BPn = np.logical_or(freq<self.Fest/CoefBP, freq>self.Fest*CoefBP)[0]
        s_fft[BPn] = 0

        s_a = np.fft.ifft(s_fft)
        self.freqinst = np.diff(np.unwrap(np.angle(s_a)))*self.Fe/(2*pi)
        self.insteval = self.temps[:len(self.freqinst)]
        return self.insteval, self.freqinst, s_a

    evaluate = eval_analytical_signal

    def plot(self, color='b'):
        import matplotlib.pyplot as plt
        self.Fig = plt.figure(None,(15/2.54, 7/2.54))
        self.Font = {'fontname' : 'Helvetica', 'fontsize' : 10}
        self.SubFreq = self.Fig.add_subplot(212)
        self.SubFreq.clear()
        self.SubFreq.grid('on')
        self.SubFreq.set_xlabel(r'$t$',self.Font)
        self.SubFreq.set_ylabel(r'$f_i (Hz)$',self.Font)
        self.SubFreq.plot(self.insteval, self.freqinst, color)
        self.SubFreq.set_ylim([0,2*self.Fest])

        self.SubSignal = self.Fig.subplot(211, sharex=self.SubFreq)
        self.SubSignal.clear()
        self.SubSignal.grid('on')
        self.SubSignal.set_ylabel(r'$x(t)$',self.Font)
        self.SubSignal.label_outer()
        self.SubSignal.plot(self.temps, self.signal, color)

    def addtoFiGraph(self,*args, **kwargs):
        kwargs.update({'scalex':False,'scaley':False})
        return self.SubFreq.plot(*args, **kwargs)
    def addtoSignalGraph(self,*args, **kwargs):
        kwargs.update({'scalex':False,'scaley':False})
        return self.SubSignal.plot(*args, **kwargs)

if __name__=='__main__':
    Obj = FrequenceInstantannee('test.wav')
    T_z, Fi_z = Obj.eval_zero_crossing(170.0, 2)
    T_a, Fi_a, s_a = Obj.eval_analytical_signal(170)
    Obj.plot()
    Obj.addtoSignalGraph(Obj.temps, s_a, 'r')
    Obj.addtoFiGraph(T_z, Fi_z, 'go')
    Obj.addtoSignalGraph(T_z, np.zeros_like(T_z), 'go')
    plt.show()
