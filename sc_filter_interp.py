import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import lfilter


CHIP_DFLT_FC = np.array([29.51388889,   49.18981481,   80.49242424,   118.05555556,   160.98484848,
    196.75925926,  265.625,       321.96969697,  379.46428571,  429.29292929,
    531.25,        643.93939394,  708.33333333,  787.03703704,  965.90909091,
    1062.5,        1287.87878788, 1416.66666667, 1574.07407407, 1770.83333333,
    2125.,         2361.11111111, 2656.25,       3035.71428571, 3541.66666667,
    3863.63636364, 4250.,         4722.22222222, 5312.5,        6071.42857143,
    7083.33333333, 8500.])

CHIP_DFLT_K = np.array([2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8])

CHIP_DFLT_FBW = 3*CHIP_DFLT_FC/np.pi*4*CHIP_DFLT_K

CHIP_DFLT_FBB = 15e3*4.2e-12/(2*np.pi*8*4.2e-12)

class SCFilter:

    # TODO: make over_samp a vector so different over_sampling values can be used depending on fc.

    def __init__(self, fs, n_input_samps=None, fc=CHIP_DFLT_FC, fbw=CHIP_DFLT_FBW, k=None, over_samp=10):
        """Waken-AI switched-capacitor analog front-end model using interpolation for sampling.

        Args:
            fs (float): Sample rate of input signals to be filtered.
            n_input_samps (int): Length of input signal vectors. Used to pre-calculate certian quantities for speed. Defaults
                to None. Even if set, you are not required to always use the same n_input_samps. See Note below.
            fc (ndarray, optional): Array of bandpass center frequencies (Hz). One for each filter channel. Use CHIP_DFLT_FC
            for the default ERB spacing.
            fbw (ndarray, optional): Filter bandwidths (Hz). One for each filter channel. Either fbw or k must be specified as
                as they each uniquely determine the bandwidth of the filter channels. Use CHIP_DFLT_FBW for the default filter
                bandwidths.
            k (ndarray, optional): Hardware parameter that sets the number of unit caps using in each N-path filter channel.
                This affects the bandwdith. Either k or fbw must be specified for each filter channel. For the actual hardware,
                k must be an integer for each filter channel (between 1:8), but here we allow floats for experimentation. 
                Use CHIP_DFLT_K for the default filter bandwidths. See note below for further explaination. Defaults to None.
            over_samp (int, optional): Advanced simulation parameter. Controls simulation accuracy. Recommended to be at least
                10, but higher values result in slower simulation times. Leave at default if you are unsure. Defaults to 10.

        Note:
            fs and n_input_samps are typically constant and allow various filter coefficients to be calcualted before passing
            a signal in. If a signal of different length than n_input_samps is passed in to filter_signal(), then all the
            filter coefficients are re-calculated.  

        Note:
            k is a chip hardware parameter that is used to set the bandwidth (fbw) of each filter channel. In the current
            chip implementation, k is an integer between 1 to 8 for each channel. A k and an fc uniquely define an fbw, 
            so if k is specified, it overides the value for fbw (fbw can be filled with dummy values in this case). The
            relationship is: fbw = 3*fc/(np.pi*4*k)

        Note:
            fbb is an approximation of 

        Example usage:

            import matplotlib.pyplot as plt
            import sc_filter_interp as sfi

            # test signal
            fs = 16000.
            fsig = 310.
            t_vec = np.arange(0, 1, 1/fs)
            signal = np.cos(2*np.pi*(fc + 10.)*t_vec)

            scfilter = sfi.SCFilter(fs, len(t_vec), fc=sfok.CHIP_DFLT_FC, fbw=sfok.CHIP_DFLT_FBW)
            output, t_vec_out = scfilter(signal)

            plt.figure()
            [plt.plot(t, out) for t, out in zip(t_vec_out, output)]
            plt.show()
        
        """

        # fixed circuit parameters
        Cu = 4.2e-12
        Csc = Cu
        Cbb = 8*Cu
        Rbb = 1./(15e3*Cu)

        

        # the bandwidth is controlled by fc and Csb = k*Cu
        # k = 3*fc/(np.pi*4*fbw)
        # self.k = k
        # self.fbw = 3*fc/(np.pi*4*k)

        if k is None:
            self.k = 3*fc/(np.pi*4*fbw)
        else:
            self.k = k

        ######## How k relates to circuit parameters:

        # Csb = k*Cu
        
        # self._alpha = Csb/Cbb
        # self._beta = Csb/(Csb + Cbb)
        # self._tau_bb = Csb*Cbb/(Csb + Cbb) * Rbb
        # self._alpha_s = (Csb + Csc)/Cbb
        # self._beta_s = (Csb + Csc)/(Csb + Csc + Cbb)
        # self._tau_bb_s = (Csc + Csb)*Cbb/(Csc + Csb + Cbb) * Rbb
        # self._eta = Csc/(Csc + Csb)

        # self._alpha = k*Cu/Cbb
        # self._beta = k*Cu/(k*Cu + Cbb)
        # self._tau_bb = k*Cu*Cbb/(k*Cu + Cbb)*Rbb
        # self._alpha_s = (k*Cu + Csc)/Cbb
        # self._beta_s = (k*Cu + Csc)/(k*Cu + Csc + Cbb)
        # self._tau_bb_s = (k*Cu + Csc)*Cbb/(k*Cu + Csc + Cbb)*Rbb
        # self._eta = Csc/(Csc + k*Cu)

        #########
        self._eta = lambda k : Csc/(Csc + k*Cu)
        
        self._h1 = lambda t, k : (k*Cu/(k*Cu + Cbb)) * (1 - np.exp(-t/(k*Cu*Cbb/(k*Cu + Cbb)*Rbb)))
        self._h2 = lambda t, k : (1 - (k*Cu/(k*Cu + Cbb))) * (1 + (k*Cu/Cbb)*np.exp(-t/(k*Cu*Cbb/(k*Cu + Cbb)*Rbb)))
        self._h3 = lambda t, k : (k*Cu/(k*Cu + Cbb)) * (1 + 1/(k*Cu/Cbb)*np.exp(-t/(k*Cu*Cbb/(k*Cu + Cbb)*Rbb)))
        self._h4 = lambda t, k : (1 - (k*Cu/(k*Cu + Cbb))) * (1 - np.exp(-t/(k*Cu*Cbb/(k*Cu + Cbb)*Rbb)))
        self._h5 = lambda t, k : ((k*Cu + Csc)/(k*Cu + Csc + Cbb)) * (1 - np.exp(-t/((k*Cu + Csc)*Cbb/(k*Cu + Csc + Cbb)*Rbb)))
        self._h6 = lambda t, k : (1 - ((k*Cu + Csc)/(k*Cu + Csc + Cbb))) * (1 + ((k*Cu + Csc)/Cbb)*np.exp(-t/((k*Cu + Csc)*Cbb/(k*Cu + Csc + Cbb)*Rbb)))
        self._h7 = lambda t, k : ((k*Cu + Csc)/(k*Cu + Csc + Cbb)) * (1 + 1/((k*Cu + Csc)/Cbb)*np.exp(-t/((k*Cu + Csc)*Cbb/(k*Cu + Csc + Cbb)*Rbb)))
        self._h8 = lambda t, k : (1 - ((k*Cu + Csc)/(k*Cu + Csc + Cbb))) * (1 - np.exp(-t/((k*Cu + Csc)*Cbb/(k*Cu + Csc + Cbb)*Rbb)))

        self._h5_1_eta = lambda t, k : (1.-(Csc/(Csc + k*Cu)))*self._h5(t, k)
        self._h5_eta = lambda t, k : (Csc/(Csc + k*Cu))*self._h5(t, k)
        self._h5_eta_m = lambda t, k : -(Csc/(Csc + k*Cu))*self._h5(t, k)

        # states that output filters belong to
        self._vo_states = [0, 1, 2, 3, 4, 5, 6, 7]
        self._c1_states = [1, 2, 5, 6]
        self._c4_states = [7, 0, 1, 3, 4]
        self._vi_states = [0, 3, 6]
        self._all_states = self._vo_states + self._c1_states + self._c4_states

        self._vo_inds = [0, 1, 2, 3, 4, 5, 6, 7]
        self._c1_inds = [8, 9, 10, 11]
        self._c4_inds = [12, 13, 14, 15, 16]

        # output state filters
        self._hfilt_funcs_vo = (self._h6, self._h6, self._h2, self._h6, self._h2, self._h2, self._h6, self._h2)
        self._hfilt_funcs_vo_len = np.array([1, 1, 2, 2, 1, 1, 2, 2])
        
        self._hfilt_funcs_c1 = (self._h5_1_eta, self._h1, self._h1, self._h5_1_eta)
        self._hfilt_funcs_c1_len = self._hfilt_funcs_vo_len[self._c1_states]
        
        self._hfilt_funcs_c4 = (self._h1, self._h5_1_eta, self._h5_eta, self._h5_1_eta, self._h1)
        self._hfilt_funcs_c4_len = self._hfilt_funcs_vo_len[self._c4_states]
        
        self._hfilt_funcs_vi = (self._h5_eta, self._h5_eta_m, self._h5_eta_m)
        self._hfilt_funcs_vi_len = self._hfilt_funcs_vo_len[self._vi_states]

        # relative delays for input signal. Each state has three delays.
        self._input_offsets = np.array([[1, 1, 1],
                                        [0, 1, 1],
                                        [0, 1, 1],
                                        [0, 1, 1],
                                        [0, 0, 1],
                                        [0, 0, 1],
                                        [0, 0, 1],
                                        [0, 0, 0]])
        self._input_offsets = self._input_offsets[self._all_states]

        # native sampling rate
        self.fs = fs

        # center frequencies
        self.fc = fc

        self._over_samp = over_samp
        # samples per Tc (1/fc) period
        self._samps_period = 12*self._over_samp

        # offsets
        self._tau_vec = np.array([2./12, 3./12, 4./12, 6./12, 8./12, 9./12, 10./12, 1.])
        self._tau_d = np.concatenate(([self._tau_vec[0]], np.diff(self._tau_vec)))

        self._filter_coeffs = [self._calc_filter_coeffs(fc, k) for fc, k in zip(self.fc, self.k)]

        if n_input_samps is None:
            self._n_input_samps = 0
        else:
            self._n_input_samps = n_input_samps
            # get templates depending on the length of the signal (n_input_samps)
            self._calc_lengths_templates()
        

    def __call__(self, signal, out_type='samples'):
        """Filter signal through SCFilter filter bank.

        Args:
            signal (ndarray): 1-D array signal input with sampling rate, fs as specified during 
                initialization. The signal need not be n_input_samps long, but this results in 
                some extra computation.
            out_type ({'raw', 'samples'}): If 'samples' returns the output of each channel as if sampled by
                an ADC in round-robin fasion (one channel at a time) using the default parameters of
                SCFilter.sample_sig_chans(sig_chans, f_chan=1000., t_start=0., num_samp_kind='max-same', interp_kind='linear', split=0.5)
                If 'raw' returns the up-sampled analog approximation for each channel dictated by the over_samp
                value set during initialization. Defaults to 'samples' which samples at 1 kHz and returns the
                same number of samples for each channel. 

        Returns:
            list (array-like): list where each element is an output signal corresponding to a filter channel
                in the order that the fc vector is specified during initialization - [channel][out_signal]
                Note that even if there is only a single filter channel, a list is returned with
                a single element. 
        """

        # check for correct out_type parameter
        out_types = {'raw', 'samples'}
        if out_type not in out_types:
            raise ValueError(f'out_type must be one of {out_types}')
        
        # check if length of signal matches what we expect. If not, update templates accordingly.
        if len(signal) != self._n_input_samps:
            self._n_input_samps = len(signal)
            self._calc_lengths_templates()

        sig_chans, t_vec_out = self._filter_signal(signal)

        if out_type == 'samples':
            sig_chans, t_vec_out = \
                self.sample_sig_chans(sig_chans, f_chan=1000., t_start=0., num_samp_kind='max-same', interp_kind='linear')

        return sig_chans, t_vec_out


    def _calc_lengths_templates(self):
        # time points of input waveform
        self._t_vec_in = np.linspace(0, (self._n_input_samps-1)/self.fs, self._n_input_samps)
        
        # total number of samples in input waveform at 1/fc rate
        # first samp is at end of first state (beginning of second state)
        self._num_samps_fc = np.floor(((self._n_input_samps-1)/self.fs*self.fc - self._tau_vec[self._vi_states[0]]) + 1).astype(int)

        # time vector for output waveform. At over_samp*fc sample rate
        # total number of output samples includes one state1 time beyond last Tc state
        n_output_samps = self._samps_period*self._num_samps_fc + self._over_samp*self._hfilt_funcs_vo_len[-1]
        self._t_vec_out = np.array([np.linspace(0., (n_out_samps-1)/(self._samps_period*fc), n_out_samps) for fc, n_out_samps in zip(self.fc, n_output_samps)])

        # indicies for each state in oversampled template vectors
        get_state_idx = lambda fc, n_samps_fc : [np.arange(0, self._over_samp*length, dtype=int) + \
            (np.arange(0, n_samps_fc, dtype=int)*self._samps_period)[..., np.newaxis] + pos \
                for pos, length in zip((self._tau_vec*self._samps_period).astype(int), self._hfilt_funcs_vo_len)]
        self._state_idx = [get_state_idx(fc, n_samps_fc) for fc, n_samps_fc in zip(self.fc, self._num_samps_fc)]

        self._out_filter_templates = [self._get_output_templates(fc, num_samps_fc, state_idx, k) \
            for fc, num_samps_fc, state_idx, k in zip(self.fc, self._num_samps_fc, self._state_idx, self.k)]


    def _get_state_output_filters(self, fc, k):
        # this also works if fc is a vector

        vo_filts = [filt(np.linspace(0, filt_len/12/fc, self._over_samp*filt_len, endpoint=False), k) \
            for filt, filt_len in zip(self._hfilt_funcs_vo, self._hfilt_funcs_vo_len)]
           
        vc1_filts = [filt(np.linspace(0, filt_len/12/fc, self._over_samp*filt_len, endpoint=False), k) \
            for filt, filt_len in zip(self._hfilt_funcs_c1, self._hfilt_funcs_c1_len)]
        
        vc4_filts = [filt(np.linspace(0, filt_len/12/fc, self._over_samp*filt_len, endpoint=False), k) \
            for filt, filt_len in zip(self._hfilt_funcs_c4, self._hfilt_funcs_c4_len)]
        
        vi_filts = [filt(np.linspace(0, filt_len/12/fc, self._over_samp*filt_len, endpoint=False), k) \
            for filt, filt_len in zip(self._hfilt_funcs_vi, self._hfilt_funcs_vi_len)]

        return vo_filts, vc1_filts, vc4_filts, vi_filts

    def _get_output_templates(self, fc, num_samps_fc, state_idx, k):
        # individual state filter templates
        vo_filts, vc1_filts, vc4_filts, vi_filts = self._get_state_output_filters(fc, k)

        # template includes one state 1 length beyond the end
        vo_template = np.zeros(self._samps_period*num_samps_fc + self._over_samp*self._hfilt_funcs_vo_len[-1])
        for filt_template, state in zip(vo_filts, self._vo_states):
            vo_template[state_idx[state]] = filt_template

        vc1_template = np.zeros(self._samps_period*num_samps_fc + self._over_samp*self._hfilt_funcs_vo_len[-1])
        for filt_template, state in zip(vc1_filts, self._c1_states):
            vc1_template[state_idx[state]] = filt_template
        
        vc4_template = np.zeros(self._samps_period*num_samps_fc + self._over_samp*self._hfilt_funcs_vo_len[-1])
        for filt_template, state in zip(vc4_filts, self._c4_states):
            vc4_template[state_idx[state]] = filt_template

        vi_template = np.zeros(self._samps_period*num_samps_fc + self._over_samp*self._hfilt_funcs_vo_len[-1])
        for filt_template, state in zip(vi_filts, self._vi_states):
            vi_template[state_idx[state]] = filt_template
        
        return vo_template, vc1_template, vc4_template, vi_template


    def _calc_state_mats(self, fc, k):
        
        tau_diffs = self._tau_d/fc
        tau1_d = tau_diffs[0]
        tau2_d = tau_diffs[1]
        tau3_d = tau_diffs[2]
        tau4_d = tau_diffs[3]
        tau5_d = tau_diffs[4]
        tau6_d = tau_diffs[5]
        tau7_d = tau_diffs[6]
        tau8_d = tau_diffs[7]

        eta = self._eta(k)
        h1 = self._h1
        h2 = self._h2
        h3 = self._h3
        h4 = self._h4
        h5 = self._h5
        h6 = self._h6
        h7 = self._h7
        h8 = self._h8

        A_tau1 = np.array([ [h2(tau1_d, k),    0,      h1(tau1_d, k),         0,      0,      0],
                            [0,             1,      0,                  0,      0,      0],
                            [h4(tau1_d, k),    0,      h3(tau1_d, k),         0,      0,      0],
                            [0,             0,      0,                  1,      0,      0],
                            [0,             0,      0,                  0,      1,      0],
                            [0,             0,      0,                  0,      0,      1] ])

        A_tau2 = np.array([ [h6(tau2_d, k),    0,      (1-eta)*h5(tau2_d, k),     eta*h5(tau2_d, k),     0,      0],
                            [0,             1,      0,                      0,                  0,      0],
                            [h8(tau2_d, k),    0,      (1-eta)*h7(tau2_d, k),     eta*h7(tau2_d, k),     0,      0],
                            [0,             0,      0,                      1,                  0,      0],
                            [0,             0,      0,                      0,                  1,      0],
                            [0,             0,      0,                      0,                  0,      1] ])

        A_tau3 = np.array([ [h6(tau3_d, k),    (1-eta)*h5(tau3_d, k),     eta*h5(tau3_d, k),         0,      0,      0],
                            [h8(tau3_d, k),    (1-eta)*h7(tau3_d, k),     eta*h7(tau3_d, k),         0,      0,      0],
                            [0,             0,                      1,                      0,      0,      0],
                            [0,             0,                      0,                      1,      0,      0],
                            [0,             0,                      0,                      0,      1,      0],
                            [0,             0,                      0,                      0,      0,      1] ])

        A_tau4 = np.array([ [h2(tau4_d, k),    h1(tau4_d, k),     0,      0,      0,      0],
                            [h4(tau4_d, k),    h3(tau4_d, k),     0,      0,      0,      0],
                            [0,             0,              1,      0,      0,      0],
                            [0,             0,              0,      1,      0,      0],
                            [0,             0,              0,      0,      1,      0],
                            [0,             0,              0,      0,      0,      1] ])

        A_tau5 = np.array([ [h6(tau5_d, k),    0,      (1-eta)*h5(tau5_d, k),     0,      -eta*h5(tau5_d, k),    0],
                            [0,             1,      0,                      0,      0,                  0],
                            [h8(tau5_d, k),    0,      (1-eta)*h7(tau5_d, k),     0,      -eta*h7(tau5_d, k),    0],
                            [0,             0,      0,                      1,      0,                  0],
                            [0,             0,      0,                      0,      1,                  0],
                            [0,             0,      0,                      0,      0,                  1] ])

        A_tau6 = np.array([ [h2(tau6_d, k),    0,      h1(tau6_d, k),     0,      0,      0],
                            [0,             1,      0,              0,      0,      0],
                            [h4(tau6_d, k),    0,      h3(tau6_d, k),     0,      0,      0],
                            [0,             0,              0,      1,      0,      0],
                            [0,             0,              0,      0,      1,      0],
                            [0,             0,              0,      0,      0,      1] ])

        A_tau7 = np.array([ [h2(tau7_d, k),    h1(tau7_d, k),     0,      0,      0,      0],
                            [h4(tau7_d, k),    h3(tau7_d, k),     0,      0,      0,      0],
                            [0,             0,              1,      0,      0,      0],
                            [0,             0,              0,      1,      0,      0],
                            [0,             0,              0,      0,      1,      0],
                            [0,             0,              0,      0,      0,      1] ])

        A_tau8 = np.array([ [h6(tau8_d, k),    (1-eta)*h5(tau8_d, k),     0,      0,      0,      -eta*h5(tau8_d, k)],
                            [h8(tau8_d, k),    (1-eta)*h7(tau8_d, k),     0,      0,      0,      -eta*h7(tau8_d, k)],
                            [0,             0,                      1,      0,      0,      0],
                            [0,             0,                      0,      1,      0,      0],
                            [0,             0,                      0,      0,      1,      0],
                            [0,             0,                      0,      0,      0,      1] ])

        # calculate the state matricies by doing a cyclic matrix multiplication
        S1 = A_tau1 @ A_tau8 @ A_tau7 @ A_tau6 @ A_tau5 @ A_tau4 @ A_tau3 @ A_tau2
        S2 = A_tau2 @ A_tau1 @ A_tau8 @ A_tau7 @ A_tau6 @ A_tau5 @ A_tau4 @ A_tau3
        S3 = A_tau3 @ A_tau2 @ A_tau1 @ A_tau8 @ A_tau7 @ A_tau6 @ A_tau5 @ A_tau4
        S4 = A_tau4 @ A_tau3 @ A_tau2 @ A_tau1 @ A_tau8 @ A_tau7 @ A_tau6 @ A_tau5
        S5 = A_tau5 @ A_tau4 @ A_tau3 @ A_tau2 @ A_tau1 @ A_tau8 @ A_tau7 @ A_tau6
        S6 = A_tau6 @ A_tau5 @ A_tau4 @ A_tau3 @ A_tau2 @ A_tau1 @ A_tau8 @ A_tau7
        S7 = A_tau7 @ A_tau6 @ A_tau5 @ A_tau4 @ A_tau3 @ A_tau2 @ A_tau1 @ A_tau8
        S8 = A_tau8 @ A_tau7 @ A_tau6 @ A_tau5 @ A_tau4 @ A_tau3 @ A_tau2 @ A_tau1

        return np.stack([S1, S2, S3, S4, S5, S6, S7, S8], axis=0)
    

    def _state_mat_coeffs(self, fc, k):
        
        S_mats = self._calc_state_mats(fc, k)
        # Laplace domain coefficients
        B0 = S_mats[:, 0, 0]
        B1 = S_mats[:, 0, 1]
        B2 = S_mats[:, 0, 2]
        A0 = S_mats[:, 1, 0]
        A1 = S_mats[:, 1, 1]
        A2 = S_mats[:, 1, 2]
        G0 = S_mats[:, 2, 0]
        G1 = S_mats[:, 2, 1]
        G2 = S_mats[:, 2, 2]
        alpha = S_mats[:, 0, 3:]
        c = S_mats[:, 1, 3:]
        p = S_mats[:, 2, 3:]

        return B0, B1, B2, A0, A1, A2, G0, G1, G2, alpha, c, p


    def _calc_state_laplace_coeffs(self, B0, B1, B2, A0, A1, A2, G0, G1, G2, alpha, c, p, state_inds=None):

        if state_inds is None:
            state_inds = [0, 1, 2, 3, 4, 5, 6, 7]
        
        # numerator coefficients 
        # note: these terms are vectors - one for each input offset       
        M0 = alpha
        M1 = (-G2 - A1)[..., np.newaxis]*alpha + B1[..., np.newaxis]*c + B2[..., np.newaxis]*p
        M2 = (A1*G2 - G1*A2)[..., np.newaxis]*alpha + (G1*B2 - G2*B1)[..., np.newaxis]*c + (A2*B1 - A1*B2)[..., np.newaxis]*p

        # denominator
        # cubic polynomial: a*x^3 + b*x^2 + c*x + d
        d = 1.*np.ones_like(A1)
        c = -(A1 + B0 + G2)
        b = A1*B0 + A1*G2 + B0*G2 - A0*B1 - A2*G1 - B2*G0
        a = A0*B1*G2 + A1*B2*G0 + A2*B0*G1 - A0*B2*G1 - A1*B0*G2 - A2*B1*G0

        return M0[state_inds], M1[state_inds], M2[state_inds], a[state_inds], b[state_inds], c[state_inds], d[state_inds]
    
    
    def _calc_filter_coeffs(self, fc, k):

        # coefficients from state matricies for all states
        B0, B1, B2, A0, A1, A2, G0, G1, G2, alpha, c, p = self._state_mat_coeffs(fc, k)
        
        # take laplace transform of state variables
        M0_vo, M1_vo, M2_vo, a_vo, b_vo, c_vo, d_vo = self._calc_state_laplace_coeffs(
            B0, B1, B2, A0, A1, A2, G0, G1, G2, alpha, c, p)
        
        M0_vc1, M1_vc1, M2_vc1, a_vc1, b_vc1, c_vc1, d_vc1 = self._calc_state_laplace_coeffs(
            A1, A0, A2, B1, B0, B2, G1, G0, G2, c, alpha, p, self._c1_states)
        
        M0_vc4, M1_vc4, M2_vc4, a_vc4, b_vc4, c_vc4, d_vc4 = self._calc_state_laplace_coeffs(
            G2, G1, G0, A2, A1, A0, B2, B1, B0, p, c, alpha, self._c4_states)

        M0 = np.concatenate([M0_vo, M0_vc1, M0_vc4])
        M1 = np.concatenate([M1_vo, M1_vc1, M1_vc4])
        M2 = np.concatenate([M2_vo, M2_vc1, M2_vc4])
        a = np.concatenate([a_vo, a_vc1, a_vc4])
        b = np.concatenate([b_vo, b_vc1, b_vc4])
        c = np.concatenate([c_vo, c_vc1, c_vc4])
        d = np.concatenate([d_vo, d_vc1, d_vc4])

        return M0, M1, M2, a, b, c, d


    def _iir_filt(self, M0, M1, M2, a, b, c, d, signal_channels):
        signal_out = lfilter(np.array([M0[0], M1[0], M2[0]]), np.array([d, c, b, a]), signal_channels[0]) + \
                        lfilter(np.array([M0[1], M1[1], M2[1]]), np.array([d, c, b, a]), signal_channels[1]) + \
                            lfilter(np.array([M0[2], M1[2], M2[2]]), np.array([d, c, b, a]), signal_channels[2])
        return signal_out


    def _shift(self, x, n):
        if n != 0:
            return np.r_[np.zeros(n), x[n-1:-1]]
        else:
            return x


    def _scale_add_templates(self, sampled_outputs, sampled_inputs, out_filter_templates, state_idx, num_samps_fc):
        # only create output vector once, and continually add to it
        output = np.zeros(self._samps_period*num_samps_fc + self._over_samp*self._hfilt_funcs_vo_len[-1])
        for samp, state in zip(sampled_outputs[self._vo_inds], self._vo_states):
            output[state_idx[state]] += samp[..., np.newaxis] * out_filter_templates[0][state_idx[state]]
        for samp, state in zip(sampled_outputs[self._c1_inds], self._c1_states):
            output[state_idx[state]] += samp[..., np.newaxis] * out_filter_templates[1][state_idx[state]]
        for samp, state in zip(sampled_outputs[self._c4_inds], self._c4_states):
            output[state_idx[state]] += samp[..., np.newaxis] * out_filter_templates[2][state_idx[state]]
        for samp, state in zip(sampled_inputs, self._vi_states):
            output[state_idx[state]] += samp[..., np.newaxis] * out_filter_templates[3][state_idx[state]]
        
        return output


    def _sample_signal(self, t_vec, signal, fc, offset=None):

        if np.isscalar(fc):
            fc = np.array([fc])
        if offset is None:
            offset = np.zeros_like(fc)
        elif np.isscalar(offset):
            offset = np.array([offset])

        Ts = t_vec[1] - t_vec[0]
        # create vectors for interpolation points for each fc/offset pair
        n_samps = np.floor(((len(t_vec)-1)*Ts - offset)*fc + 1).astype(int)
        t_samp = [np.linspace(t_off, t_off+n_samp/smpl_rate, n_samp, endpoint=False) for smpl_rate, t_off, n_samp in zip(fc, offset, n_samps)]

        # interpolation function
        sig_samples = interp1d(t_vec, signal, kind='cubic', assume_sorted=True)

        # # when interpolating, assemple t_samp into one long vector for speed
        # samps = sig_samples(np.concatenate(t_samp))
        # # array of start indicies to index vectorized samples
        # start_idx = [len(t) for t in t_samp]
        # ##### ERROR HERE!!! NEED TO CUMSUM ######
        # start_idx = [0] + start_idx[0:-1]
        # # reassemble into list of arrays
        # samps = [samps[idx:idx + len(t)] for idx, t in zip(start_idx, t_samp)]

        # its actually only a (negligible) touch slower to do a list comprehension
        samps = [sig_samples(t) for t in t_samp]

        # if len(samps) == 1:
        #     t_samp = t_samp[0]
        #     samps = samps[0]

        return t_samp, samps


    def _filter_signal(self, signal):

        # sample signal at beginning of state 2, 5, 7
        _, sig_samples = self._sample_signal(self._t_vec_in, signal, self.fc * 3, offset=self._tau_vec[0]/self.fc)
        
        # assemble samples as three different "channels" zero-padding if necessary
        samps = []
        for samples, n_samps in zip(sig_samples, self._num_samps_fc):
            samps_1 = samples[0::3] if len(samples[0::3]) == n_samps else np.r_[samples[0::3], 0.]
            samps_2 = samples[1::3] if len(samples[1::3]) == n_samps else np.r_[samples[1::3], 0.]
            samps_3 = samples[2::3] if len(samples[2::3]) == n_samps else np.r_[samples[2::3], 0.]
            samps.append(np.vstack((samps_1, samps_2, samps_3)))

        # filtering samples
        state_out_samps = []
        for this_fc in range(len(self.fc)):
            # filter each input channel with its own state filter and add together. Do this for each state.
            sampled_outputs = np.zeros((len(self._all_states), self._num_samps_fc[this_fc]))
            # unpack filter coefficients
            M0, M1, M2, a, b, c, d = self._filter_coeffs[this_fc]
            for i in range(len(self._all_states)):
                # delay inputs as necessary
                sigs = np.zeros(samps[this_fc].shape)
                for j in range(samps[this_fc].shape[0]):
                    sigs[j] = self._shift(samps[this_fc][j], self._input_offsets[i, j])
                # filter signal to get state variable and add to according state
                sampled_outputs[i] = self._iir_filt(M0[i], M1[i], M2[i], a[i], b[i], c[i], d[i], sigs)
            state_out_samps.append(sampled_outputs)

        # scale the templates with the upsampled signals
        output = [self._scale_add_templates(sampled_outputs, sampled_inputs, out_filter_templates, state_idx, num_samps_fc) \
            for sampled_outputs, sampled_inputs, out_filter_templates, state_idx, num_samps_fc \
                in zip(state_out_samps, samps, self._out_filter_templates, self._state_idx, self._num_samps_fc)]

        return output, self._t_vec_out


    def calc_chan_energies(self, sig_chans):
        """Calculate the energies, defined as the mean-square value of each signal channel.

        Args:
            sig_chans (array_like): An array where each element is a ndarray representing samples from each filter channel.

        Returns:
            [array_like]: An array of the same length as sig_chans holding the mean square value of each channel.
        """
        # mean square value of each channel
        return np.array([np.mean(np.power(chan, 2)) for chan in sig_chans])

    def sample_sig_chans(self, sig_chans, f_chan=1000., t_start=0., num_samp_kind='max-same', interp_kind='linear', split=0.5):
        """Mimic ADC sampling of analog filter channels in round-robin (one channel at a time) sampling.

        Args:
            sig_chans (array_like): An array where each element is a ndarray representing samples from each filter channel.
            f_chan (float, optional): Sampling rate per channel in Hz. Because sampling is round-robin starting at sig_chans[0], 
                the effective sampling rate for the whole filter-bank is len(sig_chans)*f_chan. Defaults to 1000..
            t_start (float, optional): Starting point in waveform to begin sampling. Defaults to 0..
            num_samp_kind (str, optional): {'max-same', 'max'} If 'max-same', returns the maximum number of samples possible
                such that each channel has the same number of samples. If a channels is able to be sampled more than this number, 
                the last n samples are taken. If 'max', returns the maximum number of samples possible per channel. In this case, 
                its possible that channels may have differing number of samples due to the round-robin sampling. Defaults to 'max-same'.
            interp_kind (str, optional): {'linear', 'quadratic', 'cubic', 'mixed'}. Samples are interpolated based on the
                amplitude and time vectors for each channel. 'linear' is the fastest to compute, but has the the highest error.
                If 'mixed', then the first n number of channels are interpolated with cubic splines, and the remaining channels
                are sampled via linear interpolation. The proportion is dictated by the split parameter. The lower frequency channels
                are typically the more error prone and thus should use cubic interpolation due to their output sample rate being
                effectively lower. The higher frequency channels have a higher sampling rate and thus have less error despite using
                linear interpolation. This strikes a balance between computational accuracy and speed. This implicitely assumes
                the fc vector is sorted from lowest to highest frequency. Defaults to 'linear'.
            split (float, optional): Used only if interp_kind='mixed'. First np.round(split*len(sig_chans)) channels are sampled
                with cubic spline interpolation, and the remaining channels are sampled with linear interpolation.Defaults to 0.5.

        Raises:
            ValueError: Invalid value if num_samp_kind != {'max-same', 'max'}
            ValueError: Invalid value if interp_types != {'linear', 'quadratic', 'cubic', 'mixed'}

        Returns:
            [array_like]: Array of same length as sig_chans containing ndarrays with samples taken at the f_chan rate.
        """

        num_samps_types = {'max-same', 'max'}
        if num_samp_kind not in num_samps_types:
            raise ValueError(f'num_samp_kind must be one of {num_samps_types}')

        interp_types = {'linear', 'quadratic', 'cubic', 'mixed'}
        if interp_kind not in interp_types:
            raise ValueError(f'interp_kind must be one of {interp_types}')
        else:
           if interp_kind in {'linear', 'quadratic', 'cubic'}:
               kind = interp_kind 
        
        # number of sampling points for each channel
        num_chan_samps = [int(np.floor((t[-1]-t[0]-t_start)*f_chan)) for t in self._t_vec_out]
        
        # get time vector sampling points assuming round robin sampling at num_chan*f_chan rate
        t_chan_samps = [np.linspace(t_start + chan_ind/f_chan, n_samps/f_chan, n_samps) for chan_ind, n_samps in enumerate(num_chan_samps)]

        # remove sample points if necessary
        if num_samp_kind == 'max-same':
            max_samps = np.min(num_chan_samps)
            for ind in range(len(t_chan_samps)):
                t_chan_samps[ind] = t_chan_samps[ind][-max_samps:]

        chan_samps = []
        if interp_kind == 'mixed':
            num_cube = np.round(len(sig_chans)*split)
            for ind in range(int(num_cube)):
                sig_samples = interp1d(self._t_vec_out[ind], sig_chans[ind], kind='cubic')
                chan_samps.append(sig_samples(t_chan_samps[ind]))
            for ind in np.arange(num_cube, len(sig_chans), dtype=int):
                sig_samples = interp1d(self._t_vec_out[ind], sig_chans[ind], kind='linear')
                chan_samps.append(sig_samples(t_chan_samps[ind]))
        else:
            for t_chan, sig_chan, t_interp_pts in zip(self._t_vec_out, sig_chans, t_chan_samps):
                sig_samples = interp1d(t_chan, sig_chan, kind=kind, assume_sorted=True)
                chan_samps.append(sig_samples(t_interp_pts))

        return chan_samps, t_chan_samps