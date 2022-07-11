#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=invalid-name

"""
Routines for performing the Lilliefors test for exponentiality.

The Lilliefors test is performed as a function of lower magnitude cutoff. The provided
p-value is eventually used to estimate a Mc-Lilliefors, which complies with the
exponential Gutenberg–Richter relation to obtain a meaningful b-value.

The accompanied Jupyter notebook demonstrates the use of the class below.

Associated publication:
    Herrmann, M. and W. Marzocchi (2020). "Inconsistencies and Lurking Pitfalls in the
        Magnitude–Frequency Distribution of High-Resolution Earthquake Catalogs".
        Seismological Research Letters 92(1). doi: 10.1785/0220200337

:copyright:
    2020 Marcus Herrmann, Università degli Studi di Napoli 'Federico II'
:license:
    European Union Public Licence (EUPL-1.2-or-later)
    (https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12)
"""

# ------------------------------------------------------------------------------
# Licensed under the EUPL, Version 1.2 or – as soon they will be approved by the
# European Commission – subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#    https://joinup.ec.europa.eu/collection/eupl/eupl-text-eupl-12
# Unless required by applicable law or agreed to in writing, software distributed
# under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
# See the Licence for the specific language governing permissions and limitations
# under the Licence.
# ------------------------------------------------------------------------------


# Python built-in modules (Standard library)
import sys
import logging
import math
from fractions import Fraction

# Helpers
from tqdm import tqdm
import pandas as pd

# Scientific
import numpy as np
import statsmodels.api as sm

# Plotting
import plotly.graph_objects as go
import plotly.colors as pco


class McLilliefors():
    """
    Class to perform the Lilliefors test for exponentiality (as function of Mc).
    """

    def __init__(self, mags, signif_lev=0.1, log_level=logging.INFO):
        """

        Parameters
        ----------
        mags : sequence of floats
            The magnitude vector of the catalog.
        signif_lev : float, optional
            The significance level, α, used for null hypothesis testing, by default 0.05.
        log_level : int, optional
            The default logging level, by default logging.INFO (20).
        """

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        self.estimates = {}

        self.update_mags(mags)

        if self.n_ev == 0:
            self.mbin_orig = np.nan
            return

        # Estimate original binning of the catalog
        self.mbin_orig = self._get_binning()

        self.logger.info("Detected magnitude binning: %s", self.mbin_orig)

        self.signif_lev = signif_lev

        self.testdistr_mcutoff = None

    def update_mags(self, mags):
        """
        Update the magnitude vector.

        Also recalculates `n_ev` property.
        """

        # If it's a pandas Series, extract the index
        # (can be used for comparing whether it's the same data)
        if isinstance(mags, pd.Series):
            self.mags_index = mags.index
            self.mags = mags.values
        else:
            self.mags_index = None
            self.mags = np.array(mags)

        self.n_ev = self.mags.size

        if self.n_ev == 0:
            self.logger.warning("Warning: No magnitudes present. "
                                "Will return NaN results.")
            self.minmag, self.maxmag = None, None
            return

        # (Re)calculate magitude range for proper bin generation
        self.minmag = np.nanmin(self.mags)
        self.maxmag = np.nanmax(self.mags)

    def _get_binning(self):
        """
        Determine binning of magnitudes (via Greatest Common Divisor).

        Note: doesn't account for an absolute shift in the binning
              (then the catalog is shit anyway)
              (e.g., [1.0, 1.4, 1.8, 3.0] will result in 0.2 instead of 0.4)
        """

        # -- Approach via greatest common divisor
        # (works also for very small catalogs, or with high-res binning)

        # Take a small random sample
        mags = np.random.choice(self.mags, 100)

        # Make ints using length of decimal place (this does NOT account for a shift)
        decis = max([str(x)[::-1].find('.') for x in mags])  # this is sensitive to float precision
        magm = (mags * 10**decis).astype(np.int)  # speedup w/ numpy

        # Make math.gcd() to work with lists / arrays; How? Iterative over it!
        gcd = magm[0]
        for c in magm[1:]:
            gcd = math.gcd(gcd, c)

        orig_bin = gcd / 10**decis

        # -- Alternative approach via mag differences
        # (this DOES account for the shift)
        # BUT it is more robust for shitty binnings (e.g., after magnitude conversions)

        if decis > 3:
            uniqmagsortdiff = np.diff(np.sort(np.unique(self.mags)))
            mindiff = np.nanmin(uniqmagsortdiff)  # with precision error
            orig_bin = Fraction(mindiff).limit_denominator()  # convert to fraction
            orig_bin = orig_bin.numerator / orig_bin.denominator

        return orig_bin

    def _get_bins(self, mbin=None):
        """
        Get the magnitude bins for a given binning `mbin`.

        If `mbin` is not specified, it will default to the one used for
        estimating Mc.
        """

        if mbin is None:
            mbin = self.mbin_orig

        bins = np.arange(np.floor(self.minmag / mbin) * mbin,
                         self.maxmag + 2 * mbin, mbin)  # give double bin margin
        binEdges = bins - mbin / 2   # to create correct bin edges
        bins = bins[:-1]  # cut margin away again

        return bins, binEdges

    def _correct_to_binning(self, val, mbin):
        """
        Correct a magnitude bin to a sensible value.

        It is of SUPER IMPORTANCE to appropriately round the Mc estimate to a multiple
        of the binning without precision errors (e.g. 1.0 instead of 1.000000000000001).
        These precision errors would otherwise affect the GR fit quite considerably,
        especially when the original magnitudes have half the binning used for GR-fitting.
        """

        if np.isnan(val):
            return val

        imbin = 1 / mbin  # to eventually create an integer
        if imbin.is_integer():
            # check if binning as a fraction like `1 / natural number`
            # Correct Mc to binning & AVOID PRECISION ERRORS
            val_bin = round(val * int(imbin)) / int(imbin)
        else:
            # do best if it's not a "natural fraction"
            # --> no guarantee to avoid precision errors
            val_bin = round(val / mbin) * mbin

        return val_bin

    def lilliefors_test(self, mags=None, log=True):
        """
        Perform a Lilliefors hypothesis test for exponentiality.

        Parameters
        ----------
        mags : 1-D array_like, optional
            The magnitude set. Defaults to the magnitude set of this FMD object.
        log : bool, optional
            Whether to output the test results to stdout, by default True.
            Only useful to switch off for batch-processing tests.

        Returns
        -------
        float
            p-value of the hypothesis test.
        """

        if mags is None:
            mags = self.mags

        if len(mags) < 3:
            return np.nan

        res = sm.stats.diagnostic.lilliefors(mags, dist='exp', pvalmethod='table')

        if log:
            self.logger.info("Lilliefors test :: test statistic: %.3f,  p-value: %.3f", *res)

        return res[1]

    def _prepare_mags_for_exponential(self, mags=None, Mc=None):
        """
        Prepare magnitudes for exponential distribution.

        Steps:
            - apply noise to each magnitude within bin size (i.e. make `mags` continuous)
            - shift magnitudes by the completeness magnitude
              (specifically, by the start, i.e. left border, of this magnitude bin)

        Parameters
        ----------
        mags : 1-D array_like, optional
            The magnitude set. Defaults to the magnitude set of this FMD object.
        Mc : float, optional
            The magnitude of completeness, or magnitude cutoff level.
            Defaults to the value calculated with .estimate_Mc(), otherwise returns None.

        Returns
        -------
        1-D np.ndarray
            The prepared magnitudes.
        """

        if mags is None:
            mags = self.mags

        if Mc is None:
            Mc = self.estimates.get('Mc', np.nan)

        if Mc is np.nan:
            raise Warning("Provide Mc or estimate it.")

        # Add noise
        while True:

            # Generate noise
            noise = np.random.random(mags.size)

            # Sample uniform in magnitude
            # Scale noise to bin size (within ±bin size)
            noise_scaled = self.mbin_orig * noise - self.mbin_orig / 2
            mags_noise = mags + noise_scaled

            # Assert that no values are equal (i.e. no duplicates)
            # (otherwise generate a new noise vector)
            if len(np.unique(mags_noise)) == len(mags):
                mags = mags_noise
                break

        # Shift Mc bin start to zero
        mags -= Mc - self.mbin_orig / 2

        # For sanity: treat negative values
        #  (can happen when `noise_scaled ~= -self.mbin_orig / 2` & mags not rounded to binning
        #   --> caused by floating point precision; will be around ~-e-08)
        mags_neg = mags < 0
        if np.count_nonzero(mags_neg):
            mags[mags_neg] = -mags[mags_neg]  # negate

        return mags

    def get_test_distribution(self, mags=None, Mc=None, n_repeats=100):
        """
        Perform a test several times and return its distribution.

        Every repetition is performed with a newly sampled random noise that is
        added to the magnitudes.
        """

        if mags is None:
            mags = self.mags

        if Mc is None:
            Mc = self.estimates.get('Mc', np.nan)

        if Mc is np.nan:
            self.logger.warning("Warning: Provide Mc, estimate it, or fit GR relation.")
            return None

        res = []
        for _ in range(n_repeats):
            _mags = self._prepare_mags_for_exponential(mags, Mc)
            res.append(self.lilliefors_test(_mags, log=False))

        return np.array(res)

    def calc_testdistr_mcutoff(self, n_repeats=100, Mstart=None, log=True):
        """
        Get test distribution (e.g., *p*-value) as function of magnitude cutoff.

        Parameters
        ----------
        n_repeats : int, optional
            Number of random initializations of noise added to mangitudes (which
            make the discrete magnitudes continuous), by default 100.
        Mstart : float, optional
            Overwrite minimum magnitude from where to start, by default starts at
            smallest magnitude present in the set.
        log : bool, optional
            Whether to log or not to log, by default `True`.
        """

        if self.n_ev == 0:
            self.testdistr_mcutoff = {
                'testdistr': np.array([np.nan]),
                'mbins': [np.nan],
                'N_samps': np.array([0]),
            }
            self.logger.warning("    Couldn't determine M_cutoff (no magnitudes present).")
            return None

        # Note: don't allow custom M-stepping, as this will cause inconsistency w/
        #       the shifting by Mc bin to zero in `self._prepare_mags_for_exponential()`
        #       --> always only use original magnitude binning as stepping
        mstep = self.mbin_orig
        mbins, _ = self._get_bins(mstep)

        if Mstart is not None:
            mbins = mbins[mbins >= Mstart]

        if log:
            self.logger.info("Getting test distributions (%dx) for %s magnitude cutoffs...",
                             n_repeats, len(mbins))

        res = []
        N_samps = []  # store number of samples
        for _mcut in tqdm(mbins, file=sys.stdout, disable=not log):

            # Extract corresponding magnitudes
            mags_cut = self.mags[self.mags >= _mcut - mstep / 2]  # `/ 2`: consider whole bin
            N_samps.append(len(mags_cut))

            res.append(self.get_test_distribution(mags_cut, _mcut, n_repeats))

        self.testdistr_mcutoff = {
            'testdistr': np.array(res),
            'mbins': mbins,
            'N_samps': np.array(N_samps),
        }

    def plot_testdist_expon_mcutoff(self, color='#000000', name=None, legendgroup=None,
                                    asfig=True):
        """
        Plot test distribution (e.g., *p*-value) as function of magnitude cutoff.

        Plots the mean, min/max, and +/- std curves.

        Parameters
        ----------
        color : str, optional
            The color to use, by default '#000000' (black).
        name : str, optional
            The name in the legend entry, by default None.
        legendgroup : bool, int, or string, optional
            Whether to group all curves into one group, by default None.
            And if yes, in which (needs to be specified as an integer, i.e. the group id).
        asfig : bool, optional
            Whether to return a plotly Figure instance (`True`), or only a dict (`False`).
            Defaults to `True`.

        Returns
        -------
        plotly.graph_objs._figure.Figure or dict
            plotly Figure instance or a dictionary to be converted into a plotly Figure.
        """

        if self.testdistr_mcutoff is None:
            # Note: need to generate test distribution first
            self.logger.warning("Warning: test distribution was not generated yet. "
                                "Executing .calc_testdistr_mcutoff() with default parameters"
                                " (Lilliefors test with 100 repeats over the whole M range).")
            self.calc_testdistr_mcutoff()

        # Convert to plotly color that can be alpha-ized
        color = pco.validate_colors(color, 'rgb')[0]

        # -- Get distributions

        res, mbins, N_samps = [self.testdistr_mcutoff[k]
                               for k in ('testdistr', 'mbins', 'N_samps')]

        if self.n_ev:
            res_means = res.mean(axis=1)
            res_std = res.std(axis=1)
            res_min, res_max = res.min(axis=1), res.max(axis=1)
        else:
            res_means, res_std, res_min, res_max = 4 * (np.array([np.nan]), )

        # -- Plot

        color_alpha_std = 'rgba' + color[3:-1] + ',0.3)'
        color_alpha_extr = 'rgba' + color[3:-1] + ',0.15)'

        legendgroup = "" if legendgroup is False or None else str(legendgroup)

        data = [
            # Mean
            go.Scatter(
                x=mbins, y=res_means,
                line_color=color, line_width=1.4,
                name='mean' if not name else '<b>' + name + '</b>',
                legendgroup=legendgroup,
                meta=np.column_stack((res_std, res_min, res_max, N_samps)),
                hovertemplate=(
                    "<i>M</i>%{x}: <i>p</i> = <b>%{y:.3g}</b> ± %{meta[0]:.3g}<br>"
                    "<i>p</i><sub>min/max</sub> = [%{meta[1]:.3g}, %{meta[2]:.3g}]<br>"
                    "N = %{meta[3]}"),
            ),

            # +/- std
            go.Scatter(x=mbins, y=res_means - res_std,
                       mode='lines', line_width=0.0,
                       name='1σ', legendgroup=legendgroup, showlegend=False,
                       hoverinfo='skip'),
            go.Scatter(x=mbins, y=res_means + res_std,
                       mode='none', fill='tonexty', fillcolor=color_alpha_std,
                       name='1σ', legendgroup=legendgroup, hoverinfo='skip'),

            # min/max
            go.Scatter(x=mbins, y=res_min,
                       mode='lines', line_width=0.0,
                       name='min / max', legendgroup=legendgroup, showlegend=False,
                       hoverinfo='skip'),
            go.Scatter(x=mbins, y=res_max,
                       mode='none', fill='tonexty', fillcolor=color_alpha_extr,
                       name='min / max', legendgroup=legendgroup, hoverinfo='skip'),
        ]

        layout = dict(  # go.Layout(
            shapes=[
                dict(type='line',
                     x0=0, x1=1, xref="paper", y0=self.signif_lev, y1=self.signif_lev,
                     line=dict(color='#D62728', width=1), layer='below'),
                dict(type='rect',
                     x0=0, x1=1, xref="paper", y0=0, y1=self.signif_lev,
                     fillcolor='#D62728', opacity=0.3, layer='below'),
            ],
            legend_traceorder='grouped',
            yaxis=dict(
                title=dict(text="Lilliefors <i>p</i>-value", standoff=8),
                showgrid=True,
                range=(0, 1),),
            xaxis=dict(
                title=dict(text="<i>M</i><sub>c</sub>", standoff=5),  # formerly "M_cutoff"
                showgrid=True,
                dtick=1,),
            height=380, width=800, margin={'l': 40, 'r': 10, 'b': 25, 't': 10},
        )

        if asfig:
            return go.Figure(data, layout)
        return {'data': data, 'layout': layout}

    def estimate_Mc_expon_test(self, mbin=None):
        """
        Get the magnitude of completeness using Lillieforst test of exponentiality.

        This means: the Mc is determined as the lowest magnitude where the p-value is above
                    the significance level, i.e. where the exponential assumption is not rejected.
        Here we will generally use the average p-value among several realizations of the random
        noise.
        For robustness, this average p-value must exceed the significance level for at least
        0.05 magnitude units, in which case the first exceedance, i.e., the smallest magnitude bin,
        yields the eventual Mc-Lilliefors.

        Parameters
        ----------
        mbin : float, optional
            Overwrite the "binning". Defaults to the magnitude binning.

        Returns
        -------
        float
            The estimated magnitude of completeness
        """

        if not self.testdistr_mcutoff:
            self.logger.warning("Warning: Test distributions were not yet calculated; "
                                "do so with .calc_testdistr_mcutoff().")
            return

        self.estimates.update({'Mc': np.nan})
        self.estimates.update({'n_compl': np.nan})

        if self.n_ev == 0:
            return np.nan

        # Determine the test statistic reference (that has to be above the significance level)
        mean = self.testdistr_mcutoff['testdistr'].mean(1)
        ref = mean  # preferred by Warner (std depends too much on tail,
                    # i.e. No of random iterations --> mean is representative)

        # Determine elements above the sign. level
        grarr = np.ma.greater(ref, self.signif_lev)

        # Determine index of first element above sign. level
        indxgr = np.where(grarr)[0]
        if indxgr.size == 0:
            return np.nan
        indx = indxgr[0]

        # Extra: only if at least 3 values are above the level --> ignore single exceedances
        #       --> then, take the first element of such a "3+ group"
        min_n_exceed = int(np.ceil(0.05 / self.mbin_orig))  # at least 0.05 of magnitude
        indxgr_split = np.split(indxgr, np.where(np.diff(indxgr) != 1)[0] + 1)
        indx = [x for x in indxgr_split if len(x) >= min_n_exceed][0][0]

        # Finally, get Mc
        Mc = self.testdistr_mcutoff['mbins'][indx]
        if not mbin:
            mbin = self.mbin_orig
        Mc = self._correct_to_binning(Mc, mbin)
        n_compl = np.count_nonzero(self.mags >= Mc - self.mbin_orig / 2)

        self.estimates.update({'Mc': Mc})
        self.estimates.update({'n_compl': n_compl})

        return Mc
