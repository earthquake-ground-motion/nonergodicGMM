"""
Ground Motion Model structure for ergodic GMMs that require calculation
of the total residuals with parts of the GMMs themselves exctracred
"""
import re
import logging
from copy import deepcopy
from typing import List, Optional
import numpy as np
import pandas as pd
from scipy.constants import g
from openquake.hazardlib import valid

# Core set of OpenQuake GMMs
import openquake.hazardlib.gsim.chiou_youngs_2014 as cy14
import openquake.hazardlib.gsim.abrahamson_2014 as ask14
import openquake.hazardlib.gsim.campbell_bozorgnia_2014 as cb14
import openquake.hazardlib.gsim.boore_2014 as bssa14
import openquake.hazardlib.gsim.akkar_2014 as asb14
import openquake.hazardlib.gsim.bindi_2014 as bi14
import openquake.hazardlib.gsim.kotha_2020 as ko20
import openquake.hazardlib.gsim.cauzzi_2014 as ca15

# Import the Flatfile
from dynamicgmm.nonergodic.grid_tools import Flatfile

# For building the contexts specific attributes require non-float datatypes
INTEGER_DTYPES = {"region", }
BOOLEAN_DTYPES = {"vs30measured", "backarc", }
BYTE_DTYPES = {"siteclass": (np.bytes_, 1),
               "ec8": (np.bytes_, 1),
               "ec8_p18": (np.bytes_, 2),
               "geology": (np.bytes_, 20)}

NUMBER_STRING = re.compile(r'^(?:\d+|\d+.|\d+\.\d+)$')


class ErgodicGMM():
    """Base class for an ergodic ground motion model containing functions
    to build the context objects, get the observations and determine their
    total residuals with respect to the GMM.

    For fitting of some non-ergodic models it is necessary to subtract certain parts
    of the ergodic models from the total residuals (e.g. anelastic attenuation,
    """
    GMM = None

    def __init__(self):
        self.REQUIRES = self.GMM.REQUIRES_RUPTURE_PARAMETERS |\
            self.GMM.REQUIRES_DISTANCES | self.GMM.REQUIRES_SITES_PARAMETERS
        self.dtypes = []
        for scenario_property in self.REQUIRES:
            if scenario_property in INTEGER_DTYPES:
                self.dtypes.append((scenario_property, np.uint32))
            elif scenario_property in BOOLEAN_DTYPES:
                self.dtypes.append((scenario_property, bool))
            elif scenario_property in BYTE_DTYPES:
                self.dtypes.append((scenario_property, BYTE_DTYPES[scenario_property]))
            else:
                self.dtypes.append((scenario_property, np.float64))
        self.dtypes = np.dtype(self.dtypes)

    def get_anelastic_attenuation_term(self, imts: List, ctx: np.recarray) -> np.ndarray:
        raise NotImplementedError

    def get_geometric_spreading_term(self, imts: List, ctx: np.recarray) -> np.ndarray:
        raise NotImplementedError

    def get_linear_site_term(self, imts: List, ctx: np.recarray) -> np.ndarray:
        raise NotImplementedError

#    def get_x_2(self, imts: List, ctx: np.recarray) -> np.ndarray:
#        raise NotImplementedError
#
#    def get_x_3(self, imts: List, ctx: np.recarray) -> np.ndarray:
#        raise NotImplementedError

    def build_ctx(self, input_flatfile):
        """
        """
        # Build the metadata context
        n = input_flatfile.shape[0]
        ctx = np.recarray(n, dtype=self.dtypes)

        scenario_mapping = {"magnitude": "mag",
                            "rupture_width": "width",
                            "rupture_length": "length",
                            "event_hypo_depth": "hypo_depth",
                            "z1": "z1pt0",
                            "depth_top_of_rupture": "ztor",
                            }
        flatfile = input_flatfile.rename(columns=scenario_mapping,
                                         inplace=False)
        flatfile["region"] = np.zeros(flatfile.shape[0], dtype=int)

        for scenario_property in self.REQUIRES:
            if scenario_property in INTEGER_DTYPES:
                ctx[scenario_property] = flatfile[scenario_property].to_numpy().astype(int)
            elif scenario_property in BOOLEAN_DTYPES:
                ctx[scenario_property] = flatfile[scenario_property].to_numpy().astype(bool)
            elif scenario_property in BYTE_DTYPES:
                ctx[scenario_property] = flatfile[scenario_property].to_numpy().astype(
                    BYTE_DTYPES[scenario_property])
            else:
                ctx[scenario_property] = flatfile[scenario_property].to_numpy().astype(float)
        return ctx

    @staticmethod
    def get_observations(
        flatfile: pd.DataFrame,
        imts: List,
        sa_units: str = "g"
) -> pd.DataFrame:
        """Retreives the requested ground motion values from the dataframe
        """
        # Get the observations
        flatfile_periods = []
        sa_columns = []
        for hdr in flatfile.columns:
            if hdr.startswith("SA("):
                flatfile_periods.append(float(hdr.replace("SA(", "").replace(")", "")))
                sa_columns.append(hdr)
            elif hdr.startswith("EAS("):
                flatfile_periods.append(float(hdr.replace("EAS(", "").replace(")", "")))
                sa_columns.append(hdr)
            elif hdr.startswith("FAS("):
                flatfile_periods.append(float(hdr.replace("EAS(", "").replace(")", "")))
                sa_columns.append(hdr)
            elif bool(NUMBER_STRING.fullmatch(hdr)):
                # Simple numerical header
                flatfile_periods.append(float(hdr))
                sa_columns.append(hdr)
            else:
                pass
        if sa_units in ("m/s", "m s^-1"):
            factor = 1.0 / g
        elif sa_units in ("cm/s/s", "cm s^-2", "gal"):
            factor = 1.0 / (100 * g)
        else:
            factor = 1.0
        n = flatfile.shape[0]
        flatfile_periods = np.array(flatfile_periods)
        ascend = np.argsort(flatfile_periods)
        flatfile_periods = flatfile_periods[ascend]
        sa_columns = [sa_columns[idx] for idx in ascend]
        for col in (["PGA",] + sa_columns):
            flatfile[col] *= factor
        flatfile_sa = flatfile[sa_columns]
        min_t, max_t = np.min(flatfile_periods), np.max(flatfile_periods)
        observations = {}
        for imt in imts:
            if str(imt) in ("PGA", "PGV"):
                if str(imt) in flatfile.columns:
                    observations[str(imt)] = np.log(flatfile[str(imt)].to_numpy())
                else:
                    logging.info("No data for IMT %s in flatfile" % str(imt))
            elif str(imt).startswith("SA("):
                period = float(str(imt).replace("SA(", "").replace(")", ""))
                if (period < min_t) or (period > max_t):
                    logging.info("Required IMT %s outside period range in flatfile (%f - %f)" &
                                 (str(imt), min_t, max_t))
                    observations[str(imt)] = np.nan * np.ones(n)
                else:
                    iloc = np.searchsorted(flatfile_periods, period)
                    if not iloc or (iloc == len(flatfile_periods)):
                        # Exactly lowest or highest value
                        observations[str(imt)] = np.log(flatfile_sa[sa_columns[iloc]])
                        continue
                    t = np.log10(period)
                    tlow = np.log10(flatfile_periods[iloc - 1])
                    thigh = np.log10(flatfile_periods[iloc])
                    dy = np.log10(flatfile_sa[sa_columns[iloc]]) -\
                        np.log10(flatfile_sa[sa_columns[iloc - 1]])
                    dx = thigh - tlow
                    observations[str(imt)] = np.log(10.0 ** (
                        np.log10(flatfile_sa[sa_columns[iloc - 1]]) + (t - tlow) * (dy / dx)))
        observations = pd.DataFrame(observations)
        return observations

    def get_total_residual(self, flatfile: pd.DataFrame, imts: List, function_type: int = 1,
                           sa_units: str = "g"):
        """
        """
        assert function_type in (1, 2, 3), "Function type should be 1, 2 or 3: %g input"\
            % function_type
        ctx = self.build_ctx(flatfile)
        obs = self.get_observations(flatfile, imts, sa_units).to_numpy().T
        n = len(ctx)
        nimts = len(imts)
        mean = np.zeros([nimts, n])
        sigma = np.zeros([nimts, n])
        phi = np.zeros([nimts, n])
        tau = np.zeros([nimts, n])
        self.GMM.compute(ctx, imts, mean, sigma, tau, phi)
        if function_type == 3:
            # Type 3 removes anelastic term, geometric spreading and linear site term
            atten_term = self.get_anelastic_attenuation_term(imts, ctx)
            geom_spread = self.get_geometric_spreading_term(imts, ctx)
            linear_site = self.get_linear_site_term(imts, ctx)
            obs -= (mean - (geom_spread + atten_term + linear_site))
        elif function_type == 2:
            # Type 2 removes the anelastic attenuation term
            atten_term = self.get_anelastic_attenuation_term(imts, ctx)
            obs -= (mean - atten_term)
        else:
            # Type 1 considers only source and site adjustment
            obs -= mean
        return pd.DataFrame(dict([(str(imts[i]), obs[i, :]) for i in range(nimts)]))

    def get_x2_x3(self, flatfile: pd.DataFrame, imts: List):
        """
        """
        nimts = len(imts)
        ctx = self.build_ctx(flatfile)
        obs = self.get_observations(flatfile, imts).to_numpy().T
        x_2 = self.get_geometric_spreading_term(imts, ctx)
        x_3 = self.get_linear_site_term(imts, ctx)
        x_2 = pd.DataFrame(dict([(str(imts[i]), x_2[i, :]) for i in range(nimts)]))
        x_3 = pd.DataFrame(dict([(str(imts[i]), x_3[i, :]) for i in range(nimts)]))
        return x_2, x_3


"""
General Template for an ergodic GMM object


class ####1234Ergodic(ErgodicGMM):
    ""Ergodic form of the ????? GMM
    ""
    GMM = valid.gsim("")

    def get_anelastic_attenuation_term(self, imts: List, ctx: np.recarray) -> np.ndarray:
        "
        "
        attenuation = np.zeros([len(imts), len(ctx)])
        for i, imt in enumerate(imts):
            C = self.GMM.COEFFS[imt]
            ????
        return attenuation

    def get_geometric_spreading_term(self, imts: List, ctx: np.recarray) -> np.ndarray:
        "
        "
        geometric_spreading = np.zeros([len(imts), len(ctx)])
        for i, imt in enumerate(imts):
            C = self.GMM.COEFFS[imt]
            ????
        return geometric_spreading


    def get_linear_site_term(self, imts: List, ctx: np.recarray) -> np.ndarray:
        "
        "
        linear_site = np.zeros([len(imts), len(ctx)])
        for i, imt in enumerate(imts):
            C = self.GMM.COEFFS[imt]
            ????
        return linear_site
"""


class ASK2014Ergodic(ErgodicGMM):
    """Ergodic form of the Abrahamson et al. (2014) GMM
    """
    GMM = valid.gsim("AbrahamsonEtAl2014")

    def get_anelastic_attenuation_term(self, imts: List, ctx: np.recarray) -> np.ndarray:
        """
        """
        attenuation = np.zeros([len(imts), len(ctx)])
        for i, imt in enumerate(imts):
            C = self.GMM.COEFFS[imt]
            attenuation[i, :] = C["a17"] * ctx.rrup
        return attenuation

    def get_geometric_spreading_term(self, imts: List, ctx: np.recarray) -> np.ndarray:
        """
        """
        geometric_spreading = np.zeros([len(imts), len(ctx)])
        for i, imt in enumerate(imts):
            C = self.GMM.COEFFS[imt]
            # Fictitious depth
            c4m = C["c4"] - (C["c4"] - 1.0) * (5.0 - ctx.mag)
            c4m[ctx.mag > 5.0] = C["c4"]
            c4m[ctx.mag < 4.0] = 1.0
            rval = np.sqrt(ctx.rrup ** 2. + c4m ** 2.)
            geometric_spreading[i, :] = (C["a2"] + C["a3"] * (ctx.mag - C["m1"])) *\
                np.log(rval)
        return geometric_spreading

    def get_linear_site_term(self, imts: List, ctx: np.recarray) -> np.ndarray:
        """
        """
        linear_site = np.zeros([len(imts), len(ctx)])
        for i, imt in enumerate(imts):
            C = self.GMM.COEFFS[imt]
            vs30star = ask14._get_vs30_star(ctx.vs30, imt)
            linear_site[i, :] = (C["a10"] + C["b"] * ask14.CONSTS["n"]) *\
                np.log(vs30star / C["vlin"])
        return linear_site


class BSSA2014Ergodic(ErgodicGMM):
    """Erogdic form of the Boore et al. (2014) GMM
    """
    GMM = valid.gsim("BooreEtAl2014")

    def get_anelastic_attenuation_term(self, imts: List, ctx: np.recarray) -> np.ndarray:
        """
        """
        attenuation = np.zeros([len(imts), len(ctx)])
        for i, imt in enumerate(imts):
            C = self.GMM.COEFFS[imt]
            rval = np.sqrt(ctx.rjb ** 2.0 + C["h"] ** 2.0)
            attenuation[i, :] = (C["c3"] + C["Dc3"]) * (rval - bssa14.CONSTS["Rref"])
        return attenuation

    def get_geometric_spreading_term(self, imts: List, ctx: np.recarray) -> np.ndarray:
        """
        """
        geometric_spreading = np.zeros([len(imts), len(ctx)])
        for i, imt in enumerate(imts):
            C = self.GMM.COEFFS[imt]
            rval = np.sqrt(ctx.rjb ** 2.0 + C["h"] ** 2.0)
            geometric_spreading[i, :] =\
                (C["c1"] + C["c2"] * (ctx.mag - bssa14.CONSTS["Mref"])) *\
                np.log(rval / bssa14.CONSTS["Rref"])
        return geometric_spreading

    def get_linear_site_term(self, imts: List, ctx: np.recarray) -> np.ndarray:
        """
        """
        linear_site = np.zeros([len(imts), len(ctx)])
        for i, imt in enumerate(imts):
            C = self.GMM.COEFFS[imt]
            linear_site[i, :] = bssa14._get_linear_site_term(C, ctx.vs30)
        return linear_site


class CB2014Ergodic(ErgodicGMM):
    """Ergodic form of the Campbell & Bozorgnia (2014) GMM
    """
    GMM = valid.gsim("CampbellBozorgnia2014")

    def get_anelastic_attenuation_term(self, imts: List, ctx: np.recarray) -> np.ndarray:
        """
        """
        attenuation = np.zeros([len(imts), len(ctx)])
        for i, imt in enumerate(imts):
            C = self.GMM.COEFFS[imt]
            attenuation[i, :] = cb14._get_anelastic_attenuation_term(C, ctx.rrup)
        return attenuation

    def get_geometric_spreading_term(self, imts: List, ctx: np.recarray) -> np.ndarray:
        """
        """
        geometric_spreading = np.zeros([len(imts), len(ctx)])
        for i, imt in enumerate(imts):
            C = self.GMM.COEFFS[imt]
            geometric_spreading[i, :] = cb14._get_geometric_attenuation_term(C, ctx.mag,
                                                                             ctx.rrup)
        return geometric_spreading

    def get_linear_site_term(self, imts: List, ctx: np.recarray) -> np.ndarray:
        """
        """
        linear_site = np.zeros([len(imts), len(ctx)])
        for i, imt in enumerate(imts):
            C = self.GMM.COEFFS[imt]
            linear_site[i, :] = (C["c11"] + C["k2"] * cb14.CONSTS["n"]) *\
                np.log(ctx.vs30 / C["k1"])
        return linear_site


class CY2014Ergodic(ErgodicGMM):
    """Ergodic form of the Chiou & Youngs (2014) GMM
    """
    GMM = valid.gsim("ChiouYoungs2014")

    def get_anelastic_attenuation_term(self, imts: List, ctx: np.recarray) -> np.ndarray:
        """
        """
        attenuation = np.zeros([len(imts), len(ctx)])
        for i, imt in enumerate(imts):
            C = self.GMM.COEFFS[imt]
            attenuation[i, :] = cy14.get_far_field_distance_scaling("CAL", C,
                                                                    ctx.mag, ctx.rrup)
        return attenuation

    def get_geometric_spreading_term(self, imts: List, ctx: np.recarray) -> np.ndarray:
        """
        """
        geometric_spreading = np.zeros([len(imts), len(ctx)])
        for i, imt in enumerate(imts):
            C = self.GMM.COEFFS[imt]
            geometric_spreading[i, :] = cy14.get_geometric_spreading(C, ctx.mag, ctx.rrup)
        return geometric_spreading

    def get_linear_site_term(self, imts: List, ctx: np.recarray) -> np.ndarray:
        """
        """
        linear_site = np.zeros([len(imts), len(ctx)])
        for i, imt in enumerate(imts):
            C = self.GMM.COEFFS[imt]
            linear_site[i, :] = cy14.get_linear_site_term("global", C, ctx)
        return linear_site

#    def get_x_2(self, imts: List, ctx: np.recarray) -> np.ndarray:
#        """
#        """
#        x_2 = np.zeros([len(imts), len(ctx)])
#        for i, imt in enumerate(imts):
#            C = self.GMM.COEFFS[imt]
#            x_2[i, :] = np.log(ctx.rrup + C["c5"] * np.cosh(C["c6"] *
#                np.clip(ctx.mag - C["chm"], 0.0, None)))
#        return x_2
#
#    def get_x_3(self, imts: List, ctx: np.recarray) -> np.ndarray:
#        """
#        """
#        x_3 = np.zeros([len(imts), len(ctx)])
#        for i, imt in enumerate(imts):
#            C = self.GMM.COEFFS[imt]
#            x_3[i, :] = np.log(ctx.vs30 / 1130).clip(-np.inf, 0.0)
#        return x_3

class KothaEtAl2020ESHM20Ergodic(ErgodicGMM):
    """Kotha et al. (2020) model adapted for ESHM20. Note that here the regional
    adjustments are all turned off and we are interested only in the base model
    """
    GMM = valid.gsim("KothaEtAl2020ESHM20")

    def get_anelastic_attenuation_term(self, imts: List, ctx: np.recarray) -> np.ndarray:
        """
        """
        attenuation = np.zeros([len(imts), len(ctx)])
        for i, imt in enumerate(imts):
            C = self.GMM.COEFFS[imt]
            h = ko20._get_h(C, ctx.hypo_depth)
            rval = np.sqrt(ctx.rjb ** 2. + h ** 2.)
            rref_val = np.sqrt(ko20.CONSTANTS["Rref"] ** 2. + h ** 2.)
            c3 =  ko20.get_distance_coefficients_2("ESHM20", None, 0.0, C, imt, ctx) / 100.0
            attenuation[i, :] = c3 * (rval - rref_val)
        return attenuation

    def get_geometric_spreading_term(self, imts: List, ctx: np.recarray) -> np.ndarray:
        """
        """
        geometric_spreading = np.zeros([len(imts), len(ctx)])
        for i, imt in enumerate(imts):
            C = self.GMM.COEFFS[imt]
            h = ko20._get_h(C, ctx.hypo_depth)
            rval = np.sqrt(ctx.rjb ** 2. + h ** 2.)
            rref_val = np.sqrt(ko20.CONSTANTS["Rref"] ** 2. + h ** 2.)
            geometric_spreading[i, :] = \
                (C["c1"] + C["c2"] * (ctx.mag - ko20.CONSTANTS["Mref"])) *\
                np.log(rval / rref_val) 
            #geometric_spreading[i, :] = cy14.get_geometric_spreading(C, ctx.mag, ctx.rrup)
        return geometric_spreading

    def get_linear_site_term(self, imts: List, ctx: np.recarray) -> np.ndarray:
        """
        """
        linear_site = np.zeros([len(imts), len(ctx)])
        for i, imt in enumerate(imts):
            C = self.GMM.COEFFS[imt]
            linear_site[i, :] = ko20.get_site_amplification("ESHM20", None, C, ctx, imt)
        return linear_site


class BindiEtAl2014Ergodic(ErgodicGMM):
    """
    """
    GMM = valid.gsim("BindiEtAl2014Rjb")

    def get_anelastic_attenuation_term(self, imts: List, ctx: np.recarray) -> np.ndarray:
        """
        """
        attenuation = np.zeros([len(imts), len(ctx)])
        for i, imt in enumerate(imts):
            C = self.GMM.COEFFS[imt]
            r_adj = np.sqrt(ctx.rjb ** 2.0 + C["h"] ** 2.0)
            attenuation[i, :] = C["c3"] * (r_adj - bi14.CONSTS["Rref"])
        return attenuation

    def get_geometric_spreading_term(self, imts: List, ctx: np.recarray) -> np.ndarray:
        """
        """
        geometric_spreading = np.zeros([len(imts), len(ctx)])
        for i, imt in enumerate(imts):
            C = self.GMM.COEFFS[imt]
            r_adj = np.sqrt(ctx.rjb ** 2.0 + C["h"] ** 2.0)
            geometric_spreading[i, :] = (
                C["c1"] + C["c2"] * (ctx.magmag - bi14.CONSTS["Mref"])
            ) * np.log10(r_adj / bi14.CONSTS["Rref"])
        return geometric_spreading
    
    def get_linear_site_term(self, imts: List, ctx: np.recarray) -> np.ndarray:
        """
        """
        linear_site = np.zeros([len(imts), len(ctx)])
        for i, imt in enumerate(imts):
            C = self.GMM.COEFFS[imt]
            linear_site[i, :] = C["gamma"] * np.log10(ctx.vs30 / bi14.CONSTS["Vref"])
        return linear_site
