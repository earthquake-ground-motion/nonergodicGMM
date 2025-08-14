"""
New EMEC catalogue harmoniser

"""

"""
Archive of regression models
"""
import numpy as np
from scipy import odr
import matplotlib.pyplot as plt


class ConversionModel(object):
    """
    """
    INPUT = None
    BOUNDS = []
    DESCR = ""
    STDDEV = 0.0
#    OUTPUT = None

    def __init__(self):
        pass

    def plot(self, data=None, xlim=(0.0, 10.), ylim=(0., 10.), figsize=(8,8),
             ax=None, col="k", filename="", filetype="png", dpi=300):
        """
        Produces a plot of the model
        """
        if not ax:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            ax.plot([0., 10.], [0., 10.], "--", lw=1., color=[0.5, 0.5,0.5])
        xrng = np.arange(xlim[0], xlim[-1] + 0.025, 0.025)
        yrng = self(xrng)
        if len(self.BOUNDS):
            idx = np.logical_and(xrng >= self.BOUNDS[0],
                                 xrng <= self.BOUNDS[1])
            ax.plot(xrng, yrng, "-.", lw=1.5, color=col)
            ax.plot(xrng[idx], yrng[idx], "-", lw=2., color=col,
                    label=self.descr())
        else:
            ax.plot(xrng, yrng, "-", lw=2., color=col,
                    label=self.descr())
        if isinstance(self.INPUT, list):
            ax.set_xlabel(self.INPUT[0], fontsize=16)
        else:
            ax.set_xlabel(self.INPUT, fontsize=16)
        ax.set_ylabel("Mw", fontsize=16)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=12)
        ax.grid(True)
        ax.legend(fontsize=14)
        if filename:
            plt.savefig(filename, format=filetype, dpi=dpi,
                        bbox_inches="tight")

        return ax

    def __call__(self, mag, h=1.0):
        """
        Executes the magnitude conversion relation for a given input magnitude
        """
        raise NotImplementedError

    def stddev(self, mag):
        return self.STDDEV

    def deriv(self, mag, h):
        """
        Returns the partial derivative of the model M* = f(M, h) with respect
        to magnitude
        df(M,h) / dM
        """
        raise NotImplementedError

    def uncertainty(self, mag, sigma_m_obs=0.0, h=1.0):
        """
        Propagate the uncertainty using standard error propagation
        
        sigma_m_out = sqrt(sigma_model ** 2 + (df/dM * sigma_m_obs) ** 2.0)
        """
        model_deriv = self.deriv(mag, h)
        return np.sqrt(self.stddev(mag) ** 2.0 + (model_deriv * sigma_m_obs) ** 2.0)

    def descr(self):
        """
        Returns a descriptive string
        """
        if self.DESCR:
            return " - ".join([self.__class__.__name__, self.DESCR])
        else:
            return self.__class__.__name__

    def __repr__(self):
        output_string = [
            self.__class__.__name__,
            self.INPUT,
            "(%s)" % self.DESCR]
        if len(self.BOUNDS):
            output_string.append("[{:.3f}:{:.3f}]".format(self.BOUNDS[0], self.BOUNDS[1]))
        return " ".join(output_string)


class Equality(ConversionModel):
    """
    Special case of a conversion model in which the input and output
    magnitude is assumed equal
    """
    
    def __init__(self, mtype="",  descr="", stddev = 0.0, bounds=None):

        """
        For equality we can set the magnitude type, bounds, description and
        standard deviation
        """
        super().__init__()
        if mtype:
            self.INPUT = mtype
        if bounds is not None:
            self.BOUNDS = bounds
        if descr:
            self.DESCR = descr
        self.STDDEV = stddev

    def __call__(self, mag, h=1.0):
        return mag

    def deriv(self, mag, h=1.0):
        return 1.0


class Offset(ConversionModel):
    """
    Special case of a conversion that take the form of an offset term,
    effectively a zero-th order polynomial
    """
    def __init__(self, mtype="",  descr="", stddev = 0.0, bounds=None,
                 offset=0.0):

        super().__init__()
        if mtype:
            self.INPUT = mtype
        if bounds is not None:
            self.BOUNDS = bounds
        if descr:
            self.DESCR = descr
        self.STDDEV = stddev
        self.offset = offset
    
    def __call__(self, mag, h=1.0):
        return mag + self.offset

    def deriv(self, mag, h=1.0):
        return 1.0



class RegressionModel():
    """
    """
    def __init__(self, B=None, C=None):
        self.B = B
        self.C = C
        self.output = None
        self.sigma_b = None
        self.cov = None

    def setup_from_data(self, x, y, C=None):
        """
        """
        raise NotImplementedError

    @staticmethod
    def model(B, x, C=None):
        raise NotImplementedError

    def __repr__(self):
        return self.__class__.__name__ + "- Abstract"
    
    def fit_model(self, x, y, sigma_x = None, sigma_y = None, B=None, C=None,
                  **kwargs):
        if C is not None:
            self.C = C
        # If B is not defined then setup initial values from data
        if self.B is None and B is None:
            self.B = self.setup_from_data(x, y, self.C, **kwargs)
        model = odr.Model(self.model)
        if sigma_x is None:
            sigma_x = np.ones(len(x))
        if sigma_y is None:
            sigma_y = np.ones(len(y))
        data = odr.RealData(x, y, sx=sigma_x, sy=sigma_y)
        runner = odr.ODR(data, model, beta0=self.B)
        self.output = runner.run()
        self.B = self.output.beta
        self.sigma_b = self.output.sd_beta
        self.stddev = np.std(y - self.model(self.B, x))

    def _execute(self, x):
        return self.model(self.B, x)

    def __call__(self, x):
        if not self.output:
            raise ValueError("Model not yet fit!")
        return self.model(self.B, x)
       


class Polynomial(RegressionModel):

    @staticmethod 
    def model(B, x, C=None):
        """
        Executes a simple polynomial funcion
        """
        y = 0.0
        for i in range(len(B)):
            y += (B[i] * (x ** float(i)))
        return y

    def setup_from_data(self, x, y, C=None, **kwargs):
        """
        """
        order = kwargs.get("order", 1)
        B = np.polyfit(x, y, deg=order)
        return B[::-1]

    def __repr__(self):
        if not self.output:
            return "Polynomial(not fit)"
        else:
            output_str = []
            for i, (b, sig_b) in enumerate(zip(self.B, self.sigma_b)):
                if not i:
                    output_str.append(r"{:.4f}($\pm${:.4f})".format(b, sig_b))
                elif i == 1:
                    output_str.append(r"{:.4f}($\pm${:.4f}) * M".format(b, sig_b))
                else:
                    output_str.append(
                        r"{:.4f}($\pm${:.4f}) * (M$^{{:g}}$)".format(b, sig_b, i)
                        )
            return " + ".join(output_str)


class NegativeSqrt(RegressionModel):
    

    @staticmethod
    def model(B, x, C=None):
        """
        Return y = a - sqrt(b - c * x)
        """
        return B[0] - np.sqrt(B[1] - B[2] * x)
    
    def setup_from_data(self, x, y, C=None, **kwargs):
        """
        """
        x0 = kwargs.get("x0", [10.0, 40.0, 7.0])
        f = lambda z, a, b, c: a - np.sqrt(b - c * z)
        res = curve_fit(f, x, y, x0)[0]
        return res

    def __repr__(self):
        if not self.output:
            return "Negative Square Root (Not Fit)"
        else:
            terms = []
            for val, sig in zip([self.B, self.sigma_b]):
                terms.append(r"{:.4f}($\pm${:.4f})".format(val, sig))
            return "%s - sqrt(%s - %s * M)" % (terms[0], terms[1], terms[2])



class Exponential(RegressionModel):

    @staticmethod
    def model(B, x, C=None):
        """
        Return y = exp(a + b * x) + c)
        """
        return np.exp(B[0] + B[1] * x) + B[2]

    def setup_from_data(self, x, y, C=None, **kwargs):
        """
        """
        x0 = np.polyfit(x, np.log(y), deg=1)
        b = x[0]
        a = x[1] / 2.0
        c = np.exp(x[1] / 2.0)
        return [a, b, c]

    def __repr__(self):
        if not self.output:
            return "Exponential(Not Fit)"
        else:
            terms = []
            for val, sig in zip([self.B, self.sigma_b]):
                terms.append(r"{:.4f}($\pm${:.4f})".format(val, sig))
            return "exp(%s + %s * M) + %s" % (terms[0], terms[1], terms[2]) 



def piecewise_linear_fixed(B, x, C):
    """
    Multi-segment piecewise linear with fixed corner points
    B: [intercept, slope1, slope2, slope3, ...]
    C: [turning_point1, turning_point2, turning_point3, ...]
    """
    assert len(C) == (len(B) - 2)
    intercepts = np.zeros(len(B) - 1)
    intercepts[0] = B[0]
    slopes = B[1:]
    n_seg = len(slopes)
    y = (slopes[0] * x) + intercepts[0]
    for i in range(1, n_seg):
        # For the x values greater than the first turning point
        idx = x >= C[i - 1]
        intercepts[i] = intercepts[i - 1] + (slopes[i - 1] * C[i - 1]) - (slopes[i] * C[i - 1])
        y[idx] = (slopes[i] * x[idx]) + intercepts[i]
    return y


def piecewise_linear_free(B, x, C=None):
    """
    Multi segment piecewise linear with free optimisation of the corner points
    B = [intercept, slope1, slope2, ..., slopen, turningpoint1, turningpoint2, ... turning_point_n-1]
    """
    
    npos = (len(B[:1]) // 2) + 2
    slopes = B[1:npos]
    mc = B[npos:]
    # Check that there is one more slope than turning point
    assert (len(slopes) - len(mc)) == 1
    intercepts = np.zeros(len(slopes))
    intercepts[0] = B[0]
    n_seg = len(slopes)
    y = slopes[0] * x + intercepts[0]
    for i in range(1, n_seg):
        # For the x values greater than the first turning point
        idx = x >= mc[i - 1]
        intercepts[i] = intercepts[i - 1] + (slopes[i - 1] * mc[i - 1]) - (slopes[i] * mc[i - 1])
        y[idx] = (slopes[i] * x[idx]) + intercepts[i]
    return y
    


class MwMs69(ConversionModel):
    INPUT = "Ms"
    BOUNDS = [4.0, 7.0]
    DESCR = "Papazachos et al. (2003)"

    def __call__(self, mag, h=1.0):
        return 0.995 * mag + 0.09

    def stddev(self, mag):
        return 0.175

    def deriv(self, mag, h=1.0):
        return 0.995


class MwI074(ConversionModel):
    INPUT = "Imax"
    DESCR = "Mezcua (2002)"

    def __call__(self, mag, h=1.0):
        return 0.6 * mag + 0.96

    def deriv(self, mag, h=1.0):
        return 0.6


class MwMs54(ConversionModel):
    INPUT = "Ms"
    DESCR = "Bungum et al. (2003)"
    def __call__(self, mag, h=1.0):
        return np.where(mag >= 5.4, 0.796 * mag + 1.280, 0.585 * mag + 2.422)
    
    def deriv(self, mag, h=1.0):
        return np.where(mag >= 5.4, 0.796, 0.585)
        

class MwML(ConversionModel):
    INPUT = "ML"
    DESCR = "Stromeyer et al. (2004)"
    def __call__(self, mag, h=1.0):
        return 0.5322  + 0.6462 * mag + 0.0376 * (mag ** 2.)

    def stddev(self, mag):
        return (0.97 * (mag ** 4.) - 12.4 * (mag ** 3.) + 58.4 * (mag ** 2.) -
                120. * mag + 921) * 1.0E-4

    def deriv(self, mag, h=1.0):
        return 0.6462 + (2.0 * 0.0376 * mag)


class MwML96(ConversionModel):
    INPUT = "ML"
    BOUNDS = [4., 7.]
    def __call__(self, mag, h=1.0):
        return 0.65 * mag + 1.90

    def stddev(self, mag):
        return 0.235
    
    def deriv(self, mag, h=1.0):
        return 0.65


class MwMLLDG(ConversionModel):
    INPUT = "ML"
    BOUNDS = [0., 4.]
    DESCR = "ML (LDG) - Mw SiHex"
    def __call__(self, mag, h=1.0):
        return np.where(mag >= 3.1, mag - 0.6, 0.664 * mag + 0.45)

    def deriv(self, mag, h=1.0):
        return np.where(mag >= 3.1, 1.0, 0.664)


class MwML92(ConversionModel):
    INPUT = "ML"
    DESCR = "ML to Ms (GW 12), then Mw"
    def __call__(self, mag, h=1.0):
        ms = 1.37 * mag - 2.19
        return np.where(ms >= 5.4, 0.796 * ms + 1.28, 0.585 * ms + 2.422)

    def deriv(self, mag, h=1.0):
        """
        chain rule: u = f(v(m))
        du/dv = (df / dv) * (dv / dm)
        """
        dfms_dmag = 1.37
        ms = 1.37 * mag - 2.19
        return np.where(ms >= 5.4, 0.796 * dfms_dmag, 0.585 * dfms_dmag)


class MwMb74a(ConversionModel):
    INPUT = "mb"
    DESCR = "Utsu (2002) - mb"
    def __call__(self, mag, h=1.0):
        return 8.17 - np.sqrt(42.04 - 6.42 * mag)

    def deriv(self, mag, h=1.0):
        """
        f(M) = a - sqrt(b - c * mag)
        df(M) / dM = c / (2.0 * sqrt(b - c * mag)
        """
        return 6.42 / (2.0 * np.sqrt(42.04 - 6.42 * mag))
     

class MwMb74b(ConversionModel):
    INPUT = "mbLg"
    DESCR = "Rueda & Mezcua (2002)"
    def __call__(self, mag, h=1.0):
        return 0.311 + 0.637 * mag + 0.061 * (mag ** 2.)

    def deriv(self, mag, h=1.0):
        return 0.637 + (2.0 * 0.061 * mag)


class MwML63(ConversionModel):
    INPUT = "ML"
    DESCR = "Gruenthal et al. (2009)"
    def __call__(self, mag, h=1.0):
        return 0.906 * mag + 0.65

    def deriv(self, mag, h=1.0):
        return 0.906


class MwML90(ConversionModel):
    INPUT = "ML"
    def __call__(self, mag, h=1.0):
        return 0.99 * mag + 0.33

    def deriv(self, mag, h=1.0):
        return 0.99


class MwMs29(ConversionModel):
    INPUT = "Ms"
    DESCR = "Utsu (2002) - Ms"
    def __call__(self, mag, h=1.0):
        return np.where(mag < 8.0, 10.85 - np.sqrt(73.74 - 8.38 * mag), 8.0)

    def deriv(self, mag, h):
        """
        """
        return np.where(mag < 8.0, 8.38 / (2.0 * np.sqrt(73.74 - 8.38 * mag)), 0.0)

class MwMs(ConversionModel):
    INPUT = "Ms"
    DESCR = "Gruenthal & Wahlstrom (2003)"
    def __call__(self, mag, h=1.0):
        return mag

    def deriv(self, mag, h=1.0):
        return 1.0


class MwI055(ConversionModel):
    INPUT = ["I0", "H"]
    def __call__(self, mag, h=1.0):
        mlio55 = 0.85 * mag + 0.76 * np.log10(h) - 1.41
        return 0.5322 + 0.6462 * mlio55 + 0.0376 * (mlio55 ** 2.)

    def deriv(self, mag, h=1.0):
        """
        chain rule = u = f(v(m))
        du / dv = (df / dv) * (dv / dm)
        """
        mlio55 = 0.85 * mag + 0.76 * np.log10(h) - 1.41
        return 0.85 * (0.6462 * (2.0 * 0.0376 * mlio55))



class MwI058(ConversionModel):
    INPUT = ["I0", "H"]
    def __call__(self, mag, h=1.0):
        ml =  0.72 * mag + 1.28 * np.log10(h) - 1.13
        mwml = MwML()
        return mwml(ml)

    def deriv(self, mag, h=1.0):
        """
        chain rule:  y = f(g(m))
        dy / dm = (df / dg) * (dg / dm)

        MwML: 0.5322  + 0.6462 * mag + 0.0376 * (mag ** 2.)
        """
        ml = 0.72 * mag + 1.28 * np.log10(h) - 1.13
        return 0.72 * (0.6462 + 2.0 * 0.0376 * ml)


class MwI067(ConversionModel):
    INPUT = ["I0", "H"]
    def __call__(self, mag, h=1.0):
        ml =  0.721 * mag + 1.283 * np.log10(h) - 1.13
        mwml = MwML()
        return mwml(ml)

    def deriv(self, mag, h=1.0):
        """
        chain rule
        MwML: 0.5322  + 0.6462 * mag + 0.0376 * (mag ** 2.)
        """
        ml =  0.721 * mag + 1.283 * np.log10(h) - 1.13
        return 0.721 * (0.6462 + 2.0 * 0.0376 * ml)


class MwMd63(ConversionModel):
    INPUT = "Md"
    def __call__(self, mag, h=1.0):
        return 1.472 * mag - 1.49

    def deriv(self, mag, h=1.0):
        return 1.472


class MwI092(ConversionModel):
    INPUT = "I0"
    def __call__(self, mag, h=1.0):
        ms = 0.63 * mag + 0.91
        msmw = MwMs54()
        return msmw(ms)

    def deriv(self, mag, h=1.0):
        """
        MwMs54: np.where(mag >= 5.4, 0.796 * mag + 1.280, 0.585 * mag + 2.422)
        """
        ms = 0.63 * mag + 0.91
        return 0.63 * np.where(ms >= 5.4, 0.796, 0.585) 


class MwMb(ConversionModel):
    INPUT = "mb"
    DESCR = "Utsu (2002) - mb"
    def __call__(self, mag, h=1.0):
        return np.where(mag < 6.584, 8.17 - np.sqrt(42.04 - 6.42 * mag), 8.17)

    def deriv(self, mag, h=1.0):
        return np.where(mag < 6.584, 6.42 / (2.0 * np.sqrt(42.04 - 6.42 * mag)), 0.0)
        

class MwML68(ConversionModel):
    INPUT = "ML"
    DESCR = "ML (LDG) -> ML,\nML -> Mw (Stromeyer et al., 2004)"
    def __call__(self, mag, h=1.0):

        ml = np.where(mag < 4.645, 1.31 * mag - 1.44, mag)
        mwml = MwMl()
        return mwml(ml)

    def deriv(self, mag, h = 1.0):
        """
        MwML: 0.5322  + 0.6462 * mag + 0.0376 * (mag ** 2.)
        """
        ml = np.where(mag < 4.645, 1.31 * mag - 1.44, mag)
        dmldm = np.where(mag < 4.645, 1.31, 1.0)
        return dmldm * (0.6462 + 2.0 * 0.0376 * ml)


class MwI0MER(ConversionModel):
    INPUT = "I0"
    DESCR = "I0 - Mw (FR) (Gruenthal et al., 2009)"
    def __call__(self, mag, h=1.0):
        return 0.682 * mag + 0.16

    def deriv(self, mag, h = 1.0):
        return 0.682


class MwI063(ConversionModel):
    INPUT = ["I0", "H"]
    def __call__(self, mag, h=1.0):
        mwml63 = MwML63()
        return mwml63(0.721 * mag + 1.283 * np.log10(h) - 1.13)

    def deriv(self, mag, h=1.0):
        """
        TO DO --- Need to check this

        MwML63:  0.906 * mag + 0.65

        """
        return 0.906 * 0.721


# ============ I got here with the derivatives! 25/02/2022

class MwML45(ConversionModel):
    INPUT = "ML"
    BOUNDS = [3., 5.6]
    DESCR = "GW2012 - IMO ML - Mw"
    def __call__(self, mag, h=1.0):
        return 0.74 * mag + 2.36

    def deriv(self, mag, h=1.0):
        return 0.74


class MwMM67(ConversionModel):
    INPUT = "ML"
    def __call__(self, mag, h=1.0):
        mwml = MwML67()
        return mwml(mag)

    def deriv(self, mag, h=1.0):
        """
        MwML67 - Don't know what this is!
        """
        # TODO
        return 1.0


class MwI004(ConversionModel):
    INPUT = ["I0", "H"]
    def __call__(self, mag, h=1.0):
        ml = 0.6 * mag + 1.8 * np.log10(h) - 1.0
        mlmw = MwML()
        return mlmw(ml)

    def deriv(self, mag, h=1.0):
        """
        MwML: 0.5322  + 0.6462 * mag + 0.0376 * (mag ** 2.)
        """
        ml = 0.6 * mag + 1.8 * np.log10(h) - 1.0
        return 0.6 * (0.6462 + 2.0 * 0.0376 * ml)


class MwI010(ConversionModel):
    INPUT = ["I0", "H"]

    def __call__(self, mag, h=1.0):
        ml = 0.696 * mag + 1.06 * np.log10(h) - 0.6
        mwml = MwML()
        return mwml(ml)

    def deriv(self, mag, h=1.0):
        """
        MwML: 0.5322  + 0.6462 * mag + 0.0376 * (mag ** 2.)
        """
        ml = 0.696 * mag + 1.06 * np.log10(h) - 0.6
        return 0.696 * (0.6462 + 2.0 * 0.0376 * ml)


class MwI021(MwI010):
    pass


class MwI033(ConversionModel):
    INPUT = ["I0", "H"]
    def __call__(self, mag, h=1.0):
        ml = 0.79 * mag + 1.19 * np.log10(h) - 1.44
        mwml = MwML()
        return mwml(ml)

    def deriv(self, mag, h=1.0):
        """
        MwML: 0.5322  + 0.6462 * mag + 0.0376 * (mag ** 2.)
        """
        ml = 0.79 * mag + 1.19 * np.log10(h) - 1.44
        return 0.79 * (0.6462 + 2.0 * 0.0376 * ml)


class MwI042(ConversionModel):
    INPUT = ["I0", "H"]
    def __call__(self, mag, h=1.0):
        ml = 0.81 * mag + 0.49 * np.log10(h) - 0.85
        mwml = MwML()
        return mwml(ml)

    def deriv(self, mag, h=1.0):
        """
        MwML: 0.5322  + 0.6462 * mag + 0.0376 * (mag ** 2.)
        """
        ml = 0.81 * mag + 0.49 * np.log10(h) - 0.85
        return 0.81 * (0.6462 + 2.0 * 0.0376 * ml)


class MwI044(ConversionModel):
    INPUT = "I0"
    def __call__(self, mag, h=1.0):
        msi044 = 0.55 * mag - 0.95
        msmw = MwMs()
        return msmw(msi044)

    def deriv(self, mag, h=1.0):
        """
        MwMs: 
        """
        msi = 0.55 * mag - 0.95
        return 0.55


class MwMd(ConversionModel):
    INPUT = "Md"
    def __call__(self, mag, h=1.0):
       # Assumes Md = ML then converts ML to Mw
       mwml = MwML()
       return mwml(mag)

    def deriv(self, mag, h=1.0):
       return 0.6462 + 2.0 * 0.0376 * mag
        

class MwML111(ConversionModel):
    INPUT = "ML"
    DESCR = "Glavatovic  - Slovenia"
    def __call__(self, mag, h=1.0):
        return 0.474 + 0.933 * mag

    def deriv(self, mag, h=1.0):
        return 0.933


class MwML111AL(ConversionModel):
    INPUT = "ML"
    DESCR = "Glavatovic  - Albania"
    def __call__(self, mag, h=1.0):
        return 2.029 + 0.656 * mag

    def deriv(self, mag, h=1.0):
        return 0.656


class MwML111CRO(ConversionModel):
    INPUT = "ML"
    DESCR = "Glavatovic  - Croatia"
    def __call__(self, mag, h=1.0):
        return 0.408 + 0.930 * mag

    def deriv(self, mag, h=1.0):
        return 0.930


class MwmbVC(ConversionModel):
    INPUT = "mb"
    DESCR = "mb-Mw Cabañas et al. (2015)"
    def __call__(self, mag, h=1.0):
        return -1.528 + 1.213 * mag
     
    def deriv(self, mag, h=1.0):
        return 1.213


class MwmbImaxIGN(ConversionModel):
    INPUT = "Imax"
    def __call__(self, mag, h=1.0):
        return 1.656 + 0.545 * mag

    def deriv(self, mag, h=1.0):
        return 0.545


class MwMLBSI(ConversionModel):
    INPUT = "ML"
    DESCR = "ML BSI - Mw Gasperini et al. (2013)"
    def __call__(self, mag, h=1.0):
        return -0.462 + 1.27 * mag 

    def deriv(self, mag, h=1.0):
        return 1.27


class MwMLCSI(ConversionModel):
    INPUT = "ML"
    DESCR = "ML CSI - Mw Gasperini et al. (2013)"

    def __call__(self, mag, h=1.0):
        return 0.302 + 0.985 * mag 

    def deriv(self, mag, h=1.0):
        return 0.985


class MwMLISIDE(ConversionModel):
    INPUT = "ML"
    DESCR = "ML ISIDE - Mw Gasperini et al. (2013)"

    def __call__(self, mag, h=1.0):
        return -0.165 + 1.066 * mag

    def deriv(self, mag, h=1.0):
        return 1.066


class MwMLBEO(ConversionModel):
    INPUT = "ML"
    DESCR = "ML (Seis. Surv. Serbia) - Mw"
    def __call__(self, mag, h=1.0):
        return 0.70 + 0.858 * mag

    def deriv(self, mag, h=1.0):
        return 0.858


class MwMLPDG(ConversionModel):
    INPUT = "ML"
    DESCR = "ML (Seis. Inst. Montenegro) - Mw"
    def __call__(self, mag, h=1.0):
        return -0.01 + 1.028 * mag

    def deriv(self, mag, h=1.0):
        return 1.028


class MwMLSKO(ConversionModel):
    INPUT = "ML"
    DESCR = "ML - Mw (Markusic et al. 2016) Skopje"
    def __call__(self, mag, h=1.0):
        return 0.56 + 0.913 * mag

    def deriv(self, mag, h=1.0):
        return 0.913


class MwMsISCWPG(ConversionModel):
    INPUT = "Ms"
    DESCR = "ISC Ms - Mw (Weatherill et al., 2016)"
    def __call__(self, mag, h=1.0):
        return np.where(mag <= 6.0, 0.616 * mag + 2.369, 0.995 * mag + 0.096)

    def deriv(self, mag, h=1.0):
        return np.where(mag <= 6.0, 0.616, 0.995)


class MwmbISCWPG(ConversionModel):
    INPUT = "mb"
    DESCR = "ISC mb - Mw (Weatherill et al., 2016)"

    def __call__(self, mag, h=1.0):
        return 1.084 * mag - 0.142

    def deriv(self, mag, h=1.0):
        return 1.084


CONVERSIONS = {
    "Equality": Equality,
    "Offset": Offset,
    "MwI004": MwI004,
    "MwI010": MwI010,
    "MwI021": MwI021,
    "MwI033": MwI033,
    "MwI042": MwI042,
    "MwI044": MwI044,
    "MwI055": MwI055,
    "MwI058": MwI058,
    "MwI063": MwI063,
    "MwI067": MwI067,
    "MwI074": MwI074,
    "MwI092": MwI092,
    "MwI0MER": MwI0MER,
    "MwML": MwML,
    "MwML111": MwML111,
    "MwML111AL": MwML111AL,
    "MwML111CRO": MwML111CRO,
    "MwML45": MwML45,
    "MwML63": MwML63,
    "MwML68": MwML68,
    "MwML90": MwML90,
    "MwML92": MwML92,
    "MwML96": MwML96,
    "MwMLBEO": MwMLBEO,
    "MwMLBSI": MwMLBSI,
    "MwMLCSI": MwMLCSI,
    "MwMLISIDE": MwMLISIDE,
    "MwMLLDG": MwMLLDG,
    "MwMLPDG": MwMLPDG,
    "MwMLSKO": MwMLSKO,
    "MwMM67": MwMM67,
    "MwMb": MwMb,
    "MwMb74a": MwMb74a,
    "MwMb74b": MwMb74b,
    "MwMd": MwMd,
    "MwMd63": MwMd63,
    "MwMs": MwMs,
    "MwMs29": MwMs29,
    "MwMs54": MwMs54,
    "MwMs69": MwMs69,
    "MwmbImaxIGN": MwmbImaxIGN,
    "MwmbVC": MwmbVC,
    "MwMsISCWPG": MwMsISCWPG,
    "MwmbISCWPG": MwmbISCWPG,
} 








#CONVERSIONS = {
#    "MwMs69": {"model": lambda ms: 0.995 * ms + 0.09,
#               "sigma": None},
#    "MwI074": {"model": lambda imax: 0.6 * imax + 0.96,
#               "sigma": None},
#    "MwMs54": {"model": lambda ms,
#         "sigma": None},
#    "": {"model": lambda,
#         "sigma": None},
#    "": {"model": lambda,
#         "sigma": None},
#    "": {"model": lambda,
#         "sigma": None},
#    "": {"model": lambda,
#         "sigma": None},
#    "": {"model": lambda,
#         "sigma": None},
#    "": {"model": lambda,
#         "sigma": None},
#    "": {"model": lambda,
#         "sigma": None},
#    "": {"model": lambda,
#         "sigma": None},
#    "": {"model": lambda,
#         "sigma": None},
#    "": {"model": lambda,
#         "sigma": None},
#    "": {"model": lambda,
#         "sigma": None},
#    "": {"model": lambda,
#         "sigma": None},
#    "": {"model": lambda,
#         "sigma": None},
#}


#class MagnitudeHarmoniser():
#    """
#    """
#    def __init__(self, catalogue_file):
#        """
#        """
#        origins = pd.read_hdf(catalogue_file, "origins")
#        self.origins = origins.groupby("master_id")
#        magnitudes = pd.read_hdf(catalogue_file, "magnitudes")
#        self.magnitudes = magnitudes.groupby("master_id")
#        self.orig_ids = list(self.origins.groups)
#        self.mag_ids = list(self.magnitudes.groups)
#        assert len(self.orig_ids) == len(self.mag_ids)
#
#    def __len__(self):
#        # Return the number of events
#        return len(self.orig_ids)
#
#    def __iter__(self):
#        # Iterate over the origins and magnitudes
#        for i_d in self.orig_ids:
#            orig = self.origins.get_group(i_d)
#            mag = self.magnitudes.get_group(i_d)
#            yield orig, mag
#
#    def harmonize(self, hierarchy):
#        pass
        
