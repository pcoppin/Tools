# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

### In a notebook
#from pathlib import Path
#pwd = Path().resolve()
### In a python script
#import os
#pwd = os.path.dirname(os.path.realpath(__file__))
### Then
#Code_foler = str(pwd.parent)
#sys.path.insert(0, Code_folder+"/Tools")
#import Tools test

pwd = os.path.dirname(os.path.realpath(__file__)) + "/"

sr_to_deg2        = (180/np.pi)**2
Sky_in_square_deg = 4*np.pi * sr_to_deg2
Code_folder       = "/mnt/2690D37B90D35043/PhD/Code"
mpl_style_file    = pwd + "matplotlib_style"

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:pink', 'tab:grey']

def logspace(min_range, max_range, counts):
    return np.logspace(np.log10(min_range), np.log10(max_range), counts)

def Reduced_chi_squared(function, x_data, y_data, y_error, params):
    from scipy.stats import chi2
    if(len(x_data)!=len(y_data)):
        raise Exception("X & Y array have different lengths")
    Ndog = len(x_data) - len(params)
    CHI2 = np.sum( np.power(y_data-function(x_data, *params),2) / np.power(y_error,2) )
    p_value = 1 - chi2.cdf(CHI2,Ndog)
    return CHI2/Ndog, p_value

def Multi_split(string, splits):
    res = [string]
    for split in splits:
        res_new = []
        for i in res:
            for value in i.split(split):
                res_new.append(value)
        res = res_new
    while( "" in res):
        res.remove("")
    return res

def Print_name_input_arguments(a, f, b):
    import inspect
    # Prints the names of the input variables
    frame = inspect.currentframe()
    frame = inspect.getouterframes(frame)[1]
    string = inspect.getframeinfo(frame[0]).code_context[0].strip()
    args = string[string.find('(') + 1:-1].split(',')
    names = []
    for i in args:
        if i.find('=') != -1:
            names.append(i.split('=')[1].strip())
        else:
            names.append(i)
    print(names)

def distance_point_to_line(point,line):
    x         = np.array(point)
    y         = np.array(line[:3])
    theta     = line[3]
    phi       = line[4]
    direction = np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
    t_min     = sum((y-x)*direction)
    d_min_vec = x+t_min*direction-y
    return np.sqrt(sum(d_min_vec*d_min_vec))

def DmpTrigger(TriggerStat, idx, MC=None):
    # TriggerStat: dc.GetDmpEvent(i).pEvtHeader().GetTriggerStatus()
    # Alternatively: Header.GeneratedTrigger(idx) ;  Header.EnabledTrigger(idx)
    # idx: 0 (unbiased), 1 & 2 (MIP), 3 (high energy), 4 (low energy)
    TriggerStatBin = format(TriggerStat,"b")[::-1]
    if( MC is None ):
        raise Exception("Specify MC as 'True' or 'False'")
    elif( MC ):
        # Generated is always True for MC, except for MIP, for which it's always False
        enabled   = 1
    else:
        enabled   = int(TriggerStatBin[idx])
    triggered = int(TriggerStatBin[8+idx])
    return enabled * triggered

def Efficiency_for_binomial_process(N,k,probability_content=0.683):
    """
    Compute the efficiency and (left & right) uncertainty for a binomial process,
       e.g. when computing the ratio between two histogram. The method in this
       function is based on https://home.fnal.gov/~paterno/images/effic.pdf

    :type   N: Integer
    :param  N: Total number of trials

    :type   k: Integer
    :param  k: Total number of successes out of the N trials

    :type   probability_content: float
    :param  probability_content: probability content contained by the
                                 uncertainty interval that his returned

    """
    if( isinstance(N,(np.int32,np.int64)) ):
        N = int(N)
    if( isinstance(k,(np.int32,np.int64)) ):
        k = int(k)
    
    if( (N>1000) and (k>50) and ((N-k)>50) ):
        from scipy.stats import norm
        sigma_gauss = norm.isf( 0.5*(1-probability_content) )
        eff = float(k)/N
        p = lambda x,y : np.power(x,y,dtype=float) # Numpy int's overflow without warning...
        sigma = sigma_gauss * np.sqrt( k/p(N,2) + p(k,2)/p(N,3) )
        if( (eff>sigma) and ((1-eff)>sigma) ):
            return np.array([eff, sigma, sigma])
    
    from decimal import Decimal
    if( (probability_content>=1) or (probability_content<=0) ):
        raise Exception("The probability content must be between 0 and 1!")
    # Function to quickly compute large binomial coefficients (exact)
    # Borrowed from: https://grocid.net/2012/07/02/quick-and-dirty-way-to-calculate-large-binomial-coefficients-in-python/
    def pow_binomial(n, k):
        def eratosthenes_simple_numbers(N):
            yield 2
            nonsimp = set()
            for i in range(3, N + 1, 2):
                if i not in nonsimp:
                    nonsimp |= {j for j in range(i * i, N + 1, 2 * i)}
                    yield i
        def calc_pow_in_factorial(a, p):
            res = 0
            while a:
                a //= p
                res += a
            return res
        ans = 1
        for p in eratosthenes_simple_numbers(n):
            ans *= p ** (calc_pow_in_factorial(n, p) - calc_pow_in_factorial(k, p) - calc_pow_in_factorial(n - k, p))
        return ans
    # Binomial function
    def Binomial_function(N,k,epsilon):
        if( (k==0) and (epsilon==Decimal(0.0)) ):
            return Decimal(1)
        elif( (k==N) and (epsilon==Decimal(1.0)) ):
            return Decimal(1)
        else:
            return Decimal(np.power(epsilon,k)*np.power(1-epsilon,N-k))
    # Find the limits
    if(k>N):
        raise Exception("Efficiency larger than one!")
    prefactor        = Decimal(N+1)*Decimal(pow_binomial(N,k))
    center           = Decimal(k)/N
    left_limit       = Decimal(center)
    right_limit      = Decimal(center)
    surface_to_left  = Decimal(0.0)
    surface_to_right = Decimal(0.0)
    steps_taken      = 0
    while( (surface_to_left+surface_to_right)<probability_content ):
        if( ((right_limit==1) or (surface_to_left<=surface_to_right)) and (not (left_limit<1e-20) )):
            value_binomial_function = prefactor*Binomial_function(N,k,left_limit)
            if(value_binomial_function>0.0):
                stepsize         = min(Decimal(0.01)/(prefactor*Binomial_function(N,k,left_limit)),Decimal(0.01))
            if(stepsize<1e-6):
                stepsize = Decimal(1e-6)
            if( (left_limit-stepsize)<0 ):
                stepsize = left_limit
            surface_to_left += prefactor*stepsize*Decimal(0.5)*(Binomial_function(N,k,left_limit)+Binomial_function(N,k,left_limit-stepsize))
            left_limit = left_limit-stepsize

        else:
            value_binomial_function = prefactor*Binomial_function(N,k,right_limit)
            if(value_binomial_function>0.0):
                stepsize         = min(Decimal(0.01)/(prefactor*Binomial_function(N,k,right_limit)),Decimal(0.01))
            if(stepsize<1e-6):
                stepsize = Decimal(1e-6)
            if( (right_limit+stepsize)>1.0 ):
                stepsize = 1-right_limit
            surface_to_right += prefactor*stepsize*Decimal(0.5)*(Binomial_function(N,k,right_limit)+Binomial_function(N,k,right_limit+stepsize))
            right_limit = right_limit + stepsize

        steps_taken += 1
    return np.array([float(center), abs(float(left_limit)-float(center)), float(right_limit)-float(center)])

def Angular_seperation(pos_1, pos_2, ra_dec=False, degrees=True):
    """
    Compute the angular seperation between point A & B on a sphere
    theta & phi can be arrays, allowing one function call to calculate the seperation between multiple points

    :type   pos_1: array or list
    :param  pos_1: (theta, phi) in spherical coordinates of point A OR (dec, ra)

    :type   pos_2: array or list
    :param  pos_2: (theta, phi) in spherical coordinates of point B OR (dec, ra)

    :type   ra_dec: boolean
    :param  ra_dec: False if pos_1/2 are in spherical coordinates, True if they are in ra & dec

    :type   degrees: boolean
    :param  degrees: False if pos_1/2 are in degrees, True if they are radians
    """
    def Spherical_to_cartesion(theta, phi):
        x = np.sin(theta)*np.cos(phi)
        y = np.sin(theta)*np.sin(phi)
        z = np.cos(theta)
        return x, y ,z
    
    if( degrees ):
        theta_1, phi_1 = np.radians(pos_1)
        theta_2, phi_2 = np.radians(pos_2)
    else:
        theta_1, phi_1 = np.array(pos_1)
        theta_2, phi_2 = np.array(pos_2)
    
    if( ra_dec ):
        theta_1 = 0.5*np.pi - theta_1
        theta_2 = 0.5*np.pi - theta_2
    
    x1, y1, z1 = Spherical_to_cartesion(theta_1, phi_1)
    x2, y2, z2 = Spherical_to_cartesion(theta_2, phi_2)
    inproduct = x1*x2+y1*y2+z1*z2
    inproduct = np.clip(inproduct, -1, 1) # Account for numerical precision
    ang_sep = np.arccos(inproduct)
    if( degrees ):
        return np.degrees(ang_sep)
    else:
        return ang_sep

def Generate_uniform_points_on_sphere(N, ra_dec=False, degrees=False):
    """
    Generate N points on the unit sphere
    Based on https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere

    :type   N: integer
    :param  N: Number of points to be generated

    :type   ra_dec: boolean
    :param  ra_dec: False if pos_1/2 are in spherical coordinates, True if they are in ra & dec

    :type   degrees: boolean
    :param  degrees: False if pos_1/2 are in degrees, True if they are radians
    """
    indices = np.arange(0, N, dtype=float) + 0.5
    theta = np.arccos(1 - 2*indices/N)
    phi = (np.pi * (1 + 5**0.5) * indices)%(2*np.pi)
    if( ra_dec ):
        theta = np.pi/2 - theta
    if( degrees ):
        return np.degrees(theta), np.degrees(phi)
    else:
        return theta, phi

def Moyal(x, *p):
    '''
    A good enough approximation to the Landau function
    that allows to estimate the MPV and sigma of the peak
    '''
    xt = (x - p[1]) / p[2]
    f = np.exp(-(xt + np.exp(-xt)) / 2)
    return p[0] * f / f.max()

def Landau(x, *p):
    '''
    Exact landau distribution (numerically integrated)
    I find that dt=0.1 and infinity=100 gives good enough approximation
    '''
    xt = (x - p[0]) / p[1]
    tt = np.linspace(0.0001, 100, 2000) # Avoid zero
    dt = tt[1]-tt[0]
    t = np.meshgrid(np.ones(len(x)), tt)[1]
    A = (1/np.pi*np.exp(-t * np.log(t) - xt * t) * np.sin(np.pi * t)).sum(axis=0)
    A[x < p[0] - 3 * p[1]] = 0.
    #A[A<0] = 0.0
    #A[np.isnan(A)] = 0.0
    return A * dt / p[1]#/ (A * dt).max()

def Gauss(x, *p):
    xt = (x - p[0]) / p[1]
    norm = 1/np.sqrt(2*np.pi*p[1]**2)
    return norm*np.exp(-0.5 * xt**2)

def Langau(x, *p):
    '''
    Landau distribution convolved with gaussian noise
    '''
    tau_arr = np.linspace(x.min() - x.mean(), 
                          x.max() - x.mean(), len(x))
    c = np.convolve(Landau(x, p[0], p[1]),
                    Gauss(tau_arr, 0, p[2]))
    xnew = np.linspace(x.min() + tau_arr.min(),
                       x.max() + tau_arr.max(),
                       len(x) + len(tau_arr) - 1)
    f = np.interp(x, xnew, c)
    f = f * (x[1] - x[0])
    return f

def kent(s, dpsi):
    """
    Compute the value of the Kent distribution

    :type   s: float
    :param  s: sigma or the angular uncertainty (in radians)

    :type   dpsi: float or np.array
    :param  dpsi: angular seperation (in radians)
    """
    if( s<0.105 ):
        # return Gaussian pdf if s<6 degrees to avoid overflows of sinh
        return Two_dim_Gaussian(s, dpsi)
    else:
        ssi = np.power(s,-2)
        norm = ssi/(4*np.pi*np.sinh(ssi))
        return norm*np.exp(np.cos(dpsi)*ssi)

def Two_dim_Gaussian(s, dpsi):
    """
    Compute the value for a 2 dimensional Gaussian distribution

    :type   s: float
    :param  s: sigma or the angular uncertainty (in radians)

    :type   dpsi: float
    :param  dpsi: angular seperation (in radians)
    """
    norm = 1.0/(2*np.pi*s**2)
    return norm*np.exp(-0.5*(dpsi/s)**2)

def kent_healpix(src_ra, src_dec, s, nside, allowed_error=0.05):
    """
    Generate a healpix for a given source following a Kent distribution

    :type   src_ra: float
    :param  src_ra: Right ascension of the source (in radians)

    :type   src_dec: float
    :param  src_dec: Declination of the source (in radians)

    :type   s: float
    :param  s: sigma or the angular uncertainty (in radians)

    :type   dpsi: integer
    :param  dpsi: nside of the healpix map
    """
    import healpy as hp
    
    npix = hp.nside2npix(nside)
    pixels = np.arange(npix)
    theta, phi = hp.pix2ang(nside, pixels)
    pos_1 = [np.pi/2-src_dec, src_ra]
    #dpsi = [Angular_seperation(pos_1, [t,p], degrees=False) for t,p in zip(theta,phi)]
    #m = [kent(s, dpsi_i)*4*np.pi/npix for dpsi_i in dpsi]
    
    dpsi = Angular_seperation(pos_1, [theta,phi], degrees=False)
    m = kent(s, dpsi)*4*np.pi/npix
    
    sum_m = sum(m)
    if( (sum_m<(1-allowed_error)) or (sum_m>(1+allowed_error)) ):
        raise Exception("Normalisation of the map deviates from unity: {}".format(sum_m))
    else:
        # renormalize due to 'rough' integration
        m /= sum_m
    return np.array(m)

def union_continuous_intervals (intervals):
    '''
    Calculate the union of a given list of continuous intervals
    '''
    sorted_intervals = sorted(intervals, key=lambda x: x[1])
    union_sorted_intervals = []
    for start, stop in sorted_intervals:
        if union_sorted_intervals and union_sorted_intervals[-1][1] >= start:
            union_sorted_intervals[-1][1] = max(union_sorted_intervals[-1][1], stop)
        else:
            union_sorted_intervals.append([start, stop])
    return union_sorted_intervals

def intersection_time_intervals (intervals_1, intervals_2):
    '''
    Calculate intersections between two sorted lists containing continuous, non-overlapping intervals
    '''
    res     = []
    for start, stop in intervals_1:
        for start_2, stop_2 in intervals_2:
            if( stop_2<start ):
                continue
            elif( start_2<stop ):
                res.append( [max(start, start_2), min(stop, stop_2)] )
            else:
                break
    return res

def pol_fun(x, *c):
    # c: coefficients, i.e. p[2]*x^2 + p[1]*x + p[0] etc.
    res = np.zeros((len(x)))
    for i in range(len(c)):
        res = res + c[i]*np.power(x, i)
    return res

def Volume_sphere(radius):
    return np.ceil(np.pi)/np.floor(np.pi)*np.pi*np.power(radius,np.floor(np.pi))

def mjd_to_year(mjd):
    """
    Conver the mjd value to the year (rough method)

    :type   mjd: array or list
    :param  mjd: Modified Julian Data values
    """
    mjd_2010_01_01 = 55197
    days_per_years = 365.2422
    return (np.array(mjd)-mjd_2010_01_01)/days_per_years + 2010

def toYearFraction(date):
    """
    Convert a datetime to the year (in float format)

    :type   date: datetime.datetime
    :param  date: date to be converted
    """
    import datetime, time
    dt = datetime.datetime
    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction

def Track_for_loop_progress(iterator, len_iterator, message=None):
    """
    Track the progress of a for loop using print statements

    :type   iterator: int
    :param  iterator: Must run from 0 to len_iterator-1
    
    :type   len_iterator: int
    :param  len_iterator: Number of times the loop is called
    
    :type   message: str
    :param  message: additional output to be printed
    """
    import timeit
    import sys
    from IPython.display import clear_output
    if( iterator==0 ):
        global Track_for_loop_progress_start
        Track_for_loop_progress_start = timeit.default_timer()
    stop = timeit.default_timer()
    if( (iterator/len_iterator)<0.05 ):
        expected_time = 0.0
    else:
        time_perc = timeit.default_timer()
        expected_time = np.round((time_perc-Track_for_loop_progress_start)/((iterator+1)/len_iterator), 2)
    clear_output(wait=True)
    print("Current progress:", np.round( (iterator+1)/len_iterator*100, 2), "%")
    print("Current run time:", int((stop-Track_for_loop_progress_start)/60),"min", int((stop-Track_for_loop_progress_start)%60), "s")
    print("Expected run time:", int(expected_time/60),"min", int(expected_time%60), "s")
    if( message ):
        print(message)
    sys.stdout.flush()
    if( iterator==(len_iterator-1) ):
        del Track_for_loop_progress_start

def Print_rounded(a, b):
    '''
    # Print a+-b rounded to 2 significant digits of the uncertainty
    
    :type   a: float
    :param  a: Value of a single measurement
    
    :type   b: float
    :param  b: Uncertainty on the measurement of a
    '''
    log_base = int(np.floor(np.log10(b)))
    if( log_base==0 ):
        return r"{:.1f} +-{:4.1f}".format(a,b)
    else:
        x, y = np.array([a,b])/np.power(10.0, log_base)
        return r"( {:.1f} +-{:4.1f} ) x 10^{:d}".format(x, y, log_base)
    return log_base

def Plot_skymap(prior):
    import healpy as hp
    nside    = prior.nside
    N_pixels = prior.npix
    pixel_area = Sky_in_square_deg/N_pixels
    Pixel_pos = hp.pix2ang(nside, np.arange(N_pixels))
    Pixel_pos_zenith, Pixel_pos_RA = Pixel_pos
    Pixel_pos_DEC = np.pi/2-Pixel_pos_zenith
    weights = prior.map / pixel_area
    fig = plt.figure(figsize=(15,7.5))
    subpl = plt.subplot(111, projection='mollweide')
    max_weight = max(weights)
    sc = subpl.scatter( np.pi-Pixel_pos_RA, Pixel_pos_DEC, c=[max(max_weight*1e-8, w) for w in weights], norm=matplotlib.colors.LogNorm(vmin=max_weight*1e-7, clip=True) )
    Fermi_ticks = np.array(range(360,-1 ,-30))
    Fermi_ticks_l = list(Fermi_ticks)
    Fermi_ticks_l[0]  = ""
    Fermi_ticks_l[-1] = ""
    for i in range(1,len(Fermi_ticks_l)-1):
        Fermi_ticks_l[i] = str(Fermi_ticks_l[i])+"°"
    xticks_pos    = np.pi - Fermi_ticks/180*np.pi
    xticks_labels = list(range(360,-1 ,-30))
    plt.xticks(ticks=xticks_pos, labels=Fermi_ticks_l, color="white")
    plt.colorbar(sc, label="Probability per square degree")
    plt.close()
    return fig

class Hist(object):
    def __init__(self, data, log=False, height=False, **kw):
        if( log ):
            if( "bins" not in kw ):
                kw["bins"] = 10
            if( isinstance(kw["bins"],int) ):
                eps = 1.01
                kw["bins"] = np.logspace(np.log10(min(data)/eps), 
                                         np.log10(max(data)*eps),
                                         kw["bins"])
        elif( ("bins" in kw) and isinstance(kw["bins"],str) and (kw["bins"]=="int") ):
            floor = int(min(data)+0.5-1e-6) - 0.5
            ceiling = int(max(data)-0.5+1e-6) + 1.5
            kw["bins"] = np.linspace(floor, ceiling, int(ceiling-floor)+1)
        
        if( height ):
            if( log or ("bins" not in kw) ):
                raise Exception("Bins must be specified explicitly when using height=True")
            self.hist, self.bins = np.array(data), np.array(kw["bins"])
        else:
            self.hist, self.bins = np.histogram(data, **kw)
        self.bin_width = self.bins[1:] - self.bins[:-1]
        self.bin_center = self.bins[:-1] + 0.5*self.bin_width
        self.set_vars()
        
    def set_vars(self):
        self.sum = np.sum(self.hist)
        self.hist_density = self.hist/self.bin_width
        self.hist_normed = self.hist/float(self.sum)
        self.pdf = self.hist_normed/self.bin_width
        self.cdf = np.minimum(1,self.hist_normed.cumsum())
        self.binomial_uncertainty = []
        if( all(self.bins>0) ):
            bins_ratio = self.bins[1:]/self.bins[:-1]
            self.log = all(np.isclose(bins_ratio,bins_ratio[0]))
        else:
            self.log = False

    def Add_counts(self, idx, counts):
        self.hist[idx] = self.hist[idx] + counts
        self.set_vars()
    
    def set_errors(self, probability_content = 0.683):
        for N_i in self.hist:
            self.binomial_uncertainty.append( self.sum*Efficiency_for_binomial_process(self.sum, int(N_i), probability_content)[1:] )
    
    def plot(self, type="hist", ax=plt, step=False, **kw):
        if( type=="hist" ):
            height = self.hist
            ylabel = "Counts"
        elif( type=="cumsum" ):
            height = self.hist.cumsum()
            ylabel = "Cumulative counts"
        elif( type=="normed" ):
            height = self.hist_normed
            ylabel = "Counts normed per bin"
        elif( type=="density" ):
            height = self.hist_density
            ylabel = "Density"
        elif( type=="pdf" ):
            height = self.pdf
            ylabel = "PDF"
        elif( type=="cdf" ):
            height = self.cdf
            ylabel = "CDF"
        elif( type=="inv_cdf" ):
            height = 1-self.cdf
            ylabel = '1-CDF'
        else:
            raise Exception("Unknown histogram type specified: {}. Please selected 'hist' (default), 'normed', 'density', 'pdf', 'cdf' or 'inv_cdf'.")
        
        if( type=="cdf" ):
            y = [0] + list(self.cdf)
            plot = ax.step(self.bins, y, where='post', **kw)
        elif( step ):
            y = list(height) + [height[-1]]
            plot = ax.step(self.bins, y, where='post', **kw)
            #return plot
        else:
            mask = height>0
            plot = ax.bar(self.bins[:-1][mask], height[mask], self.bin_width[mask], align='edge', **kw)
        ax.ylabel(ylabel)
        if( self.log ):
            ax.xscale("log")
        return plot
    
    def plot_errorbar(self, ax=plt, type="hist", linestyle="None", fmt=".", markersize=10, **kw):
        if( len(self.binomial_uncertainty)==0 ):
            self.set_errors()
        if( type=="hist" ):
            height = self.hist
            uncertainty = self.binomial_uncertainty
        elif( type=="pdf" ):
            height = self.pdf
            uncertainty = [uncertainty/self.sum/bin_width for (uncertainty,bin_width) in zip(self.binomial_uncertainty,self.bin_width)]
        return ax.errorbar(self.bin_center, height, list(zip(*uncertainty)), linestyle=linestyle, fmt=fmt, markersize=markersize, **kw)
    
    
    

    
    
### DAMPE stuff


def PSD_charge_fit_old(x, a, b, c, d):
    # Sum of a linear function and an exponential
    # Similar to Misha, he uses a + exp(b * x + c)
    return np.exp(a*(x-b)) + c + d*x

def PSD_charge_fit(loge, *p):
    return p[0] + p[1] * loge**p[2] + p[3]*loge**p[4]

def Reweight_events(z_stop, corr, nbins=1000, z_leftmost=-380, z_rightmost=480):
    """
    Generate weights for MC events to make it such that the amount of events that interact
    at any given depth (z-value) corresponds to what you would expect for simulation in
    which the cross section is rescaled
    
    :type   z_stop: Array
    :param  z_stop: List containing the 'z_stop' values of the simulated events

    :type   corr: Float
    :param  corr: Factor by which the cross section is rescaled, i.e. 1.2 corresponds to a 20% increase
    
    :type   nbins: Int
    :param  nbins: Number of bins used to divide the z-range of the detector

    :type   z_leftmost: float
    :param  z_leftmost: Defines the z-range over which the cross section will be scaled
    
    :type   z_rightmost: float
    :param  z_rightmost: Defines the z-range over which the cross section will be scaled
    """
    from scipy.interpolate import interp1d
    
    ### Bins spam the detector region from the top of PSD (z=-380) to the bottom of BGO (z=450)
    bins = np.linspace(-380, 450, nbins+1)
    bin_center = bins[:-1] + 0.5*np.diff(bins)

    ### 'w': bins (indexes) for which the bin center falls inside the z-range
    w = (bin_center>z_leftmost) * (bin_center<z_rightmost)
    ### We also calculate the indexes of the corresponding bin-edges
    w_bins = np.zeros(nbins+1, dtype=bool)
    w_bins[np.where(w)[0]] = True
    w_bins[np.where(w)[0][-1]+1] =  True

    ### Put everything in z-range into a histogram
    h = Hist(z_stop, bins=bins[w_bins])
    ### Things to the right of the z-range get put in the last bin. We do this because the weight of
    ### the remaining events is affected by the fact that more (or less events) interacted beforehand.
    h.Add_counts(-1, np.sum(z_stop>z_rightmost))
    
    cdf_normal = np.array([0] + list(h.cdf))
    cdf_rescaled = 1 - np.power(1-cdf_normal, corr)


    ### These pdf values have as bins
    pdf_normal = np.diff(cdf_normal)/np.diff(h.bins)
    pdf_rescaled = np.diff(cdf_rescaled)/np.diff(h.bins)
    
    ### The weights (number of interacting events) corresponds to the ratio of the PDFs
    weights = np.ones(nbins)
    weights[w] = pdf_rescaled / np.maximum(pdf_normal,1e-20)

    ### In bins in the z-range, in which no particle stopped, we set the weight
    ###    equal to that in the nearest bin
    w_zero = (weights==0)
    w_interp = ~w_zero[:]
    w_interp[np.where(w)[0][-1]] = False
    weights[w_zero*w] = interp1d(np.arange(len(weights))[w*w_interp], weights[w*w_interp], kind='nearest', fill_value='extrapolate')( np.where(w_zero*w)[0] )
    
    ### All bins to the right of the z-range now get the same re-scaled value as the weight of the last bin
    w_right = bin_center>z_rightmost
    weights[w_right] = weights[w][-1]

    return [h, pdf_normal, pdf_rescaled, cdf_normal, cdf_rescaled, weights, bins, bin_center]


def Combine_npy_dict(Filelist=[], keys=[],\
                     filters=['HE_trigger','NonZeroPSD','MLcontainmentBGO','MLcontainmentSTK','skim'],\
                     npy_dir="/dpnc/beegfs/users/coppinp/Simu_vary_cross_section_with_Geant4/Analysis/npy_files/",):
    import copy
    keys = copy.deepcopy(keys)

    data = {}
    for key in filters:
        if( key=="MLcontainmentBGO" ):
            ToAdd = ['BGOInterceptY', 'BGOInterceptX','BGOSlopeX', 'BGOSlopeY']
            for key_ToAdd in ToAdd:
                keys.append( key_ToAdd )
        elif( key=="MLcontainmentSTK" ):
            ToAdd = ['STKInterceptY', 'STKInterceptX','STKSlopeX', 'STKSlopeY']
            for key_ToAdd in ToAdd:
                keys.append( key_ToAdd )
        elif( key=='TrueContainment' ):
            keys.append( 'TrueContainment' )
            # ToAdd = ['start_x','start_y','start_z','stop_x','stop_y','stop_z']
            # for key_ToAdd in ToAdd:
            #     keys.append( key_ToAdd )
        elif( key=="NonZeroPSD" ):
            keys.append( "PSD_charge" )
        elif( key not in keys ):
            keys.append( key )
    keys = np.unique(keys)

    for f in Filelist:
        #print(f)
        data_i = np.load(npy_dir+f, allow_pickle=True, encoding="latin1").item()
        N_i = len(data_i['E_p']) + int( 10 * len(data_i['E_primary_non_trig']) )
        weight = (1.0/N_i) * np.ones( len(data_i['E_p'])  )
        if( "10GeV_to_10TeV" in f ):
            # Weight normally 1 per decade, so total weight is 3 if adding file that spans 3 decades
            weight *= 3
        if( "100TeV_500TeV" in f ):
            weight *= np.log10(5)
        data_i["weight"] = weight
        # Use all keys if not specified
        if( len(keys)==0 ):
            keys = data_i.keys()
        # Put data in dictionary
        for key in keys:
            if( key not in data ):
                data[key] = data_i[key]
            else:
                data[key] = np.concatenate([data[key],data_i[key]])
    for key in ["E_p","E_total_BGO", "E_total_PSD", "E_primary_non_trig"]:
        if( key in data ):
            data[key] = 1e-3 * data[key]
    
    # w = data['HE_trigger'] * data["Skimmed"]
    w = np.ones_like(data["E_total_BGO"], dtype=bool)
    for key in filters:
        if( key=='MLcontainmentBGO' ):
            # BGO prediction fiducially contained in BGO
            BGO_TopZ, BGO_BottomZ = 46., 448.
            cut = 280
            topX = data['BGOSlopeX'] * BGO_TopZ + data['BGOInterceptX']
            topY = data['BGOSlopeY'] * BGO_TopZ + data['BGOInterceptY']
            bottomX = data['BGOSlopeX'] * BGO_BottomZ + data['BGOInterceptX']
            bottomY = data['BGOSlopeY'] * BGO_BottomZ + data['BGOInterceptY']
            ml_bgo_fid = (abs(topX)<cut) * (abs(topY)<cut) * (abs(bottomX)<cut) * (abs(bottomY)<cut)
            w = w * ml_bgo_fid
            # Additional requirement: Make sure STK track is trustworthy
            TopZ = -210.
            cutTop = 500.
            topX = data['BGOSlopeX'] * TopZ + data['BGOInterceptX']
            topY = data['BGOSlopeY'] * TopZ + data['BGOInterceptY']
            w = w * (abs(topX)<cutTop) * (abs(topY)<cutTop)


        elif( key=='MLcontainmentSTK' ):
            # STK prediction fiducially contained within PSD to BGO
            TopZ, BottomZ = -325, 448.
            cutTop, cutBottom = 440, 280
            topX = data['STKSlopeX'] * TopZ + data['STKInterceptX']
            topY = data['STKSlopeY'] * TopZ + data['STKInterceptY']
            bottomX = data['STKSlopeX'] * BottomZ + data['STKInterceptX']
            bottomY = data['STKSlopeY'] * BottomZ + data['STKInterceptY']
            ml_stk_fid = (abs(topX)<cutTop) * (abs(topY)<cutTop) * (abs(bottomX)<cutBottom) * (abs(bottomY)<cutBottom)
            TopZ = 44.
            cutTop = 280.
            topX = data['STKSlopeX'] * TopZ + data['STKInterceptX']
            topY = data['STKSlopeY'] * TopZ + data['STKInterceptY']
            ml_stk_fid = ml_stk_fid * (abs(topX)<cutTop) * (abs(topY)<cutTop)
            w = w * ml_stk_fid
        # elif( key=='TrueContainment' ):
        #     TopZ, BottomZ = -325, 448.
        #     cutTop, cutBottom = 440, 280
        #     topX = (data['stop_x']-data['start_x'])/(data['stop_z']-data['start_z'])*(TopZ-data['start_z']) + data['start_x']
        #     topY = (data['stop_y']-data['start_y'])/(data['stop_z']-data['start_z'])*(TopZ-data['start_z']) + data['start_y']
        #     bottomX = (data['stop_x']-data['start_x'])/(data['stop_z']-data['start_z'])*(BottomZ-data['start_z']) + data['start_x']
        #     bottomY = (data['stop_y']-data['start_y'])/(data['stop_z']-data['start_z'])*(BottomZ-data['start_z']) + data['start_y']
        #     w_fid = (abs(topX)<cutTop) * (abs(topY)<cutTop) * (abs(bottomX)<cutBottom) * (abs(bottomY)<cutBottom)
        #     ### REQUIRE THEM ALSO TO GO THROUGH THE TOP LAYER OF BGO!!!
        #     BottomZ = 44.
        #     cutBottom = 280
        #     bottomX = (data['stop_x']-data['start_x'])/(data['stop_z']-data['start_z'])*(BottomZ-data['start_z']) + data['start_x']
        #     bottomY = (data['stop_y']-data['start_y'])/(data['stop_z']-data['start_z'])*(BottomZ-data['start_z']) + data['start_y']
        #     w_fid = w_fid * (abs(bottomX)<cutBottom) * (abs(bottomY)<cutBottom)
        #     w = w * w_fid
        elif( key=="NonZeroPSD" ):
            w = w * np.sum(data['PSD_charge']>0.1, axis=1, dtype=bool)
        else:
            w = w * data[key]
    for key in data:
        if( key=="E_primary_non_trig" ):
            continue
        elif( key=="E_p"):
            if( "E_primary_non_trig" in data ):
                data['E_primary_non_trig'] = np.concatenate( [data['E_primary_non_trig'],data['E_p'][~w][::10]] )
        data[key] = data[key][w]

    if( "PSD_charge" in data ):
        data["PSD_charge"] = np.maximum( data['PSD_charge'], 0.0 )

    return data

Proton_filelist = ["allProton-v6r0p10_10GeV_100GeV_FTFP-p2.npy",\
                   "allProton-v6r0p10_100GeV_1TeV_FTFP-p1.npy",\
                   "allProton-v6r0p10_1TeV_10TeV_FTFP.npy",\
                   "allProton-v6r0p10_10TeV_100TeV_FTFP.npy",\
                   "allProton-v6r0p12_100TeV_1PeV_EPOSLHC_FTFP_BERT.npy"]
ProtonFluka_filelist =["allProton-v6r0p15_10GeV_100GeV-FLUKA.npy",\
                       "allProton-v6r0p15_100GeV_1TeV-FLUKA.npy",\
                       "allProton-v6r0p15_1TeV_10TeV-FLUKA.npy",\
                       "allProton-v6r0p15_10TeV_100TeV-FLUKA.npy",\
                       "allProton-v6r0p15_100TeV_1PeV-FLUKA.npy"]
Helium_filelist = ["allHe4-v6r0p10_10GeV_100GeV_FTFP.npy",\
                   "allHe4-v6r0p10_100GeV_1TeV_FTFP.npy",\
                   "allHe4-v6r0p10_1TeV_10TeV-FTFP.npy",\
                   "allHe4-v6r0p10_10TeV_100TeV-FTFP.npy",\
                   "allHe4-v6r0p10_100TeV_500TeV-EPOSLHC.npy"]
HeliumFluka_filelist = ["allHe4-v6r0p10_10GeV_100GeV-FLUKA.npy",\
                        "allHe4-v6r0p10_100GeV_1TeV-FLUKA.npy",\
                        "allHe4-v6r0p10_1TeV_10TeV-FLUKA.npy",\
                        "allHe4-v6r0p10_10TeV_100TeV-FLUKA-p1.npy",\
                        "allHe4-v6r0p10_100TeV_500TeV-FLUKA.npy"]
Lithium7_filelist = ["allLi7-v6r0p10_10GeV_100GeV_QGSP.npy",\
                     "allLi7-v6r0p10_100GeV_1TeV_QGSP.npy",\
                     "allLi7-v6r0p10_1TeV_10TeV-FTFP.npy"]
Beryllium9_filelist = ["allBe9-v6r0p10_10GeV_100GeV_QGSP.npy",\
                       "allBe9-v6r0p10_100GeV_1TeV_FTFP-p1.npy",\
                       "allBe9-v6r0p10_1TeV_10TeV-FTFP.npy"]

HeliumFullSky_filelist = ["Helium_10GeV_to_10TeV_FullSky.npy"]
Proton80_filelist = ["Proton_10GeV_to_10TeV_80perc.npy",\
                     "Proton_10TeV_to_100TeV_80perc.npy"]
Proton120_filelist = ["Proton_10GeV_to_10TeV_120perc.npy",\
                      "Proton_10TeV_to_100TeV_120perc.npy"]
Helium80_filelist = ["Helium_10GeV_to_10TeV_80perc.npy",\
                     "Helium_10TeV_to_100TeV_80perc.npy"]
Helium120_filelist = ["Helium_10GeV_to_10TeV_120perc.npy",\
                      "Helium_10TeV_to_100TeV_120perc.npy"]
Helium200_filelist = ["Helium_10GeV_to_10TeV_200perc.npy",\
                      "Helium_10TeV_to_100TeV_200perc.npy"]

sample_sets = {"Proton": Proton_filelist, "Helium": Helium_filelist,\
               "ProtonFluka": ProtonFluka_filelist, "HeliumFluka": HeliumFluka_filelist,\
               "Lithium7": Lithium7_filelist, "Beryllium9": Beryllium9_filelist,\
               "Proton120": Proton120_filelist, "Proton80": Proton80_filelist,\
               "Helium120": Helium120_filelist, "Helium80": Helium80_filelist,\
               "Helium200": Helium200_filelist, "HeliumFullSky": HeliumFullSky_filelist}
    


