# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os.path

# Add dir of this file to python path

pwd = os.path.dirname(os.path.realpath(__file__)) + "/"

sr_to_deg2        = (180/np.pi)**2
Sky_in_square_deg = 4*np.pi * sr_to_deg2
mpl_style_file    = pwd + "matplotlib_style"

Code_folder = '/Users/pcoppin/Documents/Postdoc/Code/'
if( not os.path.isdir(Code_folder) ):
    # We are not on my computer, assume we are on an atlas-machine
    Code_folder = "/USERS/coppinp/Code/"

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:pink', 'tab:grey', 'tab:red',\
          'tab:brown', 'tab:olive']
markers = ['o', 'v', '^', 's', 'd', '*']
linestyles = ['-', '--', ':', '-.']

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
    that allows to estimate the MPV and sigma of the peak.
    This one, however, is not normalised. Moyal from scipy is!
    '''
    xt = (x - p[1]) / p[2]
    f = np.exp(-(xt + np.exp(-xt)) / 2)
    return p[0] * f / f.max()

# def Landau(x, mu, sigma):
#     dt = 0.01
#     t_max = 25
#     t = np.arange(1e-6, t_max, dt)
#     res = 1/(np.pi*sigma)*np.sum(np.exp(-t)*np.cos(t*(x-mu)/sigma+2*t/np.pi*np.log(t/sigma)))*dt
#     res *= int((x<(mu+20*sigma)) * (x>(mu-12*sigma)))
#     return res
# Landau = np.vectorize(Landau)
# from scipy.stats import landau
# Landau = landau.pdf

def Gauss(x, *p):
    xt = (x - p[0]) / p[1]
    norm = 1/np.sqrt(2*np.pi*p[1]**2)
    return norm*np.exp(-0.5 * xt**2)

def Langau(x, mu, sigma_landau, sigma_gauss):
    from scipy.stats import landau
    Landau = landau.pdf

    ratio = sigma_landau/sigma_gauss
    if( ratio>=20 ):
        return Landau(x, mu, sigma_landau)
    elif( ratio<=0.05 ):
        return Gauss(x, mu, sigma_gauss)
    sigma = np.sqrt(sigma_gauss**2+sigma_landau**2)
    t_min = mu-10*sigma
    t_max = mu+99*sigma
    dt = min(sigma_gauss,sigma_landau)/20
    t = np.arange(t_min, t_max, dt)
    
    y_landau = Landau(t, mu, sigma_landau)
    xx, tt = np.meshgrid(x,t)
    diff = xx-tt
    y_gauss = Gauss(diff, 0, sigma_gauss)
    
    res = dt*np.sum(y_landau[:,np.newaxis]*y_gauss, axis=0)
    
    if( hasattr(x, '__len__') ):
        return res
    else:
        return res[0]

def RMS(x):
    return np.sqrt(np.mean(np.power(x-np.mean(x),2)))

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
        return r"${:.1f} \pm{:4.1f}$".format(a,b)
    elif( log_base==1 ):
        return r"${:.0f} \pm{:4.0f}$".format(a,b)
    else:
        x, y = np.array([a,b])/np.power(10.0, log_base)
        return r"$\left({:.1f} \pm{:4.1f}\right)\cdot 10^{:d}$".format(x, y, log_base)
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
    sc = subpl.scatter(np.pi-Pixel_pos_RA, Pixel_pos_DEC, c=[max(max_weight*1e-8, w) for w in weights],
                       norm=matplotlib.colors.LogNorm(vmin=max_weight*1e-7, clip=True) )
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

def AMS_He_3_4_ratio_rigidity(R):
    # This function is based on the parametrisation from:
    # https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.123.181102
    return 0.1476 * np.power(0.25*R, -0.294)

def AMS_He_3_4_ratio_Ekin(E_kin,spectral_index=-2.65):
    Rigidity = lambda E, mass: np.sqrt(np.power(mass+E,2)-mass**2) / 2
    R_3 = Rigidity(E_kin,0.931*3.016029)
    R_4 = Rigidity(E_kin,0.931*4.002603)
    ratio_R3 = AMS_He_3_4_ratio_rigidity(R_3)
    return ratio_R3 * np.power(R_3/R_4,spectral_index)

def Helium_mass(E_kin):
    he3_to_he4 = AMS_He_3_4_ratio_Ekin(E_kin)
    return he3_to_he4 * 3 + (1 - he3_to_he4) * 4

def Helium_mass_rigidity(E_kin):
    he3_to_he4 = AMS_He_3_4_ratio_rigidity(E_kin)
    return he3_to_he4 * 3 + (1 - he3_to_he4) * 4

def Central_energy(E1, E2, g=-2.65):
    num = np.power(E2,g+1) - np.power(E1,g+1)
    den = (g+1) * (E2-E1)
    return np.power(num/den, 1./g)

def Rigidity_to_kinetic_energy(rigidity, charge=1, mass=0.931):
    return np.sqrt(np.power(charge*rigidity,2)+mass**2) - mass

def Kinetic_energy_to_rigidity(Ekin, charge=1, mass=0.931):
    return np.sqrt(np.power(Ekin+mass,2)-mass**2)/charge

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

    def Add_counts(self, counts, idx=None):
        if( idx is None ):
            self.hist += counts
        else:
            self.hist[idx] = self.hist[idx] + counts
            self.set_vars()

    def Add_values(self, values):
        new_hist, new_bin_edges = np.histogram(values, bins=self.bins)
        self.hist += new_hist
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
        try:
            ax.ylabel(ylabel)
        except:
            ax.set_ylabel(ylabel)
        if( self.log ):
            try:
                ax.xscale("log")
            except:
                ax.set_xscale('log')
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

def PSD_charge_fit_old(x, a, b, c, d):
    # Sum of a linear function and an exponential
    # Similar to Misha, he uses a + exp(b * x + c)
    return np.exp(a*(x-b)) + c + d*x

def PSD_charge_fit(loge, *p):
    return p[0] + p[1] * loge**p[2] + p[3]*loge**p[4]

def PSD_weight_fit(loge, *p):
    return p[0] + p[1]*loge + p[2]*np.power(loge,2) + p[3]*np.power(loge,p[4])

def Sample_frac(sample, E_BGO, trigger='MIP', PSD_sublayer=0, vertex=0.5):
    # Smearing splines for MIP are for x and y
    sample_to_use = sample.replace('_p15','').replace('_p12','').replace('_old', '').replace('_all','')
    if( trigger in ['MIP1','MIP2'] ):
        trigger = 'MIP'
    s_dir = Code_folder + 'PSD_smearing/Smearing_parameterisations/'
    MCtype = 'MCFLUKA' if( 'Fluka' in sample ) else 'MC'
    if( PSD_sublayer==-1 ):
        with open(s_dir+"{}_{}_Vertex_{}_FitResultsMean.pickle".format(MCtype,trigger,vertex), "rb") as f:
            Fit_results_mean = pickle.load(f)
        MPV, weight, scale, MPV_shift = Fit_results_mean[sample_to_use]
        E_bins_center = Fit_results_mean['E_bins_center']
    else:
        with open(s_dir+"{}_{}_Vertex_{}_FitResults.pickle".format(MCtype,trigger,vertex), "rb") as f:
            Fit_results = pickle.load(f)
        MPV, weight, scale, MPV_shift = list(zip(*Fit_results[(sample_to_use,PSD_sublayer)]))
        E_bins_center = Fit_results['E_bins_center']
    cutoff = -2 if( trigger=='MIP' ) else None
    return np.interp(np.log(E_BGO), np.log(E_bins_center[:cutoff]), weight[:cutoff])

def Helium_to_ProtonPlusHelium(E_BGO):
    return 1 - Proton_to_ProtonPlusHelium(E_BGO)

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

def Reweight(MCs, rescaling_factor=1., samples=None, Pickle_dir=None):
    import pickle
    from copy import deepcopy
    if samples is None:
        samples = MCs.keys()
    if( Pickle_dir is None ):
        Pickle_dir = Code_folder + 'z_classifier_PROTON_HELIUM/Reweighting/WeightingPickles/'
    for sample in samples:
        d = MCs[sample]
        if( rescaling_factor==1 ):
            if( 'Original_MC_weights' in d ):
                d['weight'] = d['Original_weight']
                d['MC_weights'] = d['Original_MC_weights']
            return None
        sample_short = sample.replace('_p15','').replace('_p12','').replace('_all','')
        if( rescaling_factor=='Geant4_to_FLUKA' ):
            Pickle_file = '{}_Geant4_to_FLUKA.pickle'.format(sample_short)
        elif( rescaling_factor=='Correct_to_measured' ):
            Pickle_file = '{}_to_Measured.pickle'.format(sample_short)
        else:
            dummy = sample.replace('_p12','').replace('_p15','').replace('_TargetDiffraction','').replace('_FullDiffraction','')
            Pickle_file = '{}_PSD{:04d}_STK{:04d}_BGO{:04d}.pickle'.format(dummy,
                                                                           int(1e3*rescaling_factor[0]),
                                                                           int(1e3*rescaling_factor[1]),
                                                                           int(1e3*rescaling_factor[2]))

        Pickle_file = Pickle_dir + Pickle_file
        with open(Pickle_file, 'rb') as f:
            interpolator = pickle.load(f)
        x = np.log10( d['E_p'] )
        y = d['pv_cos_theta']
        z = d['stop_z']
        scaling_factor = interpolator(np.dstack((x,y,z)))[0]

        if( np.any(np.isnan(scaling_factor)) ):
            raise Exception('NAN in scaling factors! Investigate interpolator and check e.g. validity energy range.')
        if( np.mean(scaling_factor<0)>0.01 ):
            raise Exception('More than 1% of events have negative weights. Investigate interpolator!')

        scaling_factor = np.maximum(scaling_factor, 0)
        d['scaling_factor'] = scaling_factor
        if 'Original_MC_weights' not in d:
            d['Original_weight'] = deepcopy(d['weight'])
            d['Original_MC_weights'] = deepcopy(d['MC_weights'])
        d['weight'] = d['Original_weight'] * scaling_factor
        d['MC_weights'] = d['Original_MC_weights'] * scaling_factor

def STK_selection(dd, primary='Proton', variance_mean=0.3, tight_cuts=False, low_high=None, n_med=None):
    cut_range = {(False,'Proton'):  [0.8, 1.3],
                 (True, 'Proton'):  [0.9, 1.15],
                 (False, 'Helium'): [1.8, 2.6],
                 (True,  'Helium'): [1.9, 2.3],
                 (False, 'None'): [-999, 1e9]}
    if( n_med is not None):
        Req_n_close_to_median = n_med
    else:
        Req_n_close_to_median = 7 if tight_cuts else 6
    if low_high is not None:
        low, high = low_high
    else:
        low, high = cut_range[(tight_cuts,primary)]
    
    STK_charge = dd['Median_STK_charge_EtaThetaCorr']
    w_close_to_median = np.abs( np.sqrt(dd['HitSignalEtaThetaCorr'])-STK_charge[:,np.newaxis] ) < variance_mean
    N_close_to_median = np.sum(w_close_to_median, axis=1)

    res = (STK_charge>low) * (STK_charge<high) * (N_close_to_median>=Req_n_close_to_median)
    # N_HitSignal = np.sum(dd['HitSignalEtaThetaCorr']>=0.1, axis=1)
    # res *= (N_HitSignal>6)
    return res

def Smear_PSD_charge_MC_to_data(dd, trigger, MC_samples, vertex=0.5):
    if( trigger in ['MIP1','MIP2'] ):
        trigger = 'MIP'
    import pickle
    splines_dir = Code_folder + 'PSD_smearing/Smearing_parameterisations/'
    MC_type = 'MCFLUKA' if('Fluka' in list(MC_samples)[0]) else 'MC'
    source_file = splines_dir+"{}_{}_Vertex_{}_FitResultsMean.pickle".format(MC_type,trigger,vertex)
    print('Reading:', source_file)
    with open(source_file, "rb") as f:
        Fit_results_mean = pickle.load(f)

    for sample in MC_samples:
        dd[sample]["PSD_charge_corr"] = np.zeros( (len(dd[sample]["PSD_charge"]),4), dtype=float)
        E = dd[sample]["E_total_BGO"]
        # No need to redo smearing for p15
        sample_to_use = sample.replace('_p15','').replace('_p12','').replace('_old', '').replace('_all','')
        MPV, weight, scale, MPV_shift = Fit_results_mean[sample_to_use]
        E_bins_center = Fit_results_mean['E_bins_center'][:len(MPV)]
        
        for i in range(4):
            MPV_pe = np.interp(np.log10(E), np.log10(E_bins_center), MPV)
            scale_pe = np.interp(np.log10(E), np.log10(E_bins_center), scale)
            MPV_shift_pe = np.interp(np.log10(E), np.log10(E_bins_center), MPV_shift)

            q = dd[sample]["PSD_charge"][:,i]
            w = q>0.1
            dd[sample]["PSD_charge_corr"][w,i] = (q[w]-MPV_pe[w])*scale_pe[w] + MPV_pe[w] + MPV_shift_pe[w]

    return None

def PSD_selection(dd, trigger, sample, sigma_left=-5, sigma_right=3):
    idx = 4 # Using Xin's charge for now

    import pickle
    splines_dir = Code_folder + 'PSD_smearing/Smearing_parameterisations/'
    splines_file = "MC_{}_fit.pickle".format(trigger) if 'E_p' in dd else "Skim_{}_fit.pickle".format(trigger)
    with open(splines_dir+splines_file, "rb") as f:
        splines = pickle.load(f)

    if( 'E_p' not in dd ):
        sample = sample.replace('Fluka','')
    E = dd["E_total_BGO"]
    Q_center = PSD_charge_fit(np.log10(E),*splines[(sample,idx)][0][0])
    Q_width = PSD_charge_fit(np.log10(E),*splines[(sample,idx)][1][0])
    q = dd["PSD_charge_Xin_pro"]
    diff = (q-Q_center)/Q_width
    w = (diff>sigma_left) * (diff<sigma_right) * dd['sel_pro_STKtrack']
    return w

def PSD_selection_proton_paper(dd, PSD_charge='Xin'):
    E = dd['E_total_BGO']
    # left = 0.6 + 0.05 * np.log10(E/10)
    # right = 1.8 + 0.002 * np.power(np.log10(E/10),4)


    left = 0.6 + 0.05 * np.log10(E/10)
    right = 1.6 + 0.005 * np.power(np.log10(E/10),4)


    if( PSD_charge=='Xin' ):
        q = dd['PSD_charge_corr'][:,4] if 'E_p' in dd else dd['PSD_charge_Xin_pro']
        return (left<q) * (q<right) * dd['sel_pro_STKtrack']
    elif( PSD_charge=='Mine_x' ):
        return (left<dd['charge_x']) * (dd['charge_x']<right)
    elif( PSD_charge=='Mine_xy' ):
        # w_x = (left<dd['charge_x']) * (dd['charge_x']<right)
        # w_y = (left<dd['charge_y']) * (dd['charge_y']<right)
        # return w_x * w_y
        # Actually in the paper they just require the average of x & y to be in the range,
        #    with addtionally (pre-selection): 1 or more hits in each sub-layer of PSD
        w = (left<dd['charge_xy']) * (dd['charge_xy']<right)
        w = w * (dd['charge_x']>0.1) * (dd['charge_y']>0.1)
        return w

    else:
        raise Exception('Type of charge: {} unknown'.format(PSD_charge))
        return 0

def PSD_selection_helium_paper(dd, PSD_charge='Xin'):
    E = dd['E_total_BGO']
    left = 1.85 + 0.02 * np.log10(E/10)
    right = 2.8 + 0.007 * np.power(np.log10(E/10),4)
    if( PSD_charge=='Xin' ):
        q = dd['PSD_charge_corr'][:,4] if 'E_p' in dd else dd['PSD_charge_Xin_pro']
        return (left<q) * (q<right) * dd['sel_pro_STKtrack']
    elif( PSD_charge=='Mine_x' ):
        return (left<dd['charge_x']) * (dd['charge_x']<right)
    elif( PSD_charge=='Mine_xy' ):
        w_x = (left<dd['charge_x']) * (dd['charge_x']<right)
        w_y = (left<dd['charge_y']) * (dd['charge_y']<right)
        ratio = dd['charge_x']/np.maximum(dd['charge_y'],1e-9)
        return w_x * w_y * (ratio>0.5) * (ratio<2)
    else:
        raise Exception('Type of charge: {} unknown'.format(PSD_charge))
        return 0

def Left_lim_carbon(E):
    left = 5.6 * np.ones(len(E))
    w_high_E = E > 1e4
    left[w_high_E] += 0.1*(np.log10(E[w_high_E])-4)
    return left

def Right_lim_carbon(E):
    right = 6.5+0.05*np.log10(E)
    w_high_E = np.log10(E) > 3.5
    right[w_high_E] += 0.6*(np.log10(E[w_high_E])-3.5)
    return right

def G4_frac_quasielastic(E):
    ### These values are derived from the Hadr03 examples
    ### See Code/GeantExamples/Processing_Hadr03_output on gridvm10
    xx = np.array([   10.        ,    12.58925412,    15.84893192,    19.95262315,
                      25.11886432,    31.6227766 ,    39.81071706,    50.11872336,
                      63.09573445,    79.43282347,   100.        ,   125.89254118,
                      158.48931925,   199.5262315 ,   251.18864315,   316.22776602,
                      398.10717055,   501.18723363,   630.95734448,   794.32823472,
                      1000.        ,  1258.92541179,  1584.89319246,  1995.26231497,
                      2511.88643151,  3162.27766017,  3981.07170553,  5011.87233627,
                      6309.5734448 ,  7943.28234724, 10000.        ])
    yy = np.array([0.1047   , 0.097465 , 0.0903725, 0.0842575, 0.07878  , 0.07443  ,
                   0.07056  , 0.0670775, 0.064135 , 0.06161  , 0.0595825, 0.05796  ,
                   0.056615 , 0.055525 , 0.0546475, 0.0538575, 0.05335  , 0.0529325,
                   0.052575 , 0.0523   , 0.052085 , 0.052015 , 0.0519725, 0.0520275,
                   0.05197  , 0.0520675, 0.0523225, 0.052495 , 0.0525175, 0.052765 ,
                   0.052995 ])
    return np.interp(np.log10(E), np.log10(xx), 1-np.array(yy))

def Corr_XS_G4(E):
    ### Method based on stop-z
    #   Taking ratio of particle that interact per layer (corrected over non-corrected)
    from scipy.interpolate import BSpline
    tck = (np.array([ 2.41771435,  2.41771435,  2.41771435,  2.41771435,  4.72029944, 7.02288453,  9.32546963,
                     10.47676217, 11.39779621, 11.39779621, 11.39779621, 11.39779621]),
           np.array([0.90601885, 0.92068368, 0.94567574, 0.94578798, 0.94624133, 0.94477405, 0.94316579, 0.94216731,
                     0.        , 0.        , 0.        , 0.        ]), 3)
    res = BSpline(*tck)(np.log(E))
    w = E<1.12201845e+01
    res[w] = BSpline(*tck)(np.log(1.12201845e+01))
    w = E>8.91250938e+04
    res[w] = BSpline(*tck)(np.log(8.91250938e+04))
    return res
def Corr_XS_G4_4(E, E_cut1=10, E_cut2 = 1e5):
    ### Method based on stop-z-2 (corrected based on pion/kaon in secondaries)
    #   Taking ratio of particle that interact per layer (corrected over non-corrected)
    from scipy.interpolate import BSpline
    tck = (np.array([2.5328436 ,  2.5328436 ,  2.5328436 ,  2.5328436 , 11.28266696,
                     11.28266696, 11.28266696, 11.28266696]),
           np.array([0.91478609, 0.97444726, 0.94366919, 0.95085184, 0.        ,
                     0.        , 0.        , 0.        ]), 3)
    res = BSpline(*tck)(np.log(E))
    w = E<E_cut1
    res[w] = BSpline(*tck)(np.log(E_cut1))
    w = E>E_cut2
    res[w] = BSpline(*tck)(np.log(E_cut2))
    return res


def Combine_npy_dict(Filelist=[], keys=[],\
                     filters=['HE_trigger','NonZeroPSD','MLcontainmentBGO','MLcontainmentSTK','skim'],\
                     npy_dir="/dpnc/beegfs/users/coppinp/Simu_vary_cross_section_with_Geant4/Analysis/npy_files/",):
    import copy
    keys = copy.deepcopy(keys)
    data = {}
    for key in filters:
        if( "MLcontainment" in key ):
            subdet  = key.split('_')[0][-3:]
            ML_conf = key.split('_')[1]
            ToAdd = [x.format(subdet,ML_conf) for x in ['{}InterceptY_{}', '{}InterceptX_{}','{}SlopeX_{}', '{}SlopeY_{}']]
            keys.extend( ToAdd )
        elif( key=='MIP_trigger' ):
            keys.append('MIP1_trigger')
            keys.append('MIP2_trigger')
        elif( key=="NonZeroPSD_default" ):
            keys.append( "PSD_charge_STKtrack_default" )
        elif( key=="NonZeroPSD_ions" ):
            keys.append( "PSD_charge_STKtrack_ions" )
        elif( 'PSD_charge_Xin_' in key ):
            # 'PSD_charge_Xin_ion_STKtrack_ions' 'PSD_charge_Xin_pro_STKtrack_default'
            conf_type = key.split('_')[3]
            keys.append( 'sel_{}_STKtrack'.format(conf_type) )
        elif( key not in keys ):
            keys.append( key )
    if( 'MIP_trigger' in keys ):
        keys.extend( ['MIP1_trigger','MIP2_trigger'] )
        
    keys = np.unique(keys)

    for files in Filelist:
        data_is = [np.load(npy_dir+f, allow_pickle=True, encoding="latin1").item() for f in files]

        ########################################################################################################################
        # Quick and dirty fix. Rename variables for old n-tuples and use proton ML tracks when asking for ion ones
        old_keys = ["PSD_charge_STKtrack","PSD_length_STKtrack", "BGO_energy_STKtrack", "BGO_length_STKtrack",\
                    "PSD_charge_BGOtrack","PSD_length_BGOtrack", "BGO_energy_BGOtrack", "BGO_length_BGOtrack",\
                    "VertexPrediction",'HitSignal','HitDistance', "ML_BGO_costheta", "ML_STK_costheta",\
                    "BGOInterceptX", "BGOInterceptY", "STKInterceptX", "STKInterceptY", 'HitSignalCombined', \
                    'HitSignalEtaThetaCorr',"STKSlopeX", "STKSlopeY", "BGOSlopeX", "BGOSlopeY",\
                    'Median_STK_charge_EtaThetaCorr', 'VertexPrediction_REG_BGO']
        for old_key in old_keys:
            for iss in range(len(data_is)):
                if( old_key in data_is[iss] ):
                    data_is[iss][old_key+'_default'] = data_is[iss].pop(old_key)
                    data_is[iss][old_key+'_ions'] = data_is[iss][old_key+'_default']
                if( (old_key+'_default' in data_is[iss]) and (old_key+'_ions' not in data_is[iss]) ):
                    data_is[iss][old_key+'_ions'] = data_is[iss][old_key+'_default']
                if( 'PSD_charge_Xin_pro_STKtrack' in data_is[iss] ):
                    data_is[iss]['PSD_charge_Xin_ion_STKtrack'] = data_is[iss]['PSD_charge_Xin_pro_STKtrack']
        ########################################################################################################################

        # if( 'E_total_BGO_quench' in data_is[0] ):
        #     if( 'E_total_BGO_quench' not in keys ):
        #         keys = list(keys)
        #         keys.append('E_total_BGO_quench')
        N_i = 0
        for data_i in data_is:
            N_i += len(data_i['E_p']) + int( 10 * len(data_i['E_primary_non_trig']) )
        for data_i in data_is:
            data_i['weight'] = (1.0/N_i) * np.ones( len(data_i['E_p'])  )
            if( any([x in files[0] for x in ['1GeV_1TeV','10GeV_10TeV','100GeV_100TeV']]) ):
                # Weight normally 1 per decade, so total weight is 3 if adding file that spans 3 decades
                data_i['weight'] *= 3
            elif( any([x in files[0] for x in ('1GeV_100GeV','10GeV_1TeV','100GeV_10TeV','1TeV_100TeV','10TeV_1PeV')]) ):
                # Weight normally 1 per decade, so total weight is 3 if adding file that spans 3 decades
                data_i['weight'] *= 2
            elif( "100TeV_500TeV" in files[0] ):
                data_i['weight'] *= np.log10(5)
            elif(  "500TeV_1PeV" in files[0] ):
                data_i['weight'] *= np.log10(2)
            elif( "1PeV_3PeV" in files[0] ):
                data_i['weight'] *= np.log10(3)
            else:
                # Don't do anything. Default scaling assuming samples that span factor 10 in energy
                pass
            # Use all keys if not specified
            if( len(keys)==0 ):
                keys = data_i.keys()
            # Put data in dictionary
            for key in keys:
                if( (key=='MIP_trigger') and (key not in data_i) ):
                    data_i[key] = data_i['MIP1_trigger'] * data_i['MIP2_trigger']
                if( key not in data ):
                    data[key] = data_i[key]
                else:
                    data[key] = np.concatenate([data[key],data_i[key]])

    for key in ["E_p","E_total_BGO", "E_total_BGO_quench", "E_total_PSD", "E_primary_non_trig"]:
        if( key in data ):
            data[key] = 1e-3 * data[key]
    
    # w = data['HE_trigger'] * data["Skimmed"]
    w = np.ones_like(data['E_p'], dtype=bool)
    for key in filters:
        if( 'MLcontainmentBGO' in key ):
            ML_conf = key.split('_')[1]
            addon = '_' + ML_conf
            
            # BGO prediction fiducially contained in BGO
            BGO_TopZ, BGO_BottomZ = 46., 448.
            cut = 280
            topX = data['BGOSlopeX'+addon] * BGO_TopZ + data['BGOInterceptX'+addon]
            topY = data['BGOSlopeY'+addon] * BGO_TopZ + data['BGOInterceptY'+addon]
            bottomX = data['BGOSlopeX'+addon] * BGO_BottomZ + data['BGOInterceptX'+addon]
            bottomY = data['BGOSlopeY'+addon] * BGO_BottomZ + data['BGOInterceptY'+addon]
            ml_bgo_fid = (abs(topX)<cut) * (abs(topY)<cut) * (abs(bottomX)<cut) * (abs(bottomY)<cut)
            w = w * ml_bgo_fid
            # Additional requirement: Make sure STK track is trustworthy
            TopZ = -210.
            cutTop = 500.
            topX = data['BGOSlopeX'+addon] * TopZ + data['BGOInterceptX'+addon]
            topY = data['BGOSlopeY'+addon] * TopZ + data['BGOInterceptY'+addon]
            w = w * (abs(topX)<cutTop) * (abs(topY)<cutTop)

        elif( 'MLcontainmentSTK' in key ):
            ML_conf = key.split('_')[1]
            addon = '_' + ML_conf
            
            # STK prediction fiducially contained within PSD to BGO
            TopZ, BottomZ = -325, 448.
            cutTop, cutBottom = 440, 280
            topX = data['STKSlopeX'+addon] * TopZ + data['STKInterceptX'+addon]
            topY = data['STKSlopeY'+addon] * TopZ + data['STKInterceptY'+addon]
            bottomX = data['STKSlopeX'+addon] * BottomZ + data['STKInterceptX'+addon]
            bottomY = data['STKSlopeY'+addon] * BottomZ + data['STKInterceptY'+addon]
            ml_stk_fid = (abs(topX)<cutTop) * (abs(topY)<cutTop) * (abs(bottomX)<cutBottom) * (abs(bottomY)<cutBottom)
            TopZ = 44.
            cutTop = 280.
            topX = data['STKSlopeX'+addon] * TopZ + data['STKInterceptX'+addon]
            topY = data['STKSlopeY'+addon] * TopZ + data['STKInterceptY'+addon]
            ml_stk_fid = ml_stk_fid * (abs(topX)<cutTop) * (abs(topY)<cutTop)
            w = w * ml_stk_fid
            
        elif( "NonZeroPSD" in key ):
            ML_conf = key.split('_')[1]
            w = w * np.sum(data['PSD_charge_STKtrack_'+ML_conf]>0.1, axis=1, dtype=bool)   
        elif( key=='MIP_trigger' ):
            w = w * (data['MIP1_trigger']+data['MIP2_trigger'])
        else:
            w = w * data[key]
            
    for key in data:
        if( key=="E_primary_non_trig" ):
            continue
        elif( key=="E_p"):
            if( "E_primary_non_trig" in data ):
                data['E_primary_non_trig'] = np.concatenate( [data['E_primary_non_trig'],data['E_p'][~w][::10]] )
        data[key] = data[key][w]
    
    for ML_conf in ['default','ions']:
        key_to_check = 'PSD_charge_STKtrack_{}'.format(ML_conf)
        if( key_to_check in data ):
            data[key_to_check] = np.maximum( data[key_to_check], 0.0 )

    return data

Proton_filelist = [["allProton-v6r0p10_10GeV_100GeV_FTFP-p2.npy",],\
                   ["allProton-v6r0p10_100GeV_1TeV_FTFP-p1.npy",],\
                   ["allProton-v6r0p10_1TeV_10TeV_FTFP.npy",],\
                   ["allProton-v6r0p10_10TeV_100TeV_FTFP.npy",],\
                   ["allProton-v6r0p12_100TeV_1PeV_EPOSLHC_FTFP_BERT.npy",]]
Proton_p15_filelist = [["allProton-v6r0p15_1GeV_10GeV_FTFP.npy",],\
                       ['allProton-v6r0p15_10GeV_100GeV_FTFP-p3.npy',],\
                       ['allProton-v6r0p15_100GeV_1TeV_FTFP-p4.npy',],\
                       ['allProton-v6r0p15_1TeV_10TeV_FTFP-p3.npy',],\
                       ['allProton-v6r0p15_10TeV_100TeV_FTFP-p2.npy',],\
                       ['allProton-v6r0p15_100TeV_1PeV-EPOSLHC_FTFP.npy','allProton-v6r0p15_100TeV_1PeV-DPMJET3_FTFP.npy']]
Proton_all_filelist = [["allProton-v6r0p15_1GeV_10GeV_FTFP.npy",],\
                       ["allProton-v6r0p10_10GeV_100GeV_FTFP-p2.npy", 'allProton-v6r0p15_10GeV_100GeV_FTFP-p3.npy',],\
                       ["allProton-v6r0p10_100GeV_1TeV_FTFP-p1.npy", 'allProton-v6r0p15_100GeV_1TeV_FTFP-p4.npy',],\
                       ["allProton-v6r0p10_1TeV_10TeV_FTFP.npy", 'allProton-v6r0p15_1TeV_10TeV_FTFP-p3.npy',],\
                       ["allProton-v6r0p10_10TeV_100TeV_FTFP.npy", 'allProton-v6r0p15_10TeV_100TeV_FTFP-p2.npy',],\
                       ["allProton-v6r0p12_100TeV_1PeV_EPOSLHC_FTFP_BERT.npy",'allProton-v6r0p15_100TeV_1PeV-EPOSLHC_FTFP.npy',\
                        'allProton-v6r0p15_100TeV_1PeV-DPMJET3_FTFP.npy']]
Proton_old_filelist = [["allProton-v6r0p0_1GeV_100GeV_FTFP.npy",],\
                       ["allProton-v6r0p0_100GeV_10TeV_FTFP_HP.npy",],\
                       ["allProton-v6r0p0_10TeV_100TeV_FTFP_HP.npy",]]
ProtonFluka_filelist =[["allProton-v6r0p15_1GeV_10GeV_FLUKA.npy",],\
                       ["allProton-v6r0p15_10GeV_100GeV-FLUKA.npy",],\
                       ["allProton-v6r0p15_100GeV_1TeV-FLUKA.npy",],\
                       ["allProton-v6r0p15_1TeV_10TeV-FLUKA.npy",],\
                       ["allProton-v6r0p15_10TeV_100TeV-FLUKA.npy",],\
                       ["allProton-v6r0p15_100TeV_1PeV-FLUKA.npy",]]
Helium_filelist = [["allHe4-v6r0p10_10GeV_100GeV_FTFP.npy",],\
                   ["allHe4-v6r0p10_100GeV_1TeV_FTFP.npy",],\
                   ["allHe4-v6r0p10_1TeV_10TeV-FTFP.npy",],\
                   ["allHe4-v6r0p10_10TeV_100TeV-FTFP.npy",],\
                   ["allHe4-v6r0p10_100TeV_500TeV-EPOSLHC.npy",]]
Helium_p12_filelist = [['allHe4-v6r0p15_1GeV_10GeV_FTFP.npy',],
                       ["allHe4-v6r0p12_10GeV_100GeV-FTFP_BGO_Quenching.npy",],\
                       ["allHe4-v6r0p12_100GeV_1TeV-FTFP_BGO_Quenching.npy",],\
                       ["allHe4-v6r0p12_1TeV_10TeV-FTFP-BGO_Quenching.npy",],\
                       ["allHe4-v6r0p12_10TeV_100TeV-FTFP_BGO_Quenching.npy",],\
                       ["allHe4-v6r0p12_100TeV_500TeV-EPOSLHC_reg-p1-p2.npy",],\
                       ["allHe4-v6r0p15_500TeV_1PeV-EPOSLHC_FTFP.npy","allHe4-v6r0p15_500TeV_1PeV_EPOSLHC_BERT-p1.npy"]]
Helium_all_filelist = [['allHe4-v6r0p15_1GeV_10GeV_FTFP.npy',],
                       ["allHe4-v6r0p10_10GeV_100GeV_FTFP.npy","allHe4-v6r0p12_10GeV_100GeV-FTFP_BGO_Quenching.npy"],\
                       ["allHe4-v6r0p10_100GeV_1TeV_FTFP.npy","allHe4-v6r0p12_100GeV_1TeV-FTFP_BGO_Quenching.npy"],\
                       ["allHe4-v6r0p10_1TeV_10TeV-FTFP.npy","allHe4-v6r0p12_1TeV_10TeV-FTFP-BGO_Quenching.npy"],\
                       ["allHe4-v6r0p10_10TeV_100TeV-FTFP.npy","allHe4-v6r0p12_10TeV_100TeV-FTFP_BGO_Quenching.npy"],\
                       ["allHe4-v6r0p10_100TeV_500TeV-EPOSLHC.npy","allHe4-v6r0p12_100TeV_500TeV-EPOSLHC_reg-p1-p2.npy"]]
HeliumFluka_filelist = [["allHe4-v6r0p15_1GeV_10GeV_FLUKA.npy",],\
                        ["allHe4-v6r0p10_10GeV_100GeV-FLUKA.npy",],\
                        ["allHe4-v6r0p10_100GeV_1TeV-FLUKA.npy",],\
                        ["allHe4-v6r0p10_1TeV_10TeV-FLUKA.npy",],\
                        ["allHe4-v6r0p10_10TeV_100TeV-FLUKA-p1.npy",],\
                        ["allHe4-v6r0p10_100TeV_500TeV-FLUKA.npy",]]
Lithium7_filelist = [["allLi7-v6r0p10_10GeV_100GeV_FTFP-p1.npy",],\
                     ["allLi7-v6r0p10_100GeV_1TeV_FTFP-p1.npy",],\
                     ["allLi7-v6r0p15_1TeV_10TeV_FTFP-p0.npy",],\
                     ["allLi7-v6r0p10_10TeV_100TeV-FTFP.npy","allLi7-v6r0p15_10TeV_100TeV-EPOSLHC_FTFP.npy"],\
                     ["allLi7-v6r0p15_100TeV_500TeV-EPOSLHC_FTFP.npy",]]
Beryllium9_filelist = [["allBe9-v6r0p10_10GeV_100GeV_FTFP-p1.npy",],\
                       ["allBe9-v6r0p10_100GeV_1TeV_FTFP-p1.npy",],\
                       ["allBe9-v6r0p10_1TeV_10TeV-FTFP.npy",],\
                       ["allBe9-v6r0p15_10TeV_100TeV-EPOSLHC_FTFP.npy",],\
                       ["allBe9-v6r0p15_100TeV_500TeV-EPOSLHC_FTFP.npy",]]
Boron_filelist = [['allB10-v6r0p15_10GeV_100GeV_FTFP-p1.npy',],\
                  ['allB10-v6r0p15_100GeV_1TeV_FTFP-p1.npy',],\
                  ['allB10-v6r0p15_1TeV_10TeV-FTFP-p1.npy',],\
                  ['allB10-v6r0p15_10TeV_100TeV-EPOSLHC_FTFP.npy',],\
                  ['allB10-v6r0p15_100TeV_500TeV_EPOSLHC_FTFP-p2.npy',]]
Carbon_filelist = [["allC12-v6r0p15_10GeV_100GeV-FTFP.npy",],\
                   ["allC12-v6r0p15_100GeV_1TeV_FTFP-BGO-Quenching-p0.npy",],\
                   ["allC12-v6r0p15_1TeV_10TeV_FTFP-BGO-Quenching-p0.npy",],\
                   ["allC12-v6r0p15_10TeV_100TeV_FTFP-BGO-Quenching-p0.npy",],\
                   ['allC12-v6r0p15_100TeV_500TeV-EPOSLHC_FTFP.npy','allC12-v6r0p15_100TeV_500TeV-EPOSLHC_FTFP-p1.npy'],\
                   ['allC12-v6r0p13-reco-v6r0p15_500TeV_1PeV-EPOSLHC_FTFP.npy',],\
                   ['allC12-v6r0p15_1PeV_3PeV-EPOSLHC_FTFP.npy']]
Nitrogen_filelist = [['allN14-v6r0p15_100GeV_1TeV_FTFP-p2.npy',],\
                     ['allN14-v6r0p10_1TeV_10TeV-FTFP.npy',],\
                     ['allN14-v6r0p15_10TeV_100TeV-EPOSLHC_FTFP.npy',],\
                     ['allN14-v6r0p15_100TeV_500TeV-EPOSLHC_FTFP.npy',]]
Oxygen_filelist = [["allO16-v6r0p15_10GeV_100GeV-FTFP.npy","allO16-v6r0p15_10GeV_100GeV-FTFP-p1.npy"],\
                   ["allO16-v6r0p15_100GeV_1TeV_FTFP-BGO-Quenching-p0.npy",],\
                   ["allO16-v6r0p15_1TeV_10TeV_FTFP-BGO-Quenching-p0.npy",],\
                   ["allO16-v6r0p15_10TeV_100TeV-EPOSLHC_FTFP.npy","allO16-v6r0p10_10TeV_100TeV_FTFP_p2.npy"],
                   ["allO16-v6r0p15_100TeV_500TeV-EPOSLHC_FTFP.npy","allO16-v6r0p15_100TeV_500TeV-EPOSLHC_FTFP-p1.npy"]]
BoronFluka_filelist = [["allB10-v6r0p15_10GeV_100GeV_FTFP-p1.npy",],\
                       ["allB10-v6r0p15_100GeV_1TeV_FTFP-p1.npy",],\
                       ["allB10-v6r0p15_1TeV_10TeV-FTFP-p1.npy",],\
                       ["allB10-v6r0p15_10TeV_100TeV-EPOSLHC_FTFP.npy",],\
                       ["allB10-v6r0p15_100TeV_500TeV_EPOSLHC_FTFP-p2.npy",]]
CarbonFluka_filelist = [['allC12-v6r0p15_10GeV_100GeV-FLUKA.npy',],\
                        ['allC12-v6r0p14-reco-v6r0p15_100GeV_1TeV-FLUKA.npy',],\
                        ['allC12-v6r0p14-reco-v6r0p15_1TeV_10TeV-FLUKA.npy',],\
                        ['allC12-v6r0p14-reco-v6r0p15_10TeV_100TeV-FLUKA.npy','allC12-v6r0p14-reco-v6r0p15_10TeV_100TeV-FLUKA-p1.npy'],\
                        ['allC12-v6r0p14-reco-v6r0p15_100TeV_500TeV-FLUKA.npy','allC12-v6r0p14-reco-v6r0p15_100TeV_500TeV-FLUKA-p1.npy']]
NitrogenFluka_filelist = [["allN14-v6r0p15_10GeV_100GeV-FLUKA.npy",],\
                          ["allN14-v6r0p15_100GeV_1TeV-FLUKA.npy",],\
                          ["allN14-v6r0p15_1TeV_10TeV-FLUKA.npy",],\
                          ["allN14-v6r0p15_10TeV_100TeV-FLUKA.npy",],\
                          ["allN14-v6r0p15_100TeV_500TeV-FLUKA.npy",]]
OxygenFluka_filelist = [["allO16-v6r0p15_10GeV_100GeV-FLUKA.npy",],\
                        ["allO16-v6r0p14-reco-v6r0p15_100GeV_1TeV-FLUKA.npy",],\
                        ["allO16-v6r0p14-reco-v6r0p15_1TeV_10TeV-FLUKA.npy",],\
                        ["allO16-v6r0p14-reco-v6r0p15_10TeV_100TeV-FLUKA.npy",],\
                        ["allO16-v6r0p14-reco-v6r0p15_100TeV_500TeV-FLUKA.npy",]]


HeliumFullSky_filelist = ["Helium_10GeV_10TeV_FullSky.npy"]
Proton80_filelist = ["Proton_10GeV_10TeV_80perc.npy",\
                     "Proton_10TeV_100TeV_80perc.npy"]
Proton120_filelist = ["Proton_10GeV_10TeV_120perc.npy",\
                      "Proton_10TeV_100TeV_120perc.npy"]
Helium80_filelist = ["Helium_10GeV_10TeV_80perc.npy",\
                     "Helium_10TeV_100TeV_80perc.npy"]
Helium120_filelist = ["Helium_10GeV_10TeV_120perc.npy",\
                      "Helium_10TeV_100TeV_120perc.npy"]
Helium200_filelist = ["Helium_10GeV_10TeV_200perc.npy",\
                      "Helium_10TeV_100TeV_200perc.npy"]

Proton_TargetDiffraction_filelist = [["allProton-v6r0p15_1GeV_1TeV_FTFP_DiffractionOn.npy",],\
                                     ["allProton-v6r0p15_1TeV_100TeV_FTFP_DiffractionOn.npy",]]
Helium_TargetDiffraction_filelist = [["allHe4-v6r0p15_1GeV_1TeV_FTFP_DiffractionOn.npy",],\
                                     ["allHe4-v6r0p15_1TeV_100TeV_FTFP_DiffractionOn.npy",]]
Proton_FullDiffraction_filelist = [["allProton-v6r0p15_1GeV_1TeV_FTFP_FullDiffractionOn.npy",],]

sample_sets = {'Proton': Proton_filelist, 'Proton_p15': Proton_p15_filelist, 'Proton_all': Proton_all_filelist,\
               'Helium': Helium_filelist, 'Helium_p12': Helium_p12_filelist, 'Helium_all': Helium_all_filelist,\
               'Proton_old': Proton_old_filelist,\
               'Proton_TargetDiffraction': Proton_TargetDiffraction_filelist, 'Helium_TargetDiffraction': Helium_TargetDiffraction_filelist,\
               'Proton_FullDiffraction': Proton_FullDiffraction_filelist,\
               'ProtonFluka': ProtonFluka_filelist, 'HeliumFluka': HeliumFluka_filelist,\
               'Lithium7': Lithium7_filelist, 'Beryllium9': Beryllium9_filelist,\
               'Boron': Boron_filelist, 'Nitrogen': Nitrogen_filelist,\
               'Proton120': Proton120_filelist, 'Proton80': Proton80_filelist,\
               'Helium120': Helium120_filelist, 'Helium80': Helium80_filelist,\
               'Helium200': Helium200_filelist, 'HeliumFullSky': HeliumFullSky_filelist,\
               'Carbon': Carbon_filelist, 'Oxygen': Oxygen_filelist,\
               'CarbonFluka': CarbonFluka_filelist, 'OxygenFluka': OxygenFluka_filelist,\
               'BoronFluka': BoronFluka_filelist, 'NitrogenFluka': NitrogenFluka_filelist}


