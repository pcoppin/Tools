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

sr_to_deg2        = (180/np.pi)**2
Sky_in_square_deg = 4*np.pi * sr_to_deg2
Code_folder       = "/mnt/2690D37B90D35043/PhD/Code"
mpl_style_file    = Code_folder + "/Tools/matplotlib_style"

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
    
    if( (N>1000) and (k>20) ):
        sigma = np.sqrt( k/np.power(N,2) + np.power(k,2)/np.power(N,3) )
        return np.array([float(k)/N, 0.5*sigma, 0.5*sigma])
    
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

def mjd_to_year(mjd):
    """
    Conver the mjd value to the year (rough method)

    :type   mjd: array or list
    :param  mjd: Modified Julian Data values
    """
    mjd_2010_01_01 = 55197
    days_per_years = 365.2422
    return (np.array(mjd)-mjd_2010_01_01)/days_per_years + 2010

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
        self.hist_normed = self.hist/self.sum
        self.pdf = self.hist_normed/self.bin_width
        self.cdf = self.hist_normed.cumsum()
        self.binomial_uncertainty = []
    
    def Add_counts(self, idx, counts):
        self.hist[idx] = self.hist[idx] + counts
        self.set_vars()
    
    def set_errors(self, probability_content = 0.683):
        for N_i in self.hist:
            self.binomial_uncertainty.append( self.sum*Efficiency_for_binomial_process(self.sum, int(N_i), probability_content)[1:] )
    
    def plot(self, ax=plt, type="hist", **kw):
        if( type=="hist" ):
            height = self.hist
        elif( type=="cumsum" ):
            height = self.hist.cumsum()
        elif( type=="normed" ):
            height = self.hist_normed
        elif( type=="density" ):
            height = self.hist_density
        elif( type=="pdf" ):
            height = self.pdf
        elif( type=="cdf" ):
            height = self.cdf
        elif( type=="inv_cdf" ):
            height = 1-self.cdf
        else:
            raise Exception("Unknown histogram type specified: {}. Please selected 'hist' (default), 'normed', 'density', 'pdf', 'cdf' or 'inv_cdf'.")
        mask = height>0
        return ax.bar(self.bins[:-1][mask], height[mask], self.bin_width[mask], align='edge', **kw)
    
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




