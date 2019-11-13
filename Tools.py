import numpy as np
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
#import Tools

sr_to_deg2        = (180/np.pi)**2
Sky_in_square_deg = 4*np.pi * sr_to_deg2

def Reduced_chi_squared(function, x_data, y_data, y_error, params):
    if(len(x_data)!=len(y_data)):
        raise Exception("X & Y array have different lengths")
    Ndog = len(x_data) - len(params)
    chi2 = np.sum( np.power(y_data-function(x_data, *params),2) / np.power(y_error,2) )
    return chi2/Ndog

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

def Angular_seperation(pos_1, pos_2, ra_dec=False, degrees=True):
    """
    Compute the angular seperation between point A & B on a sphere

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
    inproduct      = x1*x2+y1*y2+z1*z2
    if( np.any(inproduct)>1 or np.any(inproduct)<-1 ):
        raise Exception("Invalid inproduct = %f   %f"%(np.min(inproduct), np.max(inproduct)))
    ang_sep = np.arccos(inproduct)
    if( degrees ):
        return np.degrees(ang_sep)
    else:
        return ang_sep

def kent(s, dpsi):
    """
    Compute the value of the Kent distribution

    :type   s: float
    :param  s: sigma or the angular uncertainty (in radians)

    :type   dpsi: float
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
    dpsi = [Angular_seperation(pos_1, [t,p], degrees=False) for t,p in zip(theta,phi)]
    m = [kent(s, dpsi_i)*4*np.pi/npix for dpsi_i in dpsi]
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

class Hist(object):
    def __init__(self, data, log=False, **kw):
        if( log ):
            if( "bins" not in kw ):
                kw["bins"] = 10
            if( isinstance(kw["bins"],int) ):
                eps = 1.01
                kw["bins"] = np.logspace(np.log10(min(data)/eps), 
                                         np.log10(max(data)*eps),
                                         kw["bins"])
        
        self.hist, self.bins = np.histogram(data, **kw)
        self.bin_width = self.bins[1:] - self.bins[:-1]
        self.bin_center = self.bins[:-1] + 0.5*self.bin_width
        self.sum = sum(self.hist)
        
        self.hist_density = self.hist/self.bin_width
        self.hist_normed = self.hist/self.sum
        self.cdf = self.hist_normed.cumsum()
    
    def plot(self, type="normal", **kw):
        if( type=="normal" ):
            height = self.hist
        elif( type=="normed" ):
            height = self.hist_normed
        elif( type=="density" ):
            height = self.hist_density
        elif( type=="cdf" ):
            height = self.cdf
        elif( type=="inv_cdf" ):
            height = 1-self.cdf
        else:
            raise Exception("Unknown histogram type specified: {}. Please selected 'normal' (default), 'normed' or 'density'.")
        mask = height>0
        return plt.bar(self.bins[:-1][mask], height[mask], self.bin_width[mask], align='edge', **kw)
        
