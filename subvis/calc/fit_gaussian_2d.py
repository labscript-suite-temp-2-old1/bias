# from __future__ import division
# from lyse import *
from numpy import *
from scipy.optimize import leastsq
import numexpr as ne

def rebin(a, m=5):
    if m > 1:
        s = floor_divide(a.shape, m)
        sh = s[0], m, s[1], m
        return a[:s[0]*m,:s[1]*m].reshape(sh).mean(3).mean(1)
    else:
        return a

def bokeh_blur(image, r=10):
    """Convolves the image with a disc of radius r pixels"""
    from scipy.signal import fftconvolve
    # Make a disc!
    ny, nx = image.shape
    xr = min(r, floor(nx/2+1))
    yr = min(r, floor(ny/2+1))
    x = linspace(-xr,xr,2*xr+1)
    y = linspace(-yr,yr,2*yr+1)
    Y,X = meshgrid(x,y)
    disc = zeros(X.shape)
    disc[(X**2 + Y**2) < r**2] = 1
    # Normalise the disc to unit integral:
    disc /= disc.sum()
    # Convolve using the Fourier method:
    result = fftconvolve(image, disc,mode='same')
    return result
        
def moments(image, usemax=False):
    if usemax:
        # Blur by 10 pixels to diminish background noise
        image = bokeh_blur(image)
    total = image.sum()
    Y,X = indices(image.shape)
    if usemax:
        y0,x0 = [item[0] for item in where(image == image.max())]
    else:
        x0 = (X*image).sum()/total
        y0 = (Y*image).sum()/total
    col = image[:, int(x0)]
    sigma_y = sqrt(abs((arange(col.size)-y0)**2*col).sum()/abs(col.sum()))
    row = image[int(y0), :]
    sigma_x = sqrt(abs((arange(row.size)-x0)**2*row).sum()/abs(row.sum()))
    amplitude = total/(exp(-0.5*((X-x0)/sigma_x)**2 - 0.5*((Y-y0)/sigma_y)**2)).sum()
    return x0, y0, sigma_x, sigma_y, amplitude

def get_offset(image):
    Y,X = indices(image.shape)
    clipped_image = image - mean(image)
    clipped_image[clipped_image < 0] = 0
    x0, y0, sigma_x, sigma_y, amplitude = moments(clipped_image,usemax=True)
    ellipse_radius = 3 # standard deviations
    while True and ellipse_radius > 0:
        condition = ((X-x0)/sigma_x)**2 + ((Y - y0)/sigma_y)**2 > ellipse_radius**2
        if len(image[condition]):
            break
        ellipse_radius -= 1
    offset = mean(image[condition])
    return offset

def get_gaussian_guess(image):
    offset = get_offset(image)
    x0, y0, sigma_x, sigma_y, amplitude = moments(image - offset, usemax=True)
    return x0, y0, sigma_x, sigma_y, amplitude, offset

def gaussian_1d(x, x0, sigma_x, amplitude, offset):
    return amplitude * exp(-0.5*((x-x0)/sigma_x)**2) + offset

def gaussian_2d(x, y, x0, y0, sigma_x, sigma_y, amplitude, offset):
    # return amplitude * exp(-0.5*((x-x0)/sigma_x)**2 - 0.5*((y-y0)/sigma_y)**2) + offset
    return ne.evaluate("amplitude * exp(-0.5*((x-x0)/sigma_x)**2 - 0.5*((y-y0)/sigma_y)**2) + offset")

def tf_2d(X, Y, x0, y0, Rx, Ry, peak, offset):
    result = ne.evaluate('(1 - (X-x0)**2/Rx**2 - (Y-y0)**2/Ry**2)')
    result[result < 0] = 0
    result = ne.evaluate('peak*result**(3/2) + offset')
    return result
    
def get_roi_and_offset(image, params):
    ny, nx = image.shape
    Y,X = indices(image.shape)
    x0, Rx, tf_peak_x, offset_x = params['xparams']
    y0, Ry, tf_peak_y, offset_y = params['yparams']
    # Determine the extent of the image:
    xwidth = 4.2*abs(Rx)
    ywidth = 4.2*abs(Ry)
    # Get the background offset of the image:
    background = image[((Y-y0)/ywidth)**2 + ((X-x0)/xwidth)**2 > 1]
    offset = average(background)
    if isnan(offset):
        offset = 0
    # Make it square:
    xwidth = ywidth = max(xwidth,ywidth)    
    miny = int(y0 - ywidth)
    maxy = int(y0 + ywidth)
    minx = int(x0 - xwidth)
    maxx = int(x0 + xwidth)
    # Keep it in bounds (might not be square now)
    miny = max(miny,0)
    minx = max(minx,0)
    maxy = min(ny - 1, maxy)
    maxx = min(nx - 1, maxx)
    # clip it until the limits are all multiples of 12 (will allow
    # integer rebinning of image by 2,3,4 and 6 pixels):
    miny += (int(y0)-miny) % 12
    maxy -= (maxy-int(y0)) % 12
    minx += (int(x0)-minx) % 12
    maxx -= (maxx-int(x0)) % 12
    
    # return the ROI, new centroids and offset:
    return image[miny:maxy, minx:maxx], x0-minx, y0-miny, minx, miny, offset

def fit_gaussian_2d(image, scale_factor=1, binsize=1, clip=0, mask=4.0, **kwargs):
    fitfn = gaussian_2d
    # fitfn = tf_2d
    imagef = rebin(image, binsize)
    ny, nx = imagef.shape
    Y, X = indices(imagef.shape)
    params_guess = get_gaussian_guess(imagef)
    
    def residuals(params):
        fit_img = fitfn(X, Y, *params) 
        if clip > 0:
            fit_image[fit_image > clip] = clip
            imagef[imagef > clip] = clip
        err = fit_img - imagef
        err[imagef > mask] = 0
        return err.ravel()
    
    # Fit the gaussian distribution:
    params, covariance, z, z, z = leastsq(residuals, params_guess, maxfev=400000, full_output=True)
    # Rescale the first four of the fit parameters (the spatial ones):
    params[:4] *= scale_factor*binsize
    # Fix the offset due to rebin averaging
    params[:2] += (binsize-1)/2.0
    # Ensure the widths are positive
    params[2:4] = abs(params[2:4])
    if covariance is not None:
        # And their uncertainties:
        covariance[:,:4] *= scale_factor*binsize
        covariance[:4,:] *= scale_factor*binsize
    
        # compute parameter uncertainties and chi-squared
        u_params = [sqrt(abs(covariance[i,i])) if isinstance(covariance,ndarray) else inf for i in range(len(params))]
        reduced_chisquared = (residuals(params)**2).sum()/((prod(imagef.shape) - len(params)))
        try:
            covariance = reduced_chisquared*covariance
        except TypeError:
            covariance = diag(params)**2
    else:
        u_params = params * NaN

    # define dimensions and co-ordinates of original image
    ny, nx = image.shape
    Y, X = indices(image.shape) * scale_factor

    # get the cross-sections of the data and fits along these slices
    try:
        X_section = image[params[1],:]
        X_fit = fitfn(X[0,:], params[1], *params)
    except IndexError:
        if params[1] >= ny:
            X_section = image[-1,:]
            X_fit = fitfn(X[0,:], ny-1, *params)
        else:
            X_section = image[0,:]
            X_fit = fitfn(X[0,:], 0, *params)
    try:
        Y_section = image[:,params[0]]     
        Y_fit = gaussian_2d(params[0], Y[:,0], *params)
    except IndexError:
        if params[0] >= nx:
            Y_section = image[:,-1]
            Y_fit = fitfn(nx-1, Y[:,0], *params)
        else:
            Y_section = image[:,0]
            Y_fit = fitfn(0, Y[:,0], *params)
            
    # put them together to return a 2d numpy array
    X_section = array([X_section, X_fit])
    Y_section = array([Y_section, Y_fit])
       
    # append the area under the fitted Gaussian (in OD*pixel_area)
    N_int = 2*pi*params[2:5].prod()
    u_N_int = sqrt(sum((u_params[2:5]/params[2:5])**2)) * N_int
    params = append(params, N_int)
    u_params = append(u_params, u_N_int)
    
    # prepare a dictionary of param_name : (param, u_param) pairs
    params_names = ['Gaussian_X0', 'Gaussian_Y0', 'Gaussian_XW', 'Gaussian_YW', 'Gaussian_Amp', 'Gaussian_Offset', 'Gaussian_Nint']
    params_dict = dict(zip(params_names, zip(params, u_params)))
    params_dict['X_section'] = X_section
    params_dict['Y_section'] = Y_section
    
    return params_dict
