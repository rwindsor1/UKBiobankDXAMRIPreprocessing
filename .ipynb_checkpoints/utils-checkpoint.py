from zipfile import ZipFile
from scipy import ndimage as nd
import scipy.signal as sig
import numpy as np
import pydicom
import pydicom.pixel_data_handlers.gdcm_handler as gdcm_handler
pydicom.config.image_handlers = [None, gdcm_handler]

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def get_meta(meta, idx_):
    curr_filename = meta['filenames'][idx_]
    curr_series = meta['series'][idx_]
    curr_slice_thickness = meta['slice_thickness'][idx_]
    curr_row = meta['slice_rows'][idx_]
    curr_col = meta['slice_columns'][idx_]
    curr_ipp_1 = meta['ipp_1'][idx_]
    curr_slice_location = meta['slice_location'][idx_]
    curr_slice_location_min = meta['slice_location_min'][idx_]
    curr_slice_location_max = meta['slice_location_max'][idx_]

    curr_meta = {}
    curr_meta['filename'] = curr_filename
    curr_meta['series'] = curr_series
    curr_meta['slice_thickness'] = curr_slice_thickness
    curr_meta['row'] = curr_row
    curr_meta['col'] = curr_col
    curr_meta['ipp_1'] = curr_ipp_1
    curr_meta['slice_location'] = curr_slice_location
    curr_meta['slice_location_min'] = curr_slice_location_min
    curr_meta['slice_location_max'] = curr_slice_location_max
    return curr_meta

def segments_intersect(x1, x2, y1, y2):
    # Assumes x1 <= x2 and y1 <= y2; if this assumption is not safe, the code
    # can be changed to have x1 being min(x1, x2) and x2 being max(x1, x2) and
    # similarly for the ys.
    res = (x2 > y1) and (y2 > x1)

    x_t = np.abs(x2-x1)
    y_t = np.abs(y2-y1)
    if res:
        overlap = np.abs(x2 - y1)/y_t
    else:
        overlap = 0
    
    if overlap > 1.0:
        overlap = np.abs(y2 - x1)/x_t

    return res, overlap

def get_scan_in_zip(input_path, series_description):
    list_of_filenames = []
    list_of_array = {}
    list_of_ipp_0 = []
    list_of_ipp_1 = []
    list_of_ipp_2 = []
    list_of_series = []
    list_of_series_type = []
    list_of_pixel_spacing = []
    list_of_slice_thickness = []
    list_of_slice_location = []
    list_of_slice_location_min = []
    list_of_slice_location_max = []
    list_of_slice_rows = []
    list_of_slice_columns = []
    with ZipFile(input_path, 'r') as zip_obj:
        filenames = zip_obj.namelist()
        for temp_input in filenames:
            try:
                with zip_obj.open(temp_input) as dcm_file:   
                    s_temp = pydicom.dcmread(dcm_file, force=True)
                    s_temp.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
                    pxl_arr = np.array(s_temp.pixel_array)

                _idx = find(s_temp.SeriesDescription, '_')
                sd = s_temp.SeriesDescription[_idx[-1]+1:]
                if sd == series_description:
                    list_of_filenames.append(temp_input)
                    list_of_array[temp_input] = pxl_arr
                    list_of_ipp_0.append(float(s_temp.ImagePositionPatient[0]))
                    list_of_ipp_1.append(float(s_temp.ImagePositionPatient[1]))
                    list_of_ipp_2.append(float(s_temp.ImagePositionPatient[2]))
                    list_of_series.append(int(s_temp.SeriesNumber))
                    list_of_series_type.append(sd)
                    list_of_pixel_spacing.append(float(s_temp.PixelSpacing[0]))
                    list_of_slice_thickness.append(float(s_temp.SliceThickness))
                    list_of_slice_location.append(float(s_temp.SliceLocation))
                    list_of_slice_location_min.append(float(s_temp.SliceLocation) - float(s_temp.SliceThickness)/2)
                    list_of_slice_location_max.append(float(s_temp.SliceLocation) + float(s_temp.SliceThickness)/2)
                    list_of_slice_rows.append(int(s_temp.Rows))
                    list_of_slice_columns.append(int(s_temp.Columns))
            except:
                print('')

    list_of_filenames = [x for _, x in sorted(zip(list_of_slice_location,list_of_filenames))]
    list_of_ipp_0 = [x for _, x in sorted(zip(list_of_slice_location,list_of_ipp_0))]
    list_of_ipp_1 = [x for _, x in sorted(zip(list_of_slice_location,list_of_ipp_1))]
    list_of_ipp_2 = [x for _, x in sorted(zip(list_of_slice_location,list_of_ipp_2))]
    list_of_series = [x for _, x in sorted(zip(list_of_slice_location,list_of_series))]
    list_of_pixel_spacing = [x for _, x in sorted(zip(list_of_slice_location,list_of_pixel_spacing))]
    list_of_slice_thickness = [x for _, x in sorted(zip(list_of_slice_location,list_of_slice_thickness))]
    list_of_slice_location_min = [x for _, x in sorted(zip(list_of_slice_location,list_of_slice_location_min))]
    list_of_slice_location_max = [x for _, x in sorted(zip(list_of_slice_location,list_of_slice_location_max))]
    list_of_slice_rows = [x for _, x in sorted(zip(list_of_slice_location,list_of_slice_rows))]
    list_of_slice_columns = [x for _, x in sorted(zip(list_of_slice_location,list_of_slice_columns))]
    list_of_slice_location = [x for _, x in sorted(zip(list_of_slice_location,list_of_slice_location))]
    meta = {}
    meta['filenames'] = list_of_filenames
    meta['ipp_0'] = list_of_ipp_0
    meta['ipp_1'] = list_of_ipp_1
    meta['ipp_2'] = list_of_ipp_2
    meta['series'] = list_of_series
    meta['pixel_spacing'] = list_of_pixel_spacing
    meta['slice_thickness'] = list_of_slice_thickness
    meta['slice_rows'] = list_of_slice_rows
    meta['slice_columns'] = list_of_slice_columns
    meta['slice_location'] = list_of_slice_location
    meta['slice_location_min'] = list_of_slice_location_min
    meta['slice_location_max'] = list_of_slice_location_max

    if(len(np.unique(list_of_pixel_spacing)) != 1):
        error
    else:
        pixel_spacing = list_of_pixel_spacing[0]

    max_row = np.max(np.unique(list_of_slice_rows))
    max_col = np.max(np.unique(list_of_slice_columns))
    max_st = np.max(np.unique(list_of_slice_thickness))
    unique_series = np.unique(list_of_series)

    data = {}
    for idx_ in range(len(list_of_filenames)):
        curr_meta = get_meta(meta,idx_)

        if curr_meta['series'] not in data:
            data[curr_meta['series']] = {}
            data[curr_meta['series']]['scan'] = []
            data[curr_meta['series']]['slice_location'] = []
            data[curr_meta['series']]['slice_thickness'] = curr_meta['slice_thickness']
            data[curr_meta['series']]['ipp_1'] = curr_meta['ipp_1']
            data[curr_meta['series']]['row'] = curr_meta['row']
            data[curr_meta['series']]['col'] = curr_meta['col']

        scan1 = list_of_array[curr_meta['filename']]
        if(len(data[curr_meta['series']]['scan']) == 0):
            data[curr_meta['series']]['scan'] = scan1[:,:,None]
        else:
            data[curr_meta['series']]['scan'] = np.concatenate((data[curr_meta['series']]['scan'], scan1[:,:,None]), axis=2)
        data[curr_meta['series']]['slice_location'].append(curr_meta['slice_location'])

    for d in data:
        curr_scan = data[d]['scan']
        curr_slice_location = data[d]['slice_location']
        factor = data[d]['slice_thickness'] / pixel_spacing

        scan_factor = [1.0, 1.0, factor]
        new_scan = nd.interpolation.zoom(curr_scan, scan_factor, mode='nearest')
        new_slice_location = nd.interpolation.zoom(curr_slice_location, [factor], mode='nearest')

        data[d]['scan'] = new_scan
        data[d]['slice_location'] = new_slice_location
        data[d]['slice_thickness'] = pixel_spacing
    
    series_ = list(data.keys())
    for s_ in range(len(series_)-1):
        if s_ == 0:
            bottom = data[series_[s_]]
        top = data[series_[s_+1]]
        diff_in_pixel_y = np.round((bottom['ipp_1'] - top['ipp_1']) / pixel_spacing).astype(int)

        # Get intersecting slice index
        top_min_slice_location = min(top['slice_location'])
        best_val = 9999
        best_idx = 9999
        for s_min in range(len(bottom['slice_location'])):
            curr_val = np.abs(bottom['slice_location'][s_min] - top_min_slice_location)
            if curr_val < best_val:
                best_val = curr_val
                best_idx = s_min

        # Merge
        b_sc = bottom['scan']
        b_sl = bottom['slice_location']
        t_sc = top['scan']
        t_sl = top['slice_location']
        
        # Merge: Slice Location
        sl_1 = b_sl[:best_idx]
        sl_2 = b_sl[best_idx:]
        sl_3 = t_sl[:len(sl_2)]
        sl_4 = t_sl[len(sl_2):]
        sl_ = np.hstack([sl_1,(sl_2+sl_3)/2,sl_4])
        
        # Remove Pixel in Rows
        if diff_in_pixel_y > 0:
            y_end = b_sc.shape[0] + diff_in_pixel_y
            t_sc = t_sc[diff_in_pixel_y:y_end,:,:]
        elif diff_in_pixel_y < 0:
            error
        else:
            error
            if scan1.shape[0] == scan2.shape[0]:
                error
            else:
                error

        # Seamless Cloning
        mid_2_3 = int(np.round(len(sl_2)/2))
        sc_1 = b_sc[:,:,:best_idx]
        sc_2 = b_sc[:,:,best_idx:]
        sc_3 = t_sc[:,:,:len(sl_2)]
        sc_4 = t_sc[:,:,len(sl_2):]

        vol_overlap = np.concatenate((sc_1,(sc_2+sc_3),sc_4),axis=2)

        m_1 = np.zeros(b_sc[:,:,:best_idx+mid_2_3].shape)
        m_4 = np.zeros( t_sc[:,:,mid_2_3:].shape)
        m_1[:,:,-2] = 1
        m_1[:,:,-1] = 1
        m_4[:,:,0] = 1
        m_4[:,:,1] = 1
        mask = np.concatenate((m_1,m_4),axis=2)

        sc_2 = sc_2[:,:,:mid_2_3]
        sc_3 = sc_3[:,:,mid_2_3:]
        sc_mid = np.concatenate((sc_2,sc_3),axis=2)
        vol = np.concatenate((sc_1,sc_mid,sc_4),axis=2)

        vol_f = seamless_axial(vol,vol_overlap,mask)
        vol_f = hist_match(vol_f, vol)

        sc_ = vol_f

        # Save Bottom
        bottom['scan'] = sc_
        bottom['slice_location'] = sl_
    volume = bottom
    return volume

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

def axi_to_sag(A):
    A = np.moveaxis(A, -1, 0)
    A = np.flip(A,axis=0)
    return A

def sag_to_axi(A):
    A = np.flip(A,axis=0)
    A = np.moveaxis(A, 0, -1)
    return A

def seamlessclone(target_img,source_img,mask):
    # get dimensions of source image
    height, width, depth = source_img.shape

    # number of valid neighbors
    ones_ = np.ones((height,width))
    neighbour_kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    N = sig.convolve2d(ones_, neighbour_kernel, mode='same')

    # get number of pixels to determine and create index matrix
    num_pixels = sum(sum((mask > 0)))
    tmp_ = np.argwhere(np.transpose(mask)>0)
    X = tmp_[:,0]
    Y = tmp_[:,1]
    indices = np.zeros((height,width),int)
    count = 0
    for i in range(num_pixels):
        y = Y[i]
        x = X[i]
        indices[y,x] = count
        count = count + 1

    # calculate laplacian at each point in source image
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    l = []
    d = []
    for i in range(depth):
        l.append(sig.convolve(source_img[:,:,i], laplacian, mode='same'))
        d.append(target_img[:,:,i])


    # In each row of the matrix, we will have 5 elements of fewer, so we use a sparse matrix to save space
    A = np.zeros((num_pixels,num_pixels))

    # 3 because we need one for each channel
    b = np.zeros((num_pixels,depth))

    # keep track of amount of pixels added so far
    count = -1

    # Constructs the system of linear equations - iterate over each pixel in the image
    for x in range(width):
        for y in range(height):
            # only add points that are in mask
            if mask[y,x] == 1:
                count = count + 1
                A[count,count] = N[y,x]
                
                # take care of neighbors
                # top boundary
                if y != 0:
                    if mask[y-1,x] == 1:
                        index = indices[y-1,x]
                        A[count,index] = -1
                    else:
                        for channel in range(depth):
                            b[count,channel] = b[count,channel] + d[channel][y-1,x]
                
                # left boundary
                if x != 0:
                    if mask[y,x-1] == 1:
                        index = indices[y,x-1]
                        A[count,index] = -1
                    else:
                        for channel in range(depth):
                            b[count,channel] = b[count,channel] + d[channel][y,x-1]
                
                # bottom boundary
                if y != (height-1):
                    if mask[y+1,x] == 1:
                        index = indices[y+1,x]
                        A[count,index] = -1
                    else:
                        for channel in range(depth):
                            b[count,channel] = b[count,channel] + d[channel][y+1,x]
                            
                # right boundary
                if x != (width-1):
                    if mask[y,x+1] == 1:
                        index = indices[y,x+1]
                        A[count,index] = -1
                    else:
                        for channel in range(depth):
                            b[count,channel] = b[count,channel] + d[channel][y,x+1]
                
                for channel in range(depth):
                    b[count,channel] = b[count,channel] + l[channel][y,x]

    # Determines new points and fills in image.
    out_img = target_img.copy()
    for channel in range(depth):
        points = np.linalg.solve(A, b[:,channel])
        for k in range(num_pixels):
            out_img[Y[k],X[k],channel] = points[k]

    return out_img

def seamless_axial(vol,vol_overlap,mask):
    vol = axi_to_sag(vol)
    vol_overlap = axi_to_sag(vol_overlap)
    mask = axi_to_sag(mask)

    # vol = (vol - np.min(vol))/np.ptp(vol)
    # imageio.imwrite('A.jpg', vol[:,:,65]) # 156xx224x115
    # vol_overlap = (vol_overlap - np.min(vol_overlap))/np.ptp(vol_overlap)
    # imageio.imwrite('B.jpg', vol_overlap[:,:,65]) # 156xx224x115
    # mask = (mask - np.min(mask))/np.ptp(mask)
    # imageio.imwrite('C.jpg', np.median(mask,axis=2)) # 156xx224x115

    out = seamlessclone(vol,vol_overlap,np.median(mask,axis=2))
    out = sag_to_axi(out)
    return out
