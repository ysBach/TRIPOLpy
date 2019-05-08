# import warnings
from warnings import warn
from pathlib import Path
import shutil
import os

import numpy as np
import pandas as pd
from scipy.stats import itemfreq

from astropy.io import fits
from astropy.io.fits import Card
from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time
from astropy.table import Table, Column
from astropy.wcs import WCS
from astropy.nddata import CCDData, StdDevUncertainty
from astropy.modeling.functional_models import Gaussian1D
from astropy.modeling.fitting import LevMarLSQFitter

from ccdproc import (combine, trim_image,
                     subtract_bias, subtract_dark, flat_correct)

from ccdproc import sigma_func as ccdproc_mad2sigma_func

__all__ = ["KEYMAP", "GAIN_EPADU", "RDNOISE_E", "USEFUL_KEYS", "MEDCOMB_KEYS",
           "LACOSMIC",
           "imgpath", "mkdir", "cards_airmass", "cards_gain_rdnoise",
           "fits_newpath", "fitsrenamer", "load_if_exists",
           "stack_FITS", "calc_airmass", "airmass_obs", "airmass_hdr",
           "CCDData_astype", "combine_ccd", "make_errmap", "bdf_process",
           "make_summary", "Gfit2hist", "bias2rdnoise"]

MEDCOMB_KEYS = dict(overwrite=True,
                    unit='adu',
                    combine_method="median",
                    reject_method=None,
                    combine_uncertainty_function=None)

LATEST = "Jun2018"

GAIN_EPADU = dict(g=dict(default=1.82,
                         Jun2018=1.79),
                  r=dict(default=1.05,
                         Jun2018=1.71),
                  i=dict(default=2.00,
                         Jun2018=2.01))

RDNOISE_E = dict(g=dict(default=0,
                        Jun2018=32.1),
                 r=dict(default=0,
                        Jun2018=36.5),
                 i=dict(default=0,
                        Jun2018=18.7))

LACOSMIC = dict(sigclip=4.5, sigfrac=0.3, objlim=5.0,  # gain=1.0, readnoise=6.5,
                satlevel=np.inf, pssl=0.0, niter=4, sepmed=False,
                cleantype='medmask', fsmode='median', psfmodel='gauss',
                psffwhm=2.5, psfsize=7, psfk=None, psfbeta=4.765)

KEYMAP = {"EXPTIME": 'EXPOS', "GAIN": 'EGAIN', "OBJECT": 'OBJECT',
          "FILTER": 'FILTER', "EQUINOX": 'EPOCH',
          "DATE-OBS": 'DATE', "RDNOISE": None}

USEFUL_KEYS = ["EXPTIME", "FILTER", "DATE-OBS", "RET-ANG1", "CCD_TEMP",
               "CCD_COOL", "OBJECT", "EPOCH", "RA", "DEC", "ALT", "AZ",
               "AIRMASS"]


def reset_dir(topdir):
    topdir = Path(topdir)
    dirsattop = list(topdir.iterdir())
    dirsatraw = list((topdir / "rawdata").iterdir())

    for path in dirsattop:
        if path.name != "rawdata":
            if path.is_dir() and not path.name.startswith("."):
                shutil.rmtree(path)
            else:
                os.remove(path)

    for path in dirsatraw:
        if ((path.is_dir()) and (path.name != "archive")
                and (path.name != "useless") and not (path.name.startswith("."))):
            shutil.rmtree(path)
        else:
            fpaths = path.glob("*")
            for fpath in fpaths:
                os.rename(fpath, topdir / "rawdata" / fpath.name)

    shutil.rmtree(topdir / "rawdata" / "archive")
    shutil.rmtree(topdir / "rawdata" / "useless")


def imgpath(vals, delimiter='_', directory=None):
    ''' Gives the image path.
    '''
    if isinstance(vals, str):
        vals = [vals]

    imgname = ""

    for val in vals:
        imgname += str(val)
        imgname += str(delimiter)

    imgname = imgname[:-1] + ".fits"

    if directory is None:
        directory = '.'

    path = Path(directory) / imgname

    return path


def mkdir(fpath, mode=0o777, parents=True, exist_ok=True):
    ''' Convenience function for Path.mkdir()
    '''
    fpath = Path(fpath)
    Path.mkdir(fpath, mode=mode, parents=parents, exist_ok=exist_ok)


def cards_airmass(am, full):
    ''' Gives airmass and alt-az related header cards.
    '''
    amstr = ("TRIPOLpy's airmass calculation uses the same algorithm "
             + "as IRAF: From 'Some Factors Affecting the Accuracy of "
             + "Stellar Photometry with CCDs' by P. Stetson, DAO preprint, "
             + "September 1988.")

    # At some times, hdr["AIRMASS"] = am, for example, did not work for some
    # reasons which I don't know.... So I used Card. - YPBach 2018-05-04
    cs = [Card("AIRMASS", am, "Aaverage airmass (Stetson 1988)"),
          Card("ALT", full["alt"][0],
               "Altitude (start of the exposure)"),
          Card("AZ", full["az"][0], "Azimuth (start of the exposure)"),
          Card("ALT_MID", full["alt"][1],
               "Altitude (midpoint of the exposure)"),
          Card("AZ_MID", full["az"][1],
               "Azimuth (midpoint of the exposure)"),
          Card("ALT_END", full["alt"][2],
               "Altitude (end of the exposure)"),
          Card("AZ_END", full["az"][2],
               "Azimuth (end of the exposure)"),
          Card("COMMENT", amstr),
          Card("HISTORY", "ALT-AZ calculated from TRIPOLpy."),
          Card("HISTORY", "AIRMASS calculated from TRIPOLpy.")]

    return cs


def cards_gain_rdnoise(filter_str):
    cs = [Card("GAIN",
               GAIN_EPADU[filter_str][LATEST],
               f"[e-/ADU] The electron gain factor ({LATEST})."),
          Card("RDNOISE",
               RDNOISE_E[filter_str][LATEST],
               f"[e-] The (Gaussian) read noise ({LATEST})."),
          Card("COMMENT",
               (f"Gain history ({filter_str}-band CCD): "
                + f"{GAIN_EPADU[filter_str]} e/ADU.")),
          Card("COMMENT",
               (f"Read noise history ({filter_str}-band CCD): "
                + f"{RDNOISE_E[filter_str]} e."))]

    return cs


def fits_newpath(fpath, rename_by, mkdir_by=None, header=None, delimiter='_',
                 ext='fits'):
    ''' Gives the new path of the FITS file from header.
    Parameters
    ----------
    fpath: path-like
        The path to the original FITS file.
    rename_by: list of str, optional
        The keywords of the FITS header to rename by.
    mkdir_by: list of str, optional
        The keys which will be used to make subdirectories to classify files.
        If given, subdirectories will be made with the header value of the keys.
    header: Header object, optional
        The header to extract ``mkdir_by``. If ``None``, the function will do
        ``header = fits.getheader(fpath)``.
    delimiter: str, optional
        The delimiter for the renaming.
    ext: str, optional
        The extension of the file name to be returned. Normally it should be
        ``'fits'`` since this function is ``fits_newname``.
    '''

    if header is None:
        header = fits.getheader(fpath)

    # First make file name without parent path
    newname = ""
    for k in rename_by:
        newname += str(header[k])
        newname += delimiter

    newname = newname[:-1] + '.fits'

    newpath = Path(fpath.parent)

    if mkdir_by is not None:
        for k in mkdir_by:
            newpath = newpath / header[k]

    newpath = newpath / newname

    return newpath


def key_mapper(header, keymap, deprecation=False):
    ''' Update the header to meed the standard (keymap).
    Parameters
    ----------
    header: Header
        The header to be modified
    keymap: dict
        The dictionary contains ``{<standard_key>:<original_key>}`` information
    deprecation: bool, optional
        Whether to change the original keywords' comments to contain deprecation
        warning. If ``True``, the original keywords' comments will become
        ``Deprecated. See <standard_key>.``.
    '''
    newhdr = header.copy()
    for k_new, k_old in keymap.items():
        # if k_new already in the header, only deprecate k_old.
        # if not, copy k_old to k_new and deprecate k_old.
        if k_old is not None:
            if k_new in newhdr:
                if deprecation:
                    newhdr.comments[k_old] = f"Deprecated. See {k_new}"
            else:
                try:
                    comment_ori = newhdr.comments[k_old]
                    newhdr[k_new] = (newhdr[k_old], comment_ori)
                    if deprecation:
                        newhdr.comments[k_old] = f"Deprecated. See {k_new}"
                except KeyError:
                    pass

    return newhdr


def fitsrenamer(fpath=None, header=None, newtop=None, rename_by=["OBJECT"],
                mkdir_by=None, delimiter='_', archive_dir=None, keymap=None,
                key_deprecation=True,
                verbose=True, add_header=None):
    ''' Renames a FITS file by ``rename_by`` with delimiter.
    Parameters
    ----------
    fpath: path-like
        The path to the target FITS file.
    header: Header, optional
        The header of the fits file. If given, don't open ``fpath``.
    newtop: path-like
        The top path for the new FITS file. If ``None``, the new path will share
        the parent path with ``fpath``.
    rename_by: list of str, optional
        The keywords of the FITS header to rename by.
    mkdir_by: list of str, optional
        The keys which will be used to make subdirectories to classify files.
        If given, subdirectories will be made with the header value of the keys.
    delimiter: str, optional
        The delimiter for the renaming.
    archive_dir: path-like or None, optional
        Where to move the original FITS file. If ``None``, the original file
        will remain there. Deleting original FITS is dangerous so it is only
        supported to move the files. You may delete files manually if needed.
    keymap: dict or None, optional
        If not ``None``, the keymapping is done by using the dict of ``keymap``
        in the format of ``{<standard_key>:<original_key>}``.
    key_deprecation: bool, optional
        Whether to change the original keywords' comments to contain deprecation
        warning. If ``True``, the original keywords' comments will become
        ``Deprecated. See <standard_key>.``.
    add_header: header or dict
        The header keyword, value (and comment) to add after the renaming.
    '''

    # Load fits file
    hdul = fits.open(fpath)
    if header is None:
        header = hdul[0].header

    # add keyword
    if add_header is not None:
        header += add_header

    # Copy keys based on KEYMAP
    if keymap is not None:
        header = key_mapper(header, keymap, deprecation=key_deprecation)

    newhdul = fits.PrimaryHDU(data=hdul[0].data, header=header)

    # Set the new path
    newpath = fits_newpath(fpath, rename_by, mkdir_by=mkdir_by, header=header,
                           delimiter=delimiter, ext='fits')
    if newtop is not None:
        newpath = Path(newtop) / newpath

    mkdir(newpath.parent)

    if verbose:
        print(f"Rename {fpath.name} to {newpath}")

    hdul.close()
    newhdul.writeto(newpath, output_verify='fix')

    if archive_dir is not None:
        archive_dir = Path(archive_dir)
        archive_path = archive_dir / fpath.name
        mkdir(archive_path.parent)
        if verbose:
            print(f"Moving {fpath.name} to {archive_path}")
        fpath.rename(archive_path)

    return newpath


def load_if_exists(path, loader, if_not=None, **kwargs):
    ''' Load a file if it exists.
    Parameters
    ----------
    path: pathlib.Path of Path-like str
        The path to be searched.
    loader: a function
        The loader to load ``path``. Can be ``CCDData.read``, ``np.loadtxt``, etc.
    if_not: str
        Give a python code as a str to be run if the loading failed.
    Returns
    -------
    loaded:
        The loaded file. If the file does not exist, ``None`` is returned.
    '''
    path = Path(path)
    if path.exists():
        print(f'Loading the existing {str(path)}...', end='')
        loaded = loader(path, **kwargs)
        print(" Done")
    elif if_not is not None:
        loaded = eval(if_not)
    else:
        loaded = None
    return loaded


def calc_airmass(zd_deg=None, cos_zd=None, scale=750.):
    ''' Calculate airmass by nonrefracting radially symmetric atmosphere model.
    Note
    ----
    Wiki:
        https://en.wikipedia.org/wiki/Air_mass_(astronomy)#Nonrefracting_radially_symmetrical_atmosphere
    Identical to the airmass calculation at a given ZD of IRAF's
    asutil.setairmass:
        http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?setairmass

    Parameters
    ----------
    zd_deg: float, optional
        The zenithal distance in degrees
    cos_zd: float, optional
        The cosine of zenithal distance. If given, ``zd_deg`` is not used.
    scale: float, optional
        Earth radius divided by the atmospheric height (usually scale height)
        of the atmosphere. In IRAF documentation, it is mistakenly written that
        this ``scale`` is the "scale height".
    '''
    if zd_deg is None and cos_zd is None:
        raise ValueError("Either zd_deg or cos_zd should not be None.")

    if cos_zd is None:
        cos_zd = np.cos(np.deg2rad(zd_deg))

    am = np.sqrt((scale * cos_zd)**2 + 2 * scale + 1) - scale * cos_zd

    return am


def airmass_obs(targetcoord, obscoord, ut, exptime, scale=750., full=False):
    ''' Calculate airmass by nonrefracting radially symmetric atmosphere model.
    Note
    ----
    Wiki:
        https://en.wikipedia.org/wiki/Air_mass_(astronomy)#Nonrefracting_radially_symmetrical_atmosphere
    Identical to the airmass calculation for a given observational run of
    IRAF's asutil.setairmass:
        http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?setairmass
    Partly contributed by Kunwoo Kang (Seoul National University) in Apr 2018.

    '''
    if not isinstance(ut, Time):
        warn("ut is not Time object. "
             + "Assume format='isot', scale='utc'.")
        ut = Time(ut, format='isot', scale='utc')
    if not isinstance(exptime, u.Quantity):
        warn("exptime is not astropy Quantity. "
             + "Assume it is in seconds.")
        exptime = exptime * u.s

    t_start = ut
    t_mid = ut + exptime / 2
    t_final = ut + exptime

    altaz = {"alt": [], "az": [], "zd": [], "airmass": []}
    for t in [t_start, t_mid, t_final]:
        C_altaz = AltAz(obstime=t, location=obscoord)
        target = targetcoord.transform_to(C_altaz)
        alt = target.alt.to_string(unit=u.deg, sep=':')
        az = target.az.to_string(unit=u.deg, sep=':')
        zd = target.zen.to(u.deg).value
        am = calc_airmass(zd_deg=zd, scale=scale)
        altaz["alt"].append(alt)
        altaz["az"].append(az)
        altaz["zd"].append(zd)
        altaz["airmass"].append(am)

    am_simpson = (altaz["airmass"][0]
                  + 4 * altaz["airmass"][1]
                  + altaz["airmass"][2]) / 6

    if full:
        return am_simpson, altaz

    return am_simpson


def airmass_hdr(header=None, ra=None, dec=None, ut=None, exptime=None,
                lon=None, lat=None, height=None, equinox=None, frame=None,
                scale=750.,
                ra_key="RA", dec_key="DEC", ut_key="DATE-OBS",
                exptime_key="EXPTIME", lon_key="LONGITUD", lat_key="LATITUDE",
                height_key="HEIGHT", equinox_key="EQUINOX", frame_key="RADECSYS",
                ra_unit=u.hourangle, dec_unit=u.deg,
                exptime_unit=u.s, lon_unit=u.deg, lat_unit=u.deg,
                height_unit=u.m,
                ut_format='isot', ut_scale='utc',
                full=False
                ):
    ''' Calculate airmass using the header.
    Parameters
    ----------
    ra, dec: float or Quantity, optional
        The RA and DEC of the target. If not specified, it tries to find them
        in the header using ``ra_key`` and ``dec_key``.

    ut: str or Time, optional
        The *starting* time of the observation in UT.

    exptime: float or Time, optional
        The exposure time.

    lon, lat, height: str, float, or Quantity
        The longitude, latitude, and height of the observatory. See
        astropy.coordinates.EarthLocation.

    equinox, frame: str, optional
        The ``equinox`` and ``frame`` for SkyCoord.

    scale: float, optional
        Earth radius divided by the atmospheric height (usually scale height)
        of the atmosphere.

    XX_key: str, optional
        The header key to find XX if ``XX`` is ``None``.

    XX_unit: Quantity, optional
        The unit of ``XX``

    ut_format, ut_scale: str, optional
        The ``format`` and ``scale`` for Time.

    full: bool, optional
        Whether to return the full calculated results. If ``False``, it returns
        the averaged (Simpson's 1/3-rule calculated) airmass only.
    '''
    def _conversion(header, val, key, unit=None, instance=None):
        if val is None:
            val = header[key]
        elif (instance is not None) and (unit is not None):
            if isinstance(val, instance):
                val = val.to(unit).value

        return val

    ra = _conversion(header, ra, ra_key, ra_unit, u.Quantity)
    dec = _conversion(header, dec, dec_key, dec_unit, u.Quantity)
    exptime = _conversion(header, exptime, exptime_key,
                          exptime_unit, u.Quantity)
    lon = _conversion(header, lon, lon_key, lon_unit, u.Quantity)
    lat = _conversion(header, lat, lat_key, lat_unit, u.Quantity)
    height = _conversion(header, height, height_key, height_unit, u.Quantity)
    equinox = _conversion(header, equinox, equinox_key)
    frame = _conversion(header, frame, frame_key)

    if ut is None:
        ut = Time(header[ut_key], format=ut_format, scale=ut_scale)
    elif isinstance(ut, Time):
        ut = ut.isot

    targetcoord = SkyCoord(ra=ra,
                           dec=dec,
                           unit=(ra_unit, dec_unit),
                           frame=frame,
                           equinox=equinox)

    try:
        observcoord = EarthLocation(lon=lon * lon_unit,
                                    lat=lat * lat_unit,
                                    height=height * height_unit)

    except ValueError:
        observcoord = EarthLocation(lon=lon,
                                    lat=lat,
                                    height=height * height_unit)

    result = airmass_obs(targetcoord=targetcoord,
                         obscoord=observcoord,
                         ut=ut,
                         exptime=exptime * exptime_unit,
                         scale=scale,
                         full=full)

    return result


def load_ccd(path, extension=0, usewcs=True, uncertainty_ext="UNCERT",
             unit='adu'):
    '''remove it when astropy updated:
    Note
    ----
    CCDData.read cannot read TPV WCS:
    https://github.com/astropy/astropy/issues/7650
    '''
    hdul = fits.open(path)
    hdu = hdul[extension]
    try:
        unc = StdDevUncertainty(hdul[uncertainty_ext].data)
    except KeyError:
        unc = None

    w = None
    if usewcs:
        w = WCS(hdu.header)

    ccd = CCDData(data=hdu.data, header=hdu.header, wcs=w,
                  uncertainty=unc, unit=unit)
    return ccd


def stack_FITS(fitslist=None, summary_table=None, extension=0,
               unit='adu', table_filecol="file", trim_fits_section=None,
               loadccd=True, type_key=None, type_val=None):
    ''' Stacks the FITS files specified in fitslist
    Parameters
    ----------
    fitslist: path-like, list of path-like, or list of CCDData, optional.
        The list of path to FITS files or the list of CCDData to be stacked.
        It is useful to give list of CCDData if you have already stacked/loaded
        FITS file into a list by your own criteria. If ``None`` (default),
        you must give ``fitslist`` or ``summary_table``. If it is not ``None``,
        this function will do very similar job to that of ``ccdproc.combine``.
        Although it is not a good idea, a mixed list of CCDData and paths to
        the files is also acceptable.

    summary_table: pandas.DataFrame or astropy.table.Table, optional.
        The table which contains the metadata of files. If there are many
        FITS files and you want to use stacking many times, it is better to
        make a summary table by ``filemgmt.make_summary`` and use that instead
        of opening FITS files' headers every time you call this function. If
        you want to use ``summary_table`` instead of ``fitslist`` and have set
        ``loadccd=True``, you must not have ``None`` or ``NaN`` value in the
        ``summary_table[table_filecol]``.

    extension : int or str, optional.
        The extension of FITS to be stacked. For single extension, set it as 0.

    unit : `~astropy.units.Unit` or str, optional.
        The units of the data.
        Default is ``'adu'``.

    table_filecol:  str, optional.
        The column name of the ``summary_table`` which contains the path to
        the FITS files.

    trim_fits_section: str, optional
        Region of ``ccd`` to be trimmed; see ``ccdproc.subtract_overscan`` for
        details.
        Default is ``None``.

    loadccd: bool, optional
        Whether to return file paths or loaded CCDData. If ``False``, it is
        a function to select FITS files using ``type_key`` and ``type_val``
        without using much memory.

    Return
    ------
    matched: list of Path or list of CCDData
        list containing Path to files if ``loadccd`` is ``False``. Otherwise
        it is a list containing loaded CCDData after loading the files. If
        ``ccdlist`` is given a priori, list of CCDData will be returned
        regardless of ``loadccd``.
    '''
    def _parse_val(value):
        val = str(value)
        if val.lstrip('+-').isdigit():  # if int
            result = int(val)
        else:
            try:
                result = float(val)
            except ValueError:
                result = str(val)
        return result

    def _check_mismatch(row):
        mismatch = False
        for k, v in zip(type_key, type_val):
            hdr_val = _parse_val(row[k])
            parse_v = _parse_val(v)
            if (hdr_val != parse_v):
                mismatch = True
                break
        return mismatch

    if ((fitslist is not None) + (summary_table is not None) != 1):
        raise ValueError(
            "One and only one of fitslist or summary_table must be not None.")

    # If fitslist
    if (fitslist is not None) and (not isinstance(fitslist, list)):
        fitslist = [fitslist]
        # raise TypeError(
        #     f"fitslist must be a list. It's now {type(fitslist)}.")

    # If summary_table
    if summary_table is not None:
        if ((not isinstance(summary_table, Table))
                and (not isinstance(summary_table, pd.DataFrame))):
            raise TypeError(
                f"summary_table must be an astropy Table or Pandas DataFrame. It's now {type(summary_table)}.")

    # Check for type_key and type_val
    if ((type_key is None) ^ (type_val is None)):
        raise ValueError(
            "type_key and type_val must be both specified or both None.")

    # Setting whether to group
    grouping = False
    if type_key is not None:
        if len(type_key) != len(type_val):
            raise ValueError(
                "type_key and type_val must be of the same length.")
        grouping = True
        # If str, change to list:
        if isinstance(type_key, str):
            type_key = [type_key]
        if isinstance(type_val, str):
            type_val = [type_val]

    matched = []

    print("Analyzing FITS... ", end='')
    # Set fitslist and summary_table based on the given input and grouping.
    if fitslist is not None:
        if grouping:
            summary_table = make_summary(fitslist, extension=extension,
                                         verbose=True, fname_option='relative',
                                         keywords=type_key, sort_by=None)
            summary_table = summary_table.to_pandas()
    elif summary_table is not None:
        fitslist = summary_table[table_filecol].tolist()
        if isinstance(summary_table, Table):
            summary_table = summary_table.to_pandas()

    print("Done", end='')
    if load_ccd:
        print(" and loading FITS... ")
    else:
        print(".")

    # Append appropriate CCDs or filepaths to matched
    if grouping:
        for i, row in summary_table.iterrows():
            mismatch = _check_mismatch(row)
            if mismatch:  # skip this row (file)
                continue

            # if not skipped:
            # TODO: Is is better to remove Path here?
            if isinstance(fitslist[i], CCDData):
                matched.append(fitslist[i])
            else:  # it must be a path to the file
                fpath = Path(fitslist[i])
                if loadccd:
                    ccd_i = load_ccd(fpath, extension=extension, unit=unit)
                    if trim_fits_section is not None:
                        ccd_i = trim_image(
                            ccd_i, fits_section=trim_fits_section)
                    matched.append(ccd_i)
                else:
                    matched.append(fpath)
    else:
        for item in fitslist:
            if isinstance(item, CCDData):
                matched.append(item)
            else:
                if loadccd:
                    ccd_i = load_ccd(fpath, extension=extension, unit=unit)
                    if trim_fits_section is not None:
                        ccd_i = trim_image(
                            ccd_i, fits_section=trim_fits_section)
                    matched.append(ccd_i)
                else:  # TODO: Is is better to remove Path here?
                    matched.append(Path(fpath))

    # Generate warning OR information messages
    if len(matched) == 0:
        if grouping:
            warn('No FITS file had "{:s} = {:s}"'.format(str(type_key),
                                                         str(type_val))
                 + "Maybe int/float/str confusing?")
        else:
            warn('No FITS file found')
    else:
        if grouping:
            N = len(matched)
            ks = str(type_key)
            vs = str(type_val)
            if load_ccd:
                print(f'{N} FITS files with "{ks} = {vs}" are loaded.')
            else:
                print(f'{N} FITS files with "{ks} = {vs}" are selected.')
        else:
            if load_ccd:
                print('{:d} FITS files are loaded.'.format(len(matched)))

    return matched


def CCDData_astype(ccd, dtype='float32'):
    ccd.data = ccd.data.astype(dtype)
    try:
        ccd.uncertainty.array = ccd.uncertainty.array.astype(dtype)
    except AttributeError:
        pass
    return ccd


def combine_ccd(fitslist, trim_fits_section=None, output=None, unit='adu',
                subtract_frame=None, combine_method='median', reject_method=None,
                normalize_exposure=False, exposure_key='EXPTIME',
                combine_uncertainty_function=ccdproc_mad2sigma_func,
                extension=0, dtype=np.float32, type_key=None, type_val=None,
                output_verify='fix', overwrite=False,
                **kwargs):
    ''' Combining images
    Slight variant from ccdproc.
    # TODO: accept the input like ``sigma_clip_func='median'``, etc.
    # TODO: normalize maybe useless..
    Parameters
    ----------
    fitslist: list of str, path-like
        list of FITS files.

    trim_fits_section : str or None, optional
        The ``fits_section`` of ``ccdproc.trim_image``.
        Region of ``ccd`` from which the overscan is extracted; see
        `~ccdproc.subtract_overscan` for details.
        Default is ``None``.

    output : path-like or None, optional.
        The path if you want to save the resulting ``ccd`` object.
        Default is ``None``.

    unit : `~astropy.units.Unit` or str, optional.
        The units of the data.
        Default is ``'adu'``.

    subtract_frame : array-like, optional.
        The frame you want to subtract from the image after the combination.
        It can be, e.g., dark frame, because it is easier to calculate Poisson
        error before the dark subtraction and subtract the dark later.
        TODO: This maybe unnecessary.
        Default is ``None``.

    combine_method : str or None, optinal.
        The ``method`` for ``ccdproc.combine``, i.e., {'average', 'median', 'sum'}
        Default is ``None``.

    reject_method : str
        Made for simple use of ``ccdproc.combine``,
        {None, 'minmax', 'sigclip' == 'sigma_clip', 'extrema'}. Automatically
        turns on the option, e.g., ``clip_extrema = True`` or
        ``sigma_clip = True``.
        Leave it blank for no rejection.
        Default is ``None``.

    normalize_exposure : bool, optional.
        Whether to normalize the values by the exposure time before combining.
        Default is ``False``.

    exposure_key : str, optional
        The header keyword for the exposure time.
        Default is ``"EXPTIME"``.

    combine_uncertainty_function : callable, None, optional
        The uncertainty calculation function of ``ccdproc.combine``.
        If ``None`` use the default uncertainty func when using average,
        median or sum combine, otherwise use the function provided.
        Default is ``None``.

    extension: int or str, optional
        The extension to be used.
        Default is ``0``.

    dtype : str or `numpy.dtype` or None, optional
        Allows user to set dtype. See `numpy.array` ``dtype`` parameter
        description. If ``None`` it uses ``np.float64``.
        Default is ``None``.

    type_key, type_val: str, list of str
        The header keyword for the ccd type, and the value you want to match.
        For an open HDU named ``hdu``, e.g., only the files which satisfies
        ``hdu[extension].header[type_key] == type_val`` among all the ``fitslist``
        will be used.

    output_verify : str
        Output verification option.  Must be one of ``"fix"``, ``"silentfix"``,
        ``"ignore"``, ``"warn"``, or ``"exception"``.  May also be any
        combination of ``"fix"`` or ``"silentfix"`` with ``"+ignore"``,
        ``+warn``, or ``+exception" (e.g. ``"fix+warn"``).  See the astropy
        documentation below:
        http://docs.astropy.org/en/stable/io/fits/api/verification.html#verify

    **kwarg:
        kwargs for the ``ccdproc.combine``. See its documentation.
        This includes (RHS are the default values)
        ```
        weights=None,
        scale=None,
        mem_limit=16000000000.0,
        clip_extrema=False,
        nlow=1,
        nhigh=1,
        minmax_clip=False,
        minmax_clip_min=None,
        minmax_clip_max=None,
        sigma_clip=False,
        sigma_clip_low_thresh=3,
        sigma_clip_high_thresh=3,
        sigma_clip_func=<numpy.ma.core._frommethod instance>,
        sigma_clip_dev_func=<numpy.ma.core._frommethod instance>,
        dtype=None,
        combine_uncertainty_function=None, **ccdkwargs
        ```

    Returns
    -------
    master: astropy.nddata.CCDData
        Resulting combined ccd.

    '''

    def _set_reject_method(reject_method):
        ''' Convenience function for ccdproc.combine reject switches
        '''
        clip_extrema, minmax_clip, sigma_clip = False, False, False

        if reject_method == 'extrema':
            clip_extrema = True
        elif reject_method == 'minmax':
            minmax_clip = True
        elif ((reject_method == 'sigma_clip') or (reject_method == 'sigclip')):
            sigma_clip = True
        else:
            if reject_method is not None:
                raise KeyError("reject must be one of "
                               "{None, 'minmax', 'sigclip' == 'sigma_clip', 'extrema'}")

        return clip_extrema, minmax_clip, sigma_clip

    def _print_info(combine_method, Nccd, reject_method, **kwargs):
        if reject_method is None:
            reject_method = 'no'

        info_str = ('"{:s}" combine {:d} images by "{:s}" rejection')

        print(info_str.format(combine_method, Nccd, reject_method))
        print(dict(**kwargs))
        return

    def _normalize_exptime(ccdlist, exposure_key):
        _ccdlist = ccdlist.copy()
        exptimes = []

        for i in range(len(_ccdlist)):
            exptime = _ccdlist[i].header[exposure_key]
            exptimes.append(exptime)
            _ccdlist[i] = _ccdlist[i].divide(exptime)

        if len(np.unique(exptimes)) != 1:
            print('There are more than one exposure times:')
            print('\texptimes =', end=' ')
            print(np.unique(exptimes), end=' ')
            print('seconds')
        print('Normalized images by exposure time ("{:s}").'.format(
            exposure_key))

        return _ccdlist

    fitslist = list(fitslist)

    if (output is not None) and (Path(output).exists()):
        if overwrite:
            print(f"{output} already exists:\n\t", end='')
            print("But will be overridden.")
        else:
            print(f"{output} already exists:\n\t", end='')
            return load_if_exists(output, loader=CCDData.read, if_not=None)

    ccdlist = stack_FITS(fitslist=fitslist,
                         extension=extension,
                         unit=unit,
                         trim_fits_section=trim_fits_section,
                         type_key=type_key,
                         type_val=type_val)
    header = ccdlist[0].header

    _print_info(combine_method=combine_method,
                Nccd=len(ccdlist),
                reject_method=reject_method,
                dtype=dtype,
                **kwargs)

    # Normalize by exposure: useful if flat images have different exptimes.
    if normalize_exposure:
        ccdlist = _normalize_exptime(ccdlist, exposure_key)

    # Set rejection switches
    clip_extrema, minmax_clip, sigma_clip = _set_reject_method(reject_method)

    master = combine(img_list=ccdlist,
                     method=combine_method,
                     clip_extrema=clip_extrema,
                     minmax_clip=minmax_clip,
                     sigma_clip=sigma_clip,
                     combine_uncertainty_function=combine_uncertainty_function,
                     **kwargs)

    str_history = '{:d} images with {:s} = {:s} are {:s} combined '
    ncombine = len(ccdlist)
    header["NCOMBINE"] = ncombine
    header.add_history(str_history.format(ncombine,
                                          str(type_key),
                                          str(type_val),
                                          str(combine_method)))

    if subtract_frame is not None:
        subtract = CCDData(subtract_frame.copy())
        master.data = master.subtract(subtract).data
        header.add_history("Subtracted a user-provided frame")

    master.header = header
    master = CCDData_astype(master, dtype=dtype)

    if output is not None:
        master.write(output, output_verify=output_verify, overwrite=overwrite)

    return master


def get_from_header(header, key, unit=None, verbose=True,
                    default=0):
    ''' Get a variable from the header object.
    Parameters
    ----------
    header: Header
        The header to extract the value.
    key: str
        The header keyword to extract.
    unit: astropy unit
        The unit of the value.
    default: str, int, float, ..., or Quantity
        The default if not found from the header.
    '''

    def _change_to_quantity(x, desired=None):
        ''' Change the non-Quantity object to astropy Quantity.
        Parameters
        ----------
        x: object changable to astropy Quantity
            The input to be changed to a Quantity. If a Quantity is given, ``x`` is
            changed to the ``desired``, i.e., ``x.to(desired)``.
        desired: astropy Unit, optional
            The desired unit for ``x``.
        Returns
        -------
        ux: Quantity
        Note
        ----
        If Quantity, transform to ``desired``. If ``desired = None``, return it as
        is. If not Quantity, multiply the ``desired``. ``desired = None``, return
        ``x`` with dimensionless unscaled unit.
        '''
        if not isinstance(x, u.quantity.Quantity):
            if desired is None:
                ux = x * u.dimensionless_unscaled
            else:
                ux = x * desired
        else:
            if desired is None:
                ux = x
            else:
                ux = x.to(desired)
        return ux

    value = None

    try:
        value = header[key]
        if verbose:
            print(f"header: {key} = {value}")

    except KeyError:
        if default is not None:
            value = _change_to_quantity(default, desired=unit)
            warn(f"{key} not found in header: setting to {default}.")

    if unit is not None:
        value = value * unit

    return value


# NOTE: crrej should be done AFTER bias/dark and flat correction:
# http://www.astro.yale.edu/dokkum/lacosmic/notes.html
def bdf_process(ccd, output=None, mbiaspath=None, mdarkpath=None, mflatpath=None,
                fits_section=None, calc_err=False, unit='adu', gain=None,
                rdnoise=None, gain_key="GAIN", rdnoise_key="RDNOISE",
                gain_unit=u.electron / u.adu, rdnoise_unit=u.electron,
                dark_exposure=None, data_exposure=None, exposure_key="EXPTIME",
                exposure_unit=u.s, dark_scale=False,
                flat_min_value=None, flat_norm_value=None,
                do_crrej=False, verbose_crrej=False,
                verbose_bdf=True, output_verify='fix', overwrite=True,
                dtype="float32"):
    ''' Do bias, dark and flat process.
    Parameters
    ----------
    ccd: array-like
        The ccd to be processed.

    output : path-like or None, optional.
        The path if you want to save the resulting ``ccd`` object.
        Default is ``None``.

    mbiaspath, mdarkpath, mflatpath : path-like, optional.
        The path to master bias, dark, flat FITS files. If ``None``, the
        corresponding process is not done.

    fits_section: str, optional
        Region of ``ccd`` to be trimmed; see ``ccdproc.subtract_overscan`` for
        details. Default is ``None``.

    calc_err : bool, optional.
        Whether to calculate the error map based on Poisson and readnoise
        error propagation.

    unit : `~astropy.units.Unit` or str, optional.
        The units of the data.
        Default is ``'adu'``.

    gain, rdnoise : None, float, optional
        The gain and readnoise value. These are all ignored if ``calc_err=False``.
        If ``calc_err=True``, it automatically seeks for suitable gain and
        readnoise value. If ``gain`` or ``readnoise`` is specified, they are
        interpreted with ``gain_unit`` and ``rdnoise_unit``, respectively.
        If they are not specified, this function will seek for the header
        with keywords of ``gain_key`` and ``rdnoise_key``, and interprete the
        header value in the unit of ``gain_unit`` and ``rdnoise_unit``,
        respectively.

    gain_key, rdnoise_key : str, optional
        See ``gain``, ``rdnoise`` explanation above.
        These are all ignored if ``calc_err=False``.

    gain_unit, rdnoise_unit : astropy Unit, optional
        See ``gain``, ``rdnoise`` explanation above.
        These are all ignored if ``calc_err=False``.

    dark_exposure, data_exposure : None, float, astropy Quantity, optional
        The exposure times of dark and data frame, respectively. They should
        both be specified or both ``None``.
        These are all ignored if ``mdarkpath=None``.
        If both are not specified while ``mdarkpath`` is given, then the code
        automatically seeks for header's ``exposure_key``. Then interprete the
        value as the quantity with unit ``exposure_unit``.

        If ``mdkarpath`` is not ``None``, then these are passed to
        ``ccdproc.subtract_dark``.

    exposure_key : str, optional
        The header keyword for exposure time.
        Ignored if ``mdarkpath=None``.

    exposure_unit : astropy Unit, optional.
        The unit of the exposure time.
        Ignored if ``mdarkpath=None``.

    flat_min_value : float or None, optional
        min_value of `ccdproc.flat_correct`.
        Minimum value for flat field. The value can either be None and no
        minimum value is applied to the flat or specified by a float which
        will replace all values in the flat by the min_value.
        Default is ``None``.

    flat_norm_value : float or None, optional
        norm_value of `ccdproc.flat_correct`.
        If not ``None``, normalize flat field by this argument rather than the
        mean of the image. This allows fixing several different flat fields to
        have the same scale. If this value is negative or 0, a ``ValueError``
        is raised. Default is ``None``.

    output_verify : str
        Output verification option.  Must be one of ``"fix"``, ``"silentfix"``,
        ``"ignore"``, ``"warn"``, or ``"exception"``.  May also be any
        combination of ``"fix"`` or ``"silentfix"`` with ``"+ignore"``,
        ``+warn``, or ``+exception" (e.g. ``"fix+warn"``).  See the astropy
        documentation below:
        http://docs.astropy.org/en/stable/io/fits/api/verification.html#verify

    dtype : str or `numpy.dtype` or None, optional
        Allows user to set dtype. See `numpy.array` ``dtype`` parameter
        description. If ``None`` it uses ``np.float64``.
        Default is ``None``.
    '''

    proc = CCDData(ccd)
    hdr_new = proc.header
    hdr_new["PROCESS"] = ("", "The processed history: see comment.")
    hdr_new.add_comment("PROCESS key can be B (bias), D (dark), F (flat), "
                        + "T (trim), W (WCS), C(CRrej).")

    if mbiaspath is None:
        do_bias = False
        mbias = CCDData(np.zeros_like(ccd), unit=proc.unit)

    else:
        do_bias = True
        mbias = CCDData.read(mbiaspath, unit=unit)
        hdr_new["PROCESS"] += "B"
        hdr_new.add_history(f"Bias subtracted using {mbiaspath}")

    if mdarkpath is None:
        do_dark = False
        mdark = CCDData(np.zeros_like(ccd), unit=proc.unit)

    else:
        do_dark = True
        mdark = CCDData.read(mdarkpath, unit=unit)
        hdr_new["PROCESS"] += "D"
        hdr_new.add_history(f"Dark subtracted using {mdarkpath}")
        if dark_scale:
            hdr_new.add_history(
                f"Dark scaling {dark_scale} using {exposure_key}")

    if mflatpath is None:
        do_flat = False
        mflat = CCDData(np.ones_like(ccd), unit=proc.unit)

    else:
        do_flat = True
        mflat = CCDData.read(mflatpath)
        hdr_new["PROCESS"] += "F"
        hdr_new.add_history(f"Flat corrected using {mflatpath}")

    if fits_section is not None:
        proc = trim_image(proc, fits_section)
        mbias = trim_image(mbias, fits_section)
        mdark = trim_image(mdark, fits_section)
        mflat = trim_image(mflat, fits_section)
        hdr_new["PROCESS"] += "T"
        hdr_new.add_history(f"Trim by FITS section {fits_section}")

    if do_bias:
        proc = subtract_bias(proc, mbias)

    if do_dark:
        proc = subtract_dark(proc,
                             mdark,
                             dark_exposure=dark_exposure,
                             data_exposure=data_exposure,
                             exposure_time=exposure_key,
                             exposure_unit=exposure_unit,
                             scale=dark_scale)
        # if calc_err and verbose:
        #     if mdark.uncertainty is not None:
        #         print("Dark has uncertainty frame: Propagate in arithmetics.")
        #     else:
        #         print("Dark does NOT have uncertainty frame")

    if calc_err:
        if gain is None:
            gain = get_from_header(hdr_new, gain_key,
                                   unit=gain_unit,
                                   verbose=verbose_bdf,
                                   default=1.).value

        if rdnoise is None:
            rdnoise = get_from_header(hdr_new, rdnoise_key,
                                      unit=rdnoise_unit,
                                      verbose=verbose_bdf,
                                      default=0.).value

        err = make_errmap(proc,
                          gain_epadu=gain,
                          subtracted_dark=mdark)

        proc.uncertainty = StdDevUncertainty(err)
        errstr = (f"Error calculated using gain = {gain:.3f} [e/ADU] and "
                  + f"rdnoise = {rdnoise:.3f} [e].")
        hdr_new.add_history(errstr)

    if do_flat:
        if calc_err:
            if (mflat.uncertainty is not None) and verbose_bdf:
                print("Flat has uncertainty frame: Propagate in arithmetics.")
                hdr_new.add_history(
                    "Flat had uncertainty and is also propagated.")

        proc = flat_correct(proc,
                            mflat,
                            min_value=flat_min_value,
                            norm_value=flat_norm_value)

    # Do very simple L.A. Cosmic default crrejection
    if do_crrej:
        from astroscrappy import detect_cosmics

        if gain is None:
            gain = get_from_header(hdr_new, gain_key,
                                   unit=gain_unit,
                                   verbose=verbose_bdf,
                                   default=1.0).value

        if rdnoise is None:
            rdnoise = get_from_header(hdr_new, rdnoise_key,
                                      unit=rdnoise_unit,
                                      verbose=verbose_bdf,
                                      default=6.5).value

        crmask, cleanarr = detect_cosmics(proc.data, inmask=proc.mask,
                                          gain=gain, readnoise=rdnoise,
                                          **LACOSMIC, verbose=verbose_crrej)

        # create the new ccd data object
        proc.data = cleanarr
        if proc.mask is None:
            proc.mask = crmask
        else:
            proc.mask = proc.mask + crmask
        hdr_new["PROCESS"] += "C"
        hdr_new.add_history(
            f"Cosmic-Ray rejected by astroscrappy, LACOSMIC default setting.")

    proc = CCDData_astype(proc, dtype=dtype)
    proc.header = hdr_new

    if output is not None:
        proc.write(output, output_verify=output_verify, overwrite=overwrite)

    return proc


def make_errmap(ccd, gain_epadu, rdnoise_electron=0,
                subtracted_dark=None):
    ''' Calculate the usual error map.
    Parameters
    ----------
    ccd: array-like
        The ccd data which will be used to generate error map. It must be bias
        subtracted. If dark is subtracted, give ``subtracted_dark``. If the
        amount of this subtracted dark is negligible, you may just set
        ``subtracted_dark = None`` (default).
    gain: float, array-like, or Quantity
        The effective gain factor in ``electron/ADU`` unit.
    rdnoise: float, array-like, or Quantity, optional.
        The readout noise. Put ``rdnoise=0`` will calculate only the Poissonian
        error. This is useful when generating noise map for dark frames.
    subtracted_dark: array-like
        The subtracted dark map.
    '''
    data = ccd.copy()

    if isinstance(data, CCDData):
        data = data.data

    data[data < 0] = 0  # make all negative pixel to 0

    if isinstance(gain_epadu, u.Quantity):
        gain_epadu = gain_epadu.to(u.electron / u.adu).value

    if isinstance(rdnoise_electron, u.Quantity):
        rdnoise_electron = rdnoise_electron.to(u.electron)

    # Get Poisson noise
    if subtracted_dark is not None:
        dark = subtracted_dark.copy()
        if isinstance(dark, CCDData):
            dark = dark.data
        # If subtracted dark is negative, this may cause negative pixel in ``data``:
        data += dark

    var_Poisson = data / gain_epadu  # (data * gain) / gain**2 to make it ADU
    var_RDnoise = (rdnoise_electron / gain_epadu)**2

    errmap = np.sqrt(var_Poisson + var_RDnoise)

    return errmap


def make_summary(fitslist, extension=0, fname_option='relative',
                 output=None, format='ascii.csv',
                 keywords=[],
                 example_header=None, sort_by='file', verbose=True):
    """ Extracts summary from the headers of FITS files.
    Parameters
    ----------
    fitslist: list of str (path-like) or list of CCDData, optional
        The list of file paths relative to the current working directory, or
        the list of ccds to be summarized. It can be useful to give a list of
        CCDData if you have already stacked/loaded the CCDData into a list.
        Although it is not a good idea, a mixed list of CCDData and paths to
        the files is also acceptable.

    extension: int or str, optional
        The extension to be summarized.

    fname_option: str {'absolute', 'relative', 'name'}, optional
        Whether to save full absolute/relative path or only the filename.

    output: str or path-like, optional
        The directory and file name of the output summary file.

    format: str, optional
        The astropy.table.Table output format.

    keywords: list or str(``"*"``), optional
        The list of the keywords to extract (keywords should be in str).

    example_header: str or path-like, optional
        The path including the filename of the output summary text file.

    sort_by: str, optional
        The column name to sort the results. It can be any element of
        ``keywords`` or ``'file'``, which sorts the table by the file name.

    Return
    ------
    summarytab: astropy.Table

    Example
    -------
    >>> from pathlib import Path
    >>> import ysfitsutilpy as yfu
    >>> keys = ["OBS-TIME", "FILTER", "OBJECT"]  # actually it is case-insensitive
    >>> # The keywords you want to extract (from the headers of FITS files)
    >>> TOPPATH = Path(".", "observation_2018-01-01")
    >>> # The toppath
    >>> savepath = TOPPATH / "summary_20180101.csv"
    >>> # path to save summary csv file
    >>> allfits = list((TOPPATH / "rawdata").glob("*.fits"))
    >>> # list of all the fits files in Path object
    >>> summary = yfu.make_summary(pathlist=allfits, keywords=keys,
    >>>                            fname_option='name',
    >>>                            sort_by="DATE-OBS", output=savepath)
    >>> # The astropy.table.Table format.
    >>> # If you want, you may change it to pandas:
    >>> summary_pd = summary.to_pandas()`
    """

    if len(fitslist) == 0:
        print("No FITS file found.")
        return

    def _get_fname(path):
        if fname_option == 'relative':
            return str(path)
        elif fname_option == 'absolute':
            return str(path.absolute())
        else:
            return path.name

    def _get_hdr(item, extension):
        ''' Gets header from ``item``.
        '''
        if isinstance(item, CCDData):
            hdr = item.header
        else:
            hdul = fits.open(item)
            hdr = hdul[extension].header
            hdul.close()
        return hdr

    options = ['absolute', 'relative', 'name']
    if fname_option not in options:
        raise KeyError(f"fname_option must be one of {options}.")

    skip_keys = ['COMMENT', 'HISTORY']

    if verbose:
        if (keywords != []) and (keywords != '*'):
            print("Extracting keys: ", keywords)
        str_example_hdr = "Extract example header from 0-th\n\tand save as {:s}"
        str_keywords = "All {:d} keywords will be loaded."
        str_keyerror_fill = "Key {:s} not found for {:s}, filling with nan."
        str_filesave = 'Saving the summary file to "{:s}"'

    # Save example header
    if example_header is not None:
        example_fits = fitslist[0]
        if verbose:
            print(str_example_hdr.format(example_header))
        ex_hdr = _get_hdr(example_fits, extension=extension)
        ex_hdr.totextfile(example_header, overwrite=True)

    # load ALL keywords for special cases
    if (keywords == []) or (keywords == '*'):
        example_fits = fitslist[0]
        ex_hdr = _get_hdr(example_fits, extension=extension)
        N_hkeys = len(ex_hdr.cards)
        keywords = []

        for i in range(N_hkeys):
            key_i = ex_hdr.cards[i][0]
            if (key_i in skip_keys):
                continue
            elif (key_i in keywords):
                str_duplicate = "Key {:s} is duplicated! Only first one will be saved."
                print(str_duplicate.format(key_i))
                continue
            keywords.append(key_i)

        if verbose:
            print(str_keywords.format(len(keywords)))
#            except fits.VerifyError:
#                str_unparsable = '{:d}-th key is skipped since it is unparsable.'
#                print(str_unparsable.format(i))
#                continue

    # Initialize
    summarytab = dict(file=[])
    for k in keywords:
        summarytab[k] = []

    # Run through all the fits files
    for i, item in enumerate(fitslist):
        if isinstance(item, CCDData):
            summarytab["file"].append(None)
        else:
            summarytab["file"].append(_get_fname(item))
        hdr = _get_hdr(item, extension=extension)
        for k in keywords:
            try:
                summarytab[k].append(hdr[k])
            except KeyError:
                if verbose:
                    if isinstance(item, CCDData):
                        print(str_keyerror_fill.format(k, f"fitslist[{i}]"))
                    else:
                        print(str_keyerror_fill.format(k, str(item)))
                summarytab[k].append(None)

    summarytab = Table(summarytab)
    if sort_by is not None:
        summarytab.sort(sort_by)

    if output is not None:
        output = Path(output)
        if verbose:
            print(str_filesave.format(str(output)))
        summarytab.write(output, format=format)

    return summarytab


def Gfit2hist(data):
    ''' Gaussian fit to the frequency distribution of the nddata.
    '''
    freq = itemfreq(data.flatten())
    fitter = LevMarLSQFitter()
    mode = freq[freq[:, 1] == freq[:, 1].max(), 0][0]
    init = Gaussian1D(mean=mode)
    fitG = fitter(init, freq[:, 0], freq[:, 1])
    return fitG


def bias2rdnoise(data):
    ''' Infer readout noise from bias image.
    '''
    fitG = Gfit2hist(data)
    return fitG.stddev.value
