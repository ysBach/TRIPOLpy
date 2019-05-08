from warnings import warn
from pathlib import Path
from astropy.io import fits
from astropy.io.fits import Card
from astropy.io.fits.card import Undefined
from astropy.table import Table
from astropy.nddata import CCDData
import pickle
from .core import (USEFUL_KEYS, MEDCOMB_KEYS, KEYMAP,
                   mkdir, imgpath, fitsrenamer,
                   cards_gain_rdnoise,
                   airmass_hdr, cards_airmass,
                   combine_ccd, bdf_process,
                   make_summary)

__all__ = ["Preprocessor"]


class Preprocessor():
    def __init__(self, topdir, rawdir, summary_keywords=USEFUL_KEYS):
        """
        Parameters
        ----------
        topdir : path-like
            The top directory of which all the other paths will be represented
            relative to.

        rawdir : path-like
            The directory where all the FITS files are stored (without any
            subdirectory)

        summary_keywords : list of str, optional
            The keywords of the header to be used for the summary table.
        """
        topdir = Path(topdir)
        self.topdir = topdir  # e.g., Path('180412')
        self.rawdir = rawdir  # e.g., Path('180412', 'rawdata')
        self.rawpaths = list(Path(rawdir).glob('*.fits'))
        self.rawpaths.sort()
        self.summary_keywords = summary_keywords
        self.newpaths = None
        self.summary = None
        self.objpaths = None
        self.reducedpaths = None
        self.biaspaths = None
        self.darkpaths = None
        self.flatpaths = None
        # self.renamed = False
        # rawpaths: Original file paths
        # newpaths: Renamed paths
        # redpaths: Reduced frames paths excluding BDF

    def initialize_self(self):
        ''' Initialization may convenient when process was halted amid.
        '''
        if self.summary is None:
            try:
                self.summary = Table.read(str(self.topdir / "summary_raw.csv"),
                                          format='ascii.csv')
                self.newpaths = self.summary["file"].tolist()
            except FileNotFoundError:
                pass

        if self.newpaths is None:
            try:
                with open(self.topdir / "newpaths.pkl", 'rb') as pkl:
                    self.newpaths = pickle.load(pkl)
            except FileNotFoundError:
                pass

        if self.objpaths is None:
            try:
                with open(self.topdir / "objpaths.pkl", 'rb') as pkl:
                    self.objpaths = pickle.load(pkl)
            except FileNotFoundError:
                pass

        if self.biaspaths is None:
            try:
                with open(self.topdir / "biaspaths.pkl", 'rb') as pkl:
                    self.biaspaths = pickle.load(pkl)
            except FileNotFoundError:
                pass

        if self.darkpaths is None:
            try:
                with open(self.topdir / "darkpaths.pkl", 'rb') as pkl:
                    self.darkpaths = pickle.load(pkl)
            except FileNotFoundError:
                pass

        if self.flatpaths is None:
            try:
                with open(self.topdir / "flatpaths.pkl", 'rb') as pkl:
                    self.flatpaths = pickle.load(pkl)
            except FileNotFoundError:
                pass

    # TRIPOL specific
    def organize_tripol(self,
                        rename_by=["COUNTER", "FILTER",
                                   "OBJECT", "EXPOS", "RET-ANG1"],
                        mkdir_by=["FILTER", "OBJECT"], delimiter='_',
                        archive_dir=None, verbose=False):
        ''' Rename FITS files after updating theur headers.
        Parameters
        ----------
        fpath : path-like
            The path to the target FITS file.

        rename_by : list of str
            The keywords in header to be used for the renaming of FITS files.
            Each keyword values are connected by ``delimiter``.

        mkdir_by : list of str, optional
            The keys which will be used to make subdirectories to classify
            files. If given, subdirectories will be made with the header value
            of the keys.

        delimiter : str, optional
            The delimiter for the renaming.

        archive_dir : path-like or None, optional
            Where to move the original FITS file. If ``None``, the original file
            will remain there. Deleting original FITS is dangerous so it is only
            supported to move the files. You may delete files manually if
            needed.
        '''
        def _guess_hwpangle(hdr, fpath, imagetyp):
            try:
                # Best if OBJECT is [name_hwpangle] format.
                hwpangle = float(hdr[KEYMAP["OBJECT"]].split('_')[-1])
                if hwpangle > 180:
                    hwpangle = hwpangle / 10

            except ValueError:
                # If not so, set RET-ANG1 = 0.0 for non-object images
                if imagetyp != "object":
                    return 0.0

                # Otherwise, get input.
                else:
                    hwpangle = input(
                        f"{fpath.name}: HWP angle not found. Enter it (0, 22.5, 45, 67.5): ")

            return float(hwpangle)

        _valid_hwpangs = [0, 0.0, 22.5, 30, 30.0, 45, 45.0, 60, 60.0, 67.5]
        newpaths = []
        objpaths = []
        uselessdir = self.rawdir / "useless"
        mkdir(uselessdir)

        for fpath in self.rawpaths:
            # If it is TL image (e.g., ``g.fits``), delete it
            try:
                counter = fpath.name.split('_')[1][:4]

            except IndexError:
                print(f"{fpath.name} is not a regular TRIPOL FITS file. "
                      + "Maybe a TL image.")
                fpath.rename(uselessdir / fpath.name)
                continue

            hdr = fits.getheader(fpath)

            try:
                obj = hdr[KEYMAP["OBJECT"]].lower()
            except KeyError:
                print(f"{fpath} has no OBJECT! Skipping")
                continue

            cards = []
            imagetyp = None
            # is_dummay = False
            # is_object = False
            # is_flat = False

            # Do not rename useless flat/test images but move to useless directory.
            if (obj[:4].lower() == 'flat'):
                imagetyp = "flat"
                flatfor = obj[-1]
                if flatfor == hdr[KEYMAP["FILTER"]]:
                    hdr[KEYMAP["OBJECT"]] = "flat"
                    cards.append(Card("FLATFOR", obj[-1],
                                      "The band for which the flat is taken"))
                else:
                    imagetyp = "useless"

            elif obj.lower() in ["bias", "dark", "test"]:
                imagetyp = obj.lower()

            else:
                imagetyp = "object"

            # If useless, finish the loop here:
            if imagetyp == "useless":
                fpath.rename(uselessdir / fpath.name)
                continue

            # Add gain and rdnoise:
            filt = hdr[KEYMAP["FILTER"]]
            grdcards = cards_gain_rdnoise(filter_str=filt)
            [cards.append(c) for c in grdcards]

            # Add counter if there is none:
            if "COUNTER" not in hdr:
                cards.append(Card("COUNTER", counter, "Image counter"))

            # Add unit if there is none:
            if "BUNIT" not in hdr:
                cards.append(Card("BUNIT", "ADU", "Pixel value unit"))

            # Calculate airmass by looking at the first 4 chars of OBJECT
            if imagetyp in ["flat", "object"]:
                # FYI: flat MAY require airmass just for check (twilight/night)
                try:
                    am, full = airmass_hdr(hdr,
                                           ra_key="RA",
                                           dec_key="DEC",
                                           ut_key=KEYMAP["DATE-OBS"],
                                           exptime_key=KEYMAP["EXPTIME"],
                                           lon_key="LONGITUD",
                                           lat_key="LATITUDE",
                                           height_key="HEIGHT",
                                           equinox="J2000",
                                           frame='icrs',
                                           full=True)
                    amcards = cards_airmass(am, full)
                    [cards.append(c) for c in amcards]

                except ValueError:
                    if verbose:
                        print(
                            f"{fpath} failed in airmass calculation: ValueError")
                        print(am, full)

                except KeyError:
                    if verbose:
                        print(f"{fpath} failed in airmass calculation: KeyError")

            # Deal with RET-ANG1
            if ((imagetyp in ["object", "flat"])
                    and (not isinstance(hdr["RET-ANG1"], Undefined))):
                hwpangle_orig = hdr["RET-ANG1"]
                # Correctly tune the RET-ANG1 to float
                # (otherwise, it maybe understood as int...)
                if hwpangle_orig in _valid_hwpangs:
                    hdr["RET-ANG1"] = float(hdr["RET-ANG1"])

                # Sometimes it has, e.g., RET-ANG1 = "TIMEOUT" or 45.11,
                # although TRIPOL computer tries not to make these exceptions.
                else:
                    while True:
                        hwpangle = float(input(
                            f"{fpath.name}: HWP angle is now {hwpangle_orig}. Enter correct value (0, 22.5, 45, 67.5): "))
                        if hwpangle not in _valid_hwpangs:
                            accept_novalid = input(
                                "Your input is not a usual value. Still use? (y/n/?)")
                            if str(accept_novalid) == 'y':
                                break
                            elif str(accept_novalid) == '?':
                                print("Normally valid values:", _valid_hwpangs)
                            elif str(accept_novalid) != 'n':
                                continue
                        else:
                            break

                    hdr["RET-ANG1"] = float(hwpangle)
            elif isinstance(hdr["RET-ANG1"], Undefined):
                hdr["RET-ANG1"] = None

            # else:
            #     # In worst case (prior to 2019), we sometimes had to put hwp angle
            #     # in the OBJECT after ``_``.
            #     hwpangle = _guess_hwpangle(hdr, fpath, imagetyp)
            #     cards.append(Card("RET-ANG1", float(hwpangle),
            #                       "The half-wave plate angle."))

            add_hdr = fits.Header(cards)

            newpath = fitsrenamer(fpath,
                                  header=hdr,
                                  rename_by=rename_by,
                                  delimiter=delimiter,
                                  add_header=add_hdr,
                                  mkdir_by=mkdir_by,
                                  archive_dir=archive_dir,
                                  key_deprecation=True,
                                  keymap=KEYMAP,
                                  verbose=verbose)

            newpaths.append(newpath)
            if imagetyp == "object":
                objpaths.append(newpath)

        # Save list of file paths for future use.
        # It doesn't take much storage and easy to erase if you want.
        with open(self.topdir / 'newpaths.list', 'w+') as ll:
            for p in newpaths:
                ll.write(f"{str(p)}\n")

        with open(self.topdir / 'objpaths.list', 'w+') as ll:
            for p in objpaths:
                ll.write(f"{str(p)}\n")

        # Python specific pickle
        with open(self.topdir / 'newpaths.pkl', 'wb') as pkl:
            pickle.dump(newpaths, pkl)

        with open(self.topdir / 'objpaths.pkl', 'wb') as pkl:
            pickle.dump(objpaths, pkl)

        self.newpaths = newpaths
        self.objpaths = objpaths
        self.summary = make_summary(newpaths,
                                    output=self.topdir / "summary_raw.csv",
                                    format='ascii.csv',
                                    keywords=self.summary_keywords,
                                    verbose=verbose)

    # TRIPOL specific
    def make_bias(self, savedir=None, hdr_keys="OBJECT", hdr_vals="bias",
                  group_by=["FILTER"], delimiter='_', dtype='float32',
                  comb_kwargs=MEDCOMB_KEYS):
        ''' Finds and make bias frames.
        Parameters
        ----------
       savedir : path-like, optional.
            The directory where the frames will be saved.

        hdr_key : str or list of str, optional
            The header keys to be used for the identification of the bias
            frames. Each value should correspond to the same-index element of
            ``hdr_val``.

        hdr_val : str, float, int or list of such, optional
            The header key and values to identify the bias frames. Each value
            should correspond to the same-index element of ``hdr_key``.

        group_by : None, str or list str, optional.
            The header keywords to be used for grouping frames. For dark
            frames, usual choice can be ``['EXPTIME']``.

        delimiter : str, optional.
            The delimiter for the renaming.

        dtype : str or numpy.dtype object, optional.
            The data type you want for the final master bias frame. It is
            recommended to use ``float32`` or ``int16`` if there is no
            specific reason.

        comb_kwargs: dict or None, optional.
            The parameters for ``combine_ccd``.
        '''
        self.initialize_self()

        if group_by is None:
            group_by = ["OBJECT"]  # Dummy keyword

        if savedir is None:
            savedir = self.topdir

        savedir = Path(savedir)
        mkdir(savedir)

        if isinstance(hdr_keys, str):
            hdr_keys = [hdr_keys]
            hdr_vals = [hdr_vals]
            # not used isinstance(hdr_vals, str) since hdr_vals can be int, etc.

        savepaths = {}

        croptab = self.summary.copy()
        for k, v in zip(hdr_keys, hdr_vals):
            croptab = croptab[croptab[k] == v]

        grouped = croptab.group_by(group_by)

        for group in grouped.groups:
            group_by_vals = list(group[group_by][0].as_void())
            savepath = imgpath(["bias"] + group_by_vals,
                               delimiter=delimiter,
                               directory=savedir)
            _ = combine_ccd(group["file"],
                            output=savepath,
                            dtype=dtype,
                            **comb_kwargs,
                            type_key=hdr_keys,
                            type_val=hdr_vals)
            savepaths[tuple(group_by_vals)] = savepath

        # Save list of file paths for future use.
        # It doesn't take much storage and easy to erase if you want.
        with open(self.topdir / 'biaspaths.list', 'w+') as ll:
            for p in list(savepaths.values()):
                ll.write(f"{str(p)}\n")

        with open(self.topdir / 'biaspaths.pkl', 'wb') as pkl:
            pickle.dump(savepaths, pkl)

        self.biaspaths = savepaths

    def make_dark(self, savedir=None, hdr_keys="OBJECT", hdr_vals="dark",
                  bias_sub=True,
                  group_by=["FILTER", "EXPTIME"], bias_grouped_by=["FILTER"],
                  exposure_key="EXPTIME", dtype='float32',
                  delimiter='_', comb_kwargs=MEDCOMB_KEYS):
        """ Makes and saves dark (bias subtracted) images.
        Parameters
        ----------
        savedir: path-like, optional
            The directory where the frames will be saved.

        hdr_key : str or list of str, optional
            The header keys to be used for the identification of the bias
            frames. Each value should correspond to the same-index element of
            ``hdr_val``.

        hdr_val : str, float, int or list of such, optional
            The header key and values to identify the bias frames. Each value
            should correspond to the same-index element of ``hdr_key``.

        bias_sub: bool, optional
            If ``True``, subtracts bias from dark frames using self.biaspahts.

        group_by: None, str or list str, optional
            The header keywords to be used for grouping frames. For dark
            frames, usual choice can be ``['EXPTIME']``.

        bias_grouped_by: str or list of str, optional
            How the bias frames are grouped by.

        exposure_key: str, optional
            If you want to make bias from a list of dark frames, you need to
            let the function know the exposure time of the frames, so that the
            miniimum exposure time frame will be used as bias. Default is
            "EXPTIME".

        delimiter : str, optional.
            The delimiter for the renaming.

        dtype : str or numpy.dtype object, optional.
            The data type you want for the final master bias frame. It is
            recommended to use ``float32`` or ``int16`` if there is no
            specific reason.

        comb_kwargs: dict or None, optional
            The parameters for ``combine_ccd``.
        """

        # Initial settings
        self.initialize_self()

        # if bias_grouped_by is None:
        #     bias_grouped_by = ["OBJECT"]  # Dummy keyword

        # elif isinstance(bias_grouped_by, str):
        #     bias_grouped_by = [bias_grouped_by]

        # if group_by is None:
        #     group_by = ["OBJECT"]  # Dummy keyword

        # elif isinstance(group_by, str):
        #     group_by = [group_by]

        for k in bias_grouped_by:
            if k not in group_by:
                raise KeyError(
                    "bias_grouped_by must be a subset of group_by for dark.")

        if exposure_key not in group_by:
            warn("group_by is not None and does not include "
                 + f"exposure_key = {exposure_key}. Forced to append.")
            group_by.append(exposure_key)

        if savedir is None:
            savedir = self.topdir

        savedir = Path(savedir)
        mkdir(savedir)

        if isinstance(hdr_keys, str):
            hdr_keys = [hdr_keys]
            hdr_vals = [hdr_vals]
            # not used isinstance(hdr_vals, str) since hdr_vals can be int, etc.

        savepaths = {}

        croptab = self.summary.copy()
        for k, v in zip(hdr_keys, hdr_vals):
            croptab = croptab[croptab[k] == v]
        croptab.sort(exposure_key)
        grouped = croptab.group_by(group_by)
        # min_exp = darkgroups[exposure_key].astype(float).min()

        # Do dark combine:
        for group in grouped.groups:
            group_by_vals = list(group[group_by][0].as_void())
            # exptime_dark = float(group[exposure_key][0])
            darkpath = imgpath(["dark"] + group_by_vals,
                               delimiter=delimiter,
                               directory=savedir)

            mdark = combine_ccd(group["file"],
                                dtype=dtype,
                                **comb_kwargs,
                                type_key=hdr_keys + group_by,
                                type_val=hdr_vals + group_by_vals)

            if bias_sub:
                bias_vals = tuple(group[bias_grouped_by][0].as_void())
                biaspath = self.biaspaths[bias_vals]
                mdark = bdf_process(mdark, mbiaspath=biaspath, unit=None)

            mdark.write(darkpath, output_verify='fix', overwrite=True)

            savepaths[tuple(group_by_vals)] = darkpath

        # Save list of file paths for future use.
        # It doesn't take much storage and easy to erase if you want.
        with open(self.topdir / 'darkpaths.list', 'w+') as ll:
            for p in list(savepaths.values()):
                ll.write(f"{str(p)}\n")

        with open(self.topdir / 'darkpaths.pkl', 'wb') as pkl:
            pickle.dump(savepaths, pkl)

        self.darkpaths = savepaths

    # TRIPOL specific
    def make_flat(self, savedir=None,
                  hdr_keys=["OBJECT"], hdr_vals=["flat"],
                  group_by=["FILTER", "RET-ANG1"],
                  bias_sub=True, dark_sub=True,
                  bias_grouped_by=["FILTER"],
                  dark_grouped_by=["FILTER", "EXPTIME"],
                  exposure_key="EXPTIME",
                  comb_kwargs=MEDCOMB_KEYS, delimiter='_', dtype='float32'):
        '''Makes and saves flat images.
        Parameters
        ----------
        savedir: path-like, optional
            The directory where the frames will be saved.

        hdr_key : str or list of str, optional
            The header keys to be used for the identification of the bias
            frames. Each value should correspond to the same-index element of
            ``hdr_val``.

        hdr_val : str, float, int or list of such, optional
            The header key and values to identify the bias frames. Each value
            should correspond to the same-index element of ``hdr_key``.

        bias_sub, dark_sub : bool, optional
            If ``True``, subtracts bias and dark frames using ``self.biaspahts``
            and ``self.darkpaths``.

        group_by: None, str or list str, optional
            The header keywords to be used for grouping frames. For dark
            frames, usual choice can be ``['EXPTIME']``.

        bias_grouped_by, dark_grouped_by : str or list of str, optional
            How the bias and dark frames are grouped by.

        exposure_key: str, optional
            If you want to make bias from a list of dark frames, you need to
            let the function know the exposure time of the frames, so that the
            miniimum exposure time frame will be used as bias. Default is
            "EXPTIME".

        comb_kwargs: dict or None, optional
            The parameters for ``combine_ccd``.

        delimiter : str, optional.
            The delimiter for the renaming.

        dtype : str or numpy.dtype object, optional.
            The data type you want for the final master bias frame. It is
            recommended to use ``float32`` or ``int16`` if there is no
            specific reason.
        '''

        self.initialize_self()

        for k in bias_grouped_by:
            if k not in dark_grouped_by:
                raise KeyError(
                    "bias_grouped_by must be a subset of dark_grouped_by.")

        for k in bias_grouped_by:
            if k not in group_by:
                raise KeyError(
                    "bias_grouped_by must be a subset of group_by for flat.")

        if savedir is None:
            savedir = self.topdir

        savedir = Path(savedir)
        mkdir(savedir)

        if isinstance(hdr_keys, str):
            hdr_keys = [hdr_keys]
            hdr_vals = [hdr_vals]
            # not used isinstance(hdr_vals, str) since hdr_vals can be int, etc.

        savepaths = {}

        croptab = self.summary.copy()
        for k, v in zip(hdr_keys, hdr_vals):
            croptab = croptab[croptab[k] == v]

        grouped = croptab.group_by(group_by)

        # Do flat combine:
        for group in grouped.groups:
            group_by_vals = list(group[group_by][0].as_void())
            # exptime_dark = float(group[exposure_key][0])
            flatpath = imgpath(["flat"] + group_by_vals,
                               delimiter=delimiter,
                               directory=savedir)
            mflat = combine_ccd(group["file"],
                                dtype=dtype,
                                **comb_kwargs,
                                type_key=hdr_keys + group_by,
                                type_val=hdr_vals + group_by_vals)
            if bias_sub:
                bias_vals = tuple(group[bias_grouped_by][0].as_void())
                biaspath = self.biaspaths[bias_vals]
                mflat = bdf_process(mflat, mbiaspath=biaspath, unit=None)

            if dark_sub:
                dark_vals = tuple(group[dark_grouped_by][0].as_void())
                darkpath = self.darkpaths[dark_vals]
                mflat = bdf_process(mflat, mdarkpath=darkpath, unit=None)

            mflat.write(flatpath, output_verify='fix', overwrite=True)

            savepaths[tuple(group_by_vals)] = flatpath

        # Save list of file paths for future use.
        # It doesn't take much storage and easy to erase if you want.
        with open(self.topdir / 'flatpaths.list', 'w+') as ll:
            for p in list(savepaths.values()):
                ll.write(f"{str(p)}\n")

        with open(self.topdir / 'flatpaths.pkl', 'wb') as pkl:
            pickle.dump(savepaths, pkl)

        self.flatpaths = savepaths

    def do_preproc(self, savedir=None, delimiter='_', dtype='float32',
                   bias_grouped_by=["FILTER"],
                   dark_grouped_by=["FILTER", "EXPTIME"],
                   flat_grouped_by=["FILTER", "RET-ANG1"],
                   verbose_bdf=True, verbose_summary=False):
        ''' Conduct the preprocessing using simplified ``bdf_process``.
        Parameters
        ----------
        savedir: path-like, optional
            The directory where the frames will be saved.

        delimiter : str, optional.
            The delimiter for the renaming.

        dtype : str or numpy.dtype object, optional.
            The data type you want for the final master bias frame. It is
            recommended to use ``float32`` or ``int16`` if there is no
            specific reason.

        bias_grouped_by, dark_grouped_by : str or list of str, optional
            How the bias, dark, and flat frames are grouped by.
        '''
        self.initialize_self()

        for k in bias_grouped_by:
            if k not in dark_grouped_by:
                raise KeyError(
                    "bias_grouped_by must be a subset of dark_grouped_by.")

        for k in bias_grouped_by:
            if k not in flat_grouped_by:
                raise KeyError(
                    "bias_grouped_by must be a subset of group_by for flat.")

        if savedir is None:
            savedir = self.topdir

        savedir = Path(savedir)
        mkdir(savedir)

        savepaths = []
        for fpath in self.objpaths:
            savepath = savedir / fpath.name
            savepaths.append(savepath)
            objrow = self.summary[self.summary['file'] == str(fpath)]
            bias_vals = tuple(objrow[bias_grouped_by][0].as_void())
            dark_vals = tuple(objrow[dark_grouped_by][0].as_void())
            flat_vals = tuple(objrow[flat_grouped_by][0].as_void())

            try:
                biaspath = self.biaspaths[bias_vals]
            except (KeyError, TypeError):
                biaspath = None
                warn(f"Bias not available for {bias_vals}")

            try:
                darkpath = self.darkpaths[dark_vals]
            except (KeyError, TypeError):
                darkpath = None
                warn(f"Dark not available for {dark_vals}")

            try:
                flatpath = self.flatpaths[flat_vals]
            except (KeyError, TypeError):
                flatpath = None
                warn(f"Flat not available for {flat_vals}")

            objccd = CCDData.read(fpath)
            _ = bdf_process(objccd,
                            output=savepath,
                            unit=None,
                            mbiaspath=biaspath,
                            mdarkpath=darkpath,
                            mflatpath=flatpath,
                            verbose_bdf=verbose_bdf)

        self.reducedpaths = savepaths

        return make_summary(self.reducedpaths,
                            output=self.topdir / "summary_reduced.csv",
                            format='ascii.csv',
                            keywords=self.summary_keywords + ["PROCESS"],
                            verbose=verbose_summary)
