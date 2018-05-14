import warnings
from pathlib import Path
from itertools import product
from astropy.io import fits
from astropy.io.fits import Card
from astropy import table
from astropy.table import Table
from astropy.nddata import CCDData
import pickle
from .core import *

__all__=["Preprocessor"]


class Preprocessor():
    def __init__(self, topdir, rawdir, summary_keywords=USEFUL_KEYS):
        """
        Parameters
        ----------
        topdir : path-like
            The top directory of which all the other paths will be represented
            relative to.

        rawdir: path-like
            The directory where all the FITS files are stored (without any
            subdirectory)
        """
        topdir = Path(topdir)
        self.topdir = topdir  # e.g., Path('180412')
        self.rawdir = rawdir  # e.g., Path('180412', 'rawdata')
        self.rawpaths = list(Path(rawdir).glob('*.fits'))
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
                with open("newpaths.pkl", 'rb') as pkl:
                    self.newpaths = pickle.load(pkl)
            except FileNotFoundError:
                pass

        if self.objpaths is None:
            try:
                with open("objpaths.pkl", 'rb') as pkl:
                    self.objpaths = pickle.load(pkl)
            except FileNotFoundError:
                pass

        if self.biaspaths is None:
            try:
                with open("biaspaths.pkl", 'rb') as pkl:
                    self.biaspaths = pickle.load(pkl)
            except FileNotFoundError:
                pass

        if self.darkpaths is None:
            try:
                with open("darkpaths.pkl", 'rb') as pkl:
                    self.darkpaths = pickle.load(pkl)
            except FileNotFoundError:
                pass

        if self.flatpaths is None:
            try:
                with open("flatpaths.pkl", 'rb') as pkl:
                    self.flatpaths = pickle.load(pkl)
            except FileNotFoundError:
                pass


    # TRIPOL specific
    def organize_tripol(self,
                        rename_by=["FILTER", "COUNTER", "OBJECT", "EXPOS", "RET-ANG1"],
                        mkdir_by=["FILTER", "OBJECT"], delimiter='_',
                        archive_dir=None, verbose=False):
        ''' Rename FITS files after updating theur headers.
        Parameters
        ----------
        fpath: path-like
            The path to the target FITS file.
        newtop: path-like
            The top path for the new FITS file. If ``None``, the new path will
            share the parent path with ``fpath``.
        mkdir_by: list of str, optional
            The keys which will be used to make subdirectories to classify
            files. If given, subdirectories will be made with the header value
            of the keys.
        delimiter: str, optional
            The delimiter for the renaming.
        archive_dir: path-like or None, optional
            Where to move the original FITS file. If ``None``, the original file
            will remain there. Deleting original FITS is dangerous so it is only
            supported to move the files. You may delete files manually if
            needed.
        '''
        def _guess_hwpangle(hdr):
            try:
                hwpangle = float(hdr[KEYMAP["OBJECT"]].split('_')[-1])
                if hwpangle > 180:
                    hwpangle = hwpangle / 10
            except ValueError:
                hwpangle = 0

            return float(hwpangle)

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
            is_dummay = False
            is_object = False

            # Do not rename useless flat/test images but move to useless directory.
            if (obj[:4].lower() == 'flat'):
                flatfor = obj[-1]
                if flatfor == hdr[KEYMAP["FILTER"]]:
                    hdr[KEYMAP["OBJECT"]] = "flat"
                    cards.append(Card("FLATFOR", obj[-1],
                                      "The band for which the flat is taken"))
                else:
                    is_dummay = True
            elif obj.lower() == "test":
                is_dummay = True

            if is_dummay:
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

            # Add polarimetry-key (RET-ANG1) if there is none:
            if "RET-ANG1" not in hdr:
                hwpangle = _guess_hwpangle(hdr)
                cards.append(Card("RET-ANG1", hwpangle,
                                  "The half-wave plate angle."))

            # Calculate airmass by looking at the first 4 chars of OBJECT
            if obj[:4].lower() not in ['bias', 'dark', 'test']:
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
                        print(f"{fpath} failed in airmass calculation: ValueError")
                        print(am, full)
                    pass

                except KeyError:
                    if verbose:
                        print(f"{fpath} failed in airmass calculation: KeyError")
                    pass

                if obj[:4].lower() != "flat":
                    is_object = True

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
            if is_object:
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
       savedir: path-like, optional
            The directory where the frames will be saved.

        hdr_key, hdr_val: str or list of str
            The header key and values to identify the bias frames. Some
            combinations can be ``["OBJECT"]`` and ``["bias"]`` or
            ``["OBJECT", "EXPTIME"]`` and ``["dark", 0.0]``.

        group_by: None, str or list str, optional
            The header keywords to be used for grouping frames. For dark
            frames, usual choice can be ``['EXPTIME']``.

        comb_kwargs: dict or None, optional
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
            mbias = combine_ccd(group["file"],
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
        """ Makes and saves bias and dark (bias NOT subtracted) images.
        Parameters
        ----------
        savedir: path-like, optional
            The directory where the frames will be saved.

        hdr_keys, hdr_vals: str or list of str
            The header keyword and value to identify dark frames.

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
                raise KeyError("bias_grouped_by must be a subset of group_by for dark.")

        if exposure_key not in group_by:
            warnings.warn("group_by is not None and does not include "
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
            exptime_dark = float(group[exposure_key][0])
            darkpath = imgpath(["dark"] + group_by_vals,
                               delimiter=delimiter,
                               directory=savedir)

            mdark = combine_ccd(group["file"],
                                dtype=dtype,
                                **comb_kwargs,
                                type_key=hdr_keys+group_by,
                                type_val=hdr_vals+group_by_vals)

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
        '''
        Flat image must have the header key ``flat_key`` *starting* with the
        ``flat_startswith``. By default, it seeks for
        ``OBJECT = flat_<FILTER>_<HWPANGLE>``.

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
            exptime_dark = float(group[exposure_key][0])
            flatpath = imgpath(["flat"] + group_by_vals,
                               delimiter=delimiter,
                               directory=savedir)
            mflat = combine_ccd(group["file"],
                                dtype=dtype,
                                **comb_kwargs,
                                type_key=hdr_keys+group_by,
                                type_val=hdr_vals+group_by_vals)
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
                   exposure_key="EXPTIME",
                   bias_grouped_by=["FILTER"],
                   dark_grouped_by=["FILTER", "EXPTIME"],
                   flat_grouped_by=["FILTER", "RET-ANG1"],
                   verbose=True):

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

        savepaths=[]
        for fpath in self.objpaths:
            savepath = savedir / fpath.name
            savepaths.append(savepath)
            objrow = self.summary[self.summary['file'] == str(fpath)]
            bias_vals = tuple(objrow[bias_grouped_by][0].as_void())
            dark_vals = tuple(objrow[dark_grouped_by][0].as_void())
            flat_vals = tuple(objrow[flat_grouped_by][0].as_void())

            try:
                biaspath = self.biaspaths[bias_vals]
            except KeyError:
                biaspath = None
                warnings.warn(f"Bias with {bias_vals} not available.")

            try:
                darkpath = self.darkpaths[dark_vals]
            except KeyError:
                darkpath = None
                warnings.warn(f"Dark with {dark_vals} not available.")

            try:
                flatpath = self.flatpaths[flat_vals]
            except KeyError:
                flatpath = None
                warnings.warn(f"Flat with {flat_vals} not available.")

            objccd = CCDData.read(fpath)
            proc = bdf_process(objccd,
                               output=savepath,
                               unit=None,
                               mbiaspath=biaspath,
                               mdarkpath=darkpath,
                               mflatpath=flatpath)

        self.reducedpaths = savepaths

        return make_summary(self.reducedpaths,
                            output=self.topdir / "summary_reduced.csv",
                            format='ascii.csv',
                            keywords=self.summary_keywords + ["PROCESS"],
                            verbose=verbose)
