import warnings
from pathlib import Path
from itertools import product
from astropy.io import fits
from astropy.io.fits import Card
from astropy.table import Table
from .core import *

__all__=["Preprocessor"]

class Preprocessor():
    def __init__(self, toppath, summary_keywords=USEFUL_KEYS):
        self.toppath = toppath  # e.g., Path('180412', 'rawdata')
        self.rawpaths = list(Path(self.toppath).glob('*.fits'))
        self.summary_keywords = summary_keywords
        self.newpaths = None
        self.summary = None
        self.redpaths = None
        self.biaspaths = None
        self.flatpaths = None
        self.darkpaths = None
        # self.renamed = False
        # rawpaths: Original file paths
        # newpaths: Renamed paths
        # redpaths: Reduced frames paths excluding BDF

    # TRIPOL specific
    def organize_tripol(self, newtop=None,
                        rename_by=["COUNTER", "OBJECT", "EXPOS"],
                        mkdir_by=["FILTER"], delimiter='_', archive_dir=None,
                        keymapping=True, verbose=True):
        ''' Rename FITS files after updating theur headers.
        Parameters
        ----------
        fpath: path-like
            The path to the target FITS file.
        newtop: path-like
            The top path for the new FITS file. If ``None``, the new path will
            share the parent path with ``fpath``.
        rename_by: list of str, optional
            The keywords of the FITS header to rename by.
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
        keymapping: bool, optional
            Whether to add header keys based on KEYMAP.
        '''
        def _cards_airmass(am, full):
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
                       "Azimuth (end of the exposure)")
                  ]
            return cs

        newpaths = []
        for fpath in self.rawpaths:
            # If it is TL image (e.g., ``g.fits``), delete it
            try:
                counter = fpath.name.split('_')[1][:4]

            except IndexError:
                print(f"{fpath.name} is not a regular TRIPOL FITS file. "
                      + "Maybe a TL image.")
                continue

            amstr = ("TRIPOLpy's airmass calculation uses the same algorithm "
                    + "as IRAF: From 'Some Factors Affecting the Accuracy of "
                    + "Stellar Photometry with CCDs' by P. Stetson, DAO "
                    + "preprint, September 1988.")

            # Set the airmass and Alt-Az coordinates:
            # This is done outside of TRIPOL computer since it takes too much
            # time on that computer...
            hdr = fits.getheader(fpath)
            cards = []
            if hdr["OBJECT"].lower()[:4] not in ['bias', 'dark', 'test']:
                # FYI: Flat may require airmass just for check (twilight/night)
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

                    cards.append(_cards_airmass(am, full))
                    hdr.add_history("ALT-AZ calculated from TRIPOLpy.")
                    hdr.add_history("AIRMASS calculated from TRIPOLpy.")
                    hdr.add_comment(amstr)

                except ValueError:
                    if verbose:
                        print("ValueError....")
                        return hdr, am, full

                except KeyError:
                    if verbose:
                        print(f"{fpath} failed in airmass calculation")
                    pass

            # Add counter if there is none:
            if "COUNTER" not in hdr:
                cards.append(Card("COUNTER", counter, "Image counter"))

            # Add polarimetry-key (RET-ANG1) if there is none:
            if "RET-ANG1" not in hdr:
                try:
                    hwpangle = float(hdr[KEYMAP["OBJECT"]].split('_')[-1])
                    if hwpangle > 180:
                        hwpangle = hwpangle / 10
                except ValueError:
                    hwpangle = 0

                cards.append(Card("RET-ANG1", hwpangle,
                                  "The half-wave plate angle."))

            add_hdr = fits.Header(cards)

            newpath = fitsrenamer(fpath,
                                  header=hdr,
                                  newtop=newtop,
                                  rename_by=rename_by,
                                  delimiter=delimiter,
                                  add_header=add_hdr,
                                  mkdir_by=mkdir_by,
                                  archive_dir=archive_dir,
                                  keymapping=keymapping,
                                  verbose=verbose)
            newpaths.append(newpath)

        self.newpaths = newpaths
        self.summary = make_summary(newpaths,
                                    output=self.toppath / "summary.csv",
                                    format='ascii.csv',
                                    keywords=self.summary_keywords,
                                    verbose=verbose)

    # Sometimes TRIPOL specific
    # Hardcode for group_by to be exptime and filter.
    # For flat, group_by will be exptime, filter, and hwpangle.
    def make_dark(self, darkdir=None, dark_key="OBJECT", dark_val="dark",
                  group_by=["FILTER", "EXPTIME"],
                  exposure_key="EXPTIME",
                  delimiter='_', comb_kwargs=None, bias=True,
                  bias_exptime_max=0.1):
        """ Makes and saves bias and dark (bias NOT subtracted) images.
        Parameters
        ----------
        darkdir: path-like, optional
            The directory where the frames will be saved.

        dark_key, dark_val: str
            The header keyword and value to identify dark frames.

        group_by: None, str or list str, optional
            The header keywords to be used for grouping frames. For dark
            frames, usual choice can be ``['EXPTIME']``.

        exposure_key: str, optional
            If you want to make bias from a list of dark frames, you need to
            let the function know the exposure time of the frames, so that the
            miniimum exposure time frame will be used as bias. Default is
            "EXPTIME".

        comb_kwargs: dict or None, optional
            The parameters for ``combine_ccd``.

        bias: bool, optional
            If ``True``, it autoamtically finds the dark frames with exposure
            time L.E. than ``bias_exptime_max``, and the dark frame is saved
            as bias frame, as well as dark frame.

        bias_exptime_max: float, optional
            The maximum exposure time to be used to determine whether the dark
            frame is bias.
        """
        def _darkpath(group_by_vals, delimiter):
            darkname = ""
            for val in group_by_vals:
                darkname += str(val)
                darkname += str(delimiter)
            darkname = darkname[:-1] + ".fits"

            darkpath = darkdir / darkname

            return darkpath

        # Initial settings
        if self.summary is None:
            try:
                self.summary = Table.read(self.toppath / "summary.csv",
                                        format='ascii.csv')
                self.newpaths = self.summary["file"].tolist()
                self.summary_keywords = self.summary.colnames
            except:
                raise ValueError("Maybe files not 'organize'd yet?")

        if comb_kwargs is None:
            comb_kwargs = dict(overwrite=True,
                               unit=None,
                               dtype='float32',
                               combine_method="median",
                               reject_method=None,
                               combine_uncertainty_function=None)

        if group_by is None:
            group_by = []
        elif exposure_key not in group_by:
            warnings.warn("group_by is not None and does not include exposure_key. "
                          + f"Force to append it ({exposure_key}).")
            group_by.append(exposure_key)

        if dark_key not in group_by:
            group_by = [dark_key] + group_by
            # add dark_key first s.t. the output filename becomes "Dark_blahblah".

        if darkdir is None:
            darkdir = '.'

        darkdir = Path(darkdir)
        mkdir(darkdir)

        darktab = self.summary[self.summary[dark_key] == dark_val]
        darkgroups = darktab.group_by(group_by)
        min_exp = darkgroups[exposure_key].astype(float).min()

        darkpaths = []
        biaspaths = []     # Used only if bias=True

        # Do dark combine:
        for group in darkgroups.groups:
            group_by_vals = list(group[group_by][0].as_void())
            exptime_dark = float(group[exposure_key][0])
            darkpath = _darkpath(group_by_vals, delimiter)

            # Dark combine
            mdark = combine_ccd(group["file"],
                                output=darkpath,
                                **comb_kwargs,
                                type_key=group_by,
                                type_val=group_by_vals)

            darkpaths.append(darkpath)

            if bias:
                exptime_dark = float(group[exposure_key][0])
                if exptime_dark <= bias_exptime_max:
                    biasname = darkpath.name.replace(dark_val, "bias")
                    biaspath = darkpath.parent / biasname
                    if biaspath.exists():
                        raise ValueError("There are more than one dark frames "
                            + f"with exposure time <= {bias_exptime_max}!\n"
                            + "FYI, the minimum exposure time for dark frames "
                            + f"is {min_exp} seconds.")
                    biaspaths.append(biaspath)

                    bias = mdark.copy()
                    bias.header[dark_key] = "bias"
                    bias.header.add_history(f"Changed {dark_key} from {dark_val} to bias")
                    bias.write(biaspath, overwrite=True)

        if len(biaspaths) == 0:
            warnings.warn("No bias was found. \nFYI, the minimum exposure time "
                          + f"for dark frames is {min_exp} seconds.")


        # # If not, darkgroups.sort("EXPTIME") and take
        # # Find as bias when "dark" and exp=0.0
        # if bias:
        #     # Group by including exposure key
        #     darkgroups = darktab.group_by(group_by)

        #     for group in darkgroups.groups:
        #         exptime_dark = float(group[exposure_key][0])
        #         exptime_dark_min = min(dark_exptimes)
        #         exptime_min_idx = dark_exptimes.index(exptime_dark_min)
        #         # Find suitable bias image
        #         if exptime_dark_min < 1:
        #             exptime_min_darkpath = darkpaths[exptime_min_idx]
        #             biasname = exptime_min_darkpath.name.replace(dark_val, "bias")
        #             biaspath = exptime_min_darkpath.parent / biasname
        #             biaspaths.append(biaspath)

        #             bias = darks[exptime_dark_min]
        #             bias.header[dark_key] = "bias"
        #             bias.header["HISTORY"] = f"Changed from {dark_val} to bias"
        #             bias.write(biaspath, overwrite=True)

        #             print("Using dark with exposure "
        #                 + f"{min(darks.keys())}s as bias ({biaspath})")

        #         else:
        #             warnings.warn("Minimum dark exposure is too long "
        #                         + f"({exptime_dark_min} sec)!"
        #                         + " I think you have no file for bias...")

        self.biaspaths = biaspaths
        self.darkpaths = darkpaths


    # TRIPOL specific
    def make_flat(self, flatdir=None, comb_kwargs=None, flat_key="OBJECT",
                  flat_startswith="flat", flat_key_delimiter='_',
                  flat_key_info=["OBJECT", "FILTER", "HWPANGLE"],
                  polarimetry=False, hwpkey=None):
        '''
        Flat image must have the header key ``flat_key`` *starting* with the
        ``flat_startswith``. By default, it seeks for
        ``OBJECT = flat_<FILTER>_<HWPANGLE>``.

        '''

        # Initial settings
        if self.summary is None:
            try:
                self.summary = Table.read(self.toppath / "summary.csv",
                                          format='ascii.csv')
                self.newpaths = self.summary["file"].tolist()
            except:
                raise ValueError("Maybe files not 'organize'd yet?")

        if comb_kwargs is None:
            comb_kwargs = dict(overwrite=True,
                               unit=None,
                               dtype='float32',
                               combine_method="median",
                               reject_method=None,
                               combine_uncertainty_function=None)

        if flatdir is None:
            flatdir = '.'

        flatdir = Path(flatdir)
        mkdir(flatdir)

        allgroups = self.summary.group_by(flat_key)

        for group in allgroups.groups:
            obj = group[flat_key][0]
            if obj.startswith(flat_startswith):
                info = obj.split(flat_key_delimiter)

        pass


        # """ Make flats for each filter and hwp angle.
        # """
        # # Initial settings
        # if reddir is None:
        #     reddir = '.'
        # reddir = Path(reddir)

        # if self.renamed:
        #     fitspaths = self.newpaths
        # else:
        #     warnings.warn("FITS files have not yet renamed."
        #                   + f"Using ({self.toppath}).glob('*.fits')")
        #     fitspaths = self.rawpaths

        # if self.summary is None:
        #     self.summary = make_summary(fitspaths, output=None)

        # if comb_keys is None:
        #     comb_keys = dict(overwrite=True,
        #                      dtype='float32',
        #                      combine_method="median",
        #                      reject_method=None,
        #                      combine_uncertainty_function=None)

        # flattab = self.summary[self.summary["OBJECT"]]

        # # Iterate over filters
        # for filt, hwp in (self.filters, hwpangles):

        #     if hwpkey is None:
        #         # Use OBJECT to infer the HWP angle
        #         flattab = self.summary[self.summary[KEYMAP["OBJECT"]
        #                                == f"flat_{filt}_{hwp}"]

        #     else:
        #         flattab = self.summary[(self.summary[KEYMAP["OBJECT"]]
        #                                 == f"flat_{filt}")
        #                                & (self.summary[KEYMAP[hwpkey]]


        #     flatpath = reddir / f"Flat_{filt}.fits"

        #     exptime_flat = float(flattable[keymap['EXPTIME']][0])
        #     mflat = combine_ccd(flattable["file"],
        #                         **comb_keys,
        #                         type_key=[keymap["FILTER"], keymap["EXPTIME"]],
        #                         type_val=[filt, exptime_flat])

        #     # Make error map for flat (Poisson only)
        #     mflat_err = np.sqrt((mflat.data - bias.data) / gain.value)
        #     mflat.uncertainty = mflat_err

        #     # Set appropriate dark (which includes bias)
        #     try:
        #         dark = darks[exptime_flat].data

        #     except KeyError:
        #         # If no suitable exptime in darks, generate dark using linear scaling
        #         exptime_dark_max = max(darks.keys())
        #         darkmax = darks[exptime_dark_max]
        #         puredark = (darkmax.data - bias.data) * \
        #             (exptime_flat / exptime_dark_max)
        #         dark = puredark + bias.data

        #     # Do dark subtraction
        #     mflat.data = mflat.data - dark
        #     mflat = CCDData_astype(mflat, dtype='float32')
        #     mflat.write(flatpath, overwrite=True)






