import warnings
from pathlib import Path
from itertools import product
from astropy.io import fits
from astropy.io.fits import Card
from astropy import table
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
        self.darkpaths = None
        self.flatpaths = None
        # self.renamed = False
        # rawpaths: Original file paths
        # newpaths: Renamed paths
        # redpaths: Reduced frames paths excluding BDF

    def initialize_self(self):
        if self.summary is None:
            try:
                self.summary = Table.read(self.toppath / "summary.csv",
                                          format='ascii.csv')
                self.newpaths = self.summary["file"].tolist()
            except FileNotFoundError:
                pass

        if self.biaspaths is None:
            try:
                with open("mbias.list", 'r') as bl:
                    self.biaspaths = bl.read().splitlines()
            except FileNotFoundError:
                pass

        if self.darkpaths is None:
            try:
                with open("mdark.list", 'r') as dl:
                    self.darkpaths = dl.read().splitlines()
            except FileNotFoundError:
                pass

        if self.flatpaths is None:
            try:
                with open("mflat.list", 'r') as fl:
                    self.flatpaths = fl.read().splitlines()
            except FileNotFoundError:
                pass


    # TRIPOL specific
    def organize_tripol(self, newtop=None,
                        rename_by=["COUNTER", "OBJECT", "EXPOS"],
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

            hdr = fits.getheader(fpath)
            cards = []

            # If the first 4 chars of OBJECT is not object-like frames, don't
            # calculate airmass.
            try:
                obj4 = hdr["OBJECT"].lower()[:4]
            except KeyError:
                print(f"{fpath} has no OBJECT! Skipping")
                continue

            if obj4 not in ['bias', 'dark', 'test']:
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
                    amcards = _cards_airmass(am, full)
                    [cards.append(c) for c in amcards]
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
                                  keymap=KEYMAP,
                                  verbose=verbose)
            newpaths.append(newpath)

        self.newpaths = newpaths
        self.summary = make_summary(newpaths,
                                    output=self.toppath / "summary.csv",
                                    format='ascii.csv',
                                    keywords=self.summary_keywords,
                                    verbose=verbose)

    # TRIPOL specific
    def make_dark(self, darkdir=None, biasdir=None, dark_key="OBJECT", dark_val="dark",
                  bias_key="OBJECT", bias_val="bias",
                  group_by=["FILTER", "EXPTIME"],
                  exposure_key="EXPTIME",
                  delimiter='_', comb_kwargs=None):
        """ Makes and saves bias and dark (bias NOT subtracted) images.
        Parameters
        ----------
        darkdir: path-like, optional
            The directory where the frames will be saved.

        dark_key, dark_val: str
            The header keyword and value to identify dark frames.

        bias_key, bias_val: str
            The header keyword and value to identify bias frames.

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
        """

        # Initial settings
        self.initialize_self()

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

        made_bias = False

        biastab = self.summary[self.summary[bias_key] == bias_val]
        if len(biastab) > 0:
            biaspath = imgpath("bias", biasdir, delimiter)
            mbias = combine_ccd(biastab["file"],
                                **comb_kwargs,
                                type_key=bias_key,
                                type_val=bias_val)
            mbias.header[dark_key] = "bias"
            mbias.write(biaspath, overwrite=True)
            made_bias=True

        darktab = self.summary[(self.summary[dark_key] == dark_val)]
        darktab[exposure_key] = darktab[exposure_key].astype(float)
        darktab.sort(exposure_key)
        darkgroups = darktab.group_by(group_by)
        min_exp = darkgroups[exposure_key].astype(float).min()

        darkpaths = []
        biaspaths = []     # Used only if bias=True

        # Do dark combine:
        for group in darkgroups.groups:
            group_by_vals = list(group[group_by][0].as_void())
            exptime_dark = float(group[exposure_key][0])
            darkpath = imgpath(group_by_vals, darkdir, delimiter)

            if (not made_bias) and (exptime_dark == min_exp):
                if min_exp > 1:
                    warnings.warn(
                        f"The minimum exposure is long ({min_exp} s)")
                biasname = darkpath.name.replace(dark_val, "bias")
                biaspath = darkpath.parent / biasname
                biaspaths.append(biaspath)

                mbias = combine_ccd(group["file"],
                                    **comb_kwargs,
                                    type_key=group_by,
                                    type_val=group_by_vals)
                mbias.header[dark_key] = "bias"
                mbias.header.add_history(f"Changed {dark_key} from "
                                         + "{dark_val} to bias")
                mbias.write(biaspath, overwrite=True)
                made_bias=True


            # Dark combine
            mdark = combine_ccd(group["file"],
                                **comb_kwargs,
                                type_key=group_by,
                                type_val=group_by_vals)
            mdark_b = bdf_process(mdark,
                                  output=darkpath,
                                  mbiaspath=biaspath)

            darkpaths.append(darkpath)


        if len(biaspaths) == 0:
            warnings.warn("No bias was made. \nFYI, the minimum exposure time "
                          + f"for dark frames is {min_exp} seconds.")

        with open(self.toppath / 'mbias.list', 'w+') as bl:
            for bp in biaspaths:
                bl.write(f"{str(bp)}\n")

        with open(self.toppath / 'mdark.list', 'w+') as dl:
            for dp in darkpaths:
                dl.write(f"{str(dp)}\n")

        self.biaspaths = biaspaths
        self.darkpaths = darkpaths


    # TRIPOL specific

    def make_flat(self, flatdir=None, comb_kwargs=None, delimiter='_',                              flat_key="OBJECT", flat_startswith="flat",
                 group_by=["FILTER", "RET-ANG1"],
                 polarimetry=False, hwpkey=None):
        '''
        Flat image must have the header key ``flat_key`` *starting* with the
        ``flat_startswith``. By default, it seeks for
        ``OBJECT = flat_<FILTER>_<HWPANGLE>``.

        '''

        # Initial settings: If somehow pipeline stopped, renewing process should
        # work in a safer way if we include these ``if self.xx is None`` lines.
        if self.summary is None:
            try:
                self.summary = Table.read(self.toppath / "summary.csv",
                                          format='ascii.csv')
                self.newpaths = self.summary["file"].tolist()
            except:
                raise ValueError("Maybe files not 'organize'd yet?")

        if self.biaspaths is None:
            try:
                with open("mbias.list", 'r') as bl:
                    self.biaspaths = bl.read().splitlines()
            except FileNotFoundError:
                pass

        if self.darkpaths is None:
            try:
                with open("mdark.list", 'r') as dl:
                    self.darkpaths = dl.read().splitlines()
            except FileNotFoundError:
                pass

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

        allgroups = self.summary.group_by(group_by)

        for group in allgroups.groups:
            obj = group["OBJECT"][0]
            imgfilt = group["FILTER"][0]
            flatfilt = obj[-1] # The last character

            if not obj.startswith(flat_startswith):
                continue

            elif flatfilt != imgfilt:
                continue

            group_by_vals = list(group[group_by][0].as_void())
            flatpath = imgpath(group_by_vals, flatdir, delimiter)
            mflat = combine_ccd()

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






