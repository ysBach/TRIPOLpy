from . import core

MEDCOMB_KEYS = dict(overwrite=True,
                    unit=None,
                    dtype='float32',
                    combine_method="median",
                    reject_method=None,
                    combine_uncertainty_function=None)

LATEST = "Apr2018"

GAIN_EPADU = dict(g=dict(default=1.82,
                         Apr2018=1.82),
                  r=dict(default=1.05,
                         Apr2018=1.05),
                  i=dict(default=2.00,
                         Apr2018=2.00))

RDNOISE_E = dict(g=dict(default=0,
                        Apr2018=1),
                 r=dict(default=0,
                        Apr2018=1),
                 i=dict(default=0,
                        Apr2018=1))

KEYMAP = {"EXPTIME": 'EXPOS', "GAIN": 'EGAIN', "OBJECT": 'OBJECT',
          "FILTER": 'FILTER', "EQUINOX": 'EPOCH',
          "DATE-OBS": 'DATE', "RDNOISE": None}

USEFUL_KEYS = ["EXPTIME", "FILTER", "DATE-OBS", "RET-ANG1",
               "OBJECT", "EPOCH", "RA", "DEC", "ALT", "AZ", "AIRMASS"]
