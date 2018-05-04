"""
This is a code snippet used for TRIPOL computer, and not usually for the output
FITS file from TRIPOL.
"""
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--tmpheader')
parser.add_argument('--tmpfits')
args = parser.parse_args()

tmphdrpath = args.tmpheader
tmpfitspath = args.tmpfits

try:
    counter = tmpfitspath.split('/')[-1].split('_')[1][:4]
except IndexError:
    counter = -1 # When TL image

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
          "EQUINOX": "EPOCH",
          "FILTER": 'FILTER', "DATE-OBS": 'DATE', "RDNOISE": None}

KEYORI = list(KEYMAP.values())
KEYNEW = list(KEYMAP.keys())
KEYMAP2 = dict(zip(KEYORI, KEYNEW))

# First, update the header keys accordingly:
newlines = [f"counter = {counter} / Image counter",
            "bunit = 'ADU' / "]
# cards = {}
with open(tmphdrpath, 'r') as tmphdr:
    lines = tmphdr.read()
    lines = lines.replace('\r\n', '\n')
    lines = lines.replace('\r', '\n')
    lines = lines.split('\n')[:-1]
    for i, line in enumerate(lines):
        k_ori, vc_ori = line.split(' = ', maxsplit=1)
        v_ori, c_ori = vc_ori.split(' / ', maxsplit=1)
        v_ori = v_ori.replace("'", '').replace('"', '')

        try:
            v_ori = int(v_ori)
        except ValueError:
            try:
                v_ori = float(v_ori)
            except ValueError:
                pass

        if k_ori.upper() in KEYORI:
            k_new = KEYMAP2[k_ori.upper()]
            newline = f"{k_new} = {vc_ori}"
            line = f"{k_ori} = {v_ori} / Deprecated. See {k_new}."
            # cards[k_new] = (v_ori, c_ori)
            # cards[k_ori] = (v_ori, f"Deprecated. See {k_new}")
            newlines.append(line)
            newlines.append(newline)
            if k_ori.upper() == "FILTER":
                filt = v_ori.lower()
        else:
            # cards[k_ori] = (v_ori, c_ori)
            newlines.append(line)

# Appending at the last stage will overwrite the original value
# when transformed into FITS header.
newlines.append(f"GAIN = {GAIN_EPADU[filt][LATEST]} "
                + f"/ [e-/ADU] The electron gain factor ({LATEST}).")
newlines.append(f"RDNOISE = {RDNOISE_E[filt][LATEST]} "
                + f"/ [e-] The (Gaussian) readout noise ({LATEST}).")

with open(tmphdrpath, 'w+') as newhdr:
    newhdr.write('\n'.join(newlines))
