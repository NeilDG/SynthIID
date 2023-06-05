import sys

import gdown
from optparse import OptionParser

parser = OptionParser()
parser.add_option('--server_config', type=int, help="Is running on COARE?", default=0)

def main(argv):
    (opts, args) = parser.parse_args(argv)

    if(opts.server_config == 0):
        output_dir = "/scratch3/neil.delgallego/SynthV3_Raw/"
    elif(opts.server_config == 4):
        output_dir = "D:/NeilDG/Datasets/SynthV3_Raw/"
    elif(opts.server_config == 5):
        output_dir = "/home/neildelgallego/SynthV3_Raw/"
    else:
        output_dir = "/home/jupyter-neil.delgallego/SynthV3_Raw/"

    # z00
    direct_link = "https://drive.google.com/file/d/1mCCeIKNhqgrxjtXy50VswQ29dbZQ96oZ/view?usp=drive_link"
    id = direct_link.split("/d/")[1].split("/")[0]
    url = "https://drive.google.com/uc?id="+id
    gdown.download(url, output=output_dir, use_cookies=False)

if __name__ == "__main__":
    main(sys.argv)

