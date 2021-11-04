# build file

import parlai.core.build_data as build_data
import os

RESOURCES = [
    (
        'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json',
        'train-v1.1.json',
        '<checksum for this file>',
        #zipped=False,
    ),
    (
        'https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json',
        'dev-v1.1.json',
        '<checksum for this file>',
        #zipped=False,
    ),
]


def build(opt):
    dpath = os.path.join(opt['datapath'], 'SQuAD')
    version = None

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # Download the data.
        for downloadable_file in RESOURCES[:2]:
            downloadable_file.download_file(dpath)

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
