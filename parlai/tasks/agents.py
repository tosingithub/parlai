# agents file
import os
import csv
import argparse
from parlai.utils.io import PathManager
from parlai.scripts.display_data import DisplayData
from parlai.core.teachers import register_teacher, DialogTeacher
from parlai.scripts.display_model import DisplayModel



parser = argparse.ArgumentParser(description="Dataset Information")
parser.add_argument('--data', type=str, default='data/yoruba_dialog.csv', help='location of the dataset')
args = parser.parse_args()


@register_teacher("yo_teacher")
class YoTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        #build(opt)  # NOTE: the call to build here
        #suffix = 'train' if opt['datatype'].startswith('train') else 'dev'
        opt['datafile'] = args.data #os.path.join(opt['datapath'], 'yoruba_dialog.csv')
        self.id = 'yodialog'
        super().__init__(opt, shared)


    def setup_data(self, path):
    # note that path is the value provided by opt['datafile']
        prompt = ""                 # initialize prompt message
        cnter = 0                   # initialize conversation counter
        line_skipper = False        # to skip 1st row of curated dailog data with header: line
        new_episode = False
        print('loading: ' + path)
        with PathManager.open(path) as data_file:
            #self.yodialog = csv.reader(data_file, delimiter=',')
            for row in data_file.readlines():
                if line_skipper:
                    row_array = row.rstrip('\n').split(';')
                    #first_item = row_array[1]
                    if cnter == 0:
                        prompt = row_array[1]
                        cnter += 1
                        new_episode = True if str.lower(row_array[0]) == 'true' else False
                    else:
                        yield {"text": prompt, "labels": row_array[1]}, new_episode
                        cnter = 0
                else:
                    line_skipper = True


class DefaultTeacher(YoTeacher):
    pass


#DisplayData.main(task="yo_teacher")

DisplayModel.main(task='yo_teacher', model_file='../../from_pretrained/model', skip_generation=False)
