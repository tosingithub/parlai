#import os
#import csv
import argparse
from parlai.utils.io import PathManager
from parlai.core.teachers import register_teacher, DialogTeacher
from parlai.scripts.display_model import DisplayModel
from parlai.scripts.train_model import TrainModel


parser = argparse.ArgumentParser(description="Dataset Information")
parser.add_argument('--data', type=str, default='parlai/tasks/data/yoruba_dialog.csv', help='location of the dataset')
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


TrainModel.main(
    # similar to before
    task='yo_teacher', 
    model='transformer/generator',
    model_file='from_pretrained/model_yo',
    
    # initialize with a pretrained model
    init_model='zoo:tutorial_transformer_generator/model',
    # zoo:wizard_of_wikipedia/full_dialogue_retrieval_model/model
    # zoo:light/biranker_dialogue/model
    # zoo:pretrained_transformers/poly_model_huge_reddit/model


    # BlenderBot 90M
    n_heads=16, n_layers=8, n_positions=512, text_truncate=512,
    label_truncate=128, ffn_size=2048, embedding_size=512,
    activation='gelu', variant='xlm',
    dict_lower=True, dict_tokenizer='bpe',
    dict_file='zoo:tutorial_transformer_generator/model.dict',
    learn_positional_embeddings=True,
    #dropout=0.1,
    #gradient_clip=0.1,
    #lr_scheduler='reduceonplateau',

    
    # BlenderBot 3B
    # arguments we get from the pretrained model.
    # Unfortunately, these must be looked up separately for each model. variant='xlm',
    # n_heads=32, n_layers=24, n_positions=128, text_truncate=128,
    # label_truncate=128, ffn_size=10240, embedding_size=2560,
    # activation='gelu',
    # dict_lower=True, dict_tokenizer='bpe',
    # dict_file='zoo:blender/reddit_3B/model.dict',
    # learn_positional_embeddings=True,
    # variant='prelayernorm',
    # n_encoder_layers=2,
    # n_decoder_layers=24,
    # delimiter='  ',
    # lr_scheduler='reduceonplateau',
    # model_parallel=True,
    
    # some training arguments, specific to this fine-tuning
    # use a small learning rate with ADAM optimizer 1e-5,
    lr=1e-05, optimizer='adam',
    warmup_updates=100,
    # early stopping on perplexity
    validation_metric='ppl',
    # train at most 10 minutes, and validate every 0.25 epochs
    max_train_time=120 * 60, validation_every_n_epochs=0.25,
    
    # depend on your gpu. If you have a V100, this is good
    batchsize=6, fp16=True, fp16_impl='mem_efficient',
    
    # speeds up validation
    skip_generation=False,
    
    # helps us cram more examples into our gpu at a time
    dynamic_batching='full',
)

DisplayModel.main(task='yo_teacher', model_file='from_pretrained/model_yo', num_examples=5, skip_generation=False)
