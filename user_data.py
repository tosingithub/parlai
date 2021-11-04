from parlai.core.teachers import register_teacher, DialogTeacher
from parlai.scripts.display_data import DisplayData
from parlai.scripts.display_model import DisplayModel

@register_teacher("my_teacher")
class MyTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        # opt is the command line arguments.
        
        # What is this shared thing?
        # We make many copies of a teacher, one-per-batchsize. Shared lets us store 
        
        # We just need to set the "datafile".  This is boilerplate, but differs in many teachers.
        # The "datafile" is the filename where we will load the data from. In this case, we'll set it to
        # the fold name (train/valid/test) + ".txt"
        opt['datafile'] = opt['datatype'].split(':')[0] + ".txt"
        super().__init__(opt, shared)
    
    def setup_data(self, datafile):
        # filename tells us where to load from.
        # We'll just use some hardcoded data, but show how you could read the filename here:
        print(f" ~~ Loading from {datafile} ~~ ")
        
        # setup_data should yield tuples of ((text, label), new_episode)
        # That is ((str, str), bool)
        
        # first episode
        # notice how we have call, response, and then True? The True indicates this is a first message
        # in a conversation
        yield ('Hello', 'Hi'), True
        # Next we have the second turn. This time, the last element is False, indicating we're still going
        yield ('How are you', 'I am fine'), False
        yield ("Let's say goodbye", 'Goodbye!'), False
        
        # second episode. We need to have True again!
        yield ("Hey", "hi there"), True
        yield ("Deja vu?", "Deja vu!"), False
        yield ("Last chance", "This is it"), False
        
DisplayData.main(task="my_teacher")

# We can now use our teacher in the standard places! Let's see how the model we trained earlier behaves with it:
DisplayModel.main(task='my_teacher', model_file='from_pretrained/model', skip_generation=False)