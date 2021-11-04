from parlai.core.agents import register_agent, Agent
from parlai.scripts.display_model import DisplayModel

from parlai.core.teachers import register_teacher, DialogTeacher
from parlai.scripts.display_data import DisplayData
from parlai.scripts.interactive import Interactive

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
        

@register_agent("hello")
class HelloAgent(Agent):
    @classmethod
    def add_cmdline_args(cls, parser, partial_opt):
        parser.add_argument('--name', type=str, default='Alice', help="The agent's name.")
        return parser
        
    def __init__(self, opt, shared=None):
        # similar to the teacher, we have the Opt and the shared memory objects!
        super().__init__(opt, shared)
        self.id = 'HelloAgent'
        self.name = opt['name']
    
    def observe(self, observation):
        # Gather the last word from the other user's input
        words = observation.get('text', '').split()
        if words:
            self.last_word = words[-1]
        else:
            self.last_word = "stranger!"
    
    def act(self):
        # Always return a string like this.
        return {
            'id': self.id,
            'text': f"Hello {self.last_word}, I'm {self.name}",
        }

DisplayModel.main(task='my_teacher', model='hello')

# Let's interact with it
Interactive.main(model='hello', name='Bob')