""" # Import the Interactive script
from parlai.scripts.interactive import Interactive

# call it with particular args
Interactive.main(
    # the model_file is a filename path pointing to a particular model dump.
    # Model files that begin with "zoo:" are special files distributed by the ParlAI team.
    # They'll be automatically downloaded when you ask to use them.
    model_file='zoo:tutorial_transformer_generator/model'
)
 """

 
 # Alternative: Bot dialogue with itself.
from parlai.scripts.display_model import DisplayModel
DisplayModel.main(
    task='empathetic_dialogues',
    model_file='from_pretrained/model',
    num_examples=2,
    skip_generation=False,
)