from parlai.scripts.display_model import DisplayModel
DisplayModel.main(
    task='empathetic_dialogues',
    model_file='from_pretrained/model',
    num_examples=2,
    skip_generation=False,
)