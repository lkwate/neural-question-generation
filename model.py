import torch.nn as nn
from transformers import T5ForConditionalGeneration

# model
class QGModel(nn.Module):
    def __init__(self, configT5, path_t5_question_generation = None):
        super(QGModel, self).__init__()
        if path_t5_question_generation is not None:
            self.t5_model = T5ForConditionalGeneration.from_pretrained(path_t5_question_generation, config=configT5)
        else:
            self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-base', config=configT5)

    def forward(self, input_ids_ctx, attention_mask_ctx, input_ids_qt=None, attention_mask_qt=None):
        output = self.t5_model(input_ids=input_ids_ctx, attention_mask=attention_mask_ctx,
                               decoder_attention_mask=attention_mask_qt, lm_labels=input_ids_qt)
        return output

    def predict(self, intput_ids_ctx):
        output = self.t5_model.generate(intput_ids_ctx)
        return output