import torch
import transformers
from transformers import BartTokenizer, BartForConditionalGeneration

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

torch_device = 'cpu'

# rawtext = '''A major breakthrough has been announced by US scientists in the race to recreate nuclear fusion.
# Physicists have pursued the technology for decades as it promises a potential source of near-limitless clean energy.
# On Tuesday researchers confirmed they have overcome a major barrier - producing more energy from a fusion experiment than was put in .
# But experts say there is still some way to go before fusion powers homes.'''


def bart_summarize(rawtext):

    rawtext = rawtext.replace('\n', '')
    text_input_ids = tokenizer.batch_encode_plus(
        [rawtext], return_tensors='pt', max_length=1024)['input_ids'].to(torch_device)
    summary_ids = model.generate(text_input_ids, num_beams=4, length_penalty=2,
                                 max_length=142, min_length=56, no_repeat_ngram_size=3)
    summary_txt = tokenizer.decode(
        summary_ids.squeeze(), skip_special_tokens=True)

    return summary_txt, rawtext, len(rawtext), len(summary_txt)
