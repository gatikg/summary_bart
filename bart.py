from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize


# sample_text = '''A major breakthrough has been announced by US scientists in the race to recreate nuclear fusion.
# Physicists have pursued the technology for decades as it promises a potential source of near-limitless clean energy.
# On Tuesday researchers confirmed they have overcome a major barrier - producing more energy from a fusion experiment than was put in .
# But experts say there is still some way to go before fusion powers homes.'''

def bart_test(sample_text):
    nltk.download('punkt')
    pipe = pipeline("summarization", model="facebook/bart-large-cnn")

    summaries = {}

    pipe_out = pipe(sample_text)

    summaries["bart"] = "\n".join(sent_tokenize(pipe_out[0]["summary_text"]))

    return summaries["bart"], sample_text, len(sample_text), len(summaries["bart"])
