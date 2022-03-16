from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import random
import torch
import time

device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("prithivida/informal_to_formal_styletransfer")
model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/informal_to_formal_styletransfer").to(device)


def empathize(input_string, max_length=32, max_candidate=1, quality_filter=0.95):
    """
    Empathise a string.

    Parameters
    ----------
    input_string : string
        Sentence to empathise.
    max_length : int
        maximum number of words per sentence.
    quality_filter: float
        How similar must the output string be to the input string.
        Setting this too high might result in no good solutions being found, and so the unaltered input string will be returned.

    Returns
    -------
    string
        The empathized string, or, if no suitable answer can be found, the input string.
    """
    tokens = tokenizer.encode(input_string, return_tensors="pt").to(device)

    preds = model.generate(
        tokens,
        do_sample=True,
        max_length=max_length,
        top_k=50,
        top_p=quality_filter,
        early_stopping=True,
        num_return_sequences=max_candidate
    )

    for pred in preds:
        return tokenizer.decode(pred, skip_special_tokens=True).strip()
    return input_string
