---
language: 
- en
- sp
- ja
- pe
- hi
- fr
- ch
- be
- gu
- ge
- te
- it
- ar
- po
- ta
- ma
- ma
- or
- pa
- po
- ur
- ga
- he
- ko
- ca
- th
- du
- in
- vi
- bu
- fi
- ce
- la
- tu
- ru
- cr
- sw
- yo
- ku
- bu
- ma
- cz
- fi
- so
- ta
- sw
- si
- ka
- zh
- ig
- xh
- ro
- ha
- es
- sl
- li
- gr
- ne
- as
- no

tags:
- summarization
- translation

datasets:
- svakulenk0/qrecc
- taskmaster2
- djaym7/wiki_dialog
- deepmind/code_contests
- lambada
- gsm8k
- aqua_rat
- esnli
- quasc
- qed

license: apache-2.0
---

# Model Card for FLAN-T5 large

![model image](https://s3.amazonaws.com/moonup/production/uploads/1666363435475-62441d1d9fdefb55a0b7d12c.png)

#  Table of Contents

1. [Model Details](#model-details)
2. [Usage](#usage)
3. [Uses](#uses)
4. [Bias, Risks, and Limitations](#bias-risks-and-limitations)
5. [Training Details](#training-details)
6. [Evaluation](#evaluation)
7. [Environmental Impact](#environmental-impact)
8. [Citation](#citation)
9. [Model Card Authors](#model-card-authors)
10. [How To Get Started With the Model](#how-to-get-started-with-the-model)

# Model Details

## Model Description

The developers of the Text-To-Text Transfer Transformer (T5) [write](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html): 

> With T5, we propose reframing all NLP tasks into a unified text-to-text-format where the input and output are always text strings, in contrast to BERT-style models that can only output either a class label or a span of the input. Our text-to-text framework allows us to use the same model, loss function, and hyperparameters on any NLP task.

T5-Base is the checkpoint with 220 million parameters. 

- **Developed by:** Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane, Gu Zhuyun, Dai Mirac, Suzgun Xinyun, Chen Aakanksha, Chowdhery Sharan, Narang Gaurav, Mishra Adams, Yu Vincent, Zhao Yanping, Huang Andrew, Dai Hongkun, Yu Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts; Denny Zhou, Quoc V. Le, Jason Wei∗ See [associated paper](https://arxiv.org/pdf/2210.11416.pdf) and [GitHub repo](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-t5-checkpoints)
- **Model type:** Language model
- **Language(s) (NLP):** English, French, Romanian, German
- **License:** Apache 2.0
- **Related Models:** [All T5 Checkpoints](https://huggingface.co/models?search=t5)
- **Resources for more information:**
  - [Research paper](https://jmlr.org/papers/volume21/20-074/20-074.pdf)
  - [Google's T5 Blog Post](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html) 
  - [GitHub Repo](https://github.com/google-research/text-to-text-transfer-transformer)
  - [Hugging Face T5 Docs](https://huggingface.co/docs/transformers/model_doc/t5)

# Usage

Find below some example scripts on how to use the model in `transformers`:

## Using the Pytorch model

### Running the model on a CPU

<details>
<summary> Click to expand </summary>

```python

from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

input_text = "translate English to German: How old are you?"
input_ids = tokenizer.encode(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```

</details>

### Running the model on a GPU

<details>
<summary> Click to expand </summary>

```python

from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map="auto")

input_text = "translate English to German: How old are you?"
input_ids = tokenizer.encode(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```

</details>

### Running the model on a GPU using different precisions

#### FP16

<details>
<summary> Click to expand </summary>

```python
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map="auto", torch_dtype=torch.float16)

input_text = "translate English to German: How old are you?"
input_ids = tokenizer.encode(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```

</details>

#### INT8

<details>
<summary> Click to expand </summary>

```python
# pip install bitsandbytes
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", device_map="auto", load_in_8bit=True)

input_text = "translate English to German: How old are you?"
input_ids = tokenizer.encode(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```

</details>

# Uses

## Direct Use and Downstream Use

The authors write in [the original paper's model card](https://arxiv.org/pdf/2210.11416.pdf) that: 

> The primary use is research on language models, including: research on zero-shot NLP tasks and in-context few-shot learning NLP tasks, such as reasoning, and question answering; advancing fairness and safety research, and understanding limitations of current large language models

See the [research paper](https://arxiv.org/pdf/2210.11416.pdf) for further details.

## Out-of-Scope Use

More information needed.

# Bias, Risks, and Limitations

The information below in this section are copied from the model's [official model card](https://arxiv.org/pdf/2210.11416.pdf):

> Language models, including Flan-T5, can potentially be used for language generation in a harmful way, according to Rae et al. (2021). Flan-T5 should not be used directly in any application, without a prior assessment of safety and fairness concerns specific to the application.

## Ethical considerations and risks

> Flan-T5 is fine-tuned on a large corpus of text data that was not filtered for explicit content or assessed for existing biases. As a result the model itself is potentially vulnerable to generating equivalently inappropriate content or replicating inherent biases in the underlying data.

## Known Limitations

> Flan-T5 has not been tested in real world applications.

## Sensitive Use:

> Flan-T5 should not be applied for any unacceptable use cases, e.g., generation of abusive speech.

# Training Details

## Training Data

The model was trained on a mixture of tasks, that includes the tasks described in the table below (from the original paper, figure 2):

![table.png](https://s3.amazonaws.com/moonup/production/uploads/1666363265279-62441d1d9fdefb55a0b7d12c.png)


## Training Procedure

According to the model card from the [original paper](https://arxiv.org/pdf/2210.11416.pdf):

> These models are based on pretrained T5 (Raffel et al., 2020) and fine-tuned with instructions for better zero-shot and few-shot performance. There is one fine-tuned Flan model per T5 model size.


# Evaluation

## Testing Data, Factors & Metrics

The developers evaluated the model on 88 tasks and 10 languages. See the table below for quantitative evaluation:
![image.png](https://s3.amazonaws.com/moonup/production/uploads/1666361983550-62441d1d9fdefb55a0b7d12c.png)
For full details, please check the [research paper](https://arxiv.org/pdf/2210.11416.pdf).

## Results 

For full results for FLAN-T5-Large, see the [research paper](https://arxiv.org/pdf/2210.11416.pdf), Table 3.

# Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** Google Cloud TPU Pods
- **Hours used:** More information needed
- **Cloud Provider:** GCP
- **Compute Region:** More information needed
- **Carbon Emitted:** More information needed

# Citation

**BibTeX:**

```bibtex
@misc{https://doi.org/10.48550/arxiv.2210.11416,
  doi = {10.48550/ARXIV.2210.11416},
  
  url = {https://arxiv.org/abs/2210.11416},
  
  author = {Chung, Hyung Won and Hou, Le and Longpre, Shayne and Zoph, Barret and Tay, Yi and Fedus, William and Li, Eric and Wang, Xuezhi and Dehghani, Mostafa and Brahma, Siddhartha and Webson, Albert and Gu, Shixiang Shane and Dai, Zhuyun and Suzgun, Mirac and Chen, Xinyun and Chowdhery, Aakanksha and Narang, Sharan and Mishra, Gaurav and Yu, Adams and Zhao, Vincent and Huang, Yanping and Dai, Andrew and Yu, Hongkun and Petrov, Slav and Chi, Ed H. and Dean, Jeff and Devlin, Jacob and Roberts, Adam and Zhou, Denny and Le, Quoc V. and Wei, Jason},
  
  keywords = {Machine Learning (cs.LG), Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Scaling Instruction-Finetuned Language Models},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
```

**APA:**
- Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., ... & Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 21(140), 1-67.

# Model Card Authors

This model card was written by the team at Hugging Face.

# How to Get Started with the Model

Use the code below to get started with the model.

<details>
<summary> Click to expand </summary>

```python
from transformers import T5Tokenizer, T5Model

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5Model.from_pretrained("t5-base")

input_ids = tokenizer(
    "Studies have been shown that owning a dog is good for you", return_tensors="pt"
).input_ids  # Batch size 1
decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

# forward pass
outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
last_hidden_states = outputs.last_hidden_state
```

See the [Hugging Face T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Model) docs and a [Colab Notebook](https://colab.research.google.com/github/google-research/text-to-text-transfer-transformer/blob/main/notebooks/t5-trivia.ipynb) created by the model developers for more examples.
</details>