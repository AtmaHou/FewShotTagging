# Few-shot Slot Tagging
Code usage and other instructions.

## Get Started

### Requirement
```
python >= 3.6
pytorch >= 0.4.1
pytorch_pretrained_bert >= 0.6.1
allennlp >= 0.8.2
```

### Prepare pre-trained embedding:
#### BERT
Down the pytorch bert model, or convert tensorflow param yourself as follow:
```bash
export BERT_BASE_DIR=/users4/ythou/Projects/Resources/bert-base-uncased/uncased_L-12_H-768_A-12/

pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch
  $BERT_BASE_DIR/bert_model.ckpt
  $BERT_BASE_DIR/bert_config.json
  $BERT_BASE_DIR/pytorch_model.bin
```
Set BERT path in the ./utils/config.py

### Prepare data
Original data is available by contacting me, or you can generate it:
Set test, train, dev data file path in ./scripts/

### Train and test the main model
Run command line:
```bash
source ./scripts/run_main.sh [gpu id split with ','] [data set name]
```

Example:
```bash
source ./scripts/run_main.sh 1,2 snips
```

### Scripts For Main Model

We provide scripts of four main models: 
- Tap-Net,
- Tap-Net + CDT
- L-WPZ + CDT
- L-Tap-Net + CDT 

, which are in the folder named `scripts` 

> Notice: you should change the BERT PATH and the DATA PATH in scripts


## Project Architecture

### overview

```
- FewShotTagging
    - `models`: the neural network architectures
    - `scripts`: training scripts for main models
    - `utils`: some tools for Assisting training
    - `main.py`: the entry file of the whole project
```

### `models`

In `models` folder, there is a model file named `few_shot_seq_labeler.py`, which combines all components of Model Architecture,
include Embedder Module,  Emission Module, Transition Module and so on, which are all in the sub-folder named `fewshot_seqlabel`.

As mentioned before, there are three main modules in our model.
1. The Embedder Module is implemented in `context_embedder_base.py`;
2. The Emission Module is implemented in `emission_scorer_base.py`, because of the metric-based method, the most important part of which is the similarity calculation functions implemented in `similarity_scorer_base.py`;
3. The Transition Module is implemented in `transition_scorer.py`;

The Emission Module and Transition Module consist the CRF Module implemented in `conditional_random_field.py`, which is decoded by Viterbi Algorithm.
When not using Transition Module, we just decode the emission score directly implemented in `seq_labeler.py`.

When we combine the Emission Module and Transition Module, we implement a Scale Module implemented in `scale_controller.py` for better training.


### `utils`

In `utils` folder, there are some assistance modules for 
- data processing(`data_helper.py`, `preprocessor.py`), 
- constructing model architecture(`model_helper,py`), 
- controlling training process(`trainer,py`), 
- controlling testing process(`tester,py`), 
- controllable parameters definition(`opt.py`), 
- controllable device definition(`device_helper`) 
- and config(`config.py`).




