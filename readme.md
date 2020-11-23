# Few-shot Slot Tagging

This is the code of the ACL 2020 paper: [Few-shot Slot Tagging with Collapsed Dependency Transfer and Label-enhanced Task-adaptive Projection Network](https://atmahou.github.io/attachments/atma's_acl2020_FewShot.pdf).


## Notice: Better implementation availiable now!

- A new and powerfull platform is now availiable for general few-shot learning problems!!

- It fully support current experiments with better interface and flexibility~ (E.g. supoort newer [huggingface/transformers](https://github.com/huggingface/transformers))

Try it at: https://github.com/AtmaHou/MetaDialog

## Get Started

### Requirement
```
python >= 3.6
pytorch >= 0.4.1
pytorch_pretrained_bert >= 0.6.1
allennlp >= 0.8.2
pytorch-nlp
```

### Step1: Prepare BERT embedding:
- Download the pytorch bert model, or convert tensorflow param by yourself as follow:
```bash
export BERT_BASE_DIR=/users4/ythou/Projects/Resources/bert-base-uncased/uncased_L-12_H-768_A-12/

pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch
  $BERT_BASE_DIR/bert_model.ckpt
  $BERT_BASE_DIR/bert_config.json
  $BERT_BASE_DIR/pytorch_model.bin
```
- Set BERT path in the file `./scripts/run_L-Tapnet+CDT.sh` to your setting:
```bash
bert_base_uncased=/your_dir/uncased_L-12_H-768_A-12/
bert_base_uncased_vocab=/your_dir/uncased_L-12_H-768_A-12/vocab.txt
```


### Step2: Prepare data
- Download few-shot data at [my homepage](https://atmahou.github.io/) or click here: [download](https://atmahou.github.io/attachments/ACL2020data.zip)

> Tips: The numbers in file name denote cross-evaluation id, you can run a complete experiment by only using data of id=1.

- Set test, train, dev data file path in `./scripts/run_L-Tapnet+CDT.sh` to your setting.
  
> For simplicity, your only need to set the root path for data as follow:
```bash
base_data_dir=/your_dir/ACL2020data/
```

### Step3: Train and test the main model
- Build a folder to collect running log
```bash
mkdir result
```

- Execute cross-evaluation script with two params: -[gpu id] -[dataset name]

##### Example for 1-shot Snips:
```bash
source ./scripts/run_L-Tapnet+CDT.sh 0 snips
```  
##### Example for 1-shot NER:
```bash
source ./scripts/run_L-Tapnet+CDT.sh 0 ner
```

> To run 5-shots experiments, use `./scripts/run_L-Tapnet+CDT_5.sh`

## Model for Other Setting

We also provide scripts of four model settings as follows: 
- Tap-Net
- Tap-Net + CDT
- L-WPZ + CDT
- L-Tap-Net + CDT 
> You can find their corresponding scripts in `./scripts/` with the same usage as above.


## Project Architecture

### `Root`
- the project contains three main parts:
    - `models`: the neural network architectures
    - `scripts`: running scripts for cross evaluation
    - `utils`: auxiliary or tool function files
    - `main.py`: the entry file of the whole project

### `models`
- Main Model  
    - Sequence Labeler (`few_shot_seq_labeler.py`): a framework that integrates modules below to perform sequence labeling.
- Modules
    - Embedder Module (`context_embedder_base.py`): modules that provide embeddings.
    - Emission Module (`emission_scorer_base.py`): modules that compute emission scores. 
    - Transition Module (`transition_scorer.py`): modules that compute transition scores.
    - Similarity Module (`similarity_scorer_base.py`): modules that compute similarities for metric learning based emission scorer.
    - Output Module (`seq_labeler.py`, `conditional_random_field.py`): output layer with normal mlp or crf.
    - Scale Module (`scale_controller.py`): a toolkit for re-scale and normalize logits.

### `utils`

- `utils` contains assistance modules for:
    - data processing (`data_helper.py`, `preprocessor.py`), 
    - constructing model architecture (`model_helper.py`), 
    - controlling training process (`trainer.py`), 
    - controlling testing process (`tester.py`), 
    - controllable parameters definition (`opt.py`), 
    - device definition (`device_helper`) 
    - config (`config.py`).


## Updates - New branch: `fix_TapNet_svd_issue`
Thanks [Wangpeiyi9979](https://github.com/Wangpeiyi9979) for pointing out the problem of TapNet implementation ([issue](https://github.com/AtmaHou/FewShotTagging/issues/20)), which is caused by port differences of `cupy.linalg.svd` and `svd() in pytorch`. 

The corrected codes are included in new branch named `fix_TapNet_svd_issue`, because we found correction of TapNet will slightly degrade performance (still the best).



