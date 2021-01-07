# Transformer
This is a repository for implentation of Transformer ([repo](https://github.com/dmlc/dgl/tree/master/examples/pytorch/transformer)).
Architecture resembles one explained in paper.
Several noticeable points are:
* Requirements: Python 3
* Dataset: Multi30k
* Features: Model, Framework agnostic
* hyperparemeters
   * epochs = 100
   * batch = 512
   * optimizer = Adam
   * learning_rate = 1e-5
* No dropout & schedulers are implemented

# Result
* BLEU Score
      
      BLEU = 25.76, 57.7/31.7/19.5/12.3 (BP=1.000, ratio=1.061, hyp_len=12992, ref_len=12242)
  
* Training Loss Curve   
   ![trainingloss](/results/train/Loss.png)
      
# Run

      sh run.sh

### References
[1] Transformer in DGL (https://github.com/dmlc/dgl/tree/master/examples/pytorch/transformer)

[2] Multi30k (https://www.aclweb.org/anthology/W16-3210/)
