# Pandering in a (Flexible) Representative Democracy

This is the official implementaion of paper [Pandering in a (Flexible) Representative Democracy](https://arxiv.org/abs/2211.09986).

## Getting Started
### Install requirements
```
pip install -r requirements.txt
```
### Use of WOLFRAM MATHEMATICA
You will need a subscription of WOLFRAM MATHEMATICA in order to run the code.

### Start Training 
All environments are described in `train_voting.py`. Choose the environment by chaning the `main` function in `train_voting.py`.
```
python3 train_voting.py
```

### Testing
All test codes are in `test_mip.py`. You can choose the coresponding function to test your trained agent.

### Cite 
```
@misc{sun2023pandering,
      title={Pandering in a Flexible Representative Democracy}, 
      author={Xiaolin Sun and Jacob Masur and Ben Abramowitz and Nicholas Mattei and Zizhan Zheng},
      year={2023},
      eprint={2211.09986},
      archivePrefix={arXiv},
      primaryClass={cs.MA}
}
```


