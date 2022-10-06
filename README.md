# Ensemble-deep-learning
This repository contains a pseudo code for the ensemble deep learning method developed in [1]. The ensemble deep learning method can be used on a classification problem in which the input data is a time series of (also possible multi-channel) images. The ensemble method is based on three steps: (1) train a deep neural network which provides probability outcomes as output (2) choose the optimal threshold which converts the probability outcomes in binary outcomes by optimizing a desired skill score computed on the training set (3) choose a level alpha which defines the goodness of predictions on the validation set (predictors which provide the desired skill score higher that the level alpha are involved in the definition of the ensemble prediction) (4) choose the subset of epochs providing valuable predictions on the validation set (i.e. the subset of predictors providing the desired skill score computed on the validation set higher than the level alpha) (4) the ensemble prediction on a new input is defined as the median value of the predictions computed  by the selcted predictors. The ensemble method is based on the optimization of a desired skill score: the implementation comprises the use of standard skill scores as the True skill Statistic (TSS) or the use of the value-weighted skill scores as value-weighted TSS (wTSS) introduced in [2].

[1] Guastavino Sabrina, Piana Michele, Tizzi Marco, Cassola Federico, Iengo Antonio, Sacchetti Davide, Solazzo Enrico, Benvenuto Federico, Prediction of severe thunderstorm events with ensemble deep learning and radar data, 2021, arXiv:2109.09791

[2] Guastavino Sabrina, Piana Michele and Benvenuto Federico, Bad and good errors: value-weighted skill scores in deep ensemble learning, IEEE Transactions on Neural Networks and Learning Systems, 2022, doi: 10.1109/TNNLS.2022.3186068.

## Using the code

Major softwares: Keras 2.4.3, Tensorflow 2.3.0, Python 2.7.15. The requirements.txt file lists all the packages necessary to run the notebooks and functions in this repository.


## Contents

This repository contains a pseudo code for the ensemble method used in [1]:
- utilities_nowcasting.py mainly includes all the necessary functions for implementing the ensemble strategy based on the optimization of classical skill scores and value-weighted skill scores and assessing the performance of predictions.
- main_nowcasting.py is a demo which contains the definition of of the Long-Term Recurrent neural Network (LRCN) model used in the application shown in [1], and the code for the ensemble deep learning method described above. 

## Citation

If you use the codes, please consider citing [1,2]. Here are the bibitex entries

@article{guastavino2021prediction,
  
  title={Prediction of severe thunderstorm events with ensemble deep learning and radar data},
  
  author={Guastavino, Sabrina and Piana, Michele and Tizzi, Marco and Cassola, Federico and Iengo, Antonio and Sacchetti, Davide and Solazzo, Enrico and Benvenuto, Federico},
  
  journal={arXiv preprint arXiv:2109.09791},
  
  year={2021}
}


@article{guastavino2022bad,

  author={Guastavino, Sabrina and Piana, Michele and Benvenuto, Federico},

  journal={IEEE Transactions on Neural Networks and Learning Systems}, 

  title={Bad and Good Errors: Value-Weighted Skill Scores in Deep Ensemble Learning}, 

  year={2022},

  volume={},

  number={},

  pages={1-10},

  doi={10.1109/TNNLS.2022.3186068}}
  


