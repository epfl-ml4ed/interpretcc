# InterpretCC

This repository is the official implementation of the preprint entitled ["InterpretCC: Conditional Computation for Inherently Interpretable Neural Networks"](https://arxiv.org/abs/2402.02933) written by [Vinitra Swamy](http://github.com/vinitra), [Julian Blackwell](https://ch.linkedin.com/in/julian-blackwell-93407a13b), [Jibril Frej](https://github.com/Jibril-Frej), [Martin Jaggi](https://github.com/martinjaggi), and [Tanja Käser](https://people.epfl.ch/tanja.kaeser/?lang=en).

InterpretCC (interpretable conditional computation) is a family of interpretable-by-design neural networks that guarantee human-centric interpretability while maintaining comparable performance to state-of-the-art models by adaptively and sparsely activating features before prediction. We extend this idea into an interpretable mixture-of-experts model that allows humans to specify topics of interest, discretely separates the feature space for each data point into topical subnetworks, and adaptively and sparsely activates these topical subnetworks. We demonstrate variations of the InterpretCC architecture for text and tabular data across several real-world benchmarks: six online education courses, news classification, breast cancer diagnosis, and review sentiment.

## Quick Start Guide
1. `git clone https://github.com/epfl-ml4ed/interpretcc.git`
2. `cd interpretcc`
3. `pip install -r requirements.txt`
4. `jupyter notebook`
5. Run feature gating or interpretable mixture-of-experts experiments as Jupyter Notebooks in the `notebooks/` folder!

## Repository Structure

Experiments are located in `notebooks/`, corresponding directly to the three model variations (feature gating, MoE gated routing, MoE top-k routing) mentioned in the paper. The full pipeline is included for the AG News dataset. The InterpretCC models were written in PyTorch.

We also provide a modularized TensorFlow implementation of InterpretCC in the `scripts/` folder, including more variations from the Appendix education experiments. For more information about each experiment, please reference the paper directly.

Public datasets used in the InterpretCC experiments (AG News, Breast Cancer Diagnosis, SST 2) are included in the `data/` folder. It also includes [a pickle file](data/ddc_subcategories.pkl) of the Dewey Decimal Categories used for text routing.  

The Gumbel Sigmoid implementation was borrowed from [AngelosNal/PyTorch-Gumbel-Sigmoid](https://github.com/AngelosNal/PyTorch-Gumbel-Sigmoid/blob/main/gumbel_sigmoid.py).

## Contributing 

This code is provided for educational purposes and aims to facilitate reproduction of our results, and further research 
in this direction. We have done our best to document, refactor, and test the code before publication.

If you find any bugs or would like to contribute new models, training protocols, etc, please let us know. Feel free to file issues and pull requests on the repo and we will address them as we can.
  
## Citations  
If you find this code useful in your work, please cite our paper:
```
Swamy, V., Blackwell, J., Frej, J., Jaggi, M., Käser, T. (2024). 
InterpretCC: Conditional Computation for Inherently Interpretable Neural Networks. 
```
## License
This code is free software: you can redistribute it and/or modify it under the terms of the [MIT License](LICENSE).

This software is distributed in the hope that it will be useful, but without any warranty; without even the implied warranty of merchantability or fitness for a particular purpose. See the [MIT License](LICENSE) for details.
