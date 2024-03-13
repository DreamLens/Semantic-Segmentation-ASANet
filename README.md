#  Affinity Space Adaptation for Semantic Segmentation Across Domains\n---\nPytorch implementation of the paper. The objective of this implementation is to resolve the problem of unsupervised domain adaptation (UDA) in semantic segmentation, thereby reaching peak performance on established benchmarks. \n\n## Reference\nIf you find this resource helpful in your research, please reference:\n\n```
@ARTICLE{9184275,\n  author={W. {Zhou} and Y.{Wang} and J. {Chu} and J. {Yang} and X. {Bai} and Y. {Xu}},\n  journal={IEEE Transactions on Image Processing}, \n  title={Affinity Space Adaptation for Semantic Segmentation Across Domains}, \n  year={2020},\n  volume={},\n  number={},\n  pages={1-1},}\n```\n\n## Example Results and Quantitative Results\nThe images below provide a visual representation of the output.\n![](figs/teaser.png)\n![](figs/gta5_rst.png)\n![](figs/syn_rst.png)\n\n## How-To\n### Datasets\nThe source dataset, [GTA5 Dataset](https://download.visinf.tu-darmstadt.de/data/from_games/), and the target dataset, [Cityscapes Dataset](https://www.cityscapes-dataset.com/), should be downloaded.\n\n### Initial Weights\nInitial weights and trained models are available for download from [[Google Drive](https://drive.google.com/drive/folders/1F-kmmV89uJK7-IGzeAox3BSkH014GpdL?usp=sharing)] or [[Baidu Drive](https://pan.baidu.com/s/1NOKDNWVd5-kd2w0rhzwM_w)]. Position the weights in the 'Semantic-Segmentation-ASANet/pretrained' directory. \n\n### 