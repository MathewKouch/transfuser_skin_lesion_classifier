# transfuser_skin_lesion_classifier
Bi-modal cnn-transformer feature extractor with prototype decision tree classifier for accurate and explainable skin lesion diagnosis.

## Architecture
![alt text](https://github.com/MathewKouch/transfuser_skin_lesion_classifier/blob/main/transfuser_architecture.png)
Model consists of:
1. A Feature Extractor (FX) with two ResNet34 CNN branch for feature map extraction of clinical and dermoscopic images, and four vision based transformer encoders for richer feature and representations. 
Inspired by https://arxiv.org/abs/2104.09224

2. Prototypes of every skin lesion class (65) in the dataset, that are guided with class hierarchical information. Benefit of injecting hierarchy to training prototype is enabling model to make better mistakes where its representation vector is misclassified (closest to an incorrect prototype) but still relatively close or within the same higher hiercharchial level as the true class.
Inspired by https://arxiv.org/abs/2007.03047

3. Induced decision tree built from the same prototypes for hierarchical decision making, and enables visual and explainable diagnosis. 
Inspired by https://arxiv.org/abs/2004.00221

## Diagnosis
![alt text](https://github.com/MathewKouch/transfuser_skin_lesion_classifier/blob/main/transfuser_diagnosis.png)
Decision tree (right) provides more meaningful diagnostic information than typical saliency map (left), visually explaining decision probabilities for every probable diagnosis. 

## Transformer 
![alt text](https://github.com/MathewKouch/transfuser_skin_lesion_classifier/blob/main/transformer.png)
Cross modality features are generated via attention mechanism with transformer encoder, and injected back into the cnn.
