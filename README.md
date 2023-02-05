# transfuser_skin_lesion_classifier
Bi-modal cnn-transformer feature extractor with prototype decision tree classifier

![alt text](https://github.com/MathewKouch/transfuser_skin_lesion_classifier/blob/main/transfuser_architecture.png)

Model consists of two ResNet34 CNN branch for feature map extraction of clinical and dermoscopic images. 
To extract and inject cross modality features, vision based transformer encoders are used thorughout four resolutions along the CNN branches.
Feature fusion from both branches are concatenated and projected to a 512 dimensional representation to compare with skin lesion prototypes for classification.

![alt text](https://github.com/MathewKouch/transfuser_skin_lesion_classifier/blob/main/transfuser_diagnosis.png)
An induced decision tree is built from the concurrently trained prototypes and provides visual and better explainable diagnosis than saliency maps.
