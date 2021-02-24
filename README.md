## Face Mask Detection
Dl4J Implementation of Tiny-Yolov2

* Project based on [DL4J-example](https://github.com/eclipse/deeplearning4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/advanced/modelling/objectdetection)

(click for full video)


[![Watch the video](https://i.ibb.co/PFX33NV/index.gif)](https://youtu.be/CMewb5FUtt4)


### How to run

* Import project dependencies with Gradle
* download model from g-drive watch [model.txt](https://github.com/BadlyDrunkScotsman/FaceMaskDetection-Yolo-Dl4j/blob/main/model.txt)
* If you own an NVIDIA card with a CUDA support you can enable the usage in Gradle file

### Training
* If you wish to use your own data simply paste it into the /data folder, link for an example set is in [data.txt](https://github.com/BadlyDrunkScotsman/FaceMaskDetection-Yolo-Dl4j/blob/main/data/data.txt)
* The project is using a Keras Yolo data-format

If model.zip isn't available in project then root directory then the training will start.

![Training](https://i.ibb.co/FJwQgvW/Score.png)

### Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


