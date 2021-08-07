package org.fr4j;

import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_core.Scalar;
import org.bytedeco.opencv.opencv_core.Size;
import org.datavec.api.records.metadata.RecordMetaDataImageURI;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.objdetect.ImageObjectLabelProvider;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;

import org.fr4j.labelProvider.LabelProvider;
import org.fr4j.modelUtils.ModelUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.List;
import java.util.Random;

import static org.bytedeco.opencv.global.opencv_core.CV_8U;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class DisplayTest {
    private static final Logger log = LoggerFactory.getLogger(DisplayTest.class);

    public static void main(String[] args) throws Exception {
        // parameters matching the pretrained TinyYOLO model
        int width = 416;
        int height = 416;
        int nChannels = 3;
        int gridWidth = 13;
        int gridHeight = 13;

        // parameters for the Yolo2OutputLayer
        double detectionThreshold = 0.3;

        Random rng = new Random();

        File testDir = new File("data\\test");

        log.info("Load DataSet...");

        FileSplit testData = new FileSplit(testDir, new String[]{".png", ".jpg"}, rng);

        ImageObjectLabelProvider labelProvider = new LabelProvider(testDir);

        ObjectDetectionRecordReader recordReaderTest = new ObjectDetectionRecordReader(height, width,nChannels, gridHeight, gridWidth, labelProvider);
        recordReaderTest.initialize(testData);

        // ObjectDetectionRecordReader performs regression, so we need to specify it here
        RecordReaderDataSetIterator testIterator = new RecordReaderDataSetIterator(recordReaderTest, 1, 1, 1, true);
        testIterator.setPreProcessor(new ImagePreProcessingScaler());


        ComputationGraph model;
        String modelFilename = "model.zip";

        File modelFile = new File(modelFilename);

        assert modelFile.exists();

        log.info("Loading model...");
        model = ModelUtils.load(modelFile);
        log.info(model.summary(InputType.convolutional(height, width, nChannels))); //logging model summary


        // visualize results on the test set
        NativeImageLoader imageLoader = new NativeImageLoader();
        CanvasFrame frame = new CanvasFrame("Face Mask Detection");
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer)model.getOutputLayer(0);
        List<String> labels = testIterator.getLabels();
        testIterator.setCollectMetaData(true);

        while (testIterator.hasNext() && frame.isVisible()) {
            org.nd4j.linalg.dataset.DataSet ds = testIterator.next();
            RecordMetaDataImageURI metadata = (RecordMetaDataImageURI)ds.getExampleMetaData().get(0);
            INDArray features = ds.getFeatures();
            INDArray results = model.outputSingle(features);
            List<DetectedObject> objs = yout.getPredictedObjects(results, detectionThreshold);
            File file = new File(metadata.getURI());
            log.info(file.getName() + ": " + objs);

            Mat mat = imageLoader.asMat(features);
            Mat convertedMat = new Mat();
            mat.convertTo(convertedMat, CV_8U, 255, 0);
            int w = metadata.getOrigW() * 2;
            int h = metadata.getOrigH() * 2;
            Mat image = new Mat();
            resize(convertedMat, image, new Size(w, h));

            for (DetectedObject obj : objs) {
                if(obj == null)
                    continue;

                double[] xy1 = obj.getTopLeftXY();
                double[] xy2 = obj.getBottomRightXY();

                int x1 = (int) Math.round(w * xy1[0] / gridWidth);
                int y1 = (int) Math.round(h * xy1[1] / gridHeight);

                int x2 = (int) Math.round(w * xy2[0] / gridWidth);
                int y2 = (int) Math.round(h * xy2[1] / gridHeight);

                rectangle(image, new Point(x1, y1), new Point(x2, y2), Scalar.RED);
                putText(image, labels.get(obj.getPredictedClass()), new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 0.6, Scalar.RED);
            }

            frame.setTitle("Tumor Detection");
            frame.setCanvasSize(w, h);
            frame.showImage(converter.convert(image));
            frame.waitKey();
        }
        frame.dispose();
    }
}
