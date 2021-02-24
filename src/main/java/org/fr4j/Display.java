package org.fr4j;

import org.bytedeco.javacv.*;
import org.bytedeco.opencv.opencv_core.*;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.io.File;
import java.util.Arrays;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_imgproc.*;


public class Display implements Runnable {
    private static final Logger log = LoggerFactory.getLogger(Display.class);
    private static final CanvasFrame frame = new CanvasFrame("Face Mask Detection");

    public Display()
    {
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

    @Override
    public void run()
    {
        OpenCVFrameGrabber grabber = new OpenCVFrameGrabber(0);
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        NativeImageLoader imageLoader = new NativeImageLoader();
        ImagePreProcessingScaler preProcessor = new ImagePreProcessingScaler(0, 1);

        ComputationGraph model;
        String modelFilename = "model.zip";
        double detectionThreshold = 0.3;
        int gridWidth = 13;
        int gridHeight = 13;

        int croppedHeight = 360;
        int croppedWidth = 416;

        long lastTime;
        double fps = 0;


        int w = croppedWidth * 2;
        int h = croppedHeight * 2;

        try
        {
            log.info("Load model...");
            model = ComputationGraph.load(new File(modelFilename), true);

            org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer)model.getOutputLayer(0);
            List<String> labels = Arrays.asList("mask", "no-mask");
            Scalar[] colormap = { Scalar.GREEN, Scalar.RED };

            Frame cvimg;

            grabber.start();

            while (frame.isVisible()) {

                cvimg = grabber.grab();
                //BufferedImage image;
                if (cvimg != null) {

                    lastTime = System.nanoTime();

                    Mat mat = converter.convertToMat(cvimg);
                    // Do some processing on mat with OpenCV

                    //Cropping Mat
                    Rect rectCrop = new Rect(112, 0, croppedWidth, croppedHeight);
                    Mat processedMat = new Mat(mat, rectCrop);

                    INDArray indArray = imageLoader.asMatrix(processedMat);
                    preProcessor.preProcess(indArray);

                    INDArray results = model.outputSingle(indArray);

                    Mat image = new Mat();
                    resize(mat, image, new Size(w, h));

                    List<DetectedObject> detectedObjects = yout.getPredictedObjects(results, detectionThreshold);

                    for (DetectedObject obj : detectedObjects) {
                        double[] xy1 = obj.getTopLeftXY();
                        double[] xy2 = obj.getBottomRightXY();
                        String label = labels.get(obj.getPredictedClass());
                        int x1 = (int) Math.round(w * xy1[0] / gridWidth);
                        int y1 = (int) Math.round(h * xy1[1] / gridHeight);
                        int x2 = (int) Math.round(w * xy2[0] / gridWidth);
                        int y2 = (int) Math.round(h * xy2[1] / gridHeight);
                        rectangle(image, new Point(x1, y1), new Point(x2, y2), colormap[obj.getPredictedClass()]);
                        putText(image, label +" C:"+String.format("%.2f", obj.getConfidence()), new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 0.7, colormap[obj.getPredictedClass()]);
                    }
                    putText(image, "FPS:" + String.format("%.2f", fps), new Point(10,15), FONT_HERSHEY_DUPLEX, 0.5, Scalar.GRAY);

                    fps = 1E9 / (System.nanoTime() - lastTime); //one second(nano) divided by amount of time it takes for one frame to finish

                    frame.showImage(converter.convert(image));

                    System.gc();
                }
            }

            grabber.stop();
            frame.dispose();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String... args)
    {
        Display webcam = new Display();
        webcam.start();
    }

    public void start()
    {
        new Thread(this).start();
    }
}