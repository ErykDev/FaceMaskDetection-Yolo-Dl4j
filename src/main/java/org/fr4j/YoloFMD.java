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
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.fr4j.component.UIServerComponent;
import org.fr4j.labelProvider.FaceLabelProvider;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.List;
import java.util.Random;

import static org.bytedeco.opencv.global.opencv_core.CV_8U;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

/**
 * References: <br>
 * - YOLO: Real-Time Object Detection: https://pjreddie.com/darknet/yolo/ <br>
 * <p>
 * Please note, cuDNN should be used to obtain reasonable performance: https://deeplearning4j.konduit.ai/config/backends/config-cudnn#using-deeplearning-4-j-with-cudnn
 */
public class YoloFMD {
    private static final Logger log = LoggerFactory.getLogger(YoloFMD.class);

    public static void main(String[] args) throws java.lang.Exception {

        UIServerComponent uiServerComponent = new UIServerComponent();

        // parameters matching the pretrained TinyYOLO model
        int width = 416;
        int height = 416;
        int nChannels = 3;
        int gridWidth = 13;
        int gridHeight = 13;


        int nClasses = 2;

        // parameters for the Yolo2OutputLayer
        int nBoxes = 5;
        double lambdaNoObj = 0.5;
        double lambdaCoord = 5.0;
        double[][] priorBoxes = { { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 } };
        double detectionThreshold = 0.3;

        // parameters for the training phase
        int batchSize = 2;
        int nEpochs = 100;
        double learningRate = 1e-3;
        double lrMomentum = 0.9;

        Random rng = new Random(System.currentTimeMillis());


        File trainDir = new File("data\\train");
        File testDir = new File("data\\test");

        log.info("Load data...");

        FileSplit trainData = new FileSplit(trainDir, new String[]{".jpg"}, rng);
        FileSplit testData = new FileSplit(testDir, new String[]{".jpg"}, rng);

        ObjectDetectionRecordReader recordReaderTrain = new ObjectDetectionRecordReader(height, width, nChannels,
                gridHeight, gridWidth, new FaceLabelProvider(trainDir));
        recordReaderTrain.initialize(trainData);

        ObjectDetectionRecordReader recordReaderTest = new ObjectDetectionRecordReader(height, width, nChannels,
                gridHeight, gridWidth, new FaceLabelProvider(testDir));
        recordReaderTest.initialize(testData);

        // ObjectDetectionRecordReader performs regression, so we need to specify it here
        RecordReaderDataSetIterator train = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, 1, true);
        train.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        RecordReaderDataSetIterator test = new RecordReaderDataSetIterator(recordReaderTest, 1, 1, 1, true);
        test.setPreProcessor(new ImagePreProcessingScaler(0, 1));


        ComputationGraph model;
        String modelFilename = "model.zip";

        if (new File(modelFilename).exists()) {
            log.info("Loading model...");
            model = ComputationGraph.load(new File(modelFilename), true);
        } else {
            log.info("Building model...");

            ComputationGraph pretrained = (ComputationGraph) TinyYOLO.builder().build().initPretrained();
            INDArray priors = Nd4j.create(priorBoxes);

            FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                    .seed(rng.nextInt())
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                    .gradientNormalizationThreshold(1.0)
                    .updater(new Nesterovs.Builder().learningRate(learningRate).momentum(lrMomentum).build())
                    .activation(Activation.IDENTITY)
                    .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                    .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                    .build();

            model = new TransferLearning.GraphBuilder(pretrained)
                    .fineTuneConfiguration(fineTuneConf)

                    .removeVertexKeepConnections("conv2d_9")
                    .removeVertexKeepConnections("outputs")

                    .addLayer("conv2d_9",
                            new ConvolutionLayer.Builder(1,1)
                                    .nIn(1024).nOut(nBoxes * (5 + nClasses))
                                    .stride(1,1).convolutionMode(ConvolutionMode.Same)
                                    .weightInit(WeightInit.UNIFORM).activation(Activation.IDENTITY).hasBias(false)
                                    .build(),
                            "leaky_re_lu_8")
                    .addLayer("outputs",
                            new Yolo2OutputLayer.Builder()
                                    .lambdaCoord(lambdaCoord)
                                    .lambdaNoObj(lambdaNoObj)
                                    .boundingBoxPriors(priors)
                                    .build(),
                            "conv2d_9")
                    .setOutputs("outputs")
                    .build();

            log.info(model.summary(InputType.convolutional(height, width, nChannels)));
        }

        log.info("Training model...");

        model.setListeners(new ScoreIterationListener(1));
        uiServerComponent.reinitialize(model);

        model.fit(train, nEpochs);

        log.info("Saving model...");
        ModelSerializer.writeModel(model, modelFilename, true);


        // visualize results on the test set
        NativeImageLoader imageLoader = new NativeImageLoader();
        CanvasFrame frame = new CanvasFrame("Face Mask Detection");
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer yout = (org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer)model.getOutputLayer(0);
        List<String> labels = test.getLabels();
        test.setCollectMetaData(true);
        Scalar[] colormap = { Scalar.GREEN, Scalar.RED };

        while (test.hasNext() && frame.isVisible()) {
            org.nd4j.linalg.dataset.DataSet ds = test.next();
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
                double[] xy1 = obj.getTopLeftXY();
                double[] xy2 = obj.getBottomRightXY();
                String label = labels.get(obj.getPredictedClass());
                int x1 = (int) Math.round(w * xy1[0] / gridWidth);
                int y1 = (int) Math.round(h * xy1[1] / gridHeight);
                int x2 = (int) Math.round(w * xy2[0] / gridWidth);
                int y2 = (int) Math.round(h * xy2[1] / gridHeight);
                rectangle(image, new Point(x1, y1), new Point(x2, y2), colormap[obj.getPredictedClass()]);
                putText(image, label, new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, colormap[obj.getPredictedClass()]);

            }
            frame.setTitle("FaceDetection");
            frame.setCanvasSize(w, h);
            frame.showImage(converter.convert(image));
            frame.waitKey();
        }
        frame.dispose();
    }
}