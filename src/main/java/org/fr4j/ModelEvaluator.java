package org.fr4j;

import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.objdetect.ImageObjectLabelProvider;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.fr4j.evaluation.YoloBoxEvaluation;
import org.fr4j.labelProvider.LabelProvider;
import org.fr4j.modelUtils.ModelUtils;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Random;

public class ModelEvaluator {
    private static final Logger log = LoggerFactory.getLogger(DisplayTest.class);

    public static void main(String[] args) throws Exception {
        // parameters matching the pretrained TinyYOLO model
        int width = 416;
        int height = 416;
        int nChannels = 3;
        int gridWidth = 13;
        int gridHeight = 13;


        Random rng = new Random();

        File evalDir = new File("data\\test");

        log.info("Loading DataSet...");

        FileSplit testData = new FileSplit(evalDir, new String[]{".png", ".jpg"}, rng);

        ImageObjectLabelProvider labelProvider = new LabelProvider(evalDir);

        ObjectDetectionRecordReader recordReaderTest = new ObjectDetectionRecordReader(height, width, nChannels, gridHeight, gridWidth, labelProvider);
        recordReaderTest.initialize(testData);

        // ObjectDetectionRecordReader performs regression, so we need to specify it here
        RecordReaderDataSetIterator evalIterator = new RecordReaderDataSetIterator(recordReaderTest, 1, 1, 1, true);
        evalIterator.setPreProcessor(new ImagePreProcessingScaler());
        evalIterator.setCollectMetaData(true);

        ComputationGraph model;
        String modelFilename = "model.zip";

        File modelFile = new File(modelFilename);

        assert modelFile.exists();

        log.info("Loading model...");
        model = ModelUtils.load(modelFile);

        log.info("Evaluating model...");
        YoloBoxEvaluation yoloBoxEvaluation = new YoloBoxEvaluation(model, evalIterator, labelProvider, width, height, gridWidth, gridHeight);
        yoloBoxEvaluation.print();
    }
}
