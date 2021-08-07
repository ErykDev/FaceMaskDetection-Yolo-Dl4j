package org.fr4j;


import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.objdetect.ImageObjectLabelProvider;
import org.datavec.image.recordreader.objdetect.ObjectDetectionRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.fr4j.component.UIServerComponent;
import org.fr4j.evaluation.YoloBoxEvaluation;
import org.fr4j.labelProvider.LabelProvider;
import org.fr4j.modelUtils.ModelUtils;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Random;

/**
 * References: <br>
 * - YOLO: Real-Time Object Detection: https://pjreddie.com/darknet/yolo/ <br>
 * <p>
 * Please note, cuDNN should be used to obtain reasonable performance: https://deeplearning4j.konduit.ai/config/backends/config-cudnn#using-deeplearning-4-j-with-cudnn
 */
public class YoloFMDTraining {
    private static final Logger log = LoggerFactory.getLogger(YoloFMDTraining.class);

    public static void main(String[] args) throws Exception {

        UIServerComponent uiServerComponent = new UIServerComponent();

        // parameters matching the pretrained TinyYOLO model
        int width = 416;
        int height = 416;
        int nChannels = 3;
        int gridWidth = 13;
        int gridHeight = 13;

        // parameters for the training phase
        int batchSize = 2;
        int nEpochs = 220;

        Random rng = new Random();

        File trainDir = new File("data\\train");
        File evalDir = new File("data\\test");

        log.info("Load DataSets...");

        FileSplit trainData = new FileSplit(trainDir, new String[]{".png", ".jpg"}, rng);
        FileSplit evalData = new FileSplit(evalDir, new String[]{".png", ".jpg"}, rng);

        ImageObjectLabelProvider trainLabelProvider = new LabelProvider(trainDir);
        ImageObjectLabelProvider evalLabelProvider = new LabelProvider(evalDir);

        ObjectDetectionRecordReader recordReaderTrain = new ObjectDetectionRecordReader(height, width, nChannels, gridHeight, gridWidth, trainLabelProvider);
        recordReaderTrain.initialize(trainData);

        ObjectDetectionRecordReader recordReaderEval = new ObjectDetectionRecordReader(height, width, nChannels, gridHeight, gridWidth, evalLabelProvider);
        recordReaderEval.initialize(evalData);

        DataSetPreProcessor preProcessingScaler = new ImagePreProcessingScaler();

        // ObjectDetectionRecordReader performs regression, so we need to specify it here
        RecordReaderDataSetIterator trainIterator = new RecordReaderDataSetIterator(recordReaderTrain, batchSize, 1, 1, true);
        trainIterator.setPreProcessor(preProcessingScaler);

        RecordReaderDataSetIterator evalIterator = new RecordReaderDataSetIterator(recordReaderEval, 1, 1, 1, true);
        evalIterator.setPreProcessor(preProcessingScaler);
        evalIterator.setCollectMetaData(true);

        ComputationGraph model;
        String modelFilename = "model.zip";

        File modelFile = new File(modelFilename);

        double bestScore = 0.0;

        if (modelFile.exists()) {
            log.info("Loading model...");
            model = ModelUtils.load(modelFile);

            bestScore = new YoloBoxEvaluation(model, evalIterator, evalLabelProvider, width, height, gridWidth, gridHeight).getF1();
            log.info("Current F1 score :" + bestScore);
        } else {
            log.info("Building model...");
            model = ModelUtils.build();
            log.info(model.summary(InputType.convolutional(height, width, nChannels)));
        }

        log.info("Initializing uiServerComponent...");
        uiServerComponent.reinitialize(model);

        log.info("Training model...");

        for(int i = 0; i < nEpochs; i++) {
            model.fit(trainIterator);

            log.info("Evaluating model...");
            double tempScore = new YoloBoxEvaluation(model, evalIterator, evalLabelProvider, width, height, gridWidth, gridHeight).getF1();

            if (tempScore > bestScore) {
                bestScore = tempScore;
                log.info("New best F1 score: " + bestScore + " reached at Epoch: " + i);
                log.info("Saving model...");
                ModelSerializer.writeModel(model, modelFilename, true);
            }else
                log.info("Current F1 score :" + bestScore);
        }
        uiServerComponent.stop();
    }
}