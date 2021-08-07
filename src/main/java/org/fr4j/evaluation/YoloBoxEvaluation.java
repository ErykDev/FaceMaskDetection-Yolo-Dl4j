package org.fr4j.evaluation;


import org.datavec.api.records.metadata.RecordMetaDataImageURI;
import org.datavec.image.recordreader.objdetect.ImageObject;
import org.datavec.image.recordreader.objdetect.ImageObjectLabelProvider;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer;
import org.fr4j.evaluation.display.ConsoleTable;
import org.nd4j.common.primitives.Counter;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.LinkedList;
import java.util.List;

public class YoloBoxEvaluation {

    private static final Logger log = LoggerFactory.getLogger(YoloBoxEvaluation.class);

    protected final double IoU_threshold;
    protected final double confidence_threshold;

    protected final int w;
    protected final int h;

    protected final int gridWidth;
    protected final int gridHeight;


    protected Counter<String> truePositives = new Counter<>();
    protected Counter<String> falsePositives = new Counter<>();
    // protected Counter<Integer> trueNegatives = new Counter<>(); //TN is every part of the image where we did not predict an object. This metrics is not useful for object detection, hence we ignore TN.
    protected Counter<String> falseNegatives = new Counter<>();

    protected List<String> labelsList;

    public YoloBoxEvaluation(ComputationGraph Yolo2_model, RecordReaderDataSetIterator dataSetIterator, ImageObjectLabelProvider labelProvider, int w, int h, int gridWidth, int gridHeight){
        this(Yolo2_model, dataSetIterator,labelProvider, 0.5, 0.4, w, h, gridWidth, gridHeight);
    }

    public YoloBoxEvaluation(ComputationGraph Yolo2_model, RecordReaderDataSetIterator dataSetIterator, ImageObjectLabelProvider labelProvider, double confidence_threshold, int w, int h, int gridWidth, int gridHeight){
        this(Yolo2_model, dataSetIterator,labelProvider, 0.5, confidence_threshold, w, h, gridWidth, gridHeight);
    }

    public YoloBoxEvaluation(ComputationGraph Yolo2_model, RecordReaderDataSetIterator dataSetIterator, ImageObjectLabelProvider labelProvider, double IoU_threshold, double confidence_threshold, int w, int h, int gridWidth, int gridHeight){
        this.IoU_threshold = IoU_threshold;
        this.confidence_threshold = confidence_threshold;
        this.gridHeight = gridHeight;
        this.gridWidth = gridWidth;
        this.w = w;
        this.h = h;

        dataSetIterator.reset();

        labelsList = dataSetIterator.getLabels();

        Yolo2OutputLayer yout = (Yolo2OutputLayer)Yolo2_model.getOutputLayer(0);

        while (dataSetIterator.hasNext()) {
            org.nd4j.linalg.dataset.DataSet ds = dataSetIterator.next();
            RecordMetaDataImageURI metadata = (RecordMetaDataImageURI)ds.getExampleMetaData().get(0);

            INDArray features = ds.getFeatures();
            INDArray results = Yolo2_model.outputSingle(features);

            List<ImageObject> expectedObjects = labelProvider.getImageObjectsForPath(metadata.getURI());
            LinkedList<DetectedObject> detectedObjects = new LinkedList<>(yout.getPredictedObjects(results, confidence_threshold));

            //ImageObject expected : expectedObjects
            for (ImageObject expected : expectedObjects) {

                double best_iou = 0.0;
                int detectedObjectIndex = -1;

                if (detectedObjects.size() > 0) {
                    for (DetectedObject detectedObj : detectedObjects) {
                        if (detectedObj == null)
                            continue;
                        if (!labelsList.get(detectedObj.getPredictedClass()).equals(expected.getLabel()))
                            continue;

                        if (calc_iou(expected, detectedObj) > best_iou && (detectedObj.getConfidence() > confidence_threshold)) {
                            best_iou = calc_iou(expected, detectedObj);
                            detectedObjectIndex = detectedObjects.indexOf(detectedObj);
                        }
                    }

                    if (best_iou >= IoU_threshold) {
                        truePositives.incrementCount(expected.getLabel(),1); //True Positives
                    } else {
                        falsePositives.incrementCount(expected.getLabel(),1); //False Positive
                    }

                    if (detectedObjectIndex != -1)
                        detectedObjects.remove(detectedObjectIndex); //removing detected object to avoid repetition
                }else {
                    falseNegatives.incrementCount(expected.getLabel(),1); //False Negative
                }
            }
        }
    }

    public double getPrecision() {
        return (truePositives.totalCount()) / (truePositives.totalCount() + falsePositives.totalCount());
    }

    public double getRecall() {
        return (truePositives.totalCount()) / (truePositives.totalCount() + falseNegatives.totalCount());
    }

    public double getF1() {
        return (0.5)*(getPrecision() + getRecall());
    }

    public List<String> getLabelsList() {
        return labelsList;
    }

    public void print(){
        LinkedList<String> headers = new LinkedList<String>();
        headers.add("Labels");
        headers.addAll(labelsList);

        LinkedList<LinkedList<String>> content = new LinkedList<>();

        LinkedList<String> True_Positives = new LinkedList<>();
        True_Positives.add("True Positives");
        for (String label : labelsList)
            True_Positives.add(String.valueOf(truePositives.getCount(label)));

        LinkedList<String> False_Positives = new LinkedList<>();
        False_Positives.add("False Positives");
        for (String label : labelsList)
            False_Positives.add(String.valueOf(falsePositives.getCount(label)));

        LinkedList<String> False_Negatives = new LinkedList<>();
        False_Negatives.add("False Negatives");
        for (String label : labelsList)
            False_Negatives.add(String.valueOf(falseNegatives.getCount(label)));

        content.add(True_Positives);
        content.add(False_Positives);
        content.add(False_Negatives);

        ConsoleTable ct = new ConsoleTable(headers, content);
        log.info(ct.toString());

        log.info("Total True Positives:   " + truePositives.totalCount());
        log.info("Total False Positives:  " + falsePositives.totalCount());
        log.info("Total False Negatives:  " + falseNegatives.totalCount());
        log.info("Precision:  " + getPrecision());
        log.info("Recall:  " + getRecall());
        log.info("F1 Score:  " + getF1());

    }

    //https://towardsdatascience.com/evaluating-performance-of-an-object-detection-model-137a349c517b
    private double calc_iou(ImageObject expected, DetectedObject obj){
        assert expected != null;
        assert obj != null;

        double[] xy1 = obj.getTopLeftXY();
        double[] xy2 = obj.getBottomRightXY();

        int x1 = (int) Math.round(w * xy1[0] / gridWidth);
        int y1 = (int) Math.round(h * xy1[1] / gridHeight);

        int x2 = (int) Math.round(w * xy2[0] / gridWidth);
        int y2 = (int) Math.round(h * xy2[1] / gridHeight);


        //if the GT bbox and predcited BBox do not overlap then iou=0
        // If bottom right of x-coordinate  GT  bbox is less than or above the top left of x coordinate of  the predicted BBox
        if(expected.getX2()  < x1)
            return 0.0;

        // If bottom right of y-coordinate  GT  bbox is less than or above the top left of y coordinate of  the predicted BBox
        if(expected.getY2() < y1)
            return 0.0;

        // If bottom right of x-coordinate  GT  bbox is greater than or below the bottom right  of x coordinate of  the predcited BBox
        if(expected.getX1() > x2)
            return 0.0;

        // If bottom right of y-coordinate  GT  bbox is greater than or below the bottom right  of y coordinate of  the predcited BBox
        if(expected.getY1() > y2)
            return 0.0;

        double GT_bbox_area = (expected.getX2() - expected.getX1() + 1) * (expected.getY2() - expected.getY1() + 1);
        double Pred_bbox_area = (x2 - x1 + 1) * (y2 - y1 + 1);

        double x_top_left = Math.max(expected.getX1(), x1);
        double y_top_left = Math.max(expected.getY1(), y1);
        double x_bottom_right = Math.min(expected.getX2(), x2);
        double y_bottom_right = Math.min(expected.getY2(), y2);

        double intersection_area = (x_bottom_right- x_top_left + 1) * (y_bottom_right-y_top_left + 1);
        double union_area = (GT_bbox_area + Pred_bbox_area - intersection_area);

        return intersection_area / union_area;
    }
}