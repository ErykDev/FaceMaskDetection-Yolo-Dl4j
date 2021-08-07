package org.fr4j.modelUtils;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public final class ModelUtils {
    private ModelUtils(){}

    public static ComputationGraph load(File model) throws IOException {
        return ComputationGraph.load(model, true);
    }

    public static ComputationGraph build() throws IOException {
        int width = 416;
        int height = 416;
        int nChannels = 3;

        int nClasses = 2;

        // parameters for the Yolo2OutputLayer
        int nBoxes = 5;
        double lambdaNoObj = 0.5;
        double lambdaCoord = 5.0;
        double[][] priorBoxes = { { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 }, { 2, 2 } };

        // parameters for the training phase
        double learningRate = 1e-3;
        double lrMomentum = 0.9;

        Random rng = new Random(System.currentTimeMillis());


        ComputationGraph pretrained = (ComputationGraph) TinyYOLO.builder().build().initPretrained();
        INDArray priors = Nd4j.create(priorBoxes);

        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .seed(rng.nextInt())
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(1.0)
                .updater(new Nesterovs(learningRate, lrMomentum))
                .activation(Activation.IDENTITY)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .build();

        return new TransferLearning.GraphBuilder(pretrained)
                .fineTuneConfiguration(fineTuneConf)

                .removeVertexKeepConnections("conv2d_9")
                .addLayer("conv2d_9",
                        new ConvolutionLayer.Builder(1,1)
                                .nIn(1024).nOut(nBoxes * (5 + nClasses)).hasBias(false)
                                .stride(1,1).convolutionMode(ConvolutionMode.Same)
                                .weightInit(WeightInit.UNIFORM)
                                .activation(Activation.IDENTITY)
                                .build(),
                        "leaky_re_lu_8")

                .removeVertexKeepConnections("outputs")
                .addLayer("outputs",
                        new Yolo2OutputLayer.Builder()
                                .lambdaCoord(lambdaCoord)
                                .lambdaNoObj(lambdaNoObj)
                                .boundingBoxPriors(priors)
                                .build(),
                        "conv2d_9")
                .setInputTypes(InputType.convolutional(height, width, nChannels))
                .setOutputs("outputs")
                .build();
    }
}