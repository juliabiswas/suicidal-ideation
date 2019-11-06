package com.srp;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomialText;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.RandomForest;
import weka.core.Debug;
import weka.core.Instances;

/**
 * Uses the ModelGenerator and ModelClassifier
 * classes to generate the three models and use them for
 * prediction.
 *
 * @author Julia Biswas
 * @version 12 March 2019
 */
public class Test {
    public static final String DATAFILE = "/Users/juliabiswas/Downloads/weka-3-8-3/dataset.arff";

    public static void main(String[] args) throws Exception
    {
        ModelGenerator modGen = new ModelGenerator();

        Instances data = modGen.loadData(DATAFILE);

        //splits percent of data that goes to training (90%) and
        // percent of data that goes to testing (10%)
        int trainingSize = (int) Math.round(data.numInstances() * .9);
        int testingSize = data.numInstances() - trainingSize;

        //randomizes the data
        data.randomize(new Debug.Random(1));

        //gets the data into a training set and a testing set
        Instances trainingSet = new Instances(data, 0, trainingSize);
        Instances testingSet = new Instances(data, trainingSize, testingSize);

        //creates the Naive Bayes classifier & evaluates
        NaiveBayes classifNB = (NaiveBayes) modGen.buildClassifierNB(trainingSet);
        String evalSummaryNB = modGen.evaluateModel(classifNB, trainingSet, testingSet);
        System.out.println("Evaluation for NB" + evalSummaryNB);

        //creates the SVM classifier & evaluates
        SMO classifSVM = (SMO) modGen.buildClassifierSVM(trainingSet);
        String evalSummarySVM = modGen.evaluateModel(classifSVM, trainingSet, testingSet);
        System.out.println("Evaluation for SVM" + evalSummarySVM);

        //creates the Random Forest classifier & evaluates
        RandomForest classifRF = (RandomForest) modGen.buildClassifierRF(trainingSet);
        String evalSummaryRF = modGen.evaluateModel(classifRF, trainingSet, testingSet);
        System.out.println("Evaluation for RF" + evalSummaryRF);
    }
}
