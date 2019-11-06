package com.srp;

import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * Generates the three models.
 *
 * @author Julia Biswas
 * @version 12 March 2019
 */
public class ModelGenerator
{
    /**
     * Retrieves the data from the given file.
     * @param file      the file with the data
     * @return          the data that has been retrieved
     */
    public Instances loadData(String file)
    {
        Instances data = null;
        try {
            data = DataSource.read(file);
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }
        }
        catch (Exception excep) {
            //get a logger and then log a message with a severe level and excep's stacktrace.
            Logger.getLogger(ModelGenerator.class.getName()).log(Level.SEVERE, null, excep);
        }
        return data;
    }

    /**
     * Builds the classifier using the Naive Bayes model.
     *
     * @param trainingSet       the training data
     * @return                  the classifier
     */
    public Classifier buildClassifierNB(Instances trainingSet)
    {
        NaiveBayes nB = new NaiveBayes();
        try {
            nB.buildClassifier(trainingSet);
        }
        catch (Exception excep) {
            //get a logger and then log a message with a severe level and excep's stacktrace.
            Logger.getLogger(ModelGenerator.class.getName()).log(Level.SEVERE, null, excep);
        }
        return nB;
    }

    /**
     * Builds the classifier using the Support Vector Machine model.
     *
     * @param trainingSet       the training data
     * @return                  the classifier
     */
    public Classifier buildClassifierSVM(Instances trainingSet) throws Exception
    {
        SMO sVM = new SMO();

        try {
            sVM.buildClassifier(trainingSet);
        }
        catch (Exception excep) {
            //get a logger and then log a message with a severe level and excep's stacktrace.
            Logger.getLogger(ModelGenerator.class.getName()).log(Level.SEVERE, null, excep);
        }
        return sVM;
    }

    /**
     * Builds the classifier using the Random Forest model.
     *
     * @param trainingSet       the training data
     * @return                  the classifier
     */
    public Classifier buildClassifierRF(Instances trainingSet) throws Exception
    {
        RandomForest rF = new RandomForest();

        try {
            rF.buildClassifier(trainingSet);
        }
        catch (Exception excep) {
            //get a logger and then log a message with a severe level and excep's stacktrace.
            Logger.getLogger(ModelGenerator.class.getName()).log(Level.SEVERE, null, excep);
        }
        return rF;
    }

    /**
     * Checks the accuracy of the classifier.
     *
     * @param model             the model to check the accuracy of
     * @param trainingSet       the training data set to use
     * @param testingSet        the testing data set to use
     * @return                  a summary of the accuracy testing results
     */
    public String evaluateModel(Classifier model, Instances trainingSet, Instances testingSet)
    {
        Evaluation eval = null;
        try {
            eval = new Evaluation(trainingSet);
            eval.evaluateModel(model, testingSet);
        }
        catch (Exception excep) {
            //get a logger and then log a message with a severe level and excep's stacktrace.
            Logger.getLogger(ModelGenerator.class.getName()).log(Level.SEVERE, null, excep);
        }
        return eval.toSummaryString("", false);
    }
}
