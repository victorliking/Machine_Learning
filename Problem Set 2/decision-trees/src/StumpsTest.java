package cs446.homework2;

import weka.core.FastVector;
import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import cs446.weka.classifiers.trees.Id3;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.util.Random;
import cs446.homework2.DecisionStumps;


public class StumpsTest{

	public static void main(String[] args) throws Exception {

	if (args.length != 1) {
	    System.err.println("Usage: WekaTesterSGD train-arff-file test-arrf-file test-blind-file");
	    System.exit(-1);
	}

	// Load the data
	Instances data = new Instances(new FileReader(new File(args[0])));

	// The last attribute is the class label
	data.setClassIndex(data.numAttributes() - 1);
	
	DecisionStumps stumps = new DecisionStumps();
	stumps.buildClassifier(data);

	Instances features = stumps.CreateNewFeature(data);
	features.setClassIndex(features.numAttributes()-1);

	double totalCorrect = 0;
	double totalPossible = 0;
	for(int i = 1; i <= 5; i++) {
	    // Train on 80% of the data and test on 20%
	    Instances train = new Instances(new FileReader(new File("/Users/Victor/Desktop/CS446/hw2/Data/trainToPredict_" + Integer.toString(i) +".arff")));
	    Instances test = new Instances(new FileReader(new File("/Users/Victor/Desktop/CS446/hw2/Data/testToPredict_" + Integer.toString(i) +".arff")));
	    train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes() - 1);
	    SGD classifier = new SGD();
	    classifier.buildClassifier(train);

	    Evaluation evaluation = new Evaluation(test);
	    evaluation.evaluateModel(classifier, test);
	    System.out.println(evaluation.toSummaryString());
	    totalCorrect += evaluation.correct();
	    totalPossible += evaluation.correct() + evaluation.incorrect();
	}

	System.out.println("Average 5 cross validation stumps:");
	System.out.println(totalCorrect / totalPossible * 100 + " %");

	

	}
}