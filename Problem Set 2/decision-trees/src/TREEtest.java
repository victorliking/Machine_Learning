package cs446.homework2;

import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import cs446.weka.classifiers.trees.Id3;
import cs446.homework2.SGD;
import weka.core.Instance;
import weka.classifiers.*;
import java.lang.*;
import java.util.*;

public class TREEtest{
	public static void main(String[] args) throws Exception {

	if (args.length != 1) {
	    System.err.println("Usage: WekaTesterSGD train-arff-file test-arrf-file test-blind-file");
	    System.exit(-1);
	}

	// Load the data
	Instances data = new Instances(new FileReader(new File(args[0])));

	// The last attribute is the class label
	data.setClassIndex(data.numAttributes() - 1);
	
	double totalCorrect = 0;
	double totalPossible = 0;
	//Instances train;
	
	for(int i = 1; i <= 5; i++) {
	    // Train on 80% of the data and test on 20%
	  
	    //if(i<4) 
	    Instances train = new Instances(new FileReader(new File("/Users/Victor/Desktop/CS446/hw2/Data/trainToPredict_" + Integer.toString(i) +".arff")));
	    Instances test = new Instances(new FileReader(new File("/Users/Victor/Desktop/CS446/hw2/Data/testToPredict_" + Integer.toString(i) +".arff")));
	    train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes() - 1);

	    //else 
	    	//train = data.trainCV(5,0);
	    //Instances test = data.testCV(5, i);
	    Id3 classifier = new Id3();
	    classifier.setMaxDepth(-1);
	    classifier.buildClassifier(train);
	    System.out.println(classifier);
	    System.out.println();
	    Evaluation evaluation = new Evaluation(test);
	    evaluation.evaluateModel(classifier, test);
	    System.out.println(evaluation.toSummaryString());
	    totalCorrect += evaluation.correct();
	    totalPossible += evaluation.correct() + evaluation.incorrect();
		
	//}
	}
	System.out.println("Average percentage across five-fold cross validation for depth -1:");
	System.out.println(totalCorrect / totalPossible * 100 + " %");
	
	
	totalCorrect = 0;
	totalPossible = 0;
	for(int i = 1; i <= 5; i++) {
	    Instances train = new Instances(new FileReader(new File("/Users/Victor/Desktop/CS446/hw2/Data/trainToPredict_" + Integer.toString(i) +".arff")));
	    Instances test = new Instances(new FileReader(new File("/Users/Victor/Desktop/CS446/hw2/Data/testToPredict_" + Integer.toString(i) +".arff")));
	    train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes() - 1);

	    //else 
	    	//train = data.trainCV(5,0);
	    //Instances test = data.testCV(5, i);
	    Id3 classifier = new Id3();
	    classifier.setMaxDepth(4);
	    classifier.buildClassifier(train);
	    System.out.println(classifier);
	    System.out.println();
	    Evaluation evaluation = new Evaluation(test);
	    evaluation.evaluateModel(classifier, test);
	    System.out.println(evaluation.toSummaryString());
	    System.out.println();
	    totalCorrect += evaluation.correct();
	    totalPossible += evaluation.correct() + evaluation.incorrect();
		
	}
	System.out.println("Average percentage across five-fold cross validation for depth 4:");
	System.out.println(totalCorrect / totalPossible * 100 + " %");

	
	totalCorrect = 0;
	totalPossible = 0;
	//Instances train;
	for(int i = 1; i <= 5; i++) {
	    // Train on 80% of the data and test on 20%
	    
	   
	    //if(i<4) 
	    Instances train = new Instances(new FileReader(new File("/Users/Victor/Desktop/CS446/hw2/Data/trainToPredict_" + Integer.toString(i) +".arff")));
	    Instances test = new Instances(new FileReader(new File("/Users/Victor/Desktop/CS446/hw2/Data/testToPredict_" + Integer.toString(i) +".arff")));
	    train.setClassIndex(train.numAttributes() - 1);
        test.setClassIndex(test.numAttributes() - 1);

	    //else 
	    	//train = data.trainCV(5,0);
	    //Instances test = data.testCV(5, i);
	    Id3 classifier = new Id3();
	    classifier.setMaxDepth(8);
	    classifier.buildClassifier(train);
	    System.out.println(classifier);
	    System.out.println();
	    Evaluation evaluation = new Evaluation(test);
	    evaluation.evaluateModel(classifier, test);
	    System.out.println(evaluation.toSummaryString());
	    System.out.println();	
	    totalCorrect += evaluation.correct();
	    totalPossible += evaluation.correct() + evaluation.incorrect();
		
	}
	System.out.println("Average percentage across five-fold cross validation for depth 8:");
	System.out.println(totalCorrect / totalPossible * 100 + " %");


	

}


}

/*	
	totalCorrect = 0; 
	totalPossible = 0;
	for(int i = 0; i < 5; i++) {
	    // Train on 80% of the data and test on 20%
	    Instances train = data.trainCV(5,i);
	    Instances test = data.testCV(5, i);
	    Id3 classifier = new Id3();
	    classifier.setMaxDepth(4);
	    classifier.buildClassifier(train);

            //System.out.println(classifier);
            //System.out.println();

	    Evaluation evaluation = new Evaluation(test);
	    evaluation.evaluateModel(classifier, test);
	    //System.out.println(evaluation.toSummaryString());
	    totalCorrect += evaluation.correct();
	    totalPossible += evaluation.correct() + evaluation.incorrect();
	}

	System.out.println("Average percentage across five-fold cross validation:");
	System.out.println(totalCorrect / totalPossible * 100 + " %");

	totalCorrect = 0; 
	totalPossible = 0;
	for(int i = 0; i < 5; i++) {
	    // Train on 80% of the data and test on 20%
	    Instances train = data.trainCV(5,i);
	    Instances test = data.testCV(5, i);
	    Id3 classifier = new Id3();
	    classifier.setMaxDepth(8);
	    classifier.buildClassifier(train);

            //System.out.println(classifier);
            //System.out.println();

	    Evaluation evaluation = new Evaluation(test);
	    evaluation.evaluateModel(classifier, test);
	    //System.out.println(evaluation.toSummaryString());
	    totalCorrect += evaluation.correct();
	    totalPossible += evaluation.correct() + evaluation.incorrect();
	}

	System.out.println("Average percentage across five-fold cross validation:");
	System.out.println(totalCorrect / totalPossible * 100 + " %");
*/
	/*for(int i =0;i<5;i++){
		SGD classifier = new SGD();
		Instances train = data.trainCV(5,i);
		Instances test  = data.testCV(5,i);

		//train
		classifier.buildClassifier(train);
		System.out.println(classifier);
		System.out.println();

		//evaluate
		Evaluation evaluation = new Evaluation(test);
		evaluation.evaluateModel(classifier, test);
		System.out.println(evaluation.toSummaryString());

		totalCorrect += evaluation.correct();
		totalCase += evaluation.correct()+evaluation.incorrect();
		System.out.println("file "+i +"accuracy:=" + evaluation.correct()/(evaluation.correct()+evaluation.incorrect()));
		System.out.println();
	}

	System.out.println("Avg accuracy:");
	System.out.println(totalCorrect / totalCase * 100 + " %");
	}	*/

