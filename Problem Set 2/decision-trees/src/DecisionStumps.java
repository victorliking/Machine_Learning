package cs446.homework2;

import weka.core.Attribute;
import weka.classifiers.*;
import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import cs446.weka.classifiers.trees.Id3;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.util.Random;
import weka.core.Instance;
import weka.core.FastVector;
public class DecisionStumps extends Classifier{

	private boolean trained = false;
	private Id3 stumps[];
	Random randnum;

	public DecisionStumps() {
		stumps = new Id3[100];

	}
	
	@Override
	public void buildClassifier(Instances data) throws Exception {
		
		for(int i=0; i<100; i++){
			data.randomize(new java.util.Random());
			int size = (int) Math.round(data.numInstances() *0.5);
			Instances new_data = new Instances(data, 0, size);
				
			stumps[i] = new Id3(); 
			stumps[i].setMaxDepth(4);
			stumps[i].buildClassifier(new_data);

		}
	trained = true;	
	}

	public Instances CreateNewFeature(Instances data) throws Exception{
		
		FastVector zeroOne = new FastVector(2);
		FastVector labels = new FastVector(2);
		zeroOne.addElement("1");
		zeroOne.addElement("0");

		labels.addElement("-1");
		labels.addElement("1");
			
		FastVector attributes = new FastVector();
		for(int i=0; i<100; i++){
			Attribute attr = new Attribute("stump "+i, zeroOne);
			attributes.addElement(attr);
		}
		Attribute classLabel = new Attribute("Class", labels);
		attributes.addElement(classLabel);

		Instances stumpData = new Instances("Decison Stumps", attributes, data.numInstances());
		
		stumpData.setClass(classLabel);

		int numInst = data.numInstances();
		for(int i=0; i<numInst; i++) {
			Instance cur = new Instance(stumpData.numAttributes());
			cur.setDataset(stumpData);
			
			for(int j=0; j<100; j++) {
				double attrVal = stumps[j].classifyInstance(data.instance(i));
				cur.setValue(j, attrVal);
			}
			//String label;
			if(data.instance(i).classValue() == 0.0)
				cur.setClassValue("-1");
			else
				cur.setClassValue("1");
			
			stumpData.add(cur);

		}
	return stumpData;
	}

}
