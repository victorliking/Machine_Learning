package cs446.homework2;

import weka.classifiers.*;
import weka.core.Instances;
import weka.core.Instance;
import java.lang.*;
import java.util.*;
import java.lang.Boolean;
import java.lang.Exception;
import java.lang.Math;
import java.util.Arrays;
import java.util.Vector;



public class SGD extends Classifier{
	private double weight = 0.0;
	private boolean trained  =false;
	private ArrayList<Double> w;
	//private numAttr;

	

	public double cal_error(Instances insts){
		double error  =0.0;
		int numIns = insts.numInstances();
		for(int i =0;i<numIns;i++){
			Instance ins  = insts.instance(i);
			double target = ins.classValue()==0.0?-1.0:1.0;
			error += Math.pow((target- prediction(ins)),2);
		}
		return error/2;
	}

	public double cal(ArrayList<Double> a,ArrayList<Double> b){
		double ret = 0.0;
		for(int i=0;i<a.size();i++){
			ret+= a.get(i)*b.get(i);
		}
		return ret;
	}

	public ArrayList<Double> Copy_to_ArrayList(double[] temp){
		ArrayList<Double> ret = new ArrayList<Double>(temp.length);
		for(int i =0;i<temp.length;i++){
			ret.add(temp[i]);
		}
		return ret;
	}

	public double prediction(Instance ins){
		ArrayList<Double> x = Copy_to_ArrayList(ins.toDoubleArray()); 
		x.remove(w.size());
		return cal(w,x) + weight;
	}

	public void printList(){
		System.out.println("[");
		for (int i =0;i<this.w.size()-1 ;i++ ) {
			System.out.format("%.3f%n",this.w.get(i));
			//System.out.print(",");
		}
		System.out.println(this.w.get(this.w.size()-1) + "]");
	}

	@Override
	public void buildClassifier(Instances arg0) throws Exception{
		//System.out.println("error is " + cal_error(arg0));
		//System.out.println();
		double R = 0.001;
		int numAttr = arg0.numAttributes()-1;
		w = new ArrayList<Double>(numAttr);
		for (int i =0;i<numAttr ;i++ ) {
			w.add(0.0);
		}

		for(int n = 100;n>0;n--){
			//arg0.randomize(new java.util.Random());
			for (int i =0;i<arg0.numInstances() ;i++ ) {
				Instance ins = arg0.instance(i);
				double temp[] = ins.toDoubleArray();
				double target_value = ins.classValue()== 0.0 ? -1.0 : 1.0;
				double difference = prediction(ins)-target_value;
				for (int j =0;j<arg0.numAttributes()-1 ;j++ ) {
					//double attrValue = ins.value(i);
					w.set(j,w.get(j) - R* difference*temp[j]);

				}
				//weight = weight - R*difference;
				//if(error>50)
				//break;
			}
		}
		trained = true;
	}

	@Override
	public double classifyInstance(Instance instance) throws java.lang.Exception {
		if(!trained){
			throw new Exception("The classifier is not trained!");
		}
		return prediction(instance) >= 0 ? 1.0 : 0.0;
		
	}
}