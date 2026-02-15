package Aleeza_08_2024;

// Created by Aleeza Bhatti & Yousra Esseddiqi, July 2024 
// Updated by Taesik Kim


import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.DenseInstance;
import weka.core.FastVector;


public class swotAI_RealTime {  // real time training


	public static NominalPrediction np;
	public static J48 tree;
	public static Instances data;
	public static String[] key = {"S","W","O","T"};
	public static String[] attribute= {"Marketing_Condition","Financial_Performance","Customer_Feedback","Industry_Competition",
	 "Product_Service_Quality","Consumer_Behavior","Expansion_Ability","Uncontrollable_Factors","Market_Saturation",
     "Marketing_Strategies"};
	
	
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;
 
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
 
		return inputReader;
	}
 
	public static Evaluation classify(Classifier model, Instances trainingSet, Instances testingSet) throws Exception {
	

		Evaluation evaluation = new Evaluation(trainingSet);	
				
		model.buildClassifier(trainingSet);		
		
		evaluation.evaluateModel(model, testingSet);

		return evaluation;
	}
 
	public static double calculateAccuracy(FastVector predictions) {
		// public static double calculateAccuracy(FastVector predictions) {
			
		double correct = 0;
		
		System.out.println("Key: Predicted");
		System.out.println("-----------------");
		
		for (int i = 0; i < predictions.size(); i++) {
			np = (NominalPrediction) predictions.elementAt(i);
			//np = (NominalPrediction) predictions.elementAt(i);
			
			//System.out.print(predictions.get(i));
			System.out.print(key[(int)np.actual()] + "  : " + key[(int)np.predicted()] );
		//	System.out.print("  predicted : " + np.predicted() + " Key : "+np.actual());
			
			
			if (np.predicted() == np.actual()) {
				correct++;
				System.out.println();
			}
			else {
				System.out.print(" ***\n");
			}
		}
 
		System.out.println();
		System.out.println(correct + "/"+predictions.size());
		return 100 * correct / predictions.size();
	}
 
	
	public static void predictionSingleData() throws Exception {
		
		
		// ask the user to input	
		double[] userAttribute = new double[10];
		
		userAttribute[0]=0.62;  // Marketing_Condition
		userAttribute[1]=0.2;   // Financial_Performance
		userAttribute[2]=0.45;  // Customer_Feedback
		userAttribute[3]=0.15;  // Industry_Competition
		userAttribute[4]=0.62;  // Product_Service_Quality
		userAttribute[5]=0.22;  // Consumer_Behavior
		userAttribute[6]=0.57;  // Expansion_Ability
		userAttribute[7]=0.5;	// Uncontrollable_Factors
		userAttribute[8]=0.55;  // Market_Saturation
		userAttribute[9]=0.27;  // Marketing_Strategies
		
		DenseInstance instance = new DenseInstance(1.0, userAttribute);
		
		instance.setDataset(data);
		//instance.setDataset(normalizedData);
		

		// Make prediction
		double prediction = tree.classifyInstance(instance); 
		
		System.out.println("\nUser Input");
		for (int i= 0; i<attribute.length; i++) {
			System.out.printf("%-25s %6.2f \n" ,attribute[i],userAttribute[i] );
		}

		System.out.println("\nPrediction: " + key[(int)prediction]);
	}
	
	public static void main(String[] args) throws Exception {
		
		BufferedReader datafile = readDataFile("C:/Program Files/Weka-3-8-6/data/swot.arff");
	
		data = new Instances(datafile);
		//Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
 

                data.randomize(new java.util.Random(1));
                int trainSize = (int) Math.round(data.numInstances() * 0.66);
                System.out.println("Training size : "+trainSize);
                int testSize = data.numInstances() - trainSize;
                System.out.println("Test size : "+testSize);
                Instances train = new Instances(data, 0, trainSize);
                Instances test = new Instances(data, trainSize, testSize);


               tree = new J48();
              //  J48 tree = new J48();
                
                
               FastVector predictions = new FastVector();
               
                String[] options;

                options = weka.core.Utils.splitOptions("-C 0.25 -M 2");
                tree.setOptions(options);
			
                Evaluation validation = classify(tree, train, test);

                predictions.appendElements(validation.predictions());
 
                // Calculate overall accuracy of current classifier on all splits
                double accuracy = calculateAccuracy(predictions);

                // Print current classifier's name and accuracy in a complicated,
                // but nice-looking way.
                System.out.println("Accuracy of " + tree.getClass().getSimpleName() + ": "
                                + String.format("%.2f%%", accuracy)
                                + "\n---------------------------------");
 
                predictionSingleData();
                
	}
}


