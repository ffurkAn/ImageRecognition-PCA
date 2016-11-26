package ytu.ml.pca;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.Logger;

import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.RealMatrix;

import user.furkan.util.ListUtils;
import user.furkan.util.StringUtils;

/**
 * Class for responsible for holding the samples separated for train and valitation.
 * @author furkan
 *
 */
public class DataTable {

	Logger logger = Logger.getGlobal();
	
	private double[][] trainSamples;
	private double[][] testSamples;
	
	private List<Integer> trainImageValueList;
	private List<Integer> testImageValueList;
	
	private double[] trainMeanVector;
	
	private double[][] covarianceMatrix;
	private EigenDecomposition eigen;
	private double threshold;
	private int numberOfNewFeatures;
	
	private double[][] trainEigenSpace;
	private double[][] testEigenSpace;
	private double[][] meanCenteredTrainMatrix;
	
	private Map<Integer,List<SampleObject>> classifierCounterMap;
	
	private int trainVectorCounter = 0;
	private int testVectorCounter = 0;
	
	public DataTable() {
		trainSamples = new double[Constants.NUMBER_OF_TRAIN_VECTORS][Constants.NUMBER_OF_PIXEL_VALUES];
		testSamples = new double[Constants.NUMBER_OF_TEST_VECTORS][Constants.NUMBER_OF_PIXEL_VALUES];
		trainMeanVector = new double[Constants.NUMBER_OF_PIXEL_VALUES];
		trainImageValueList = new ArrayList<>();
		testImageValueList = new ArrayList<>();
		classifierCounterMap = new HashMap<>();
	}
	
	

	public double[] getTrainMeanVector() {
		return trainMeanVector;
	}



	public void setTrainMeanVector(double[] trainMeanVector) {
		this.trainMeanVector = trainMeanVector;
	}



	public void setTrainSamples(double[][] trainSamples) {
		this.trainSamples = trainSamples;
	}


	public double[][] getTestSamples() {
		return testSamples;
	}

	public void setTestSamples(double[][] testSamples) {
		this.testSamples = testSamples;
	}

	public Map<Integer, List<SampleObject>> getClassifierCounterMap() {
		return classifierCounterMap;
	}

	public void setClassifierCounterMap(Map<Integer, List<SampleObject>> classifierCounterMap) {
		this.classifierCounterMap = classifierCounterMap;
	}

	public double[][] getTrainSamples() {
		return trainSamples;
	}

	public EigenDecomposition getEigen() {
		return eigen;
	}

	public void setEigen(EigenDecomposition eigen) {
		this.eigen = eigen;
	}

	public List<Integer> getTrainImageValueList() {
		return trainImageValueList;
	}

	public void setTrainImageValueList(List<Integer> trainImageValueList) {
		this.trainImageValueList = trainImageValueList;
	}

	public List<Integer> getTestImageValueList() {
		return testImageValueList;
	}

	public void setTestImageValueList(List<Integer> testImageValueList) {
		this.testImageValueList = testImageValueList;
	}

	public int getNumberOfNewFeatures() {
		return numberOfNewFeatures;
	}

	public void setNumberOfNewFeatures(int numberOfNewFeatures) {
		this.numberOfNewFeatures = numberOfNewFeatures;
	}

	public double getThreshold() {
		return threshold;
	}

	public void setThreshold(double threshold) {
		this.threshold = threshold;
	}


	public double[][] getCovarianceMatrix() {
		return covarianceMatrix;
	}



	public void setCovarianceMatrix(double[][] covarianceMatrix) {
		this.covarianceMatrix = covarianceMatrix;
	}



	public double[][] getTrainEigenSpace() {
		return trainEigenSpace;
	}



	public void setTrainEigenSpace(double[][] trainEigenSpace) {
		this.trainEigenSpace = trainEigenSpace;
	}



	public double[][] getTestEigenSpace() {
		return testEigenSpace;
	}



	public void setTestEigenSpace(double[][] testEigenSpace) {
		this.testEigenSpace = testEigenSpace;
	}



	public double[][] getMeanCenteredTrainMatrix() {
		return meanCenteredTrainMatrix;
	}



	public void setMeanCenteredTrainMatrix(double[][] meanCenteredTrainMatrix) {
		this.meanCenteredTrainMatrix = meanCenteredTrainMatrix;
	}



	@Override
	public String toString() {
		return "DataTable [logger=" + logger + ", trainSamples=" + Arrays.toString(trainSamples) + ", testSamples="
				+ Arrays.toString(testSamples) + ", trainImageValueList=" + trainImageValueList
				+ ", testImageValueList=" + testImageValueList + ", trainMeanVector=" + Arrays.toString(trainMeanVector)
				+ ", covarianceMatrix=" + Arrays.toString(covarianceMatrix) + ", eigen=" + eigen + ", threshold="
				+ threshold + ", numberOfNewFeatures=" + numberOfNewFeatures + ", trainEigenSpace="
				+ Arrays.toString(trainEigenSpace) + ", testEigenSpace=" + Arrays.toString(testEigenSpace)
				+ ", meanCenteredTrainMatrix=" + Arrays.toString(meanCenteredTrainMatrix) + ", classifierCounterMap="
				+ classifierCounterMap + "]";
	}

	public void loadDatFileToDataTable(String fileName) throws Exception{
		
		int sampleId = 0;
		
		Scanner s = new Scanner(new File(fileName));

		logger.info("Started to parsing file..");
		while (s.hasNext()) {

			String line = s.nextLine().trim();

			// if it is not a empty line
			// StringUtils is a library defined by user @furkan
			if(!StringUtils.isEmpty(line)){
				
				try {
					SampleObject sample = new SampleObject();
					sample.setLineId(sampleId);
					sample.setSampleValues(new ArrayList<>());
					
					Scanner valueScanner = new Scanner(line);
					valueScanner.useDelimiter(","); // separate values by using ','
					
					while(valueScanner.hasNext()){
						
						String pixelValue = valueScanner.next().trim();
						
						if(!StringUtils.isEmpty(pixelValue)){
							
							sample.getSampleValues().add(Double.parseDouble(pixelValue));
						}
					}
					
					// remove the classifier from sample
					Double classLabel = sample.getSampleValues().remove(sample.getSampleValues().size()-1);
					sample.setClassifierNumberStr(classLabel.toString());
					sample.setClassifierNumber((int)classLabel.doubleValue());
					
					// add sample into appropriate set
					addSampleIntoAppropriateSet(sample);
					
				} catch (Exception e) {
					logger.info("Erroc occured while parsing file: " + line + "\n" + e.toString() +  "///// " + e.getMessage());
				}
			}
			
			sampleId++;
		}
	}

	private void addSampleIntoAppropriateSet(SampleObject sample) {
		
		// If this is the first sample of the class
		if(classifierCounterMap.get(sample.getClassifierNumber()) == null){
			
			for(int x = 0 ; x < sample.getSampleValues().size() ; x++){
				trainSamples[trainVectorCounter][x] = sample.getSampleValues().get(x).doubleValue();
			}
			
			trainVectorCounter++;
			
			List<SampleObject> list = new ArrayList<>();
			list.add(sample);
			
			// first value added
			classifierCounterMap.put(sample.getClassifierNumber(),list);
			
			// to compare real iamges with test images, I store classifiers in list
			trainImageValueList.add(sample.getClassifierNumber());
		
		}
		// If there are not 5 examples yet in train set
		else if(classifierCounterMap.get(sample.getClassifierNumber()).size() < Constants.NUMBER_OF_VECTORS_USED_IN_TRAINING_FOR_EACH_CLASS){
			
			for(int x = 0 ; x < sample.getSampleValues().size() ; x++){
				trainSamples[trainVectorCounter][x] = sample.getSampleValues().get(x).doubleValue();
			}
			
			trainVectorCounter++;
			classifierCounterMap.get(sample.getClassifierNumber()).add(sample);
			
			// to compare real iamges with test images, I store classifiers in list
			trainImageValueList.add(sample.getClassifierNumber());
		}
		// add into train set
		else{

			for(int x = 0 ; x < sample.getSampleValues().size() ; x++){
				testSamples[testVectorCounter][x] = sample.getSampleValues().get(x).doubleValue();
			}
			
			testVectorCounter++;
			// to compare real iamges with test images, I store classifiers in list
			testImageValueList.add(sample.getClassifierNumber());
		}
	}
	
	
}
