package ytu.ml.pca;

import java.io.File;
import java.util.ArrayList;
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
	public static final int NUMBER_OF_CLASSES = 10;
	public static final int NUMBER_OF_TRAIN_VECTORS = 5;
	public static final int NUMBER_OF_PIXEL_VALUES = 64;
	
	
	
	private Map<Integer,List<SampleObject>> trainSamples;
	private List<SampleObject> validationSamples;
	private List<SampleObject> allSamples;
	private List<Integer> trainImageValueList;
	private List<Integer> testImageValueList;
	private List<Double> trainMeanVector;
	private Map<Integer,List<List<Double>>> meanCenteredVectorMap;
	private RealMatrix covarianceMatrix;
	private EigenDecomposition eigen;
	private double threshold;
	private int numberOfNewFeatures;
	private RealMatrix trainEigenSpace;
	private RealMatrix testEigenSpace;
	private RealMatrix meanCenteredTrainMatrix;
	
	public DataTable() {
		trainSamples = new HashMap<>();
		validationSamples = new ArrayList<>();
		allSamples = new ArrayList<>();
		trainMeanVector = new ArrayList<>();
		meanCenteredVectorMap = new HashMap<>();
		trainImageValueList = new ArrayList<>();
		testImageValueList = new ArrayList<>();
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

	public RealMatrix getCovarianceMatrix() {
		return covarianceMatrix;
	}

	public void setCovarianceMatrix(RealMatrix covarianceMatrix) {
		this.covarianceMatrix = covarianceMatrix;
	}

	public RealMatrix getTrainEigenSpace() {
		return trainEigenSpace;
	}

	public void setTrainEigenSpace(RealMatrix trainEigenSpace) {
		this.trainEigenSpace = trainEigenSpace;
	}

	public RealMatrix getTestEigenSpace() {
		return testEigenSpace;
	}

	public void setTestEigenSpace(RealMatrix testEigenSpace) {
		this.testEigenSpace = testEigenSpace;
	}

	public RealMatrix getMeanCenteredTrainMatrix() {
		return meanCenteredTrainMatrix;
	}

	public void setMeanCenteredTrainMatrix(RealMatrix meanCenteredTrainMatrix) {
		this.meanCenteredTrainMatrix = meanCenteredTrainMatrix;
	}

	public Map<Integer, List<List<Double>>> getMeanCenteredVectorMap() {
		return meanCenteredVectorMap;
	}

	public void setMeanCenteredVectorMap(Map<Integer, List<List<Double>>> meanCenteredVectorMap) {
		this.meanCenteredVectorMap = meanCenteredVectorMap;
	}

	public List<Double> getTrainMeanVector() {
		return trainMeanVector;
	}

	public void setTrainMeanVector(List<Double> trainMeanVector) {
		this.trainMeanVector = trainMeanVector;
	}

	public List<SampleObject> getAllSamples() {
		return allSamples;
	}

	public void setAllSamples(List<SampleObject> allSamples) {
		this.allSamples = allSamples;
	}

	public Map<Integer, List<SampleObject>> getTrainSamples() {
		return trainSamples;
	}

	public void setTrainSamples(Map<Integer, List<SampleObject>> trainSamples) {
		this.trainSamples = trainSamples;
	}

	public List<SampleObject> getValidationSamples() {
		return validationSamples;
	}

	public void setValidationSamples(List<SampleObject> validationSamples) {
		this.validationSamples = validationSamples;
	}

	@Override
	public String toString() {
		
		StringBuilder sb = new StringBuilder();
		
		for (SampleObject sample : allSamples) {
			
			for (Double s : sample.getSampleValues()) {
				System.out.print(s+",");
			}
			System.out.print(sample.getClassifierNumberStr() + "\n");
		}
		
		return sb.toString();
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
					allSamples.add(sample);
					
				} catch (Exception e) {
					logger.info("Erroc occured while parsing file: " + line + "\n" + e.toString() +  "///// " + e.getMessage());
				}
			}
			
			sampleId++;
		}
	}

	private void addSampleIntoAppropriateSet(SampleObject sample) {
		
		// If this is the first sample of the class
		if(trainSamples.get(sample.getClassifierNumber()) == null){
			List<SampleObject> samples = new ArrayList<>();
			samples.add(sample);
			trainSamples.put(sample.getClassifierNumber(),samples);
			// to compare real iamges with test images, I store classifiers in list
			trainImageValueList.add(sample.getClassifierNumber());
		
		}
		// If there are not 5 examples yet in train set
		else if(trainSamples.get(sample.getClassifierNumber()).size()< NUMBER_OF_TRAIN_VECTORS){
			trainSamples.get(sample.getClassifierNumber()).add(sample);
			// to compare real iamges with test images, I store classifiers in list
			trainImageValueList.add(sample.getClassifierNumber());
		}
		// add into train set
		else{
			validationSamples.add(sample);
			// to compare real iamges with test images, I store classifiers in list
			testImageValueList.add(sample.getClassifierNumber());
		}
	}
	
	
}
