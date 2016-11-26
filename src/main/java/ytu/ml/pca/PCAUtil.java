package ytu.ml.pca;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.logging.Logger;

import javax.xml.crypto.Data;

import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.correlation.Covariance;
import org.apache.commons.math3.stat.descriptive.summary.Sum;
import org.apache.commons.math3.util.MathUtils;
import org.omg.CORBA.TRANSACTION_MODE;

/**
 * Class responsible for all the calcualtion needed for PCA
 * @author furkan
 *
 */
public class PCAUtil {

	public static Logger logger = Logger.getGlobal();

	/**
	 * calculates the mean vector of each train sample set
	 * 
	 * i.e, 5 sample line for classifier '0'
	 * @param trainSamples
	 * @return
	 */
	public static List<Double> calculateMeanVectors(Map<Integer, List<SampleObject>> trainSamples) {

		List<Double> returnMap = new ArrayList<>();

		// for every pixel index
		for(int x = 0 ; x < 64 ; x++){

			double total = 0.0;
			// iterates the train samples for each classifier
			for (Entry<Integer,List<SampleObject>> entry : trainSamples.entrySet()) {

				for (SampleObject sample : entry.getValue()) {

					double value = sample.getSampleValues().get(x);
					total += value;
				}
			}

			double mean = total / 50;
			returnMap.add(mean);
		}

		return returnMap;
	}

	/**
	 * subtract mean vector values from every train values
	 * @param trainSamples
	 * @param meanVector
	 * @return
	 */
	public static Map<Integer,List<List<Double>>> calculateSubtractVectors(Map<Integer, List<SampleObject>> trainSamples, List<Double> meanVector){

		Map<Integer,List<List<Double>>> returnMap = new HashMap<>();

		// Iterates the sample set for each classifier
		for (Entry<Integer,List<SampleObject>> entry : trainSamples.entrySet()) {

			List<List<Double>> listOfMeanCenteredSamples = new ArrayList<>();

			// Iterates the samples and calculate the meanCentered values for each pixel value
			for (SampleObject sample : entry.getValue()) {

				List<Double> meanCenteredValueList = new ArrayList<>();

				for (int i = 0; i < sample.getSampleValues().size(); i++) {

					double meanValue = meanVector.get(i);
					double trainValue = sample.getSampleValues().get(i);

					double meanCenteredValue = trainValue - meanValue;
					meanCenteredValueList.add(meanCenteredValue);
				}

				listOfMeanCenteredSamples.add(meanCenteredValueList);
			}

			returnMap.put(entry.getKey(), listOfMeanCenteredSamples);
		}
		return returnMap;

	}


	/**
	 * Creates the matrix which includes double values for the pixels and
	 * @param meanCenteredVectors
	 * @return covariance matrix of the created matrix
	 */
	public static RealMatrix findCovarianceMatrix(Map<Integer,List<List<Double>>> meanCenteredVectors){

		// intialize 64x50 matrix
		double[][] meanCenteredMatrix = getCenteredTrainMatrix(meanCenteredVectors);
		return covariance(meanCenteredMatrix);

	}

	private static double[][] getCenteredTrainMatrix(Map<Integer, List<List<Double>>> meanCenteredVectors) {
		double[][] meanCenteredMatrix = new double[DataTable.NUMBER_OF_PIXEL_VALUES][DataTable.NUMBER_OF_CLASSES*DataTable.NUMBER_OF_TRAIN_VECTORS];

		int column = 0;
		for (Entry<Integer,List<List<Double>>> entry : meanCenteredVectors.entrySet()) {
			for (List<Double> meanCenteredVector: entry.getValue()) {

				int row = 0;
				for (Double pixelValue : meanCenteredVector) {

					meanCenteredMatrix[row][column] = pixelValue;
					row++;
				}
				column++;
			}
		}
		return meanCenteredMatrix;
	}

	/**
	 *  creates covariance matrix of matrix which is formed by mean centered vectors
	 * @param returnMatrix
	 * @return
	 */
	private static RealMatrix covariance(double[][] meanCenteredMatrix) {
		RealMatrix mx = MatrixUtils.createRealMatrix(meanCenteredMatrix);
		RealMatrix covarianceMatrix = new Covariance(mx.transpose()).getCovarianceMatrix();
		return covarianceMatrix;
	}

	public static RealMatrix createTrainEigenSpace(DataTable dataTable) {

		RealMatrix covarianceMatrix = dataTable.getCovarianceMatrix();

		EigenDecomposition eigen = new EigenDecomposition(covarianceMatrix);
		dataTable.setEigen(eigen);
		double[] eigenValues = eigen.getRealEigenvalues(); // it gives the eigenvalues sorted

		double sumOfEigenValues = sumOfValues(eigenValues);
		double total = 0.0;
		int i = 0; // last index of the eligible eigen value

		while(total/sumOfEigenValues < dataTable.getThreshold()){
			total = total + eigenValues[i];
			i++;
		}

		dataTable.setNumberOfNewFeatures(i-1);

		// creating eigenspace by forming eigenvectors as matrix
		// rows as feature, columns as value
		logger.info("");

		double[][] eigenvectorMatrix = getEigenVectorMatrix(dataTable);
		RealMatrix eigenspaceVectorMatrix = MatrixUtils.createRealMatrix(eigenvectorMatrix); // eigen vector matrix - feature sayýsý X eigen value sayýsý (64)
		RealMatrix meanCenteredTrainMatrix = MatrixUtils.createRealMatrix(getCenteredTrainMatrix(dataTable.getMeanCenteredVectorMap())); // mean centered train matrix - pixel value X train data sayýsý

		RealMatrix trainEigenSpace = eigenspaceVectorMatrix.multiply(meanCenteredTrainMatrix);

		//satýrlarý özellik dizisi yapmak için transpoze aldýk
		return trainEigenSpace.transpose();

	}

	private static double[][] getEigenVectorMatrix(DataTable dataTable) {

		EigenDecomposition eigen = dataTable.getEigen();
		int eigenVectorLength = eigen.getRealEigenvalues().length;

		double[][] eigenvectorMatrix = new double[dataTable.getNumberOfNewFeatures()][eigenVectorLength];

		for(int featureCounter = 0 ; featureCounter < dataTable.getNumberOfNewFeatures() ; featureCounter++){

			RealVector vector = eigen.getEigenvector(featureCounter); 
			for (int columnCounter = 0; columnCounter < eigenVectorLength; columnCounter++) {
				eigenvectorMatrix[featureCounter][columnCounter] = vector.getEntry(columnCounter);
			}
		}
		return eigenvectorMatrix;
	}

	public static RealMatrix createTestEigenSpace(DataTable dataTable) {

		// feature sayýsý X eigen value sayýsý
		double[][] eigenvectorMatrix = getEigenVectorMatrix(dataTable);

		// test data sayýsý X pixel sayýsý
		double[][] meanCenteredTestMatrix = new double[dataTable.getValidationSamples().size()][dataTable.NUMBER_OF_PIXEL_VALUES]; 

		int testSampleCounter = 0;
		for (SampleObject validationSample : dataTable.getValidationSamples()) {

			meanCenteredTestMatrix[testSampleCounter] = getSubtractedVector(validationSample, dataTable.getTrainMeanVector()); 

		}

		RealMatrix eigenSpaceVectorMatrix = MatrixUtils.createRealMatrix(eigenvectorMatrix); // feat
		RealMatrix meanCenteredMatrix = MatrixUtils.createRealMatrix(meanCenteredTestMatrix); // meanCenteredTestMatrix

		RealMatrix testEigenSpace = eigenSpaceVectorMatrix.multiply(meanCenteredMatrix.transpose());

		//satýrlarý özellik dizisi yapmak için transpoze aldýks
		return testEigenSpace.transpose();
	}

	private static double[] getSubtractedVector(SampleObject sampleObject, List<Double> meanVector) {

		double[] subtractedVector = new double[sampleObject.getSampleValues().size()];

		for (int valueCounter = 0; valueCounter < sampleObject.getSampleValues().size(); valueCounter++) {
			subtractedVector[valueCounter] = sampleObject.getSampleValues().get(valueCounter) - meanVector.get(valueCounter);
		}

		return subtractedVector;
	}

	private static double sumOfValues(double[] eigenValues) {
		double result = 0;
		for (double value:eigenValues)
			result += value;
		return result;
	}

	public static void test(RealMatrix trainEigenSpace, List<Integer> list, RealMatrix testEigenSpace, List<Integer> list2) {

		// test datasýný iterate ediyoruz, her özellik için traindeki 50 veriye ait 3 özellik ile distance alýorz
		
		double[][] trainMatrix = trainEigenSpace.getData();
		double[][] testMatrix = testEigenSpace.getData();
		
		int numberOfTrueRecognition = 0;
		int numberOfFalseRecognition = 0;
		
		for(int testDataCounter = 0; testDataCounter<testMatrix[0].length; testDataCounter++){

			double minDistance = 999999;
			int minDistanceIndex = -1;
			
			for(int trainDataCounter = 0; trainDataCounter < trainMatrix[0].length ; trainDataCounter++){

				double distance = getDistance(testMatrix[testDataCounter],trainMatrix[trainDataCounter]);

				if(distance < minDistance){
					minDistance = distance;
					minDistanceIndex = trainDataCounter;
				}
			}
			
			// Compare wheter its true or false
			if(list.get(minDistanceIndex).intValue() == list2.get(testDataCounter).intValue()){
				
				numberOfTrueRecognition++;
			}else{
				numberOfFalseRecognition++;
			}
			
			
		}
		
		double accuracy = ((double)numberOfTrueRecognition / (numberOfTrueRecognition + numberOfFalseRecognition)) * 100;
		
		System.out.printf("\nAccuracy of the tree is: %.3f\n",accuracy);


	}

	/**
	 * Euclidean Distance
	 * @param testVector
	 * @param trainVector
	 * @return
	 */
	private static double getDistance(double[] testVector, double[] trainVector) {
		double diff_square_sum = 0.0;
		for (int i = 0; i < testVector.length; i++) {
			diff_square_sum += (testVector[i] - trainVector[i]) * (testVector[i] - trainVector[i]);
		}
		return Math.sqrt(diff_square_sum);
	}


}
