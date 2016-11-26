package ytu.ml.pca;

import java.awt.Robot;
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

import Jama.EigenvalueDecomposition;
import Jama.Matrix;

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
	public static double[] calculateMeanVector(double[][] trainSamples) {

		double[] returnMap = new double[64];

		// for every pixel index
		for(int x = 0 ; x < 64 ; x++){

			double total = 0.0;
			// iterates the train samples for each classifier
			
			for(int trainRow = 0; trainRow < trainSamples.length ; trainRow++){
				double value = trainSamples[trainRow][x];
				total += value;
			}
			
			double mean = total / 50;
			returnMap[x] = mean;
		}

		return returnMap;
	}

	/**
	 * subtract mean vector values from every train values
	 * @param trainSamples
	 * @param meanVector
	 * @return
	 */
	public static double[][] calculateMeanCenteredTrainMatrix(double[][] trainSamples, double[] meanVector){

		double[][] meanCenteredTrainMatrix = new double[Constants.NUMBER_OF_TRAIN_VECTORS][Constants.NUMBER_OF_PIXEL_VALUES];

		// Iterates the sample set for each classifier
		for(int row = 0;row < trainSamples.length ; row++){
			
			double[] subtractedValue = getSubtractedVector(trainSamples[row], meanVector);
			meanCenteredTrainMatrix[row] = subtractedValue;
		}
			
		return meanCenteredTrainMatrix;

	}


	/**
	 *  creates covariance matrix of matrix which is formed by mean centered vectors
	 * @param returnMatrix
	 * @return
	 */
	public static double[][] findCovarianceMatrix(double[][] meanCenteredVectors){

//		yazdir(meanCenteredVectors);
		RealMatrix mx = MatrixUtils.createRealMatrix(meanCenteredVectors);
		RealMatrix covarianceMatrix = new Covariance(mx).getCovarianceMatrix();
		
		return covarianceMatrix.getData();
	}

	
	private static void yazdir(double[][] meanCenteredVectors) {

		for(int x = 0; x< meanCenteredVectors.length ; x++){
			
			for(int y=0 ; y < meanCenteredVectors[x].length; y++){
				
				System.out.print(meanCenteredVectors[x][y]+",");
			}
			
			System.out.print("\n");
		}
		
	}

	public static double[][] createTrainEigenSpace(DataTable dataTable) {

		RealMatrix covarianceMatrix = MatrixUtils.createRealMatrix(dataTable.getCovarianceMatrix());

		EigenDecomposition eigen = new EigenDecomposition(covarianceMatrix);
		dataTable.setEigen(eigen);
		
		double d = eigen.getRealEigenvalue(0);
		double[] eigenValues = eigen.getRealEigenvalues();// it gives the eigenvalues sorted

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
		RealMatrix meanCenteredTrainMatrix = MatrixUtils.createRealMatrix(dataTable.getMeanCenteredTrainMatrix()); // mean centered train matrix - pixel value X train data sayýsý

		RealMatrix trainEigenSpace = eigenspaceVectorMatrix.multiply(meanCenteredTrainMatrix.transpose());

		//satýrlarý özellik dizisi yapmak için transpoze aldýk
		return trainEigenSpace.transpose().getData();

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

	public static double[][] createTestEigenSpace(DataTable dataTable) {

		// feature sayýsý X eigen value sayýsý
		double[][] eigenvectorMatrix = getEigenVectorMatrix(dataTable);

		// test data sayýsý X pixel sayýsý
		double[][] meanCenteredTestMatrix = new double[Constants.NUMBER_OF_TEST_VECTORS][Constants.NUMBER_OF_PIXEL_VALUES]; 

		int testSampleCounter = 0;
		for (double[] validationSample : dataTable.getTestSamples()) {

			meanCenteredTestMatrix[testSampleCounter] = getSubtractedVector(validationSample, dataTable.getTrainMeanVector()); 

		}

		RealMatrix eigenSpaceVectorMatrix = MatrixUtils.createRealMatrix(eigenvectorMatrix); // feat
		RealMatrix meanCenteredMatrix = MatrixUtils.createRealMatrix(meanCenteredTestMatrix); // meanCenteredTestMatrix

		RealMatrix testEigenSpace = eigenSpaceVectorMatrix.multiply(meanCenteredMatrix.transpose());

		//satýrlarý özellik dizisi yapmak için transpoze aldýks
		return testEigenSpace.transpose().getData();
	}

	private static double[] getSubtractedVector(double[] vector, double[] meanVector) {

		double[] subtractedVector = new double[vector.length];

		for (int valueCounter = 0; valueCounter < vector.length; valueCounter++) {
			subtractedVector[valueCounter] = vector[valueCounter] - meanVector[valueCounter];
		}

		return subtractedVector;
	}

	private static double sumOfValues(double[] eigenValues) {
		double result = 0;
		for (double value:eigenValues)
			result += value;
		return result;
	}

	public static void test(double[][] trainSpaceMatrix, List<Integer> trainImageValue, double[][] testEigenMatrix, List<Integer> testImageValue) {

		// test datasýný iterate ediyoruz, her özellik için traindeki 50 veriye ait 3 özellik ile distance alýorz
		
		int numberOfTrueRecognition = 0;
		int numberOfFalseRecognition = 0;
		
		for(int testDataCounter = 0; testDataCounter<testEigenMatrix.length; testDataCounter++){

			double minDistance = 999999;
			int minDistanceIndex = -1;
			
			for(int trainDataCounter = 0; trainDataCounter < trainSpaceMatrix.length ; trainDataCounter++){

				double distance = getDistance(testEigenMatrix[testDataCounter],trainSpaceMatrix[trainDataCounter]);

				if(distance < minDistance){
					minDistance = distance;
					minDistanceIndex = trainDataCounter;
				}
			}
			
			// Compare wheter its true or false
			if(trainImageValue.get(minDistanceIndex).intValue() == testImageValue.get(testDataCounter).intValue()){
				
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
			diff_square_sum += (trainVector[i] - testVector[i]) * (trainVector[i] - testVector[i]);
		}
		return Math.sqrt(diff_square_sum);
	}


}
