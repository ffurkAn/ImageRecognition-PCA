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
	public static double[][] findCovarianceMatrix(Map<Integer,List<List<Double>>> meanCenteredVectors){

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
	private static double[][] covariance(double[][] meanCenteredMatrix) {
		RealMatrix mx = MatrixUtils.createRealMatrix(meanCenteredMatrix);
		RealMatrix covarianceMatrix = new Covariance(mx.transpose()).getCovarianceMatrix();
		return covarianceMatrix.getData();
	}

	public static double[][] createEigenSpaceForKFeature(DataTable dataTable) {

		double[][] covarianceMatrix = dataTable.getCovarianceMatrix();
		RealMatrix matrix = MatrixUtils.createRealMatrix(covarianceMatrix);

		EigenDecomposition eigen = new EigenDecomposition(matrix);
		double[] eigenValues = eigen.getRealEigenvalues(); // it gives the eigenvalues sorted

		int eigenVectorLength = eigenValues.length;
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
		
		double[][] eigenvectorMatrix = new double[dataTable.getNumberOfNewFeatures()][eigenVectorLength];
		
		for(int featureCounter = 0 ; featureCounter < dataTable.getNumberOfNewFeatures() ; featureCounter++){
			
				RealVector vector = eigen.getEigenvector(featureCounter); 
				for (int columnCounter = 0; columnCounter < eigenVectorLength; columnCounter++) {
					eigenvectorMatrix[featureCounter][columnCounter] = vector.getEntry(columnCounter);
				}
		}
		
		RealMatrix eigenspaceVectorMatrix = MatrixUtils.createRealMatrix(eigenvectorMatrix);
		RealMatrix meanCenteredMeanMatrix = MatrixUtils.createRealMatrix(getCenteredTrainMatrix(dataTable.getMeanCenteredVectorMap()));
		
		RealMatrix eigenspaceMatrix = eigenspaceVectorMatrix.multiply(meanCenteredMeanMatrix);
		
		return eigenspaceMatrix.getData();
		
	}

	
	public static void test(DataTable dataTable) {
		
		List<SampleObject> validationSamples = dataTable.getValidationSamples();
		List<Double> meanVector = dataTable.getTrainMeanVector();
		
		// computing mean centered validation vectors
		SampleObject[][] meanCenteredValidationMatrix = new SampleObject[dataTable.getNumberOfNewFeatures()][validationSamples.size()];
		
		for(int row = 0; row < dataTable.getNumberOfNewFeatures() ; row++){
			
			for (int validationSampleCounter = 0; validationSampleCounter < validationSamples.size(); validationSampleCounter++) {
				
				SampleObject validationSample = validationSamples.get(validationSampleCounter);
				List<Double> subtractedVector = getSubtractedVector(validationSample,meanVector);
				validationSamples.get(validationSampleCounter).setSampleValues(subtractedVector);
				
				meanCenteredValidationMatrix[row][validationSampleCounter] = validationSample;
			}
		}
		
		System.out.println("asdas");
		
	}

	private static List<Double> getSubtractedVector(SampleObject sampleObject, List<Double> meanVector) {
	
		List<Double> subtractedVector = new ArrayList<>();
		
		for (int valueCounter = 0; valueCounter < sampleObject.getSampleValues().size(); valueCounter++) {
			double subtractedValue = sampleObject.getSampleValues().get(valueCounter) - meanVector.get(valueCounter);
			
			subtractedVector.add(subtractedValue);
		}
		
		return subtractedVector;
			
	}
	
	private static double sumOfValues(double[] eigenValues) {
		double result = 0;
		for (double value:eigenValues)
			result += value;
		return result;
	}


}
