package ytu.ml.pca;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
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

	Logger logger = Logger.getGlobal();

	/**
	 * calculates the mean vector of each train sample set
	 * 
	 * i.e, 5 sample line for classifier '0'
	 * @param trainSamples
	 * @return
	 */
	public static List<String> calculateMeanVectors(Map<Integer, List<SampleObject>> trainSamples) {

		List<String> returnMap = new ArrayList<>();

		// for every pixel index
		for(int x = 0 ; x < 64 ; x++){

			double total = 0.0;
			// iterates the train samples for each classifier
			for (Entry<Integer,List<SampleObject>> entry : trainSamples.entrySet()) {

				for (SampleObject sample : entry.getValue()) {

					double value = Double.parseDouble(sample.getSampleValues().get(x));
					total += value;
				}
			}

			double mean = total / 50;
			returnMap.add(Double.toString(mean));
		}

		return returnMap;
	}

	/**
	 * subtract mean vector values from every train values
	 * @param trainSamples
	 * @param meanVector
	 * @return
	 */
	public static Map<Integer,List<List<String>>> calculateSubtractVectors(Map<Integer, List<SampleObject>> trainSamples, List<String> meanVector){

		Map<Integer,List<List<String>>> returnMap = new HashMap<>();

		// Iterates the sample set for each classifier
		for (Entry<Integer,List<SampleObject>> entry : trainSamples.entrySet()) {

			List<List<String>> listOfMeanCenteredSamples = new ArrayList<>();

			// Iterates the samples and calculate the meanCentered values for each pixel value
			for (SampleObject sample : entry.getValue()) {

				List<String> meanCenteredValueList = new ArrayList<>();

				for (int i = 0; i < sample.getSampleValues().size(); i++) {

					double meanValue = Double.parseDouble(meanVector.get(i));
					double trainValue = Double.parseDouble(sample.getSampleValues().get(i));

					double meanCenteredValue = trainValue - meanValue;
					meanCenteredValueList.add(Double.toString(meanCenteredValue));
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
	public static double[][] findCovarianceMatrix(Map<Integer,List<List<String>>> meanCenteredVectors){

		// intialize 64x50 matrix
		double[][] meanCenteredMatrix = new double[DataTable.NUMBER_OF_PIXEL_VALUES][DataTable.NUMBER_OF_CLASSES*DataTable.NUMBER_OF_TRAIN_VECTORS];

		int column = 0;
		for (Entry<Integer,List<List<String>>> entry : meanCenteredVectors.entrySet()) {
			for (List<String> meanCenteredVector: entry.getValue()) {

				int row = 0;
				for (String pixelValue : meanCenteredVector) {

					meanCenteredMatrix[row][column] = Double.parseDouble(pixelValue);
					row++;
				}
				column++;
			}
		}
		return covariance(meanCenteredMatrix);

	}

	/**
	 *  creates covariance matrix of matrix which is formed by mean centered vectors
	 * @param returnMatrix
	 * @return
	 */
	private static double[][] covariance(double[][] meanCenteredMatrix) {
		RealMatrix mx = MatrixUtils.createRealMatrix(meanCenteredMatrix);
		RealMatrix covarianceMatrix = new Covariance(mx).getCovarianceMatrix();
		return covarianceMatrix.getData();
	}

	public static void compute(double[][] covarianceMatrix, double threshold) {

		RealMatrix matrix = MatrixUtils.createRealMatrix(covarianceMatrix);

		EigenDecomposition eigen = new EigenDecomposition(matrix);
		double[] eigenValues = eigen.getRealEigenvalues(); // it gives the eigenvalues sorted

		double sumOfEigenValues = sumOfValues(eigenValues);
		double total = 0.0;
		int i = 0; // last index of the eligible eigen value
		
		while(total/sumOfEigenValues < threshold){
			total = total + eigenValues[i];
			i++;
		}
		
		int lastIndex = i - 2; 
		
		
		
	}

	private static double sumOfValues(double[] eigenValues) {
		double result = 0;
		for (double value:eigenValues)
			result += value;
		return result;
	}

}
