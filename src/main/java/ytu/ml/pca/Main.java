package ytu.ml.pca;

import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.RealMatrix;

public class Main {

	Logger logger = Logger.getGlobal();
	
	public static void main(String[] args) throws Exception {
		
		Logger logger = Logger.getGlobal();
		
		long trainingStartTime = System.currentTimeMillis();
		String fileName = "C:/Users/furkan/desktop/sayi.dat";
		
		double threshold = 0.4;
		
		DataTable dataTable = new DataTable();
		dataTable.loadDatFileToDataTable(fileName);

		// Step 1 Calculate the mean face vector
		double[] meanVectors = PCAUtil.calculateMeanVector(dataTable.getTrainSamples());
		dataTable.setTrainMeanVector(meanVectors);
		
		// Step 2 Calculate mean centered vectors
		double[][] subtractVectors = PCAUtil.calculateMeanCenteredTrainMatrix(dataTable.getTrainSamples(), dataTable.getTrainMeanVector());
		dataTable.setMeanCenteredTrainMatrix(subtractVectors);
		
		// Step 3 calculate Covariance
		double[][] covarianceMatrix = PCAUtil.findCovarianceMatrix(dataTable.getMeanCenteredTrainMatrix());
		dataTable.setCovarianceMatrix(covarianceMatrix);
		
		dataTable.setThreshold(threshold);
		double[][] trainEigenSpace = PCAUtil.createTrainEigenSpace(dataTable);
		dataTable.setTrainEigenSpace(trainEigenSpace);
		
		double[][] testEigenSpace = PCAUtil.createTestEigenSpace(dataTable);
		dataTable.setTestEigenSpace(testEigenSpace);
		
		PCAUtil.test(dataTable.getTrainEigenSpace(), dataTable.getTrainImageValueList(),dataTable.getTestEigenSpace(),dataTable.getTestImageValueList());
		
		logger.info("Total time of building, pruning and testing of algotihm is: "+ TimeUnit.MILLISECONDS.toSeconds((System.currentTimeMillis() - trainingStartTime)) + " seconds.");
	}

}
