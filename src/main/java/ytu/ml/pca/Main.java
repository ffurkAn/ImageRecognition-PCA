package ytu.ml.pca;

import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import org.apache.commons.math3.linear.EigenDecomposition;

public class Main {

	Logger logger = Logger.getGlobal();
	
	public static void main(String[] args) throws Exception {
		
		
		long trainingStartTime = System.currentTimeMillis();
		String fileName = "C:/Users/furkan/desktop/sayi.dat";
		
		double threshold = 0.6;
		
		DataTable dataTable = new DataTable();
		dataTable.loadDatFileToDataTable(fileName);

		// Step 1 Calculate the mean face vector
		List<String> meanVectors = PCAUtil.calculateMeanVectors(dataTable.getTrainSamples());
		dataTable.setTrainMeanVector(meanVectors);
		
		// Step 2 Calculate mean centered vectors
		Map<Integer,List<List<String>>> subtractVectors = PCAUtil.calculateSubtractVectors(dataTable.getTrainSamples(), dataTable.getTrainMeanVector());
		dataTable.setMeanCenteredVectorMap(subtractVectors);
		
		// Step 3 calculate Covariance
		double[][] covarianceMatrix = PCAUtil.findCovarianceMatrix(dataTable.getMeanCenteredVectorMap());
		dataTable.setCovarianceMatrix(covarianceMatrix);
		
		PCAUtil.compute(dataTable.getCovarianceMatrix(),threshold);
		
	}

}
