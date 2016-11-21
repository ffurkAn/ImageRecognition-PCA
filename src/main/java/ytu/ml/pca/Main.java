package ytu.ml.pca;

import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

public class Main {

	Logger logger = Logger.getGlobal();
	
	public static void main(String[] args) throws Exception {
		
		
		long trainingStartTime = System.currentTimeMillis();
		String fileName = "C:/Users/furkan/desktop/sayi.dat";
		
		
		DataTable dataTable = new DataTable();
		dataTable.loadDatFileToDataTable(fileName);

		// Step 1 Calculate the mean face vector
		List<String> meanVectors = PCAUtil.calculateMeanVectors(dataTable.getTrainSamples());
		dataTable.setTrainMeanVector(meanVectors);
		
		// Step 2
		Map<Integer,List<List<String>>> subtractVectors = PCAUtil.calculateSubtractVectors(dataTable.getTrainSamples(), dataTable.getTrainMeanVector());
		dataTable.setSubtractVectorsMap(subtractVectors);
		
		// Step 3 calculate Covariance
		
		
	}

}
