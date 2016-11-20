package ytu.ml.pca;

import java.util.logging.Logger;

public class Main {

	Logger logger = Logger.getGlobal();
	
	public static void main(String[] args) throws Exception {
		
		
		long trainingStartTime = System.currentTimeMillis();
		String fileName = "C:/Users/furkan/desktop/sayi.dat";
		
		
		DataTable dataTale = new DataTable();
		dataTale.loadDatFileToDataTable(fileName);
		
		dataTale.toString();
	}

}
