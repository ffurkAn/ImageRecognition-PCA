package ytu.ml.pca;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.logging.Logger;

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
			
			float total = 0.0f;
			// iterates the train samples for each classifier
			for (Entry<Integer,List<SampleObject>> entry : trainSamples.entrySet()) {
				
				for (SampleObject sample : entry.getValue()) {
					
					float value = Float.parseFloat(sample.getSampleValues().get(x));
					total += value;
				}
			}
			
			float mean = total / 50;
			returnMap.add(Float.toString(mean));
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
			
			List<List<String>> listOfSubtractedSamples = new ArrayList<>();
			
			// Iterates the samples and calculate the subtracted values for each pixel value
			for (SampleObject sample : entry.getValue()) {
				
				List<String> subtractedValueList = new ArrayList<>();
				
				for (int i = 0; i < sample.getSampleValues().size(); i++) {
					
					float meanValue = Float.parseFloat(meanVector.get(i));
					float trainValue = Float.parseFloat(sample.getSampleValues().get(i));
					
					float subtractedValue = trainValue - meanValue;
					subtractedValueList.add(Float.toString(subtractedValue));
				}
				
				listOfSubtractedSamples.add(subtractedValueList);
			}
			
			returnMap.put(entry.getKey(), listOfSubtractedSamples);
		}
		return returnMap;
		
	}
	
	
}
