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
	public static Map<Integer, List<String>> calculateMeanVectors(Map<Integer, List<SampleObject>> trainSamples) {

		Map<Integer,List<String>> returnMap = new HashMap<>();
		
		// iterates the train samples for each classifier
		for (Entry<Integer,List<SampleObject>> entry : trainSamples.entrySet()) {
			
			List<String> meanVector = new ArrayList<>();
			
			//iterates all the pixel values
			for (int i = 0; i< 64;i++) {
				
				int total = 0;
				
				// iterates train sets
				for(int j = 0; j<5 ; j++){
					
					// increase the total by adding jth sample's ith pixel value
					total += Integer.parseInt(entry.getValue().get(j).getSampleValues().get(i));
				}
				
				float avaragePixelValue = (float)total / 5;
				meanVector.add(Float.toString(avaragePixelValue));
			}
			returnMap.put(entry.getKey(), meanVector);
		}
		
		return returnMap;
	}
	
	/**
	 * subtract mean vector values from every train values
	 * @param trainSamples
	 * @param meanVectors
	 * @return
	 */
	public static Map<Integer,List<List<String>>> calculateSubtractVectors(Map<Integer, List<SampleObject>> trainSamples, Map<Integer,List<String>> meanVectors){
		
		Map<Integer,List<List<String>>> returnMap = new HashMap<>();
		
		// Iterates the sample set for each classifier
		for (Entry<Integer,List<SampleObject>> entry : trainSamples.entrySet()) {
			
			List<List<String>> listOfSubtractedSamples = new ArrayList<>();
			
			// Iterates the samples and calculate the subtract values for each pixel value
			for (SampleObject sample : entry.getValue()) {
				
				List<String> subtractedValueList = new ArrayList<>();
				
				for (int i = 0; i < sample.getSampleValues().size(); i++) {
					
					float meanValue = Float.parseFloat(meanVectors.get(entry.getKey()).get(i));
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
