package ytu.ml.pca;

import java.util.List;

/**
* This class holds the line of data with its classifier number
*
*/
public class SampleObject {
	
	private int lineId;
	private String classifierNumberStr;
	private Integer classifierNumber; // 0-9
	private List<Double> sampleValues; // holds the values separated with comma
	
	public SampleObject(int lineId, String classifierNumberStr, int classifierNumber, List<Double> sampleValues) {
		super();
		this.lineId = lineId;
		this.classifierNumberStr = classifierNumberStr;
		this.classifierNumber = classifierNumber;
		this.sampleValues = sampleValues;
	}

	public SampleObject() {
		super();
		// TODO Auto-generated constructor stub
	}

	public int getLineId() {
		return lineId;
	}

	public void setLineId(int lineId) {
		this.lineId = lineId;
	}

	public String getClassifierNumberStr() {
		return classifierNumberStr;
	}

	public void setClassifierNumberStr(String classifierNumberStr) {
		this.classifierNumberStr = classifierNumberStr;
	}

	public Integer getClassifierNumber() {
		return classifierNumber;
	}

	public void setClassifierNumber(Integer classifierNumber) {
		this.classifierNumber = classifierNumber;
	}

	public List<Double> getSampleValues() {
		return sampleValues;
	}

	public void setSampleValues(List<Double> sampleValues) {
		this.sampleValues = sampleValues;
	}

	@Override
	public String toString() {
		return "SampleObject [lineId=" + lineId + ", classifierNumberStr=" + classifierNumberStr + ", classifierNumber="
				+ classifierNumber + ", sampleValues=" + sampleValues + "]";
	}
	
	
	
	
	
}
