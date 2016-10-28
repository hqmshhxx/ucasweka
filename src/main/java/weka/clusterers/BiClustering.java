package weka.clusterers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;

import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class BiClustering extends AbstractClusterer
		implements WeightedInstancesHandler, TechnicalInformationHandler, OptionHandler {

	private static final long serialVersionUID = -8665636089245404045L;

	/** 属性的最少个数 */
	private int minAttributes = 5;
	/** instance的最少个数 */
	private int minInstances = 5;
	/** 目标函数值改变量范围 */
	private double endValue = 1e-4;

	private double mValue = 0;

	private int mIterations = 0;
	private int maxIterations = 500;
	protected ReplaceMissingValues m_ReplaceMissingFilter;
	protected boolean m_dontReplaceMissing = false;
	
	

	@Override
	public void buildClusterer(Instances data) throws Exception {
		// can clusterer handle the data?
		getCapabilities().testWithFail(data);

		m_ReplaceMissingFilter = new ReplaceMissingValues();
		Instances instances = new Instances(data);

		instances.setClassIndex(-1);
		if (!m_dontReplaceMissing) {
			m_ReplaceMissingFilter.setInputFormat(instances);
			instances = Filter.useFilter(instances, m_ReplaceMissingFilter);
		}
		boolean converged = false;
		while (!converged) {
			mIterations++;

			mValue = findScore(instances);

			if (mValue <= endValue) {
				converged = true;
			}
			if (mIterations == maxIterations) {
				converged = true;
			}
		}
	}

	public double findScore(Instances instances) {
		double score = 0;
		for (int i = 0; i < instances.numInstances(); i++) {
			for (int j = 0; j < instances.numAttributes(); j++) {
				double[] values=calculateScore(instances,i,j);
				score+= Math.pow(instances.get(i).value(j)-values[0]-values[1]+values[2],2);
			}
		}
		score=score/(instances.numAttributes()*instances.numInstances());
		return score;
	}

	public double[] calculateScore(Instances instances, int row, int col) {
		double scoreI = 0;
		double scoreJ = 0;
		double[] values = new double[3];
		int rowNum = instances.numInstances();
		int colNum = instances.numAttributes();
		for (int i = 0; i < instances.numInstances(); i++) {
			scoreI += instances.get(i).value(col);
		}
		for (int j = 0; j < instances.numAttributes(); j++) {
			scoreJ += instances.get(row).value(j);
		}
		values[0] += scoreI * 1.0 / rowNum;
		values[1] += scoreJ * 1.0 / colNum;
		values[2] = (scoreI + scoreJ) * 1.0 / (rowNum * colNum);
		
		return values;

	}

	@Override
	public int clusterInstance(Instance instance) throws Exception {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public TechnicalInformation getTechnicalInformation() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Enumeration<Option> listOptions() {
		// TODO Auto-generated method stub
		return super.listOptions();
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		// TODO Auto-generated method stub
		String optionString = Utils.getOption('I', options);
		if (optionString.length() != 0) {
			setMaxIterations(Integer.parseInt(optionString));
		}
		optionString=Utils.getOption("MI", options);
		if (optionString.length() != 0) {
			setMinInstances(Integer.parseInt(optionString));
		}
		optionString=Utils.getOption("MA", options);
		if (optionString.length() != 0) {
			setMinAttributes(Integer.parseInt(optionString));
		}
		optionString=Utils.getOption("EV", options);
		if (optionString.length() != 0) {
			setMinAttributes(Integer.parseInt(optionString));
		}
		super.setOptions(options);
		Utils.checkForRemainingOptions(options);
	}

	@Override
	public String[] getOptions() {
		// TODO Auto-generated method stub
		ArrayList<String> result = new ArrayList<>();
		result.add("-EV");
		result.add(""+ getEndValue());
		result.add("-I");
		result.add("" + getMaxIterations());
		result.add("-MI");
		result.add("" + getMinInstances());
		result.add("-MA");
		result.add("" + getMinAttributes());

	

		Collections.addAll(result, super.getOptions());
		return result.toArray(new String[result.size()]);
	}

	@Override
	public Capabilities getCapabilities() {
		// TODO Auto-generated method stub
		Capabilities result = super.getCapabilities();
		result.disableAll();
		result.enable(Capability.NO_CLASS);

		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);

		return result;
	}

	@Override
	public int numberOfClusters() throws Exception {
		// TODO Auto-generated method stub
		return 0;
	}
	

	public int getMinAttributes() {
		return minAttributes;
	}

	public void setMinAttributes(int minAttributes) {
		this.minAttributes = minAttributes;
	}

	public int getMinInstances() {
		return minInstances;
	}

	public void setMinInstances(int minInstances) {
		this.minInstances = minInstances;
	}

	public int getMaxIterations() {
		return maxIterations;
	}

	public void setMaxIterations(int maxIterations) {
		this.maxIterations = maxIterations;
	}

	public double getEndValue() {
		return endValue;
	}

	public void setEndValue(double endValue) {
		this.endValue = endValue;
	}

	@Override
	public String toString() {
		// TODO Auto-generated method stub
		StringBuilder temp = new StringBuilder();
		temp.append("\nBiClustering\n======\n");
		temp.append("\nNumber of iterations: " + mIterations);
		temp.append("\nthe value is: "+mValue);
		temp.append("\n\n");

		return temp.toString();
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}
}
