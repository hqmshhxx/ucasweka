package weka.clusterers;


import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;

import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.Capabilities.Capability;

public class BCFCM extends RandomizableClusterer implements NumberOfClustersRequestable, 
							WeightedInstancesHandler,TechnicalInformationHandler{
	
	private static final long serialVersionUID = 1L;
	
	private AluBiCluster bic;
	private AluFCM fcm;
	protected Instances m_ClusterCentroids;
	private Instances mInitClusters;
	/**
	 * Holds the squared errors for all clusters.
	 */
	private double[] mSquaredErrors;
	
	private int mNumCluster = 3;
	
	
	public BCFCM (){
		bic = new AluBiCluster();
	}
	

	@Override
	public void buildClusterer(Instances data) throws Exception {
		getCapabilities().testWithFail(data);
		
		bic.buildClusterer(data);
		mInitClusters = bic.getFullCentroids();
		fcm = new AluFCM(mInitClusters);
		fcm.buildClusterer(data);
		m_ClusterCentroids = fcm.m_ClusterCentroids;
		mSquaredErrors = fcm.m_squaredErrors;
		
	}
	
	
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		// TODO Auto-generated method stub
		return fcm.distributionForInstance(instance);
	}
	public double getSquaredError() {
		return Utils.sum(mSquaredErrors);
	}
	
	public Instances getCentroids() {
		return fcm.getClusterCentroids();
	}
	
	@Override
	public Enumeration<Option> listOptions() {
		// TODO Auto-generated method stub
		return fcm.listOptions();
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		// TODO Auto-generated method stub
		String optionString = Utils.getOption('N', options);

		if (optionString.length() != 0) {
			setNumClusters(Integer.parseInt(optionString));
		}

		super.setOptions(options);
		Utils.checkForRemainingOptions(options);
	}

	@Override
	public String[] getOptions() {
		// TODO Auto-generated method stub
		ArrayList<String> result = new ArrayList<>();
				
		result.add("-N");
		result.add("" + mNumCluster);

		result.add("-I");
		result.add("" + getMaxIterations());

		Collections.addAll(result, super.getOptions());

		return result.toArray(new String[result.size()]);
	}

	@Override
	public void setSeed(int value) {
		// TODO Auto-generated method stub
		fcm.setSeed(value);
	}

	@Override
	public int getSeed() {
		// TODO Auto-generated method stub
		return 10;
	}
	public int getMaxIterations() {
		return bic.getMaxIterations();
	}

	public void setMaxIterations(int maxIterations) {
		bic.setMaxIterations(maxIterations);
	}
	public double getEndValue() {
		return bic.getEndValue();
	}

	public void setEndValue(double endValue) {
		bic.setEndValue(endValue);
	}
	@Override
	public Capabilities getCapabilities() {
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
	public void setNumClusters(int numClusters) throws Exception {
		// TODO Auto-generated method stub
		mNumCluster = numClusters;
	}

	@Override
	public int numberOfClusters() throws Exception {
		// TODO Auto-generated method stub
		return mNumCluster;
	}
	
	@Override
	public TechnicalInformation getTechnicalInformation() {
		// TODO Auto-generated method stub
		return fcm.getTechnicalInformation();
	}
	
	@Override
	public String toString() {
		return fcm.toString();
		
/*		
		if (m_ClusterCentroids == null) {
			return "No clusterer built yet!";
		}
		int maxWidth = 0;
		int maxAttWidth = 0;
		boolean containsNumeric = false;
		for (int i = 0; i < mNumCluster; i++) {
			for (int j = 0; j < m_ClusterCentroids.numAttributes(); j++) {
				if (m_ClusterCentroids.attribute(j).name().length() > maxAttWidth) {
					maxAttWidth = m_ClusterCentroids.attribute(j).name().length();
				}
				if (m_ClusterCentroids.attribute(j).isNumeric()) {
					containsNumeric = true;
					double width = Math.log(Math.abs(m_ClusterCentroids.instance(i).value(j))) / Math.log(10.0);
					if (width < 0) {
						width = 1;
					}
					// decimal + # decimal places + 1
					width += 6.0;
					if ((int) width > maxWidth) {
						maxWidth = (int) width;
					}
				}
			}
		}
		for (int i = 0; i < m_ClusterCentroids.numAttributes(); i++) {
			if (m_ClusterCentroids.attribute(i).isNominal()) {
				Attribute a = m_ClusterCentroids.attribute(i);
				for (int j = 0; j < m_ClusterCentroids.numInstances(); j++) {
					String val = a.value((int) m_ClusterCentroids.instance(j).value(i));
					if (val.length() > maxWidth) {
						maxWidth = val.length();
					}
				}
				for (int j = 0; j < a.numValues(); j++) {
					String val = a.value(j) + " ";
					if (val.length() > maxAttWidth) {
						maxAttWidth = val.length();
					}
				}
			}
		}
		
		
		StringBuffer temp = new StringBuffer();
		temp.append("\nBCFCM\n======\n");
		temp.append("\nNumber of iterations: " + fcm.m_Iterations);
		temp.append("\tthe seed: " + m_Seed);
		
		temp.append("\n");
		if (fcm.m_DistanceFunction instanceof EuclideanDistance) {
			temp.append("Within cluster sum of squared errors: "+ Utils.sum(fcm.m_squaredErrors));
		} else {
			temp.append("Sum of within cluster distances: "+ Utils.sum(fcm.m_squaredErrors));
		}
		

		temp.append("\n\nFinal cluster centroids:\n");
		temp.append(pad("Cluster#", " ", (maxAttWidth + (maxWidth * 2 + 2))
				- "Cluster#".length(), true));

		temp.append("\n");
		temp.append(pad("Attribute", " ", maxAttWidth - "Attribute".length(),false));

		temp.append(pad("Full Data", " ", maxWidth + 1 - "Full Data".length(),true));
		// cluster numbers
		for (int i = 0; i < mNumCluster; i++) {
			String clustNum = "" + i;
			temp.append(pad(clustNum, " ", maxWidth + 1 - clustNum.length(),true));
		}
		temp.append("\n");

		// cluster sizes
		String cSize = "(" + Utils.sum(fcm.m_ClusterSizes) + ")";
		temp.append(pad(cSize, " ",maxAttWidth + maxWidth + 1 - cSize.length(), true));
		for (int i = 0; i < mNumCluster; i++) {
			cSize = "(" + fcm.m_ClusterSizes[i] + ")";
			temp.append(pad(cSize, " ", maxWidth + 1 - cSize.length(), true));
		}
		temp.append("\n");

		temp.append(pad("", "=",maxAttWidth+ (maxWidth * (m_ClusterCentroids.numInstances() + 1)
								+ m_ClusterCentroids.numInstances() + 1), true));
		temp.append("\n");
		for (int i = 0; i < m_ClusterCentroids.numAttributes(); i++) {
			String attName = m_ClusterCentroids.attribute(i).name();
			temp.append(attName);
			for (int j = 0; j < maxAttWidth - attName.length(); j++) {
				temp.append(" ");
			}

			String strVal;
			String valMeanMode;
			
			for (int j = 0; j < mNumCluster; j++) {
				if (m_ClusterCentroids.attribute(i).isNominal()) {
					if (m_ClusterCentroids.instance(j).isMissing(i)) {
						valMeanMode = pad("missing", " ", maxWidth + 1- "missing".length(), true);
					} else {
						valMeanMode = pad((strVal = m_ClusterCentroids.attribute(i)
								.value((int) m_ClusterCentroids.instance(j).value(i))),
								" ",maxWidth + 1 - strVal.length(), true);
					}
				} else {
					if (m_ClusterCentroids.instance(j).isMissing(i)) {
						valMeanMode = pad("missing", " ", maxWidth + 1- "missing".length(), true);
					} else {
						valMeanMode = pad((strVal = Utils.doubleToString(m_ClusterCentroids.instance(j).value(i), maxWidth, 4)
										.trim()), " ",maxWidth + 1 - strVal.length(), true);
					}
				}
				temp.append(valMeanMode);
			}
			temp.append("\n");
			
		}

		temp.append("\n");
		return temp.toString();
*/		
	}

	private String pad(String source, String padChar, int length,
			boolean leftPad) {
		StringBuffer temp = new StringBuffer();

		if (leftPad) {
			for (int i = 0; i < length; i++) {
				temp.append(padChar);
			}
			temp.append(source);
		} else {
			temp.append(source);
			for (int i = 0; i < length; i++) {
				temp.append(padChar);
			}
		}
		return temp.toString();
	}

	
	public static void main(String[] args){
		
		runClusterer(new BCFCM(), args);
//		BCFCM bcm = new BCFCM();
		
	}
}