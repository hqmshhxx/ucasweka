package weka.clusterers;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;

public class BCFCM extends RandomizableClusterer implements NumberOfClustersRequestable, 
							WeightedInstancesHandler,TechnicalInformationHandler{
	
	private FastFCM fcm;
	protected Instance[] mCentroids;
	private Instances[] mClusters;
	/**
	 * Holds the squared errors for all clusters.
	 */
	private double[] mSquaredErrors;
	
	
	private Instances[] mFcmClusters;
	private Instances mFcmCentroids;
	/**
	 * Holds the squared errors for all clusters.
	 */
	private double[] mFcmSquaredErrors;
	private int mNumCluster = 3;
	/** Number of threads to run */
	protected int mExecutionSlots = 3;
	/** For parallel execution mode */
	protected transient ExecutorService mExecutorPool;
	
	public BCFCM (FastFCM fcm){
		this.fcm = fcm;
	}
	protected void startExecutorPool() {
		if (mExecutorPool != null) {
			mExecutorPool.shutdownNow();
		}
		mExecutorPool = Executors.newFixedThreadPool(mExecutionSlots);
	}
	@Override
	public void buildClusterer(Instances data) throws Exception {
		getCapabilities().testWithFail(data);
		startExecutorPool();
		List<Future<BiClustering>> results = new ArrayList<>();
		fcm.buildClusterer(data);
		mFcmClusters = fcm.getClusters();
		mFcmCentroids = fcm.getClusterCentroids();
		mFcmSquaredErrors = fcm.m_squaredErrors;
		for(int i = 0; i < mNumCluster; i++){
			results.add(mExecutorPool.submit(new BiCTask(mFcmClusters[i])));
		}
		
		try{
			for(int j = 0; j < results.size(); j++){
				BiClustering bi = results.get(j).get();
				mClusters[j] = bi.getCluster();
				mCentroids[j] = bi.getCentroid();
				mSquaredErrors[j] = bi.getSquaredError();
			}
		}catch(Exception e){
			e.printStackTrace();
		}
		
		mExecutorPool.shutdown();
	}
	
	private class BiCTask implements Callable<BiClustering>{
		private Instances ins;
		private BiClustering bic;
		public BiCTask(Instances ins){
			this.ins = ins;
			bic = new BiClustering();
		}
		@Override
		public BiClustering call() throws Exception {
			// TODO Auto-generated method stub
			bic.buildClusterer(ins);
			return bic;
		}
		
	}
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		// TODO Auto-generated method stub
		return fcm.distributionForInstance(instance);
	}
	public double getSquaredError() {
		return Utils.sum(mSquaredErrors);
	}
	public double getFcmSquaredError() {
		return Utils.sum(mFcmSquaredErrors);
	}
	
	public Instance[] getCentroids() {
		return mCentroids;
	}
	public Instances getFcmCentroids() {
		return mFcmCentroids;
	}
	@Override
	public Enumeration<Option> listOptions() {
		// TODO Auto-generated method stub
		return fcm.listOptions();
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		// TODO Auto-generated method stub
		fcm.setOptions(options);
	}

	@Override
	public String[] getOptions() {
		// TODO Auto-generated method stub
		return fcm.getOptions();
	}

	@Override
	public void setSeed(int value) {
		// TODO Auto-generated method stub
		fcm.setSeed(value);
	}

	@Override
	public int getSeed() {
		// TODO Auto-generated method stub
		return fcm.getSeed();
	}

	@Override
	public Capabilities getCapabilities() {
		// TODO Auto-generated method stub
		return fcm.getCapabilities();
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
		if (mCentroids == null) {
			return "No clusterer built yet!";
		}

		int maxWidth = 0;
		int maxAttWidth = 0;
		boolean containsNumeric = false;
		for (int i = 0; i < mNumCluster; i++) {
			for (int j = 0; j < mCentroids[i].numAttributes(); j++) {
				if (mCentroids[i].attribute(j).name().length() > maxAttWidth) {
					maxAttWidth = mCentroids[i].attribute(j).name()
							.length();
				}
				if (mCentroids[i].attribute(j).isNumeric()) {
					containsNumeric = true;
					double width = Math.log(Math.abs(mCentroids[i].value(j))) / Math.log(10.0);

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

		for (int i = 0; i < mCentroids[i].numAttributes(); i++) {
			if (mCentroids[i].attribute(i).isNominal()) {
				Attribute a = mCentroids[i].attribute(i);
				String val = a.value((int) mCentroids[i].value(i));
				if (val.length() > maxWidth) {
					maxWidth = val.length();
				}
				for (int j = 0; j < a.numValues(); j++) {
					String valu = a.value(j) + " ";
					if (valu.length() > maxAttWidth) {
						maxAttWidth = valu.length();
					}
				}
			}
		}
		if ( maxAttWidth < "missing".length()) {
			maxAttWidth = "missing".length();
		}

		String plusMinus = "+/-";
		maxAttWidth += 2;
		if (containsNumeric) {
			maxWidth += plusMinus.length();
		}
		if (maxAttWidth < "Attribute".length() + 2) {
			maxAttWidth = "Attribute".length() + 2;
		}

		StringBuffer temp = new StringBuffer();
		temp.append("\nBCFCM\n======\n");
		temp.append("\nNumber of iterations: " + fcm.m_Iterations);

		temp.append("Sum of within cluster distances: "+ Utils.sum(mSquaredErrors));
		
		
		temp.append("\n\nFinal cluster centroids:\n");
		temp.append(pad("Cluster#", " ", (maxAttWidth + (maxWidth * 2 + 2))
				- "Cluster#".length(), true));

		temp.append("\n");
		temp.append(pad("Attribute", " ", maxAttWidth - "Attribute".length(),
				false));

		temp.append(pad("Full Data", " ", maxWidth + 1 - "Full Data".length(),
				true));

		// cluster numbers
		for (int i = 0; i < mNumCluster; i++) {
			String clustNum = "" + i;
			temp.append(pad(clustNum, " ", maxWidth + 1 - clustNum.length(),
					true));
		}
		temp.append("\n");

		temp.append(pad("", "=",maxAttWidth+ (maxWidth * (mCentroids.length + 1)
								+ mCentroids.length + 1), true));
		temp.append("\n");
		temp.append("\n");
		return temp.toString();
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
		FastFCM fcm = new FastFCM();
		
		runClusterer(new BCFCM(fcm), args);
	}
}