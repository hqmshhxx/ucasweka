package weka.clusterers;

import java.util.Enumeration;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import weka.clusterers.NumberOfClustersRequestable;
import weka.clusterers.RandomizableClusterer;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.WeightedInstancesHandler;

public class BCFCM extends RandomizableClusterer implements
NumberOfClustersRequestable, WeightedInstancesHandler,
TechnicalInformationHandler{
	
	private FastFCM fcm;
	protected Instances mClusterCentroids;
	private Instances[] mClusters;
	private int mNumCluster = 3;
	/** Number of threads to run */
	protected int mExecutionSlots = 3;
	/** For parallel execution mode */
	protected transient ExecutorService mExecutorPool;
	
	public BCFCM (FastFCM fcm, BiClustering bic){
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
		fcm.buildClusterer(data);
		mClusters = fcm.getClusters();
		for(int i = 0; i < mNumCluster; i++){
			mExecutorPool.submit(new BiCTask(mClusters[i]));
		}
		
		mExecutorPool.shutdown();
	}
	
	private class BiCTask implements Callable<Boolean>{
		private Instances ins;
		private BiClustering bic;
		public BiCTask(Instances ins){
			this.ins = ins;
			bic = new BiClustering();
		}
		@Override
		public Boolean call() throws Exception {
			// TODO Auto-generated method stub
			bic.buildClusterer(ins);
			return null;
		}
		
	}
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		// TODO Auto-generated method stub
		return super.distributionForInstance(instance);
	}

	@Override
	public Enumeration<Option> listOptions() {
		// TODO Auto-generated method stub
		return super.listOptions();
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		// TODO Auto-generated method stub
		super.setOptions(options);
	}

	@Override
	public String[] getOptions() {
		// TODO Auto-generated method stub
		return super.getOptions();
	}

	@Override
	public void setSeed(int value) {
		// TODO Auto-generated method stub
		super.setSeed(value);
	}

	@Override
	public int getSeed() {
		// TODO Auto-generated method stub
		return super.getSeed();
	}

	@Override
	public Capabilities getCapabilities() {
		// TODO Auto-generated method stub
		return fcm.getCapabilities();
	}

	@Override
	public TechnicalInformation getTechnicalInformation() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void setNumClusters(int numClusters) throws Exception {
		// TODO Auto-generated method stub
		
	}

	@Override
	public int numberOfClusters() throws Exception {
		// TODO Auto-generated method stub
		return 0;
	}
	public static void main(String[] args){
		
	}
}