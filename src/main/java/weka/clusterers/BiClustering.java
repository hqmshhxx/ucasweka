package weka.clusterers;


import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class BiClustering extends AbstractClusterer
		implements WeightedInstancesHandler, TechnicalInformationHandler, OptionHandler {

	private static final long serialVersionUID = -8665636089245404045L;

	/** 属性的最少个数 */
	private int minAttNum= 5;
	/** instance的最少个数 */
	private int minInsNum = 50;
	/** 目标函数值改变量范围 */
	private double endValue = 1e-4;

	private double mValue = 0;
	private double alpha = 2;
	
	private int mRows = 0;
	private int mCols = 0;
	/**
	 * 整个训练集的均值
	 */
	private double mMeans = 0.0d;
	/**
	 * Keep track of the number of iterations completed before convergence. 迭代次数
	 */
	private int mIterations = 0;
	
	/**
	 * Maximum number of iterations to be executed. 最大迭代次数
	 */
	private int maxIterations = 500;
	protected ReplaceMissingValues mReplaceMissingFilter;
	protected boolean mDontReplaceMissing = false;
	
	protected Future<Double> mFutureMeans;
	
	/** Number of threads to run */
	protected int mExecutionSlots = 6;

	/** For parallel execution mode */
	protected transient ExecutorService mExecutorPool;
	
	/**
	 * Start the pool of execution threads
	 */
	protected void startExecutorPool() {
		if (mExecutorPool != null) {
			mExecutorPool.shutdownNow();
		}
		mExecutorPool = Executors.newFixedThreadPool(mExecutionSlots);
	}
	@Override
	public void buildClusterer(Instances data) throws Exception {
		// can clusterer handle the data?
		getCapabilities().testWithFail(data);

		mReplaceMissingFilter = new ReplaceMissingValues();
		Instances instances = new Instances(data);
		instances.setClassIndex(-1);
		if (!mDontReplaceMissing) {
			mReplaceMissingFilter.setInputFormat(instances);
			instances = Filter.useFilter(instances, mReplaceMissingFilter);
		}
		startExecutorPool();
		multipleNodeDeletion(instances);
		mExecutorPool.shutdown();
		
	}
	protected void multipleNodeDeletion(Instances instances) {
		int count;
		while (true) {
			mIterations++;
			System.out.println("mIterations = "+mIterations);
			count=0;
			mRows = instances.numInstances();
			mCols = instances.numAttributes();
		
			/*compute H(I,J)*/
//			mMeans=calculateMeans(instances);
			mValue = calculateH(instances);
			mFutureMeans = mExecutorPool.submit(new ComputeMeans(instances));
			
			System.out.println("single deletion\nmValue = "+ mValue+" mRows = "+mRows+" mCols = "+mCols);
			
			if (mValue <= endValue) {
				System.out.println("找到了最小值");
				break;
			}
			if(mRows<=minAttNum||mCols<=minAttNum){
				break;
			}
			instances = multiDelRows(instances);
			
			mRows = instances.numInstances();
			mCols = instances.numAttributes();
	
			/*compute H(I,J)*/
			mFutureMeans = mExecutorPool.submit(new ComputeMeans(instances));
			mValue = calculateH(instances);
		
			System.out.println("mutiple deletion mRows\nmValue = "+ mValue+" mRows = "+mRows+" mCols = "+mCols);
			
			if(mRows<minInsNum||mCols<minAttNum){
				break;
			}
			mScores = new double[mCols];
			for (int j = 0; j < instances.numAttributes(); j++) {
				double Ij=calculate_Ij(instances,j);
				mScores[j] = calculateAttributes(instances,j,Ij);
			}
			
			for (int j = 0; j < mCols; j++) {
				if (mScores[j] > alpha * mValue) {
					instances.deleteAttributeAt(j - count);
					++count;
				}
			}
			mRows = instances.numInstances();
			mCols = instances.numAttributes();
			/*compute H(I,J)*/
			mMeans = calculateMeans(instances);
			mValue = calculateH(instances);
			System.out.println("mutiple deletion mCols\nmValue = "+ mValue+" mRows = "+mRows+" mCols = "+mCols);

			if (mValue <= endValue) {
				System.out.println("找到了最小值");
				break;
			}

			if (mRows == instances.numInstances() && mCols == instances.numAttributes()) {
				singleNodeDeletion(instances);
			}

			if (mIterations == maxIterations) {
				System.out.println("到达最大循环数");
				break;
			}
			System.out.println("==============================");
		}
	}
	protected Instances multiDelRows(Instances instances){
		int numPerTask = instances.numInstances() / mExecutionSlots;
		List<Future<double[]>> results = new ArrayList<>();
		for (int i = 0; i < mExecutionSlots; i++) {
			int start = i * numPerTask;
			int end = start + numPerTask;
			if (i == mExecutionSlots - 1) {
				end = instances.numInstances();
			}
			Future<double[]> futureMR = mExecutorPool.submit(new MDRowsTask(instances, start, end));
			results.add(futureMR);
		}
		try{
			for(Future<double[]> task : results){
				double[] scores = task.get();
				
				for(int i=0; i<scores.length; i++){
					if (scores[i] > alpha * mValue) {
						instances.instance(i).setValue(0, null);
					}
				}
			}
		}catch(Exception e){
			e.printStackTrace();
		}
		instances.deleteWithMissing(0);
		return instances;
	}
	private class MDRowsTask implements Callable<double[]>{
		private Instances instances;
		private int start;
		private int end;
		private double[] scores= new double[end - start];
		
		public MDRowsTask(Instances instances, int start, int end){
			this.instances = instances;
			this.start = start;
			this.end = end;
		}
		@Override
		public double[] call() throws Exception {
			for (int i = start; i < end; i++) {
				Future<Double> iJ = mExecutorPool.submit(new ComputeiJ(instances, i));
				scores[i-start] = calculateInstances(instances, i,iJ);
			}
			return scores;
		}
	}
	
	protected Instances multiDelCols(Instances instances){
		int numPerTask = instances.numAttributes() / mExecutionSlots;
		List<Future<double[]>> results = new ArrayList<>();
		for (int i = 0; i < mExecutionSlots; i++) {
			int start = i * numPerTask;
			int end = start + numPerTask;
			if (i == mExecutionSlots - 1) {
				end = instances.numAttributes();
			}
			Future<double[]> futureMR = mExecutorPool.submit(new MDColsTask(instances, start, end));
			results.add(futureMR);
		}
		try{
			for(Future<double[]> task : results){
				double[] scores = task.get();
				for(int i=0; i<scores.length; i++){
					if (scores[i] > alpha * mValue) {
						instances.instance(i).setValue(0, null);
					}
				}
			}
		}catch(Exception e){
			e.printStackTrace();
		}
		instances.deleteWithMissing(0);
		return instances;
	}
	private class MDColsTask implements Callable<double[]>{
		private Instances instances;
		private int start;
		private int end;
		private double[] scores= new double[end - start];
		
		public MDColsTask(Instances instances, int start, int end){
			this.instances = instances;
			this.start = start;
			this.end = end;
		}
		@Override
		public double[] call() throws Exception {
			for (int i = start; i < end; i++) {
				Future<Double> Ij = mExecutorPool.submit(new ComputeIj(instances, i));
				scores[i-start] = calculateAttributes(instances, i,Ij);
			}
			return scores;
		}
		
	}
	protected void singleNodeDeletion(Instances instances) {

		double[] mScores = new double[instances.numInstances()];
		int maxRowIndex = 0;
		int maxColIndex = 0;
		double[] maxH = new double[2];

		for (int i = 0; i <  instances.numInstances(); i++) {
			double iJ=calculate_iJ(instances,i);
			mScores[i] = calculateInstances(instances,i,iJ);
		}
		maxRowIndex = Utils.maxIndex(mScores);
		maxH[0] = mScores[maxRowIndex];

		mScores = new double[instances.numAttributes()];
		for (int j = 0; j <  instances.numAttributes(); j++) {
			double Ij=calculate_Ij(instances,j);
			mScores[j] = calculateAttributes(instances,j,Ij);
		}
		maxColIndex = Utils.maxIndex(mScores);
		maxH[1] = mScores[maxColIndex];

		if (maxH[0] > maxH[1]) {
			instances.delete(maxRowIndex);
		} else {
			instances.deleteAttributeAt(maxColIndex);
		}
	}

	protected double calculateInstances(Instances instances, int row,Future<Double> iJ) {
		double score = 0;
		try{
			for (int j = 0; j < instances.numAttributes(); j++) {
				Future<Double> Ij= mExecutorPool.submit(new ComputeIj(instances,j));
				score += Math.pow(instances.instance(row).value(j) - iJ.get() - Ij.get() + mFutureMeans.get(), 2);
			}
		}catch(Exception e){
			e.printStackTrace();
		}
		score /= instances.numAttributes();
		return score;
	}

	protected double calculateAttributes(Instances instances, int col, Future<Double> Ij) {
		double score = 0;
		try{
			for (int i = 0; i < instances.numInstances(); i++) {
				Future<Double>  iJ = mExecutorPool.submit(new ComputeiJ(instances,i));
				score += Math.pow(instances.instance(i).value(col) - iJ.get() - Ij.get() + mFutureMeans.get(), 2);
			}
		}catch(Exception e){
			e.printStackTrace();
		}
		score /= instances.numInstances();
		return score;
	}
	
	protected double calculate_iJ(Instances instances, int row){
		double iJ=0.0;
		for (int j = 0; j < instances.numAttributes(); j++) {
			iJ += instances.instance(row).value(j);
		}
		iJ /= instances.numAttributes();
		return iJ;
	}
	private class ComputeiJ implements Callable<Double>{
		private Instances instances;
		private int row;
		
		public ComputeiJ(Instances ins, int row){
			this.instances = ins;
			this.row = row;
		}
		@Override
		public Double call() throws Exception {
			double iJ=0.0;
			for (int j = 0; j < instances.numAttributes(); j++) {
				iJ += instances.instance(row).value(j);
			}
			iJ /= instances.numAttributes();
			return iJ;
		}
	}

	protected double calculate_Ij(Instances instances, int col){
		double Ij=0.0;
		
		for (int i = 0; i < instances.numInstances(); i++) {
			Ij += instances.instance(i).value(col);
		}
		Ij /= instances.numInstances();
		return Ij;
	}
	private class ComputeIj implements Callable<Double>{
		private Instances instances;
		private int col;
		
		public ComputeIj(Instances ins, int col){
			this.instances = ins;
			this.col = col;
		}
		@Override
		public Double call() throws Exception {
			double Ij=0.0;
			for (int i = 0; i < instances.numInstances(); i++) {
				Ij += instances.instance(i).value(col);
			}
			Ij /= instances.numInstances();
			return Ij;
		}
	}
	protected double calculateH(Instances instances) {
		double score = 0;
		for (int i = 0; i < instances.numInstances(); i++) {
			for (int j = 0; j < instances.numAttributes(); j++) {
				double[] values = calculateIJ(instances, i, j);
				score += Math.pow(instances.instance(i).value(j) - values[0] - values[1] + mMeans, 2);
			}
		}
		score /= (instances.numAttributes() * instances.numInstances());
		return score;
	}
	private class ComputeH implements Callable<Double>{
		private Instances instances;
		
		public ComputeH(Instances ins){
			this.instances = ins;
		}
		@Override
		public Double call() throws Exception {
			double score = 0;
			for (int i = 0; i < instances.numInstances(); i++) {
				for (int j = 0; j < instances.numAttributes(); j++) {
					double[] values = calculateIJ(instances, i, j);
					score += Math.pow(instances.instance(i).value(j) - values[0] - values[1] + mMeans, 2);
				}
			}
			score /= (instances.numAttributes() * instances.numInstances());
			return score;
		}
	}

	protected double calculateMeans(Instances instances) {
		double mean = 0.0;
		for (int i = 0; i < instances.numInstances(); i++) {
			for (int j = 0; j < instances.numAttributes(); j++) {
				mean += instances.instance(i).value(j);
			}
		}
		mean /= instances.numAttributes() * instances.numInstances();
		return mean;
	}
	private class ComputeMeans implements Callable<Double>{
		private Instances instances;
		
		public ComputeMeans(Instances ins){
			this.instances = ins;
		}
		@Override
		public Double call() throws Exception {
			double mean = 0.0;
			for (int i = 0; i < instances.numInstances(); i++) {
				for (int j = 0; j < instances.numAttributes(); j++) {
					mean += instances.instance(i).value(j);
				}
			}
			mean /= instances.numAttributes() * instances.numInstances();
			return mean;
		}
	}
	protected double[] calculateIJ(Instances instances, int row, int col) {
		double scoreI = 0;
		double scoreJ = 0;
		double[] values = new double[2];
		int rowNum = instances.numInstances();
		int colNum = instances.numAttributes();
		
		for (int i = 0; i < instances.numInstances(); i++) {
			scoreI += instances.instance(i).value(col);
		}
		
		for (int j = 0; j < instances.numAttributes(); j++) {
			scoreJ += instances.instance(row).value(j);
		}
	
		values[0] += scoreI * 1.0 / rowNum;
		values[1] += scoreJ * 1.0 / colNum;
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
