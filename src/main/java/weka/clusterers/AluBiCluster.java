package weka.clusterers;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class AluBiCluster implements Serializable{

	private static final long serialVersionUID = 1L;
	/** 属性的最少个数 */
	private int minAttNum= 8;
	/** instance的最少个数 */
	private int minInsNum = 1000;
	/** 目标函数值改变量范围 */
	private double endValue = 1e-3;
	private double alpha = 2;
	
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
	protected double mMeans;
	
	private Instances mLastIns;
	private Instance mCentroid;
	private double[] mSquaredErrors;
	private Instances delIns;
	private Instances mClusterCendroids;
	
	
//	private Future<Double> mFutureValue;
	private double mValue = 0;
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
	public void buildClusterer(Instances data) throws Exception {
		mReplaceMissingFilter = new ReplaceMissingValues();
		Instances instances = new Instances(data);
		delIns = new Instances(instances,0);
		mClusterCendroids = new Instances(instances,0);
		instances.setClassIndex(-1);
		if (!mDontReplaceMissing) {
			mReplaceMissingFilter.setInputFormat(instances);
			instances = Filter.useFilter(instances, mReplaceMissingFilter);
		}
		startExecutorPool();
		multipleNodeDeletion(instances);
		
		double[] centroidVal = calculateCentroid(mLastIns);
		mCentroid = new DenseInstance(1.0,centroidVal);
		mSquaredErrors = mLastIns.variances();
		mExecutorPool.shutdown();
		
		
	}
	public double[] calculateCentroid(Instances instances){
		double[] centroidVal = new double[instances.numAttributes()];
		for(int i=0; i < instances.numAttributes(); i++){
			 centroidVal[i] = instances.meanOrMode(i);
		}
		return centroidVal;
	}
	protected void multipleNodeDeletion(Instances instances) throws InterruptedException, ExecutionException{
		// rows and cols 是多点删除前时instans的行数和列数
		int rows = instances.numInstances();
		int cols = instances.numAttributes();
//		mFutureMeans = mExecutorPool.submit(new ComputeMeans(instances));
//		mValue = calculateH(instances);// 第一次运行multiDelRows需要用到mValue
		mMeans = calculateMeans(instances);
		mValue = calculateH(instances);
		while (true) {
			mIterations++;
			System.out.println("mIterations = "+mIterations);
			//delete rows
//			mFutureMeans = mExecutorPool.submit(new ComputeMeans(instances));
			
			instances = multiDelRows(instances);
			mMeans = calculateMeans(instances);
			mValue = calculateH(instances);//删除行后需要从新计算mValue
			System.out.println("mutiple deletion rows\nmValue = "+ mValue+" rows = "+instances.numInstances()+" cols = "+instances.numAttributes());

			if (mValue <= endValue) {
				System.out.println("找到了最小值");
				break;
			}
			if(rows<minInsNum){
				break;
			}
			//delete cols
//			mFutureMeans = mExecutorPool.submit(new ComputeMeans(instances));
/*			
			instances = multiDelCols(instances);
			mMeans = calculateMeans(instances);
			mValue = calculateH(instances);//删除列后需要重新计算mValue
			System.out.println("mutiple deletion cols\nmValue = "+ mValue+" rows = "+instances.numInstances()+" cols = "+instances.numAttributes());
			
			if (mValue <= endValue) {
				System.out.println("找到了最小值");
				break;
			}
			if(cols<minAttNum){
				break;
			}
*/			
			// 单点删除
			if (rows == instances.numInstances() && cols == instances.numAttributes()) {
				instances = singleNodeDeletion(instances);
				mMeans = calculateMeans(instances);
				mValue = calculateH(instances);//单点删除后需要重新计算mValue，多行删除时，需要次mValue
				System.out.println("single deletion\nmValue = "+ mValue+" rows = "+instances.numInstances()+" cols = "+instances.numAttributes());
				if (mValue <= endValue) {
					System.out.println("找到了最小值");
					break;
				}
				if(cols < minAttNum || rows < minInsNum){
					break;
				}
			}
		
			if (mIterations == maxIterations) {
				System.out.println("到达最大循环数");
				break;
			}
			
			cols = instances.numAttributes();
			rows = instances.numInstances();
			System.out.println("==============================");
		}
		mLastIns = instances;
		instances = null;
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
			Future<double[]> futureMR = mExecutorPool.submit(new MDRowsTask(instances, start, end, mMeans));
			results.add(futureMR);
		}
		try{
			for(Future<double[]> task : results){
				double[] scores = task.get();
				for(int i=0; i<scores.length; i++){
					if (scores[i] > alpha * mValue) {
						delIns.add(instances.instance(i));
						instances.instance(i).setValue(0, Double.NaN);
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
		private double mean;
		private double[] scores;
		
		public MDRowsTask(Instances instances, int start, int end, double mean){
			this.instances = instances;
			this.start = start;
			this.end = end;
			this.mean = mean;
			scores= new double[end - start];
		}
		@Override
		public double[] call() throws Exception {
			for (int i = start; i < end; i++) {
				double iJ = calculate_iJ(instances,i);
//				Future<Double> iJ = mExecutorPool.submit(new ComputeiJ(instances, i));
				scores[i-start] = calculateInstances(instances, i,iJ,mean);
			}
			return scores;
		}
	}
	
	protected Instances multiDelCols(Instances instances){
		int numPerTask = instances.numAttributes() / mExecutionSlots;
		List<Future<double[]>> results = new ArrayList<>();
		if(numPerTask == 0){
			results.add(mExecutorPool.submit(new MDColsTask(instances, 0, instances.numAttributes(), mMeans)));
		}else{
			for (int i = 0; i < mExecutionSlots; i++) {
				int start = i * numPerTask;
				int end = start + numPerTask;
				if (i == mExecutionSlots - 1) {
					end = instances.numAttributes();
				}
				Future<double[]> futureMR = mExecutorPool.submit(new MDColsTask(instances, start, end, mMeans));
				results.add(futureMR);
			}
		}
		try{
			double[] attVals = new double[instances.numAttributes()];
			int j=0;
			for(Future<double[]> task : results){
				double[] scores = task.get();
				for(int i=0; i<scores.length; i++){
					attVals[j++] = scores[i];
				}	
			}
			int delCount = 0;
			for(j = 0;j < attVals.length; j++){
				if (attVals[j] > alpha * mValue) {
					instances.deleteAttributeAt(j - delCount);
					delCount++;
				}
			}
		}catch(InterruptedException ie){
			ie.printStackTrace();
		}catch(ExecutionException ee){
			ee.printStackTrace();
		}
		return instances;
	}
	private class MDColsTask implements Callable<double[]>{
		private Instances instances;
		private int start;
		private int end;
		private double mean;
		private double[] scores;
		
		public MDColsTask(Instances instances, int start, int end, double mean){
			this.instances = instances;
			this.start = start;
			this.end = end;
			this.mean = mean;
			scores= new double[end - start];
		}
		@Override
		public double[] call() throws Exception {
			for (int i = start; i < end; i++) {
				double Ij = calculate_Ij(instances,i);
//				Future<Double> Ij = mExecutorPool.submit(new ComputeIj(instances, i));
				scores[i-start] = calculateAttributes(instances, i,Ij,mean);
			}
			return scores;
		}
		
	}
	protected Instances singleNodeDeletion(Instances instances) {
		Future<double[]> rowMax = mExecutorPool.submit(new MaxRowScores(instances, mMeans));
		try{
			double[] rowVal = rowMax.get();
			delIns.add(instances.instance((int)rowVal[0]));
			instances.delete((int)rowVal[0]);
		}catch(InterruptedException ie){
			ie.printStackTrace();
		}catch(ExecutionException ee){
			ee.printStackTrace();
		}
		return instances;
		
/*		
		Future<double[]> rowMax = mExecutorPool.submit(new MaxRowScores(instances, mMeans));
		Future<double[]> colMax = mExecutorPool.submit(new MaxColScores(instances, mMeans));
		try{
			double[] rowVal =  rowMax.get();
			double[] colVal = colMax.get();
			if (rowVal[1] > colVal[1]) {
				instances.delete((int)rowVal[0]);
			} else {
				instances.deleteAttributeAt((int)colVal[0]);
			}
		}catch(InterruptedException ie){
			ie.printStackTrace();
		}catch(ExecutionException ee){
			ee.printStackTrace();
		}
		return instances;
*/		
	}
	
	private class MaxRowScores implements Callable<double[]>{
		private Instances instances;
		private double mean;
		public MaxRowScores(Instances ins,double mean){
			instances = ins;
			this.mean = mean;
		}
		@Override
		public double[] call() throws Exception {
			int numPerTask = instances.numInstances() / mExecutionSlots;
			List<Future<double[]>> results = new ArrayList<>();
			for (int i = 0; i < mExecutionSlots; i++) {
				int start = i * numPerTask;
				int end = start + numPerTask;
				if (i == mExecutionSlots - 1) {
					end = instances.numInstances();
				}
				Future<double[]> futureMR = mExecutorPool.submit(new MDRowsTask(instances, start, end,mean));
				results.add(futureMR);
			}
			double[] maxRow = new double[2];
			int num = 0;
			Arrays.fill(maxRow, Double.MAX_VALUE);
			try{
				for(Future<double[]> task : results){
					double[] values = task.get();
					int index = Utils.maxIndex(values);
					if(values[index] > maxRow[1]){
						maxRow[1] = values[index];
						maxRow[0] = index + num;
					}
					num += values.length;
				}
			}catch(Exception e){
				e.printStackTrace();
			}
			return maxRow;
		}
		
	}
	
	private class MaxColScores implements Callable<double[]>{
		private Instances instances;
		private double mean;
		public MaxColScores(Instances ins, double mean){
			instances = ins;
			this.mean = mean;
		}
		@Override
		public double[] call() throws Exception {
			int numPerTask = instances.numAttributes() / mExecutionSlots;
			List<Future<double[]>> results = new ArrayList<>();
			if(numPerTask == 0){
				results.add(mExecutorPool.submit(new MDColsTask(instances, 0, instances.numAttributes(), mean)));
			}else{
				for (int i = 0; i < mExecutionSlots; i++) {
					int start = i * numPerTask;
					int end = start + numPerTask;
					if (i == mExecutionSlots - 1) {
						end = instances.numAttributes();
					}
					Future<double[]> futureMR = mExecutorPool.submit(new MDColsTask(instances, start, end, mean));
					results.add(futureMR);
				}
			}
			double[] maxCol = new double[2];
			int num = 0;
			Arrays.fill(maxCol, Double.MAX_VALUE);
			try{
				for(Future<double[]> task : results){
					double[] values = task.get();
					int index = Utils.maxIndex(values);
					if(values[index] > maxCol[1]){
						maxCol[1] = values[index];
						maxCol[0] = index + num;
					}
					num += values.length;
				}
			}catch(InterruptedException ie){
				ie.printStackTrace();
			}catch(ExecutionException ee){
				ee.printStackTrace();
			}
			return maxCol;
		}
	}

	protected double calculateInstances(Instances instances, int row, double iJ, double mean) {
		double score = 0;
		try{
			for (int j = 0; j < instances.numAttributes(); j++) {
				double Ij = calculate_Ij(instances,j);
//				Future<Double> Ij= mExecutorPool.submit(new ComputeIj(instances,j));
				score += Math.pow(instances.instance(row).value(j) - iJ - Ij + mean, 2);
			}
		}catch(Exception e){
			e.printStackTrace();
		}
		score /= instances.numAttributes();
		return score;
	}

	protected double calculateAttributes(Instances instances, int col, double Ij, double mean) {
		double score = 0;
		try{
			for (int i = 0; i < instances.numInstances(); i++) {
				double iJ = calculate_iJ(instances,i);
//				Future<Double>  iJ = mExecutorPool.submit(new ComputeiJ(instances,i));
				score += Math.pow(instances.instance(i).value(col) - iJ - Ij + mean, 2);
			}
		}catch(Exception e){
			e.printStackTrace();
		}
		score /= instances.numInstances();
		return score;
	}
	public double calculate_iJ(Instances instances,int row){
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

	public double calculate_Ij(Instances instances, int col){
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
	public double calculateH(Instances instances) {
		double score = 0;
		try{
			for (int i = 0; i < instances.numInstances(); i++) {
				for (int j = 0; j < instances.numAttributes(); j++) {
					double[] values = calculateIJ(instances, i, j);
					score += Math.pow(instances.instance(i).value(j) - values[0] - values[1] + mMeans, 2);
				}
			}
		}catch(Exception e){
			e.printStackTrace();
		}finally{
			System.out.println("Compute H over");
		}
		
		score /= (instances.numAttributes() * instances.numInstances());
		return score;
	}
/*	
	private class ComputeH implements Callable<Double>{
		private Instances instances;
		
		public ComputeH(Instances ins){
			this.instances = ins;
		}
		@Override
		public Double call() throws Exception {
			double score = 0;
			try{
				for (int i = 0; i < instances.numInstances(); i++) {
					for (int j = 0; j < instances.numAttributes(); j++) {
						double[] values = calculateIJ(instances, i, j);
						score += Math.pow(instances.instance(i).value(j) - values[0] - values[1] + mFutureMeans.get(), 2);
					}
				}
			}catch(Exception e){
				e.printStackTrace();
			}
			score /= (instances.numAttributes() * instances.numInstances());
			return score;
		}
	}
*/
	public double calculateMeans(Instances instances) {
		double mean = 0.0;
		for (int i = 0; i < instances.numInstances(); i++) {
			for (int j = 0; j < instances.numAttributes(); j++) {
				mean += instances.instance(i).value(j);
			}
		}
		mean /= instances.numAttributes() * instances.numInstances();
		
		System.out.println("Compute Means over");
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
			System.out.println("Compute Means over");
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
	public Instances getFullCentroids(){
		
		int cap = delIns.numInstances();
		Instances first = new Instances(delIns,0,cap/2);
		Instance firstCentroid = new DenseInstance(1.0,calculateCentroid(first));
		mClusterCendroids.add(firstCentroid);
		Instances second = new Instances(delIns,cap/2,cap-cap/2);
		Instance secondCentroid = new DenseInstance(1.0,calculateCentroid(second));
		mClusterCendroids.add(secondCentroid);
		mClusterCendroids.add(mCentroid);
		return mClusterCendroids;
	}
	public double getSquaredError() {
		return Utils.sum(mSquaredErrors);
	}
	public double getStandardDevs(){
		return Math.sqrt(getSquaredError());
	}
	public Instance getCentroid() {
		return mCentroid;
	}
	public Instances getCluster() {
		return mLastIns;
	}
	public Instances getLastIns(){
		return mLastIns;
	}
	public Instances getDelIns(){
		return delIns;
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
	@Override
	public String toString() {
		if (mCentroid == null) {
			return "No clusterer built yet!";
		}
		int maxWidth = 0;
		int maxAttWidth = 0;
		boolean containsNumeric = false;
		
		for (int j = 0; j < mCentroid.numAttributes(); j++) {
			if (mCentroid.attribute(j).name().length() > maxAttWidth) {
				maxAttWidth = mCentroid.attribute(j).name().length();
			}
			if (mCentroid.attribute(j).isNumeric()) {
				containsNumeric = true;
				double width = Math.log(Math.abs(mCentroid.value(j))) / Math.log(10.0);
				if (width < 0) {
					width = 1;
				}
				width += 6.0;
				if ((int) width > maxWidth) {
					maxWidth = (int) width;
				}
			}
		}
		for (int i = 0; i < mCentroid.numAttributes(); i++) {
			if (mCentroid.attribute(i).isNominal()) {
				Attribute a = mCentroid.attribute(i);
				String val = a.value((int) mCentroid.value(i));
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
		String plusMinus = "+/-";
		maxAttWidth += 2;
		if (containsNumeric) {
			maxWidth += plusMinus.length();
		}
		if (maxAttWidth < "Attribute".length() + 2) {
			maxAttWidth = "Attribute".length() + 2;
		}
		StringBuffer temp = new StringBuffer();
		temp.append("\nFastFCM\n======\n");
		temp.append("\nNumber of iterations: " + mIterations);
		temp.append("\n");
		temp.append("Sum of within cluster distances: "+ Utils.sum(mSquaredErrors));

		temp.append("\n\nFinal cluster centroids:\n");
		temp.append(pad("Cluster#", " ", (maxAttWidth + (maxWidth * 2 + 2))
				- "Cluster#".length(), true));

		temp.append("\n");
		temp.append(pad("Attribute", " ", maxAttWidth - "Attribute".length(),false));

		temp.append(pad("Full Data", " ", maxWidth + 1 - "Full Data".length(),true));
		// cluster numbers
		String clustNum = "" + 0;
		temp.append(pad("", " ", maxWidth + 1 - clustNum.length(),true));
		temp.append("\n");

		temp.append(pad("", "=",maxAttWidth+ (maxWidth * 2+ 3), true));
		temp.append("\n");
		for (int i = 0; i < mCentroid.numAttributes(); i++) {
			String attName = mCentroid.attribute(i).name();
			temp.append(attName);
			for (int j = 0; j < maxAttWidth - attName.length(); j++) {
				temp.append(" ");
			}
		}
		temp.append("\n\n");
		return temp.toString();
	}

	public static void main(String[] args) {
		
	}
}
