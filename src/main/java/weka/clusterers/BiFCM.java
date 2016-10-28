package weka.clusterers;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.core.matrix.Matrix;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class BiFCM {

	/** 属性的最少个数 */
	private int minAttributes = 2;
	/** instance的最少个数 */
	private int minInstances = 2;
	/** 目标函数值改变量范围 */
	private double endValue = 1e-3;

	private double mValue = 0;
	private double alpha = 1.4;

	private Instances instances;
	private int mRows = 0;
	private int mCols = 0;
	private double means=0.0;
	
	private Instances[] array;
	private int plainRows=0;
	private int plainCols=0;
	private Instances plain;
	private Instances centroids;
	private Matrix matrix;

	private int mIterations = 0;
	private int maxIterations = 500;
	protected ReplaceMissingValues m_ReplaceMissingFilter;
	protected NominalToBinary m_NominalToBinary;
	protected boolean m_dontReplaceMissing = false;

	public void loadData(String fileName) {
		ArffLoader loader = new ArffLoader();
		try {
			loader.setFile(new File(fileName));
			plain = loader.getDataSet();
		} catch (IOException e) {
			e.printStackTrace();
			plain = null;
		}
	}

	public void multipleNodeDeletion() {
		instances=new Instances(plain);
		instances.deleteAttributeAt(instances.numAttributes()-1);
		m_ReplaceMissingFilter = new ReplaceMissingValues();
		instances.setClassIndex(-1);
		
		int count;
		array=new Instances[3];
		array[0]=new Instances(plain,0);
		array[1]=new Instances(plain,0);
	

		plainRows=instances.numInstances();
		plainCols=instances.numAttributes();
		int rows,cols;
		rows=cols=0;
		while (true) {
			mIterations++;
			System.out.println("mIterations = "+mIterations);
			count=0;
			mRows=rows = instances.numInstances();
			mCols=cols = instances.numAttributes();
		
			/*compute H(I,J)*/
			means=calculateMeans();
			mValue = calculateH();
			
			System.out.println("mValue = "+ mValue+" mRows = "+mRows+" mCols = "+mCols);
			
			if (mValue <= endValue) {
				System.out.println("找到了最小值");
				break;
			}
			if(rows<=minInstances||cols<=minAttributes){
				break;
			}
			
			double[] mScores = new double[mRows];
		
			for (int i = 0; i <  mRows; i++) {
				double iJ=calculate_iJ(i);
				mScores[i] = calculateInstances(i,iJ);
			}

			for (int j = 0; j <  mRows; j++) {
				if (mScores[j] > alpha * mValue) {
					if(plainCols==mCols){
						array[0].add(plain.instance(j-count));
					}else{
						array[1].add(plain.instance(j-count));
					}
					plain.delete(j-count);
					instances.delete(j - count);
					++count;
				}
			}
			mScores=null;
			count = 0;
	
			mRows=rows = instances.numInstances();
			mCols=cols = instances.numAttributes();
			/*compute H(I,J)*/
			means = calculateMeans();
			mValue = calculateH();
		
			System.out.println("mutiple deletion rows\nmValue = "+ mValue+" mRows = "+mRows+" mCols = "+mCols);
			
			if(rows<minInstances||cols<minAttributes){
				break;
			}
			mScores = new double[mCols];
			for (int j = 0; j < mCols; j++) {
				double Ij=calculate_Ij(j);
				mScores[j] = calculateAttributes(j,Ij);
			}
			
			for (int j = 0; j < mCols; j++) {
				if (mScores[j] > alpha * mValue) {
					instances.deleteAttributeAt(j - count);
					++count;
				}
			}
			mRows=rows = instances.numInstances();
			mCols=cols = instances.numAttributes();
			/*compute H(I,J)*/
			means = calculateMeans();
			mValue = calculateH();
			
			System.out.println("mutiple deletion cols\nmValue = "+ mValue+" mRows = "+mRows+" mCols = "+mCols);
			
			if (mValue <= endValue) {
				System.out.println("找到了最小值");
				break;
			}

			if (rows == mRows && cols == mCols) {
				System.out.println("使用单节点删除");
				singleNodeDeletion();
			}


			if (mIterations == maxIterations) {
				System.out.println("到达最大循环数");
				break;
			}
			System.out.println("==============================");
		}
		array[2]=new Instances(plain);
		instances=null;
		toCentroids();
		evalution();
		initMemberShipAndDatas();
//		saveIns();
	}

	public void singleNodeDeletion() {

		double[] mScores = new double[instances.numInstances()];
		int maxRowIndex = 0;
		int maxColIndex = 0;
		double[] maxH = new double[2];

		for (int i = 0; i <  instances.numInstances(); i++) {
			double iJ=calculate_iJ(i);
			mScores[i] = calculateInstances(i,iJ);
		}
		maxRowIndex = Utils.maxIndex(mScores);
		maxH[0] = mScores[maxRowIndex];

		mScores = new double[instances.numAttributes()];
		for (int j = 0; j <  instances.numAttributes(); j++) {
			double Ij=calculate_Ij(j);
			mScores[j] = calculateAttributes(j,Ij);
		}
		maxColIndex = Utils.maxIndex(mScores);
		maxH[1] = mScores[maxColIndex];

		if (maxH[0] > maxH[1]) {
			if(plainCols==instances.numAttributes()){
				array[0].add(plain.instance(maxRowIndex));
			}else{
				array[1].add(plain.instance(maxRowIndex));
			}
			plain.delete(maxRowIndex);
			instances.delete(maxRowIndex);
		} else {
			instances.deleteAttributeAt(maxColIndex);
		}
	}

	public double calculateInstances(int row,double iJ) {
		double score = 0;
			for (int j = 0; j < instances.numAttributes(); j++) {
				double Ij= calculate_Ij(j);
				score += Math.pow(instances.instance(row).value(j) - iJ - Ij+ means, 2);
			}
		score /= instances.numAttributes();
		return score;
	}

	public double calculateAttributes(int col,double Ij) {
		double score = 0;
		for (int i = 0; i < instances.numInstances(); i++) {
			double iJ = calculate_iJ(i);
			score += Math.pow(instances.instance(i).value(col) - iJ - Ij + means, 2);
		}
		score /= instances.numInstances();
		return score;
	}
	
	public double calculate_iJ(int row){
		double iJ=0.0;
		for (int j = 0; j < instances.numAttributes(); j++) {
			iJ += instances.instance(row).value(j);
		}
		iJ /= instances.numAttributes();
		return iJ;
	}

	public double calculate_Ij(int col){
		double Ij=0.0;
		for (int i = 0; i < instances.numInstances(); i++) {
			Ij += instances.instance(i).value(col);
		}
		Ij /= instances.numInstances();
		return Ij;
	}
	public double calculateH() {
		double score = 0;
		for (int i = 0; i < instances.numInstances(); i++) {
			for (int j = 0; j < instances.numAttributes(); j++) {
				double[] values = calculateIJ( i, j);
				score += Math.pow(instances.instance(i).value(j) - values[0] - values[1] + means, 2);
			}
		}
		score /= (instances.numAttributes() * instances.numInstances());
		return score;
	}

	public double calculateMeans() {
		double mean = 0.0;
		for (int i = 0; i < instances.numInstances(); i++) {
			for (int j = 0; j < instances.numAttributes(); j++) {
				mean += instances.instance(i).value(j);
			}
		}
		mean /= instances.numAttributes() * instances.numInstances();
		return mean;
	}
	public double[] calculateIJ(int row, int col) {
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
	
	public Instances getCentroids(){
		return centroids;
	}
	public Instances getData(){
		Instances ins=new Instances(plain,0);
		ins.addAll(array[0]);
		ins.addAll(array[1]);
		ins.addAll(array[2]);
//		ins.deleteAttributeAt(ins.numAttributes()-1);
		return ins;
	}
	public Matrix getMatrix(){
		return matrix;
	}

	public String toString() {
		// TODO Auto-generated method stub
		StringBuilder temp = new StringBuilder();
		temp.append("\nBiCluster\n======\n");
		temp.append("Number of iterations: " + mIterations);
		temp.append("\nthe value is: " + mValue);
		
		temp.append("\narray[0] length: "+array[0].numInstances());
		temp.append("\narray[1] length: "+array[1].numInstances());
		temp.append("\narray[2] length: "+array[2].numInstances());
/*
		temp.append("\ninstances");
		temp.append("\n"+instances.toString());
		temp.append("\n\n");
*/
		System.out.println(temp.toString());
		return temp.toString();
	}

	public void toCentroids(){
		centroids=new Instances(plain,array.length);
		System.out.println("聚类中心");
		for(int i=0; i<array.length; i++){
			Instance in=new DenseInstance(array[0].numAttributes());
			for(int j=0; j<array[0].numAttributes(); j++){
				in.setValue(j,array[i].meanOrMode(j));
				System.out.print(in.value(j)+" ");
			}
			System.out.println();
			centroids.add(in);
		}
	}
	public void saveIns(){
		ArffSaver xs =new ArffSaver();
		Instances ins=new Instances(array[0]);
		ins.addAll(array[1]);
		ins.addAll(array[2]);
		xs.setInstances(ins);
		try {
			xs.setDestination(new File("/home/ucas/see-dataset.arff"));
			xs.writeBatch();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
	}
	public void evalution(){
		int[] countOne=new int[3];
		int[] countTwo = new int[3];
		int[] countThree = new int [3];
		int[] classNum =new int[3];
	
			for(Instance in : array[0]){
				String value=in.stringValue(in.numAttributes()-1);
				if("优秀".equals(value)){
					++countOne[0];
				}
				else if("良好".equals(value)){
					++countOne[1];
				}
				else if ("中等".equals(value)){
					++countOne[2];
				}
				classNum[0]=countOne[Utils.maxIndex(countOne)];
				
			}

			for(Instance in : array[1]){
				String value=in.stringValue(in.numAttributes()-1);
				if("优秀".equals(value)){
					++countTwo[0];
				}
				else if("良好".equals(value)){
					++countTwo[1];
				}
				else if ("中等".equals(value)){
					++countTwo[2];
				}
				classNum[1]=countTwo[Utils.maxIndex(countTwo)];
				
			}
			
			for(Instance in : array[2]){
				String value=in.stringValue(in.numAttributes()-1);
				if("优秀".equals(value)){
					++countThree[0];
				}
				else if("良好".equals(value)){
					++countThree[1];
				}
				else if ("中等".equals(value)){
					++countThree[2];
				}
				classNum[2]=countThree[Utils.maxIndex(countThree)];
				
			}
		System.out.println(classNum[0]);
		for(int i: countOne){
			System.out.print(i+" ");
		}
		System.out.println();
		System.out.println(classNum[1]);
		for(int i: countTwo){
			System.out.print(i+" ");
		}
		System.out.println();
		System.out.println(classNum[2]);
		for(int i: countThree){
			System.out.print(i+" ");
		}
		System.out.println();
	
	}
	public void initMemberShipAndDatas(){
		instances=new Instances(plain,0);
	
		for(Instances ins: array){
			for(Instance in : ins){
				instances.add(in);
			}
		}
		matrix=new Matrix(instances.numInstances(),array.length);
				
		Random rand =new Random();
//		rand.setSeed(50);
//		double sum=0.0d;

		for(int i=0; i<array[0].numInstances(); i++){
			double one=8+rand.nextDouble();
			double two=rand.nextDouble();
			double three=rand.nextDouble();
			double sum=one+two+three;
			matrix.set(i, 0,one/sum);
			matrix.set(i, 1, two/sum);
			matrix.set(i, 2, three/sum);
		}
		for(int i=0; i<array[1].numInstances(); i++){
			double one=0;
			double two=8+rand.nextDouble();
			double three=6+rand.nextDouble();
			double sum=one+two+three;
			matrix.set(i, 0,three/sum);
			matrix.set(i, 1, one/sum);
			matrix.set(i, 2, two/sum);
		}
		for(int i=0; i<array[2].numInstances(); i++){
			double one=rand.nextDouble();
			double two=rand.nextDouble();
			double three=8+rand.nextDouble();
			double sum=one+two+three;
			matrix.set(i, 0,three/sum);
			matrix.set(i, 1, two/sum);
			matrix.set(i, 2, one/sum);
		}
	
/*		
		for(int i=0; i<array[0].numInstances(); i++){
			double one=0.8*rand.nextDouble();
			double two=0.4*rand.nextDouble();
			double three=0.1*rand.nextDouble();
			double sum=one+two+three;
			matrix.set(i, 0,one/sum);
			matrix.set(i, 1, two/sum);
			matrix.set(i, 2, three/sum);
		}
		for(int i=0; i<array[1].numInstances(); i++){
			double one=0.2*rand.nextDouble();
			double two=0.8*rand.nextDouble();
			double three=0.4*rand.nextDouble();
			double sum=one+two+three;
			matrix.set(i, 0,three/sum);
			matrix.set(i, 1, one/sum);
			matrix.set(i, 2, two/sum);
		}
		for(int i=0; i<array[2].numInstances(); i++){
			double one=0.1*rand.nextDouble();
			double two=0.3*rand.nextDouble();
			double three=0.9*rand.nextDouble();
			double sum=one+two+three;
			matrix.set(i, 0,three/sum);
			matrix.set(i, 1, two/sum);
			matrix.set(i, 2, one/sum);
		}
	*/	
	}


	public static void main(String[] args) {
		// TODO Auto-generated method stub
		BiFCM bi = new BiFCM();
		bi.loadData("/home/ucas/software/aluminium-electrolysis/CSV日报/CSV一厂房日报/101-102-r90-r88-less-attr.arff");
		bi.multipleNodeDeletion();
		bi.toString();
//		bi.loadData("/home/ucas/software/aluminium-electrolysis/CSV日报/CSV一厂房日报/101-102-r90-r88.arff");
		FuzzyC fc =new FuzzyC(bi.getMatrix());
//		FuzzyC fc =new FuzzyC();
		try {
			fc.buildClusterer(bi.getData());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		fc.toString();
	}
}
