package weka.clusterers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;

import weka.classifiers.rules.DecisionTableHashKey;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DenseInstance;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.matrix.Matrix;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class FuzzyCPlusPlus extends RandomizableClusterer
		implements NumberOfClustersRequestable, WeightedInstancesHandler, TechnicalInformationHandler {

	/** for serialization. */
	private static final long serialVersionUID = 159802830430835116L;

	
	/**
	 * number of clusters to generate. 产生聚类个数
	 */
	protected int m_NumClusters = 3;

	/**
	 * Holds the initial start points, as supplied by the initialization method
	 * used
	 */
	protected Instances m_initialStartPoints;

	/**
	 * holds the cluster centroids. 聚类中心
	 */
	protected Instances m_ClusterCentroids;

	/**
	 * Preserve order of instances.
	 */
	protected boolean m_PreserveOrder = true;

	/**
	 * Holds the standard deviations of the numeric attributes in each cluster.
	 * 每个聚类的标准差
	 */
	protected Instances m_ClusterStdDevs;

	/**
	 * the distance function used. 距离函数,欧几里得距离
	 */
	protected DistanceFunction m_DistanceFunction = new EuclideanDistance();

	protected double[] m_FullStdDevs;
	/**
	 * Display standard deviations for numeric atts.
	 */
	protected boolean m_displayStdDevs;

	/**
	 * The number of instances in each cluster. 每个聚类包含的实例个数
	 */
	protected double[] m_ClusterSizes;

	/**
	 * Keep track of the number of iterations completed before convergence. 迭代次数
	 */
	protected int m_Iterations = 0;

	/**
	 * Maximum number of iterations to be executed. 最大迭代次数
	 */
	protected int m_MaxIterations = 500;

	/**
	 * Holds the squared errors for all clusters. 平方误差
	 */
	protected double[] m_squaredErrors;

	/**
	 * objective function result 目标函数值
	 */
	private double m_OFR = 100;

	/**
	 * a small value used to verify if clustering has converged. 目标函数值改变量范围
	 */
	private double m_EndValue = 1e-3;

	/**
	 * uij is the degree of membership of xi in the cluster j
	 */
	private Matrix memberShip;
	
	private Instances instances;
	
	private Instances plainData;


	/**
	 * holds the fuzzifier 模糊算子(加权指数)
	 */
	private double m_fuzzifier = 2;

	/**
	 * Assignments obtained.(cluster indexes).
	 * 
	 */
	protected int[] m_Assignments = null;
	
	protected Instances[] clusters;
	protected Instances[] plainClusters;
	/**
	 * assume that a cluster has 68% probability
	 */
	private double probCluster = 0.68;

	public FuzzyCPlusPlus() {
		super();
		m_SeedDefault = 10;
//		setSeed(m_SeedDefault);
	}

	public FuzzyCPlusPlus(Instance ins) {
		super();
		m_SeedDefault = 10;
		setSeed(m_SeedDefault);
//		m_NumClusters = matrix.getColumnDimension();
//		memberShip=matrix;
		
	
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		// TODO Auto-generated method stub
		double[] d = new double[m_NumClusters];
		double top = 0, bottom = 0, sum;
		for (int j = 0; j < m_NumClusters; j++) {
			top = m_DistanceFunction.distance(instance, m_ClusterCentroids.instance(j));
			sum = 0;
			for (int k = 0; k < m_NumClusters; k++) {
				bottom = m_DistanceFunction.distance(instance, m_ClusterCentroids.instance(k));
				sum += Math.pow(top / bottom, 2f / (m_fuzzifier - 1));
			}
			d[j] = 1f / sum;
		}
		return d;
	}

	/**
	 * Generates a clusterer. Has to initialize all fields of the clusterer that
	 * are not being set via options.
	 * 
	 * @param data
	 *            set of instances serving as training data
	 * @throws Exception
	 *             if the clusterer has not been generated successfully
	 */
	@Override
	public void buildClusterer(Instances data) throws Exception{

		m_Iterations = 0;
	
		instances = new Instances(plainData);
		instances.deleteAttributeAt(instances.numAttributes()-1);
		getCapabilities().testWithFail(instances);
		m_ClusterSizes = new double[m_NumClusters];
		m_Assignments = new int[instances.numInstances()];

	
		if (m_displayStdDevs) {
			m_FullStdDevs = instances.variances();
		}
//		m_FullMeansOrMediansOrModes = calculateMeansOrMediansOrModes(0, instances, true);

	
//		double sumOfWeights = instances.sumOfWeights();
		for (int i = 0; i < instances.numAttributes(); i++) {
			if (instances.attribute(i).isNumeric()) {
				if (m_displayStdDevs) {
					m_FullStdDevs[i] = Math.sqrt(m_FullStdDevs[i]);
				}
			}
		}
		m_DistanceFunction.setInstances(instances);
		if(m_ClusterCentroids==null){
			System.out.println("初始化聚类中心为前三个点");
			 m_ClusterCentroids = new Instances(instances, m_NumClusters);
			for (int i = 0; i < m_NumClusters; i++) {
				Instance cent = new DenseInstance(instances.instance(i));
				m_ClusterCentroids.add(cent);
			}
		}
		

		m_squaredErrors = new double[m_NumClusters];

		Matrix oldMatrix = new Matrix(instances.numInstances(), m_NumClusters);
		double difference = 0.0;
		initMemberShip(instances);
		do {
			saveMembershipMatrix(instances, oldMatrix);
			updateCentroid(instances);
			updateMemberShip(instances);
			difference = calculateMaxMembershipChange(instances, oldMatrix);
		} while (difference > m_EndValue && ++m_Iterations < m_MaxIterations);
		// 更新m_Assignments;
		updateClustersInfo(instances);
		calDis();

		// save memory!
		m_DistanceFunction.clean();

	}

	private void saveMembershipMatrix(Instances instances, Matrix matrix) {
		matrix.setMatrix(0, matrix.getRowDimension() - 1, 0, matrix.getColumnDimension() - 1, memberShip);
	}

	private double calculateMaxMembershipChange(Instances instances, Matrix matrix) {
		double maxMembership = 0.0;
		for (int i = 0; i < instances.numInstances(); i++) {
			for (int j = 0; j < m_NumClusters; j++) {
				double v = Math.abs(memberShip.get(i, j) - matrix.get(i, j));
				maxMembership = Math.max(v, maxMembership);
			}
		}
		return maxMembership;
	}


	public void initMemberShip(Instances instances) {
		/* 初始化membership也就是uij */
		memberShip = new Matrix(instances.numInstances(), m_NumClusters);

		Random rand = new Random();
//		rand.setSeed(m_Seed);
		for (int i = 0; i < instances.numInstances(); i++) {
			double sum = 0d;
			for (int j = 0; j < m_NumClusters; j++) {
				double value = 0.01d + rand.nextDouble();
				memberShip.set(i, j, value);
				sum += value;
			}
			for (int j = 0; j < m_NumClusters; j++) {
				double value = memberShip.get(i, j) / sum;
				memberShip.set(i, j, value);
			}
		}
		/*
		 * for(int i=0;i<instances.numInstances(); i++){ for(int j=0;
		 * j<m_NumClusters; j++){ System.out.print(memberShip.get(i, j)+" "); }
		 * System.out.println(); }
		 */
	}



	private void updateCentroid(Instances instances) {
		double bottom;
		// Instances newCentroids=new
		// Instances(m_ClusterCentroids,m_NumClusters);

		for (int k = 0; k < m_NumClusters; k++) {
			bottom = 0.0d;
			double[] attributes = new double[instances.numAttributes()];
			Instance in = new DenseInstance(1.0, attributes);
			for (int i = 0; i < instances.numInstances(); i++) {
				double uValue = Math.pow(memberShip.get(i, k), m_fuzzifier);
				bottom += uValue;
				for (int j = 0; j < instances.numAttributes(); j++) {
					double attValue = in.value(j);
					attValue += uValue * instances.instance(i).value(j);
					in.setValue(j, attValue);
				}
			}
			for (int m = 0; m < in.numAttributes(); m++) {
				double attValue = in.value(m);
				in.setValue(m, attValue / bottom);
			}
			m_ClusterCentroids.set(k, in);
			// newCentroids.add(in);
		}
		// m_ClusterCentroids.clear();
		// m_ClusterCentroids=newCentroids;
	}

	private void updateMemberShip(Instances instances) {
		// double top, bottom, sum;
		for (int i = 0; i < instances.numInstances(); i++) {
			/*
			 * System.out.println("第"+i+"实例"); for(int u=0;
			 * u<instances.numAttributes();u++){
			 * System.out.print(instances.instance(i).value(u)+", "); }
			 * System.out.println();
			 */
			for (int j = 0; j < m_NumClusters; j++) {
				// top = 0.0;
				/*
				 * Instance ins= m_ClusterCentroids.instance(j);
				 * System.out.println("第"+j+"聚类中心"); for(int u=0;
				 * u<ins.numAttributes();u++){ System.out.print(ins.value(u)+
				 * ", "); } System.out.println();
				 */
				final double top = m_DistanceFunction.distance(instances.instance(i), m_ClusterCentroids.instance(j));
				double sum = 0.0;
				if (top != 0.0) {
					for (int k = 0; k < m_NumClusters; k++) {
						// bottom = 0;
						final double bottom = m_DistanceFunction.distance(instances.instance(i),
								m_ClusterCentroids.instance(k));
						if (bottom == 0.0) {
							sum = Double.POSITIVE_INFINITY;
							break;
						}
						sum += Math.pow(top / bottom, 2.0d / (m_fuzzifier - 1.0));
					}
				}
				double membership;
				if (sum == 0.0) {
					membership = 1.0;
				} else if (sum == Double.POSITIVE_INFINITY) {
					membership = 0.0;
				} else {
					membership = 1.0 / sum;
				}
				memberShip.set(i, j, membership);
			}

		}

	}

	private double calculateObjectiveFunction(Instances instances) {
		double sum, dist;
		sum = dist = 0;
		for (int i = 0; i < instances.numInstances(); i++) {
			for (int j = 0; j < m_NumClusters; j++) {
				dist = m_DistanceFunction.distance(instances.instance(i), m_ClusterCentroids.instance(j));
				sum += Math.pow(memberShip.get(i, j), m_fuzzifier) * Math.pow(dist, 2);
			}
		}
		return sum;
	}

	private void updateClustersInfo(Instances instances) {
		clusters = new Instances[m_NumClusters];
		plainClusters = new Instances[m_NumClusters];
		for (int i = 0; i < m_NumClusters; i++) {
			clusters[i] = new Instances(instances, 0);
			plainClusters[i] = new Instances(plainData,0);
		}
		if (m_displayStdDevs) {
			m_ClusterStdDevs = new Instances(instances, m_NumClusters);
		}
		// update m_Assignments and m_ClusterSizes m_squaredErrors
		for (int i = 0; i < instances.numInstances(); i++) {
			double max = 0;
			int index = 0;
			double dist = 0f;
			for (int j = 0; j < m_NumClusters; j++) {
				if (max < memberShip.get(i, j)) {
					max = memberShip.get(i, j);
					index = j;
				}
			}
			m_Assignments[i] = index;
			clusters[index].add(instances.instance(i));
			plainClusters[index].add(plainData.instance(i));
			m_ClusterSizes[index] += 1;
			
			Instance in=new DenseInstance(instances.instance(i));
			in.deleteAttributeAt(in.numAttributes()-1);

			dist = m_DistanceFunction.distance(in, m_ClusterCentroids.instance(index));
			m_squaredErrors[index] = dist * dist ;
		}

		// update m_ClusterStdDevs 
		for (int i = 0; i < m_NumClusters; i++) {
			if (m_displayStdDevs) {
				double[] vals2 = clusters[i].variances();
				for (int j = 0; j < instances.numAttributes(); j++) {
					if (instances.attribute(j).isNumeric()) {
						vals2[j] = Math.sqrt(vals2[j]);
					} else {
						vals2[j] = Utils.missingValue();
					}
				}
				m_ClusterStdDevs.add(new DenseInstance(1.0, vals2));
			}
		}
	}
	public void calDis(){
		double[] dis=new double[3];
		Instance center=new DenseInstance(1.0, calculateMeans(instances));
	
		Instances partIns=new Instances(instances,0);
	    for(int i=0; i<3;i++){
	    	partIns.addAll(clusters[(i+1)%3]);
	    	partIns.addAll(clusters[(i+2)%3]);
	    	Instance part=new DenseInstance(1.0,calculateMeans(partIns));
	    	dis[i] = m_DistanceFunction.distance(part, center);
	    	partIns.clear();
	    }
	    int minIndex=Utils.minIndex(dis);
	    plainData.clear();
	    
	    if(clusters[minIndex].numInstances()*1.0/instances.numInstances()<0.3){
  	    	plainData.addAll(plainClusters[minIndex]);
  	    }
	    for(int i=0;i<3;i++){
	    	if(i!=minIndex){
	    		plainData.addAll(plainClusters[i]);
	    	}
	    }
	}
	
	public Instances getNewData(){
		
		toString();
		System.out.println(plainData.size());
		return plainData;
	}
	
	public double[] calculateMeans(Instances members) {
		int size = members.numInstances();
		double[] sum = new double[members.numAttributes()];

		for (Instance ins : members) {
			double[] temp=ins.toDoubleArray();
			for(int i =0; i< temp.length; i++){
				sum[i]+=temp[i];
			}
		}
		for(int i=0; i<sum.length; i++){
			sum[i]/=size;
		}

		return sum;
	}
	
	public void setPlainData(Instances data){
		plainData=data;
	}
	
	public void statisticClass(){
		int[][] statis =new int[3][3];
		for(int i=0; i<3; i++){
			for(Instance in : plainClusters[i]){
				String value=in.stringValue(in.numAttributes()-1);
				if("优秀".equals(value)){
					++statis[i][0];
				}
				else if("良好".equals(value)){
					++statis[i][1];
				}
				else if ("中等".equals(value)){
					++statis[i][2];
				}
			}
		}
		for(int i=0; i<3; i++){
			for(int j=0; j<3; j++){
				System.out.print(statis[i][j]+" ");
			}
			System.out.println();
		}
		
	}
	

	/**
	 * Returns a string describing this clusterer.
	 * 
	 * @return a description of the evaluator suitable for displaying in the
	 *         explorer/experimenter gui
	 */
	public String globalInfo() {
		return "Cluster data using fuzzy k means algorithm";
	}

	/**
	 * Returns default capabilities of the clusterer.
	 * 
	 * @return the capabilities of this clusterer
	 */
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
	public TechnicalInformation getTechnicalInformation() {
		// TODO Auto-generated method stub
		TechnicalInformation result;

		result = new TechnicalInformation(Type.INPROCEEDINGS);
		result.setValue(Field.AUTHOR, "D. Arthur and S. Vassilvitskii");
		result.setValue(Field.TITLE, "k-means++: the advantages of carefull seeding");
		result.setValue(Field.BOOKTITLE,
				"Proceedings of the eighteenth annual " + "ACM-SIAM symposium on Discrete algorithms");
		result.setValue(Field.YEAR, "2007");
		result.setValue(Field.PAGES, "1027-1035");

		return result;
	}

	@Override
	public void setNumClusters(int n) throws Exception {
		// TODO Auto-generated method stub
		if (n <= 0) {
			throw new Exception("Number of clusters must be > 0");
		}
		m_NumClusters = n;

	}

	/**
	 * gets the number of clusters to generate.
	 * 
	 * @return the number of clusters to generate
	 */
	public int getNumClusters() {
		return m_NumClusters;
	}

	/**
	 * Returns the number of clusters.
	 * 
	 * @return the number of clusters generated for a training dataset.
	 * @throws Exception
	 *             if number of clusters could not be returned successfully
	 */
	@Override
	public int numberOfClusters() throws Exception {
		// TODO Auto-generated method stub
		return m_NumClusters;
	}

	/**
	 * set the maximum number of iterations to be executed.
	 * 
	 * @param n
	 *            the maximum number of iterations
	 * @throws Exception
	 *             if maximum number of iteration is smaller than 1
	 */
	public void setMaxIterations(int n) throws Exception {
		if (n <= 0) {
			throw new Exception("Maximum number of iterations must be > 0");
		}
		m_MaxIterations = n;
	}

	/**
	 * gets the number of maximum iterations to be executed.
	 * 
	 * @return the number of clusters to generate
	 */
	public int getMaxIterations() {
		return m_MaxIterations;
	}

	/**
	 * Sets whether standard deviations and nominal count. Should be displayed
	 * in the clustering output.
	 * 
	 * @param stdD
	 *            true if std. devs and counts should be displayed
	 */
	public void setDisplayStdDevs(boolean stdD) {
		m_displayStdDevs = stdD;
	}

	/**
	 * Gets whether standard deviations and nominal count. Should be displayed
	 * in the clustering output.
	 * 
	 * @return true if std. devs and counts should be displayed
	 */
	public boolean getDisplayStdDevs() {
		return m_displayStdDevs;
	}

	/**
	 * Sets whether order of instances must be preserved.
	 * 
	 * @param r
	 *            true if missing values are to be replaced
	 */
	public void setPreserveInstancesOrder(boolean r) {
		m_PreserveOrder = r;
	}

	/**
	 * Gets whether order of instances must be preserved.
	 * 
	 * @return true if missing values are to be replaced
	 */
	public boolean getPreserveInstancesOrder() {
		return m_PreserveOrder;
	}

	
	/**
	 * Gets the the cluster centroids.
	 * 
	 * @return the cluster centroids
	 */
	public Instances getClusterCentroids() {
		return m_ClusterCentroids;
	}

	/**
	 * Gets the standard deviations of the numeric attributes in each cluster.
	 * 
	 * @return the standard deviations of the numeric attributes in each cluster
	 */
	public Instances getClusterStandardDevs() {
		return m_ClusterStdDevs;
	}

	/**
	 * Gets the sum of weights for all the instances in each cluster.
	 * 
	 * @return The number of instances in each cluster
	 */
	public double[] getClusterSizes() {
		return m_ClusterSizes;
	}

	/**
	 * Gets the assignments for each instance.
	 * 
	 * @return Array of indexes of the centroid assigned to each instance
	 * @throws Exception
	 *             if order of instances wasn't preserved or no assignments were
	 *             made
	 */
	public int[] getAssignments() throws Exception {
		if (!m_PreserveOrder) {
			throw new Exception("The assignments are only available when order of instances is preserved (-O)");
		}
		if (m_Assignments == null) {
			throw new Exception("No assignments made.");
		}
		return m_Assignments;
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		// TODO Auto-generated method stub
		String optionString = Utils.getOption('N', options);

		if (optionString.length() != 0) {
			setNumClusters(Integer.parseInt(optionString));
		}

		optionString = Utils.getOption("I", options);
		if (optionString.length() != 0) {
			setMaxIterations(Integer.parseInt(optionString));
		}
		m_PreserveOrder = Utils.getFlag("O", options);
		super.setOptions(options);
		Utils.checkForRemainingOptions(options);
	}
	
	

	@Override
	public String[] getOptions() {
		// TODO Auto-generated method stub
		ArrayList<String> result = new ArrayList<>();
		if (m_displayStdDevs) {
			result.add("-V");
		}

		result.add("-N");
		result.add("" + getNumClusters());

		result.add("-A");
		result.add((m_DistanceFunction.getClass().getName() + " " + Utils.joinOptions(m_DistanceFunction.getOptions()))
				.trim());

		result.add("-I");
		result.add("" + getMaxIterations());

		if (m_PreserveOrder) {
			result.add("-O");
		}

		Collections.addAll(result, super.getOptions());

		return result.toArray(new String[result.size()]);
	}

	private String pad(String source, String padChar, int length, boolean leftPad) {
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
		if (m_ClusterCentroids == null) {
			return "No clusterer built yet!";
		}
		int maxWidth = 0;
		int maxAttWidth = 0;
		boolean containsNumeric = false;
		for (int i = 0; i < m_NumClusters; i++) {
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

	
		// check for size of cluster sizes
		for (double m_ClusterSize : m_ClusterSizes) {
			String size = "(" + m_ClusterSize + ")";
			if (size.length() > maxWidth) {
				maxWidth = size.length();
			}
		}

		if (m_displayStdDevs && maxAttWidth < "missing".length()) {
			maxAttWidth = "missing".length();
		}
		String plusMinus = "+/-";
		maxAttWidth += 2;
		if (m_displayStdDevs && containsNumeric) {
			maxWidth += plusMinus.length();
		}
		if (maxAttWidth < "Attribute".length() + 2) {
			maxAttWidth = "Attribute".length() + 2;
		}

		if (maxWidth < "Full Data".length()) {
			maxWidth = "Full Data".length() + 1;
		}

		if (maxWidth < "missing".length()) {
			maxWidth = "missing".length() + 1;
		}
		StringBuffer temp = new StringBuffer();
		temp.append("\nFuzzyCMeans\n======\n");
		temp.append("\nNumber of iterations: " + m_Iterations);

		temp.append("\n\nFinal cluster centroids:\n");
		temp.append(pad("Cluster#", " ", (maxAttWidth + (maxWidth * 2 + 2)) - "Cluster#".length(), true));

		temp.append("\n");
		temp.append(pad("Attribute", " ", maxAttWidth - "Attribute".length(), false));

		temp.append(pad("Full Data", " ", maxWidth + 1 - "Full Data".length(), true));
		// cluster numbers
		for (int i = 0; i < m_NumClusters; i++) {
			String clustNum = "" + i;
			temp.append(pad(clustNum, " ", maxWidth + 1 - clustNum.length(), true));
		}
		temp.append("\n");

		// cluster sizes
		String cSize = "(" + Utils.sum(m_ClusterSizes) + ")";
		temp.append(pad(cSize, " ", maxAttWidth + maxWidth + 1 - cSize.length(), true));
		for (int i = 0; i < m_NumClusters; i++) {
			cSize = "(" + m_ClusterSizes[i] + ")";
			temp.append(pad(cSize, " ", maxWidth + 1 - cSize.length(), true));
		}
		temp.append("\n");

		temp.append(pad("", "=",
				maxAttWidth
						+ (maxWidth * (m_ClusterCentroids.numInstances() + 1) + m_ClusterCentroids.numInstances() + 1),
				true));
		temp.append("\n");
		for (int i = 0; i < m_ClusterCentroids.numAttributes(); i++) {
			String attName = m_ClusterCentroids.attribute(i).name();
			temp.append(attName);
			for (int j = 0; j < maxAttWidth - attName.length(); j++) {
				temp.append(" ");
			}

			String strVal;
			String valMeanMode;
			
			for (int j = 0; j < m_NumClusters; j++) {
				if (m_ClusterCentroids.attribute(i).isNominal()) {
					if (m_ClusterCentroids.instance(j).isMissing(i)) {
						valMeanMode = pad("missing", " ", maxWidth + 1 - "missing".length(), true);
					} else {
						valMeanMode = pad(
								(strVal = m_ClusterCentroids.attribute(i)
										.value((int) m_ClusterCentroids.instance(j).value(i))),
								" ", maxWidth + 1 - strVal.length(), true);
					}
				} else {
					if (m_ClusterCentroids.instance(j).isMissing(i)) {
						valMeanMode = pad("missing", " ", maxWidth + 1 - "missing".length(), true);
					} else {
						valMeanMode = pad((strVal = Utils
								.doubleToString(m_ClusterCentroids.instance(j).value(i), maxWidth, 4).trim()), " ",
								maxWidth + 1 - strVal.length(), true);
					}
				}
				temp.append(valMeanMode);
			}
			temp.append("\n");
			if (m_displayStdDevs) {
				// Std devs/max nominal
				String stdDevVal = "";

			
				
						stdDevVal = pad(
								(strVal = plusMinus + Utils.doubleToString(m_FullStdDevs[i], maxWidth, 4).trim()), " ",
								maxWidth + maxAttWidth + 1 - strVal.length(), true);
					
					temp.append(stdDevVal);

					// Clusters
					for (int j = 0; j < m_NumClusters; j++) {
						if (m_ClusterCentroids.instance(j).isMissing(i)) {
							stdDevVal = pad("--", " ", maxWidth + 1 - 2, true);
						} else {
							stdDevVal = pad(
									(strVal = plusMinus + Utils
											.doubleToString(m_ClusterStdDevs.instance(j).value(i), maxWidth, 4).trim()),
									" ", maxWidth + 1 - strVal.length(), true);
						}
						temp.append(stdDevVal);
					}
					temp.append("\n\n");
				
			}
		}

		temp.append("\n\n");
		System.out.println(temp.toString());
		statisticClass();
		return temp.toString();
	}


}

	

