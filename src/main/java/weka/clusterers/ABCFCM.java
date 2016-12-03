package weka.clusterers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import weka.clusterers.NumberOfClustersRequestable;
import weka.clusterers.RandomizableClusterer;
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
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class ABCFCM extends RandomizableClusterer implements
		NumberOfClustersRequestable, WeightedInstancesHandler,
		TechnicalInformationHandler {

	/** for serialization. */
	private static final long serialVersionUID = 159802830430835116L;

	/**
	 * replace missing values in training instances. 替换训练集中的缺省值
	 */
	protected ReplaceMissingValues m_ReplaceMissingFilter;

	/**
	 * Replace missing values globally?
	 */
	protected boolean m_dontReplaceMissing = false;

	/**
	 * number of clusters to generate. 产生聚类个数
	 */
	protected int m_NumClusters = 3;
	
	/**
	 * holds the cluster centroids. 聚类中心
	 */
	protected Instances m_ClusterCentroids;
	
	protected Instances[] mClusters;

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

	/**
	 * For each cluster, holds the frequency counts for the values of each
	 * nominal attribute. 每个聚类中，名词属性的频率计数值
	 */
	protected double[][][] m_ClusterNominalCounts;
	protected double[][] m_ClusterMissingCounts;

	/**
	 * Stats on the full data set for comparison purposes. In case the attribute
	 * is numeric the value is the mean if is being used the Euclidian distance
	 * or the median if Manhattan distance and if the attribute is nominal then
	 * it's mode is saved.
	 */
	protected double[] m_FullMeansOrMediansOrModes;
	protected double[] m_FullStdDevs;
	protected double[][] m_FullNominalCounts;
	protected double[] m_FullMissingCounts;

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
	 * Holds the squared errors for all clusters. 平方误差
	 */
	protected double[] m_squaredErrors;

	/** whether to use fast calculation of distances (using a cut-off). */
	protected boolean m_FastDistanceCalc = false;

	/**
	 * uij is the degree of membership of xi in the cluster j
	 */
	private Matrix bestMS;
	
	private Instances instances;

	/**
	 * holds the fuzzifier 模糊算子(加权指数)
	 */
	private double m_fuzzifier = 2;

	/**
	 * Assignments obtained.(cluster indexes).
	 * 
	 */
	protected int[] m_Assignments = null;

	/** Number of threads to run */
	protected int m_executionSlots = 1;

	/** For parallel execution mode */
	protected transient ExecutorService m_executorPool;
	
	private Matrix[] matrixs;
	
	
	/** The number of colony size (employed bees+onlooker bees) */
	int NP = 200;
	/** The number of food sources equals the half of the colony size */
	int foodNum = NP / 2;
	/**
	 * A food source which could not be improved through "limit" trials is
	 * abandoned by its employed bee
	 */
	int limit = 20;
	/** The number of cycles for foraging {a stopping criteria} */
	int maxCycle = 100;
	int mCycle = 0;

	/** Problem specific variables */
	/** The number of parameters of the problem to be optimized */
	int dimension = 0;
	/** lower bound of the parameters. */
	double lb = 0;
	/**
	 * upper bound of the parameters. lb and ub can be defined as arrays for the
	 * problems of which parameters have different bounds
	 */
	double ub = 1;

	/** Algorithm can be run many times in order to see its robustness */
	int runCount = 1;

	/**
	 * foods is the population of food sources. Each row of foods matrix is a
	 * vector holding dimension parameters to be optimized. The number of rows
	 * of foods matrix equals to the foodNum
	 * */
	double foods[][] ;

	/**
	 * f is a vector holding objective function values associated with food
	 * sources
	 */
	double funVal[] = new double[foodNum];
	/**
	 * fitness is a vector holding fitness (quality) values associated with food
	 * sources
	 */
	double fitness[] = new double[foodNum];

	/**
	 * trial is a vector holding trial numbers through which solutions can not
	 * be improved
	 */
	double trial[] = new double[foodNum];

	/**
	 * prob is a vector holding probabilities of food sources (solutions) to be
	 * chosen
	 */
	double prob[] = new double[foodNum];


	/** Optimum solution obtained by ABC algorithm */
	double minObjFunValue = Double.MAX_VALUE;
	/**
	 * Holds the squared errors for all clusters. 平方误差
	 */
	double squaredError=0;

	/** Parameters of the optimum solution */
	double bestFood[];
	
	/** globalMins holds the minObjFunValue of each run in multiple runs */
	double globalMins[] = new double[runCount];

	public ABCFCM() {
		m_SeedDefault = 10;
		setSeed(m_SeedDefault);
	}

	/**
	 * Start the pool of execution threads
	 */
	protected void startExecutorPool() {
		if (m_executorPool != null) {
			m_executorPool.shutdownNow();
		}
		m_executorPool = Executors.newFixedThreadPool(m_executionSlots);
	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		// TODO Auto-generated method stub
		double[] d = new double[m_NumClusters];
		double top = 0, bottom = 0, sum;
		for (int j = 0; j < m_NumClusters; j++) {
			top = m_DistanceFunction.distance(instance,
					m_ClusterCentroids.instance(j));
			sum = 0;
			for (int k = 0; k < m_NumClusters; k++) {
				bottom = m_DistanceFunction.distance(instance,
						m_ClusterCentroids.instance(k));
				sum += Math.pow(top / bottom, 2.0 / (m_fuzzifier - 1));
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
	public void buildClusterer(Instances data) throws Exception {
		getCapabilities().testWithFail(data);
		startExecutorPool();
		instances = new Instances(data);
		init();
		double mean = 0;
		for (int run = 0; run < runCount; run++) {
			initial();
			memorizeBestSource();
			for (int iter = 0; iter < maxCycle; iter++) {
				mCycle = iter+1;
				calculateProbabilities();
				memorizeBestSource();
				sendOnlookerBees();
				calculateFcm();
				memorizeBestSource();
				sendScoutBees();
				System.out.println("iter="+iter+" minObjFunValue="+minObjFunValue);
			}
/*			
			Instances ins = arrayToInstances(bestFood);
			for(int i=0; i < maxCycle; i++){
				bestMS = updateMemberShip(ins);
				m_ClusterCentroids = updateCentroid(bestMS);
				System.out.println("fcm iter="+i);
			}
*/		
			updateClustersInfo();
			System.out.println("minObjFunValue = "+ minObjFunValue);
			System.out.println(toString());
			globalMins[run] = minObjFunValue;
			mean = mean + minObjFunValue;
		}
		int maxIndex = Utils.maxIndex(globalMins);
		int minIndex = Utils.minIndex(globalMins);
		double max = globalMins[maxIndex];
		double min = globalMins[minIndex];

		mean = mean / runCount;

		double stdError = 0;

		for (int j = 0; j < runCount; j++) {
			stdError += Math.pow(globalMins[j] - mean, 2);
		}
		stdError /= runCount;
		System.out.println("maxFunVal=" + max + " minFunVal=" + min + " mean="
				+ mean + " stdError=" + stdError);
		
		m_executorPool.shutdown();
		// save memory!
		m_DistanceFunction.clean();

	}
	/*
	 * Variables are initialized in the range [lb,ub]. If each parameter has
	 * different range, use arrays lb[j], ub[j] instead of lb and ub
	 */
	/* Counters of food sources are also initialized in this function */

	public void init(int index) {
		double[] solution = new double[dimension];
		for (int j = 0; j < dimension; j++) {
			foods[index][j] = Math.random() * (ub - lb) + lb;
			solution[j] = foods[index][j];
		}
		Instances centroids = arrayToInstances(solution);
		Matrix matrix = updateMemberShip(centroids);
		matrixs[index] = matrix;
		funVal[index] = calculateObjectiveFunction(centroids, matrix);
		fitness[index] = calculateFitness(funVal[index]);
		trial[index] = 0;
	}
	private class InitTask implements Callable<Boolean>{
		private int start;
		private int end;
		public InitTask(int start,int end){
			this.start=start;
			this.end = end;
		}
		public Boolean call(){
			double[] solution = new double[dimension];
			for (int i = start; i < end; i++) {
				for (int j = 0; j < dimension; j++) {
					foods[i][j] = Math.random() * (ub - lb) + lb;
					solution[j] = foods[i][j];
				}
				Instances centroids = arrayToInstances(solution);
				Matrix matrix = updateMemberShip(centroids);
				matrixs[i] = matrix;
				funVal[i] = calculateObjectiveFunction(centroids, matrix);
				fitness[i] = calculateFitness(funVal[i]);
				trial[i] = 0;
			}
			return true;
		}
	}
	/* All food sources are initialized */
	public void initial() {
		int numPerTask = foodNum / m_executionSlots;
		List<Future<Boolean>> results = new ArrayList<Future<Boolean>>();
		for (int i = 0; i < m_executionSlots; i++) {
			int start = i * numPerTask;
			int end = start + numPerTask;
			if (i == m_executionSlots - 1) {
				end = foodNum;
			}
			results.add(m_executorPool.submit(new InitTask( start, end)));
		}
		try{
			for(Future<Boolean> task : results){
				task.get();
			}
		}catch(Exception e){
			e.printStackTrace();
		}
	}


	/**
	 * mean Euclidean distances between X_{m} and the rest of solutions.
	 * @return
	 */
	public double calculateMean(int index){
		double sum=0;
		for(int i=0; i< foodNum; i++){
			double total = 0;
			if(index!=i){
			for(int j=0; j<dimension; j++){
					total+=Math.pow(foods[index][j] - foods[i][j],2);
				}
			}
			sum+=total;
		}
		return sum/(foodNum-1);
	}
	/**
	 * calculate the  neighbor of  X_{m} and itself (N_{m})
	 * @param index
	 * @return
	 */
	public List<double[]> calculateNeighbor(int index){
		List<double[]> neighbors = new ArrayList<>();
		double mean = calculateMean(index);
		for(int i=0; i<foodNum; i++){
			double total =0;
			if(index !=i){
				for(int j=0; j<dimension; j++){
					total += Math.pow(foods[index][j] - foods[i][j], 2);
				}
			}
			if(total < mean){
				neighbors.add(foods[i]);
			}
		}
		return neighbors;
	}
	/**
	 * calculate the best solution among the neighbor of  X_{m} and itself (N_{m})
	 * @param index
	 * @return X_{Nm}^best
	 */
	public double[] calculateNeighborBest(int index){
		List<double[]> neighbors = calculateNeighbor(index);
		double maxFit = lb;
		double[] maxNeighbor = null;
		for(double[] neighbor : neighbors){
			Instances centroids = arrayToInstances(neighbor);
			Matrix matrix = updateMemberShip(centroids);
			double objVal = calculateObjectiveFunction(centroids, matrix);
			double fitness = calculateFitness(objVal);
			if(maxFit<fitness){
				maxFit = fitness;
				maxNeighbor = neighbor;
			}
		}
		return maxNeighbor;
		
	}
	/** The best food source is memorized */
	public void memorizeBestSource() {
		int minIndex = Utils.minIndex(funVal);
		minObjFunValue = funVal[minIndex];
		for (int j = 0; j < dimension; j++){
			bestFood[j] = foods[minIndex][j];
		}
		m_ClusterCentroids = arrayToInstances(bestFood);
		for(int m = 0; m < instances.numInstances();m++){
			for(int n=0; n < m_NumClusters;n++){
				bestMS.set(m, n, matrixs[minIndex].get(m, n));
			}
		}
		
	}
	public void print(){
		for(double val :funVal){
			System.out.print(val+" ");
		}
		System.out.println();
	}
	
private class EmployBeeTask implements Callable<Boolean>{
	private int start;
	private int end;
	public EmployBeeTask(int start,int end){
		this.start=start;
		this.end = end;
	}
	public Boolean call(){
		Random rand = new Random();
		for (int i = start; i < end; i++) {
			/* The parameter to be changed is determined randomly */
			int dj = rand.nextInt(dimension);
			/*
			 * A randomly chosen solution is used in producing a mutant solution
			 * of the solution i
			 */
			int foodi = rand.nextInt(foodNum);
			double[] solution = new double[dimension];
			for (int j = 0; j < dimension; j++) {
				solution[j] = foods[i][j];
			}
			/* v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij}) */
			double r = rand.nextDouble() * 2 - 1;
			solution[dj] = foods[i][dj]+ (foods[i][dj] - foods[foodi][dj])
					* r*(1 + 1/(Math.exp(-maxCycle*1.0/mCycle)+1));

			/*
			 * if generated parameter value is out of boundaries, it is shifted
			 * onto the boundaries
			 */
			if (solution[dj] < lb)
				solution[dj] = lb;
			if (solution[dj] > ub)
				solution[dj] = ub;
			
			Instances centroids = arrayToInstances(solution);
			Matrix matrix = updateMemberShip(centroids);
			double objValSol = calculateObjectiveFunction(centroids, matrix);
			double fitnessSol = calculateFitness(objValSol);

			/*
			 * a greedy selection is applied between the current solution i and
			 * its mutant
			 */
			if (fitnessSol > fitness[i]) {

				/**
				 * If the mutant solution is better than the current solution i,
				 * replace the solution with the mutant and reset the trial
				 * counter of solution i
				 * */
				trial[i] = 0;
				for (int j = 0; j < dimension; j++){
					foods[i][j] = solution[j];
				}
					
				for(int m = 0; m < instances.numInstances();m++){
					for(int n=0; n < m_NumClusters;n++){
						matrixs[i].set(m, n, matrix.get(m, n));
					}
				}
				
				funVal[i] = objValSol;
				fitness[i] = fitnessSol;
			} else {
				/*
				 * if the solution i can not be improved, increase its trial
				 * counter
				 */
				trial[i] = trial[i] + 1;
			}
		}
		return true;
	}
}
	
	/**
	 * Employed Bee Phase
	 */
	public void sendEmployedBees() {
		int numPerTask = foodNum / m_executionSlots;
		List<Future<Boolean>> results = new ArrayList<Future<Boolean>>();
		for (int i = 0; i < m_executionSlots; i++) {
			int start = i * numPerTask;
			int end = start + numPerTask;
			if (i == m_executionSlots - 1) {
				end = foodNum;
			}
			results.add(m_executorPool.submit(new EmployBeeTask(start, end)));
		}
		try{
			for(Future<Boolean> task : results){
				task.get();
			}
			print();
		}catch(Exception e){
			e.printStackTrace();
		}
	}


	public void calculateProbabilities() {

		double sum = 0;

		for (int i = 0; i < foodNum; i++) {
			sum += fitness[i];
		}

		for (int i = 0; i < foodNum; i++) {
			prob[i] = fitness[i] / sum;
		}

	}
	private class OnlookerBeeTask implements Callable<Boolean>{
		private int start;
		private int end;
		private int neighbour;
		private Random rand;
		public OnlookerBeeTask(int start,int end,int neighbour,Random rand){
			this.start=start;
			this.end = end;
			this.neighbour = neighbour;
			this.rand = rand;
		}
		public Boolean call(){
			double[] solution = new double[dimension];
					for (int j = 0; j < dimension; j++){
						solution[j] = foods[start][j];
					}
					double[] bestNeighbor = calculateNeighborBest(start);
					int minFIndex = Utils.minIndex(funVal);
					/* v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij}) */
					
					double r = rand.nextDouble()-1;
					solution[end] =  bestNeighbor[end]
							+ (bestNeighbor[end] - foods[neighbour][end])* r+
							rand.nextDouble()*1.5*(foods[minFIndex][end]-bestNeighbor[end]);

					/*
					 * if generated parameter value is out of boundaries, it is
					 * shifted onto the boundaries
					 */
					if (solution[end] < lb)
						solution[end] = lb;
					if (solution[end] > ub)
						solution[end] = ub;
					Instances centroids = arrayToInstances(solution);
					Matrix matrix = updateMemberShip(centroids);
					double objValSol = calculateObjectiveFunction(centroids, matrix);
					double fitnessSol = calculateFitness(objValSol);

					/*
					 * a greedy selection is applied between the current solution i
					 * and its mutant
					 */
					if (fitnessSol > fitness[start]) {
						/*
						 * If the mutant solution is better than the current
						 * solution i, replace the solution with the mutant and
						 * reset the trial counter of solution i
						 */
						trial[start] = 0;
						for (int j = 0; j < dimension; j++){
							foods[start][j] = solution[j];
						}
							
						for(int m = 0; m < instances.numInstances();m++){
							for(int n=0; n < m_NumClusters;n++){
								matrixs[start].set(m, n, matrix.get(m, n));
							}
						}
						funVal[start] = objValSol;
						fitness[start] = fitnessSol;
					} else {
						/*
						 * if the solution i can not be improved, increase its trial
						 * counter
						 */
						trial[start] = trial[start] + 1;
					}
					return true;
				}
	}
	/** onlooker Bee Phase */
	public void sendOnlookerBees() {

		int i, j, t;
		i = 0;
		t = 0;
		Random rand = new Random();
		List<Future<Boolean>> results = new ArrayList<Future<Boolean>>();
		while (t < foodNum) {
			double r = Math.random();
//			r = ((double) Math.random() * 32767 / ((double) (32767) + (double) (1)));
			/*
			 * choose a food source depending on its probability to be chosen
			 */
			if (r < prob[i]) {
				t++;

				/* The parameter to be changed is determined randomly */
				int dj = rand.nextInt(dimension);

				/*
				 * A randomly chosen solution is used in producing a mutant
				 * solution of the solution i
				 */
				int neighbour = rand.nextInt(foodNum);

				/*
				 * Randomly selected solution must be different from the
				 * solution i
				 */
				while (neighbour == i) {
					// System.out.println(Math.random()*32767+"  "+32767);
					neighbour = rand.nextInt(foodNum);
				}
				results.add(m_executorPool.submit(new OnlookerBeeTask( i, dj,neighbour,rand)));
			}
			i++;
			if (i == foodNum)
				i = 0;
		}/* while */
		try{
			for(Future<Boolean> task : results){
				task.get();
			}
			print();
		}catch(Exception e){
			e.printStackTrace();
		}
		/* end of onlooker bee phase */
	}

	/*
	 * determine the food sources whose trial counter exceeds the "limit" value.
	 * In Basic ABC, only one scout is allowed to occur in each cycle
	 */
	public void sendScoutBees() {
		int maxtrialindex, i;
		maxtrialindex = 0;
		for (i = 1; i < foodNum; i++) {
			if (trial[i] > trial[maxtrialindex])
				maxtrialindex = i;
		}
		if (trial[maxtrialindex] >= limit) {
			init(maxtrialindex);
		}
	}
	/** Fitness function */
	public double calculateFitness(double fun) {
		double result = 0;
		if (fun >= 0) {
			result = 1 / (fun + 1);
		} else {
			result = 1 + Math.abs(fun);
		}
		return result;
	}
	private class CalFcmTask implements Callable<Boolean>{
		private int start;
		private int end;
		public CalFcmTask(int start,int end){
			this.start=start;
			this.end = end;
		}
		public Boolean call(){
			double[] solution = new double[dimension];
		
			for (int i = start; i < end; i++) {
				for (int u = 0; u < dimension; u++){
					solution[u] = foods[i][u];
				}
				Instances centroids = arrayToInstances(solution);
				Matrix matrix = updateMemberShip(centroids);
				centroids = updateCentroid(matrix);
				matrix = updateMemberShip(centroids);
				double objValSol = calculateObjectiveFunction(centroids, matrix);
				double fitnessSol = calculateFitness(objValSol);

				/*
				 * a greedy selection is applied between the current solution i and
				 * its mutant
				 */
				if (fitnessSol > fitness[i]) {

					/**
					 * If the mutant solution is better than the current solution i,
					 * replace the solution with the mutant and reset the trial
					 * counter of solution i
					 * */
					trial[i] = 0;
					double[] newSolution = instancesToArray(centroids);
					for (int j = 0; j < dimension; j++){
						foods[i][j] = newSolution[j];
					}
						
					for(int m = 0; m < instances.numInstances();m++){
						for(int n=0; n < m_NumClusters;n++){
							matrixs[i].set(m, n, matrix.get(m, n));
						}
					}
					funVal[i] = objValSol;
					fitness[i] = fitnessSol;
				} else {
					/*
					 * if the solution i can not be improved, increase its trial
					 * counter
					 */
					trial[i] = trial[i] + 1;
				}
			}
		return true;
		}
	}
	public  void calculateFcm(){
		int numPerTask = foodNum / m_executionSlots;
		List<Future<Boolean>> results = new ArrayList<Future<Boolean>>();
		for (int i = 0; i < m_executionSlots; i++) {
			int start = i * numPerTask;
			int end = start + numPerTask;
			if (i == m_executionSlots - 1) {
				end = foodNum;
			}
			results.add(m_executorPool.submit(new CalFcmTask( start, end)));
		}
		try{
			for(Future<Boolean> task : results){
				task.get();
			}
			print();
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	public double[] instancesToArray(Instances centroids) {
		
		double[] value = new double[dimension];
		int index =0;
		for(Instance ins : centroids){
			for(int i = 0; i<ins.numAttributes();i++){
				value[index++]=ins.value(i);
			}
		}
		return value;
	}
	public Instances arrayToInstances(double sol[]) {
		int count = 0;
		Instances centroids = new Instances(instances, 0);
		for(int i=0;i<m_NumClusters; i++){
			double[] val = new double[dimension/m_NumClusters];
			for(int j=0; j< dimension/m_NumClusters; j++ ){
				val[j] = sol[count++];
			}
			Instance ins = new DenseInstance(1.0,val);
			centroids.add(ins);
		}
		return centroids;
	}
	public void init() {
		dimension = instances.numAttributes()*m_NumClusters;
		foods = new double[foodNum][dimension];
		bestFood = new double[dimension];
		matrixs = new Matrix[foodNum];
		for(Matrix matrix: matrixs){
			matrix = new Matrix(instances.numInstances(), m_NumClusters);
		}
		m_DistanceFunction.setInstances(instances);
		bestMS = new Matrix(instances.numInstances(), m_NumClusters);
		mClusters = new Instances[m_NumClusters];
		m_Assignments = new int[instances.numInstances()];
		m_squaredErrors = new double[m_NumClusters];
		m_ClusterNominalCounts = new double[m_NumClusters][instances
				.numAttributes()][];
		m_ClusterMissingCounts = new double[m_NumClusters][instances
				.numAttributes()];
		if (m_displayStdDevs) {
			m_FullStdDevs = instances.variances();
		}
		m_FullMeansOrMediansOrModes = calculateMeansOrMediansOrModes(0,
				instances, true);

		m_FullMissingCounts = m_ClusterMissingCounts[0];
		m_FullNominalCounts = m_ClusterNominalCounts[0];
		double sumOfWeights = instances.sumOfWeights();

		for (int i = 0; i < instances.numAttributes(); i++) {
			if (instances.attribute(i).isNumeric()) {
				if (m_displayStdDevs) {
					m_FullStdDevs[i] = Math.sqrt(m_FullStdDevs[i]);
				}
				if (m_FullMissingCounts[i] == sumOfWeights) {
					m_FullMeansOrMediansOrModes[i] = Double.NaN; // mark missing as mean
				}
			} else {
				if (m_FullMissingCounts[i] > m_FullNominalCounts[i][Utils
						.maxIndex(m_FullNominalCounts[i])]) {
					m_FullMeansOrMediansOrModes[i] = -1; // mark missing as most common value
				}
			}
		}
		m_ClusterNominalCounts = new double[m_NumClusters][instances
				.numAttributes()][0];
		m_ClusterMissingCounts = new double[m_NumClusters][instances
				.numAttributes()];
	}

	public Instances updateCentroid(Matrix matrix) {
		Instances centroids = new Instances(instances,0);
		for (int k = 0; k < m_NumClusters; k++) {
			double bottom = 0.0d;
			double[] attributes = new double[instances.numAttributes()];
			Instance in = new DenseInstance(1.0, attributes);
			for (int i = 0; i < instances.numInstances(); i++) {
				double uValue = Math.pow(matrix.get(i, k), m_fuzzifier);
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
			centroids.add(in);
		}
		return centroids;
	}

	public Matrix updateMemberShip(Instances centroids) {
		Matrix matrix = new Matrix(instances.numInstances(), m_NumClusters);
		for (int j = 0; j < m_NumClusters; j++) {
			for (int i = 0; i < instances.numInstances(); i++) {
				double bottom = 0;
				double top = m_DistanceFunction.distance(instances.instance(i),
						centroids.instance(j));
				double sum = 0;
				for (int k = 0; k < m_NumClusters; k++) {
					bottom = m_DistanceFunction.distance(instances.instance(i),
							centroids.instance(k));
					sum += Math.pow(top / bottom, 2.0d / (m_fuzzifier - 1.0));
				}
				matrix.set(i, j, 1.0d / sum);
			}
		}
		return matrix;
	}

	public double calculateObjectiveFunction(Instances centroids,Matrix matrix) {
		double sum = 0, dist = 0;
		for (int i = 0; i < instances.numInstances(); i++) {
			for (int j = 0; j < m_NumClusters; j++) {
				dist = m_DistanceFunction.distance(instances.instance(i),
						centroids.instance(j));
				sum += Math.pow(matrix.get(i, j), m_fuzzifier)
						* Math.pow(dist, 2);
			}
		}
		return sum;
	}

	public void updateClustersInfo() {
		m_ClusterSizes = new double[m_NumClusters];
		for (int i = 0; i < m_NumClusters; i++) {
			mClusters[i] = new Instances(instances, 0);
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
				if (max < bestMS.get(i, j)) {
					max = bestMS.get(i, j);
					index = j;
				}
			}
			m_Assignments[i] = index;
			mClusters[index].add(instances.get(i));
			m_ClusterSizes[index] += 1;

			dist = m_DistanceFunction.distance(instances.instance(i),
					m_ClusterCentroids.instance(index));
			m_squaredErrors[index] = dist * dist
					* instances.instance(i).weight();
		}

		// update m_ClusterStdDevs m_ClusterNominalCounts m_ClusterMissingCounts
		for (int i = 0; i < m_NumClusters; i++) {
			calculateMeansOrMediansOrModes(i, mClusters[i], true);
			if (m_displayStdDevs) {
				double[] vals2 = mClusters[i].variances();
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

	private double[] calculateMeansOrMediansOrModes(int centroidIndex,
			Instances members, boolean updateClusterInfo) {

		double[] vals = new double[members.numAttributes()];
		double[][] nominalDists = new double[members.numAttributes()][];
		double[] weightMissing = new double[members.numAttributes()];
		double[] weightNonMissing = new double[members.numAttributes()];

		// Quickly calculate some relevant statistics
		for (int j = 0; j < members.numAttributes(); j++) {
			if (members.attribute(j).isNominal()) {
				nominalDists[j] = new double[members.attribute(j).numValues()];
			}
		}
		for (Instance inst : members) {
			for (int j = 0; j < members.numAttributes(); j++) {
				if (inst.isMissing(j)) {
					weightMissing[j] += inst.weight();
				} else {
					weightNonMissing[j] += inst.weight();
					if (members.attribute(j).isNumeric()) {
						vals[j] += inst.weight() * inst.value(j);
					} else {
						nominalDists[j][(int) inst.value(j)] += inst.weight();
					}
				}
			}
		}
		for (int j = 0; j < members.numAttributes(); j++) {
			if (members.attribute(j).isNumeric()) {
				if (weightNonMissing[j] > 0) {
					vals[j] /= weightNonMissing[j];
				} else {
					vals[j] = Utils.missingValue();
				}
			} else {
				double max = -Double.MAX_VALUE;
				double maxIndex = -1;
				for (int i = 0; i < nominalDists[j].length; i++) {
					if (nominalDists[j][i] > max) {
						max = nominalDists[j][i];
						maxIndex = i;
					}
					if (max < weightMissing[j]) {
						vals[j] = Utils.missingValue();
					} else {
						vals[j] = maxIndex;
					}
				}
			}
		}
		if (updateClusterInfo) {
			for (int j = 0; j < members.numAttributes(); j++) {
				m_ClusterMissingCounts[centroidIndex][j] = weightMissing[j];
				m_ClusterNominalCounts[centroidIndex][j] = nominalDists[j];
			}
		}
		return vals;
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
		result.setValue(Field.TITLE,
				"k-means++: the advantages of carefull seeding");
		result.setValue(Field.BOOKTITLE,
				"Proceedings of the eighteenth annual "
						+ "ACM-SIAM symposium on Discrete algorithms");
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
		maxCycle = n;
	}

	/**
	 * gets the number of maximum iterations to be executed.
	 * 
	 * @return the number of clusters to generate
	 */
	public int getMaxIterations() {
		return maxCycle;
	}
	public void setRunCount(int n) throws Exception {
		if (n <= 0) {
			throw new Exception("run count be >=1");
		}
		runCount = n;
	}
	public int getRunCount() {
		return runCount;
	}
	
	public void setLimit(int n) throws Exception {
		if (n <= 0) {
			throw new Exception("limit count be >=1");
		}
		limit = n;
	}
	public int getLimit() {
		return limit;
	}
	
	public void setFuzzifier(double val){
		m_fuzzifier = val;
	}
	public double getFuzzifier(){
		return m_fuzzifier;
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
	 * Sets whether missing values are to be replaced.
	 * 
	 * @param r
	 *            true if missing values are to be replaced
	 */
	public void setDontReplaceMissingValues(boolean r) {
		m_dontReplaceMissing = r;
	}

	/**
	 * Gets whether missing values are to be replaced.
	 * 
	 * @return true if missing values are to be replaced
	 */
	public boolean getDontReplaceMissingValues() {
		return m_dontReplaceMissing;
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
	 * Sets whether to use faster distance calculation.
	 * 
	 * @param value
	 *            true if faster calculation to be used
	 */
	public void setFastDistanceCalc(boolean value) {
		m_FastDistanceCalc = value;
	}

	/**
	 * Gets whether to use faster distance calculation.
	 * 
	 * @return true if faster calculation is used
	 */
	public boolean getFastDistanceCalc() {
		return m_FastDistanceCalc;
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
	 * Returns for each cluster the weighted frequency counts for the values of
	 * each nominal attribute.
	 * 
	 * @return the counts
	 */
	public double[][][] getClusterNominalCounts() {
		return m_ClusterNominalCounts;
	}

	/**
	 * Gets the squared error for all clusters.
	 * 
	 * @return the squared error, NaN if fast distance calculation is used
	 * @see #m_FastDistanceCalc
	 */
	public double getSquaredError() {
		if (m_FastDistanceCalc) {
			return Double.NaN;
		} else {
			return Utils.sum(m_squaredErrors);
		}
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
			throw new Exception(
					"The assignments are only available when order of instances is preserved (-O)");
		}
		if (m_Assignments == null) {
			throw new Exception("No assignments made.");
		}
		return m_Assignments;
	}

	public Instances[] getClusters(){
		return mClusters;
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
		optionString = Utils.getOption("R", options);
		if (optionString.length() != 0) {
			setRunCount(Integer.parseInt(optionString));
		}
		optionString = Utils.getOption("L", options);
		if (optionString.length() != 0) {
			setLimit(Integer.parseInt(optionString));
		}
		optionString = Utils.getOption("F", options);
		if (optionString.length() != 0) {
			setFuzzifier(Double.parseDouble(optionString));
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

		if (m_dontReplaceMissing) {
			result.add("-M");
		}

		result.add("-N");
		result.add("" + getNumClusters());

		result.add("-A");
		result.add((m_DistanceFunction.getClass().getName() + " " + Utils
				.joinOptions(m_DistanceFunction.getOptions())).trim());

		result.add("-I");
		result.add("" + getMaxIterations());
		result.add("-R");
		result.add("" + getRunCount());
		result.add("-L");
		result.add("" + getLimit());
		
		result.add("F");
		result.add("" + getFuzzifier());

		if (m_PreserveOrder) {
			result.add("-O");
		}

		Collections.addAll(result, super.getOptions());

		return result.toArray(new String[result.size()]);
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
		if (m_ClusterCentroids == null) {
			return "No clusterer built yet!";
		}
		int maxWidth = 0;
		int maxAttWidth = 0;
		boolean containsNumeric = false;
		for (int i = 0; i < m_NumClusters; i++) {
			for (int j = 0; j < m_ClusterCentroids.numAttributes(); j++) {
				if (m_ClusterCentroids.attribute(j).name().length() > maxAttWidth) {
					maxAttWidth = m_ClusterCentroids.attribute(j).name()
							.length();
				}
				if (m_ClusterCentroids.attribute(j).isNumeric()) {
					containsNumeric = true;
					double width = Math.log(Math.abs(m_ClusterCentroids
							.instance(i).value(j))) / Math.log(10.0);

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
					String val = a.value((int) m_ClusterCentroids.instance(j)
							.value(i));
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

		if (m_displayStdDevs) {
			// check for maximum width of maximum frequency count
			for (int i = 0; i < m_ClusterCentroids.numAttributes(); i++) {
				if (m_ClusterCentroids.attribute(i).isNominal()) {
					int maxV = Utils.maxIndex(m_FullNominalCounts[i]);
					/*
					 * int percent = (int)((double)m_FullNominalCounts[i][maxV]
					 * / Utils.sum(m_ClusterSizes) * 100.0);
					 */
					int percent = 6; // max percent width (100%)
					String nomV = "" + m_FullNominalCounts[i][maxV];
					// + " (" + percent + "%)";
					if (nomV.length() + percent > maxWidth) {
						maxWidth = nomV.length() + 1;
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
		temp.append("\nABCFCM\n======\n");
		temp.append("\nNumber of iterations: " + m_Iterations);
		temp.append("\tthe seed: " + m_Seed);
		if (!m_FastDistanceCalc) {
			temp.append("\n");
			if (m_DistanceFunction instanceof EuclideanDistance) {
				temp.append("Within cluster sum of squared errors: "
						+ Utils.sum(m_squaredErrors));
			} else {
				temp.append("Sum of within cluster distances: "
						+ Utils.sum(m_squaredErrors));
			}
		}
		
		if (!m_dontReplaceMissing) {
			temp.append("\nMissing values globally replaced with mean/mode");
		}
		temp.append("\nobjFunValue="+minObjFunValue);
		temp.append("\n");
		temp.append("\n\nFinal cluster centroids:\n");
		temp.append(pad("Cluster#", " ", (maxAttWidth + (maxWidth * 2 + 2))
				- "Cluster#".length(), true));

		temp.append("\n");
		temp.append(pad("Attribute", " ", maxAttWidth - "Attribute".length(),
				false));

		temp.append(pad("Full Data", " ", maxWidth + 1 - "Full Data".length(),
				true));
		// cluster numbers
		for (int i = 0; i < m_NumClusters; i++) {
			String clustNum = "" + i;
			temp.append(pad(clustNum, " ", maxWidth + 1 - clustNum.length(),
					true));
		}
		temp.append("\n");

		// cluster sizes
		String cSize = "(" + Utils.sum(m_ClusterSizes) + ")";
		temp.append(pad(cSize, " ",
				maxAttWidth + maxWidth + 1 - cSize.length(), true));
		for (int i = 0; i < m_NumClusters; i++) {
			cSize = "(" + m_ClusterSizes[i] + ")";
			temp.append(pad(cSize, " ", maxWidth + 1 - cSize.length(), true));
		}
		temp.append("\n");

		temp.append(pad("", "=",
				maxAttWidth
						+ (maxWidth * (m_ClusterCentroids.numInstances() + 1)
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
			// full data
			if (m_ClusterCentroids.attribute(i).isNominal()) {
				if (m_FullMeansOrMediansOrModes[i] == -1) { // missing
					valMeanMode = pad("missing", " ",
							maxWidth + 1 - "missing".length(), true);
				} else {
					valMeanMode = pad((strVal = m_ClusterCentroids.attribute(i)
							.value((int) m_FullMeansOrMediansOrModes[i])), " ",
							maxWidth + 1 - strVal.length(), true);
				}
			} else {
				if (Double.isNaN(m_FullMeansOrMediansOrModes[i])) {
					valMeanMode = pad("missing", " ",
							maxWidth + 1 - "missing".length(), true);
				} else {
					valMeanMode = pad(
							(strVal = Utils
									.doubleToString(
											m_FullMeansOrMediansOrModes[i],
											maxWidth, 4).trim()), " ", maxWidth
									+ 1 - strVal.length(), true);
				}
			}
			temp.append(valMeanMode);
			for (int j = 0; j < m_NumClusters; j++) {
				if (m_ClusterCentroids.attribute(i).isNominal()) {
					if (m_ClusterCentroids.instance(j).isMissing(i)) {
						valMeanMode = pad("missing", " ", maxWidth + 1
								- "missing".length(), true);
					} else {
						valMeanMode = pad(
								(strVal = m_ClusterCentroids.attribute(i)
										.value((int) m_ClusterCentroids
												.instance(j).value(i))), " ",
								maxWidth + 1 - strVal.length(), true);
					}
				} else {
					if (m_ClusterCentroids.instance(j).isMissing(i)) {
						valMeanMode = pad("missing", " ", maxWidth + 1
								- "missing".length(), true);
					} else {
						valMeanMode = pad(
								(strVal = Utils
										.doubleToString(
												m_ClusterCentroids.instance(j)
														.value(i), maxWidth, 4)
										.trim()), " ",
								maxWidth + 1 - strVal.length(), true);
					}
				}
				temp.append(valMeanMode);
			}
			temp.append("\n");
			if (m_displayStdDevs) {
				// Std devs/max nominal
				String stdDevVal = "";

				if (m_ClusterCentroids.attribute(i).isNominal()) {
					// Do the values of the nominal attribute
					Attribute a = m_ClusterCentroids.attribute(i);
					for (int j = 0; j < a.numValues(); j++) {
						// full data
						String val = "  " + a.value(j);
						temp.append(pad(val, " ",
								maxAttWidth + 1 - val.length(), false));
						double count = m_FullNominalCounts[i][j];
						int percent = (int) ((double) m_FullNominalCounts[i][j]
								/ Utils.sum(m_ClusterSizes) * 100.0);
						String percentS = "" + percent + "%)";
						percentS = pad(percentS, " ", 5 - percentS.length(),
								true);
						stdDevVal = "" + count + " (" + percentS;
						stdDevVal = pad(stdDevVal, " ", maxWidth + 1
								- stdDevVal.length(), true);
						temp.append(stdDevVal);

						// Clusters
						for (int k = 0; k < m_NumClusters; k++) {
							percent = (int) ((double) m_ClusterNominalCounts[k][i][j]
									/ m_ClusterSizes[k] * 100.0);
							percentS = "" + percent + "%)";
							percentS = pad(percentS, " ",
									5 - percentS.length(), true);
							stdDevVal = "" + m_ClusterNominalCounts[k][i][j]
									+ " (" + percentS;
							stdDevVal = pad(stdDevVal, " ", maxWidth + 1
									- stdDevVal.length(), true);
							temp.append(stdDevVal);
						}
						temp.append("\n");
					}
					// missing (if any)
					if (m_FullMissingCounts[i] > 0) {
						// Full data
						temp.append(pad("  missing", " ", maxAttWidth + 1
								- "  missing".length(), false));
						double count = m_FullMissingCounts[i];
						int percent = (int) ((double) m_FullMissingCounts[i]
								/ Utils.sum(m_ClusterSizes) * 100.0);
						String percentS = "" + percent + "%)";
						percentS = pad(percentS, " ", 5 - percentS.length(),
								true);
						stdDevVal = "" + count + " (" + percentS;
						stdDevVal = pad(stdDevVal, " ", maxWidth + 1
								- stdDevVal.length(), true);
						temp.append(stdDevVal);

						// Clusters
						for (int k = 0; k < m_NumClusters; k++) {
							percent = (int) ((double) m_ClusterMissingCounts[k][i]
									/ m_ClusterSizes[k] * 100.0);
							percentS = "" + percent + "%)";
							percentS = pad(percentS, " ",
									5 - percentS.length(), true);
							stdDevVal = "" + m_ClusterMissingCounts[k][i]
									+ " (" + percentS;
							stdDevVal = pad(stdDevVal, " ", maxWidth + 1
									- stdDevVal.length(), true);
							temp.append(stdDevVal);
						}

						temp.append("\n");
					}

					temp.append("\n");
				} else {
					// Full data
					if (Double.isNaN(m_FullMeansOrMediansOrModes[i])) {
						stdDevVal = pad("--", " ", maxAttWidth + maxWidth + 1
								- 2, true);
					} else {
						stdDevVal = pad(
								(strVal = plusMinus
										+ Utils.doubleToString(
												m_FullStdDevs[i], maxWidth, 4)
												.trim()), " ", maxWidth
										+ maxAttWidth + 1 - strVal.length(),
								true);
					}
					temp.append(stdDevVal);

					// Clusters
					for (int j = 0; j < m_NumClusters; j++) {
						if (m_ClusterCentroids.instance(j).isMissing(i)) {
							stdDevVal = pad("--", " ", maxWidth + 1 - 2, true);
						} else {
							stdDevVal = pad(
									(strVal = plusMinus
											+ Utils.doubleToString(
													m_ClusterStdDevs
															.instance(j).value(
																	i),
													maxWidth, 4).trim()), " ",
									maxWidth + 1 - strVal.length(), true);
						}
						temp.append(stdDevVal);
					}
					temp.append("\n\n");
				}
			}
		}

		temp.append("\n\n");
		return temp.toString();
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		runClusterer(new ABCFCM(), args);
	}
}
