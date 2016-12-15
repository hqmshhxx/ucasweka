package weka.classifiers.abc;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import weka.core.Instances;
import weka.core.Utils;

public class ABCANN implements Serializable{

	private static final long serialVersionUID = 2150281474585060975L;
	/** The number of colony size (employed bees+onlooker bees) */
	int NP = 50;
	/** The number of food sources equals the half of the colony size */
	int foodNum = NP / 2;
	/**
	 * A food source which could not be improved through "limit" trials is
	 * abandoned by its employed bee
	 */
	int limit = 10;
	/** The number of cycles for foraging {a stopping criteria} */
	int maxCycle = 5;
	int mCycle = 0;

	/** Problem specific variables */
	/** The number of parameters of the problem to be optimized */
	int dimension = 0;
	/** lower bound of the parameters. */
	double lb = -3;
	/**
	 * upper bound of the parameters. lb and ub can be defined as arrays for the
	 * problems of which parameters have different bounds
	 */
	double ub = 3;

	/** Algorithm can be run many times in order to see its robustness */
	int runCount = 30;

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
	
	private int ipNum = 6;
	private int hlNum = 3;
	private int opNum = 1;
	
	private Instances train;
	private BP bp;
	
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
		
		funVal[index] = calculateObjectiveFunction(solution);
		fitness[index] = calculateFitness(funVal[index]);
		trial[index] = 0;
	}

	/* All food sources are initialized */
	public void initial() {
		
		dimension = ipNum*hlNum+hlNum+hlNum*opNum+opNum;
		foods = new double[foodNum][dimension];
		bestFood = new double[dimension];
		int i;
		for (i = 0; i < foodNum; i++) {
			init(i);
		}
		minObjFunValue = funVal[0];
		for (i = 0; i < dimension; i++)
			bestFood[i] = foods[0][i];
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
			double objVal = calculateObjectiveFunction(neighbor);
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
		int i, j;
		for (i = 0; i < foodNum; i++) {
			if (funVal[i] < minObjFunValue) {
				minObjFunValue = funVal[i];
				for (j = 0; j < dimension; j++)
					bestFood[j] = foods[i][j];
			}
		}
	}

	
	/**
	 * Employed Bee Phase
	 */
	public void sendEmployedBees() {
		Random rand = new Random();
		for (int i = 0; i < foodNum; i++) {
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
			double objValSol = calculateObjectiveFunction(solution);
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

		/* end of employed bee phase */

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

	/** onlooker Bee Phase */
	public void sendOnlookerBees() {

		int i, j, t;
		i = 0;
		t = 0;
		Random rand = new Random();
		while (t < foodNum) {

			double r = rand.nextDouble();
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
				double[] solution = new double[dimension];
				for (j = 0; j < dimension; j++){
					solution[j] = foods[i][j];
				}
				
				double[] bestNeighbor = calculateNeighborBest(i);
				int minFIndex = Utils.minIndex(funVal);
				
				/* v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij}) */
				
				r = rand.nextDouble() * 2 - 1;
				solution[dj] =  bestNeighbor[dj] + (bestNeighbor[dj] - foods[neighbour][dj])* r+
						rand.nextDouble()*1.5*(foods[minFIndex][dj]-bestNeighbor[dj]);

				/*
				 * if generated parameter value is out of boundaries, it is
				 * shifted onto the boundaries
				 */
				if (solution[dj] < lb)
					solution[dj] = lb;
				if (solution[dj] > ub)
					solution[dj] = ub;
				double objValSol = calculateObjectiveFunction(solution);
				double fitnessSol = calculateFitness(objValSol);


				/*
				 * a greedy selection is applied between the current solution i
				 * and its mutant
				 */
				if (fitnessSol > fitness[i]) {
					/*
					 * If the mutant solution is better than the current
					 * solution i, replace the solution with the mutant and
					 * reset the trial counter of solution i
					 */
					trial[i] = 0;
					for (j = 0; j < dimension; j++)
						foods[i][j] = solution[j];
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
			i++;
			if (i == foodNum)
				i = 0;
		}/* while */

		/* end of onlooker bee phase */
	}

	/*
	 * determine the food sources whose trial counter exceeds the "limit" value.
	 * In Basic ABC, only one scout is allowed to occur in each cycle
	 */
	void sendScoutBees() {
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
		if (fun > 0) {
			result = 1 / (fun + 1);
		} else {
			result = 1;
		}
		return result;
	}

	
	/**
	 * calculate function value
	 * 
	 * @param sol
	 * @return
	 */
	public double calculateObjectiveFunction(double sol[]) {
//		return calculateErrors(sol);
		return buildNet(sol);

	}
	public double buildNet(double[] weights){
		double error = Double.MAX_VALUE;
		try {
			error = bp.buildNet(weights);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return  error;
	}
	public double calculateErrors(double[]solution){
		if(train == null){
			try {
				throw new Exception("train is null!");
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		double[] hiddenNodes = null;
		double[] outNodes = null;
		double errors = 0;
		for(int i=0; i<train.numInstances(); i++){
			hiddenNodes = calculateHiddenLayer(solution,i);
			outNodes = calculateOutLayer(solution,hiddenNodes);
			errors += calculateError(solution,i,outNodes);
		}
		
		return errors/(hlNum*opNum);
	}
	public double[] calculateHiddenLayer(double[] solution, int instanceIndex){
		int iwi = 0;
		double[] hiddenNodes = new double[hlNum];
		
		for(int hni=0; hni<hlNum; hni++){
			for(int j=0; j<train.numAttributes()-1; j++){
				hiddenNodes[hni] += solution[iwi++]*train.instance(instanceIndex).value(j);
			}
			hiddenNodes[hni] += solution[ipNum*hlNum+hni+1];
			hiddenNodes[hni] = 1/(1+Math.exp(-hiddenNodes[hni]));
		}
		return hiddenNodes;
	}
	public double[] calculateOutLayer(double[] solution,double[] hiddenNodes){
		int hwi = ipNum*hlNum+hlNum;
		double[] outNodes = new double[opNum];
		for(int opi=0; opi<opNum; opi++){
			for(int j=0; j<hlNum; j++){
				outNodes[opi] += solution[hwi++]*hiddenNodes[j];
			}
			outNodes[opi] += solution[hwi+opi];
			outNodes[opi] = 1/(1+Math.exp(-outNodes[opi]));
		}
		return outNodes;
	}
	public double calculateError(double[] solution,int instanceIndex,double[] outNodes){
		double error = 0;
		for(int opi=0; opi<opNum; opi++){
			error += outNodes[opi];
		}
		double realValue = train.instance(instanceIndex).value(train.numAttributes()-1);
		error = Math.pow((error-realValue), 2);
		return error;
	}
	public void setData(Instances data){
		train = new Instances(data);
	}
	public void setInputNum(int in){
		ipNum = in;
	}
	public void setHiddenNum(int hl){
		hlNum = hl;
	}
	public void setOutNum(int out){
		opNum = out;
	}
	public double getMinObjFunValue(){
		return minObjFunValue;
	}
	public double[] getBestFood(){
		return bestFood;
	}
	
	public void setBp(BP bp){
		this.bp = bp;
	}
	public void build(){
		try {
			bp.buildNetwork(train);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		System.out.println("start abc");
		initial();
		memorizeBestSource();
		for (int iter = 0; iter < maxCycle; iter++) {
			mCycle = iter + 1;
			sendEmployedBees();
			System.out.println("sendEmployedBees finished ");
			calculateProbabilities();
			sendOnlookerBees();
			System.out.println("sendOnlookerBees finished ");
			memorizeBestSource();
			sendScoutBees();
			System.out.println("sendScoutBees finished ");
			System.out.println("\nmcycle = " + mCycle+"\n");
		}
		System.out.println("人工蜂群的最小值：" + getMinObjFunValue());
		
	}
}
