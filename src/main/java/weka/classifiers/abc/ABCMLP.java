package weka.classifiers.abc;

import java.util.Collections;
import java.util.StringTokenizer;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.Capabilities.Capability;

public class ABCMLP extends AbstractClassifier implements OptionHandler,
		WeightedInstancesHandler {

	private static final long serialVersionUID = 3101525950144598975L;
	private ABCANN abcAnn;
	private BP bp;

	public ABCMLP() {
		abcAnn = new ABCANN();
		bp = new BP();
	}

	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		// class
		result.enable(Capability.NUMERIC_CLASS);

		return result;
	}

	public void buildClassifier(Instances data) throws Exception {

		abcAnn.setData(data);
		abcAnn.build();
		double[] weights = abcAnn.getBestFood();
		bp.buildNetwork(data);
		bp.initWeights(weights);
		bp.buildClassifier(data);
	}

	public void setInputNum(int in) {
		abcAnn.setInputNum(in);
	}

	public void setHiddenNum(int hl) {
		abcAnn.setHiddenNum(hl);
	}

	public void setOutNum(int out) {
		abcAnn.setOutNum(out);
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		// the defaults can be found here!!!!
		String learningString = Utils.getOption('L', options);
		if (learningString.length() != 0) {
			setLearningRate((new Double(learningString)).doubleValue());
		} else {
			setLearningRate(0.3);
		}
		String momentumString = Utils.getOption('M', options);
		if (momentumString.length() != 0) {
			setMomentum((new Double(momentumString)).doubleValue());
		} else {
			setMomentum(0.2);
		}
		String epochsString = Utils.getOption('N', options);
		if (epochsString.length() != 0) {
			setTrainingTime(Integer.parseInt(epochsString));
		} else {
			setTrainingTime(500);
		}
		String valSizeString = Utils.getOption('V', options);
		if (valSizeString.length() != 0) {
			setValidationSetSize(Integer.parseInt(valSizeString));
		} else {
			setValidationSetSize(0);
		}

		String thresholdString = Utils.getOption('E', options);
		if (thresholdString.length() != 0) {
			setValidationThreshold(Integer.parseInt(thresholdString));
		} else {
			setValidationThreshold(20);
		}
		String hiddenLayers = Utils.getOption('H', options);
		if (hiddenLayers.length() != 0) {
			setHiddenLayers(hiddenLayers);
		} else {
			setHiddenLayers("a");
		}
		if (Utils.getFlag('G', options)) {
			setGUI(true);
		} else {
			setGUI(false);
		} // small note. since the gui is the only option that can change the
			// other
			// options this should be set first to allow the other options to
			// set
			// properly
		if (Utils.getFlag('A', options)) {
			setAutoBuild(false);
		} else {
			setAutoBuild(true);
		}
		if (Utils.getFlag('B', options)) {
			setNominalToBinaryFilter(false);
		} else {
			setNominalToBinaryFilter(true);
		}
		if (Utils.getFlag('C', options)) {
			setNormalizeNumericClass(false);
		} else {
			setNormalizeNumericClass(true);
		}
		if (Utils.getFlag('I', options)) {
			setNormalizeAttributes(false);
		} else {
			setNormalizeAttributes(true);
		}

		if (Utils.getFlag('D', options)) {
			setDecay(true);
		} else {
			setDecay(false);
		}

		super.setOptions(options);

		Utils.checkForRemainingOptions(options);
	}

	/**
	 * Gets the current settings of NeuralNet.
	 * 
	 * @return an array of strings suitable for passing to setOptions()
	 */
	@Override
	public String[] getOptions() {

		Vector<String> options = new Vector<String>();

		options.add("-L");
		options.add("" + getLearningRate());
		options.add("-M");
		options.add("" + getMomentum());
		options.add("-N");
		options.add("" + getTrainingTime());
		options.add("-V");
		options.add("" + getValidationSetSize());
		options.add("-E");
		options.add("" + getValidationThreshold());
		options.add("-H");
		options.add(getHiddenLayers());
		if (getGUI()) {
			options.add("-G");
		}
		if (!getAutoBuild()) {
			options.add("-A");
		}
		if (!getNominalToBinaryFilter()) {
			options.add("-B");
		}
		if (!getNormalizeNumericClass()) {
			options.add("-C");
		}
		if (!getNormalizeAttributes()) {
			options.add("-I");
		}

		if (getDecay()) {
			options.add("-D");
		}

		Collections.addAll(options, super.getOptions());

		return options.toArray(new String[0]);
	}

	/**
	 * @param d
	 *            True if the learning rate should decay.
	 */
	public void setDecay(boolean d) {
		bp.setDecay(d);
	}

	/**
	 * @return the flag for having the learning rate decay.
	 */
	public boolean getDecay() {
		return bp.getDecay();
	}

	/**
	 * @param c
	 *            True if the class should be normalized (the class will only
	 *            ever be normalized if it is numeric). (Normalization puts the
	 *            range between -1 - 1).
	 */
	public void setNormalizeNumericClass(boolean c) {
		bp.setNormalizeNumericClass(c);
	}

	/**
	 * @return The flag for normalizing a numeric class.
	 */
	public boolean getNormalizeNumericClass() {
		return bp.getNormalizeNumericClass();
	}

	/**
	 * @param a
	 *            True if the attributes should be normalized (even nominal
	 *            attributes will get normalized here) (range goes between -1 -
	 *            1).
	 */
	public void setNormalizeAttributes(boolean a) {
		bp.setNormalizeAttributes(a);
	}

	/**
	 * @return The flag for normalizing attributes.
	 */
	public boolean getNormalizeAttributes() {
		return bp.getNormalizeAttributes();
	}

	/**
	 * @param f
	 *            True if a nominalToBinary filter should be used on the data.
	 */
	public void setNominalToBinaryFilter(boolean f) {
		bp.setNominalToBinaryFilter(f);
	}

	/**
	 * @return The flag for nominal to binary filter use.
	 */
	public boolean getNominalToBinaryFilter() {
		return bp.getNominalToBinaryFilter();
	}

	/**
	 * This sets the threshold to use for when validation testing is being done.
	 * It works by ending testing once the error on the validation set has
	 * consecutively increased a certain number of times.
	 * 
	 * @param t
	 *            The threshold to use for this.
	 */
	public void setValidationThreshold(int t) {
		if (t > 0) {
			bp.setValidationThreshold(t);
		}
	}

	/**
	 * @return The threshold used for validation testing.
	 */
	public int getValidationThreshold() {
		return bp.getValidationThreshold();
	}

	/**
	 * The learning rate can be set using this command. NOTE That this is a
	 * static variable so it affect all networks that are running. Must be
	 * greater than 0 and no more than 1.
	 * 
	 * @param l
	 *            The New learning rate.
	 */
	public void setLearningRate(double l) {
		if (l > 0 && l <= 1) {
			bp.setLearningRate(l);
		}
	}

	/**
	 * @return The learning rate for the nodes.
	 */
	public double getLearningRate() {
		return bp.getLearningRate();
	}

	/**
	 * The momentum can be set using this command. THE same conditions apply to
	 * this as to the learning rate.
	 * 
	 * @param m
	 *            The new Momentum.
	 */
	public void setMomentum(double m) {
		if (m >= 0 && m <= 1) {
			bp.setMomentum(m);
		}
	}

	/**
	 * @return The momentum for the nodes.
	 */
	public double getMomentum() {
		return bp.getMomentum();
	}

	/**
	 * This will set whether the network is automatically built or if it is left
	 * up to the user. (there is nothing to stop a user from altering an
	 * autobuilt network however).
	 * 
	 * @param a
	 *            True if the network should be auto built.
	 */
	public void setAutoBuild(boolean a) {
		bp.setAutoBuild(a);
	}

	/**
	 * @return The auto build state.
	 */
	public boolean getAutoBuild() {
		return bp.getAutoBuild();
	}

	/**
	 * This will set what the hidden layers are made up of when auto build is
	 * enabled. Note to have no hidden units, just put a single 0, Any more 0's
	 * will indicate that the string is badly formed and make it unaccepted.
	 * Negative numbers, and floats will do the same. There are also some
	 * wildcards. These are 'a' = (number of attributes + number of classes) /
	 * 2, 'i' = number of attributes, 'o' = number of classes, and 't' = number
	 * of attributes + number of classes.
	 * 
	 * @param h
	 *            A string with a comma seperated list of numbers. Each number
	 *            is the number of nodes to be on a hidden layer.
	 */
	public void setHiddenLayers(String h) {
		bp.setHiddenLayers(h);
	}

	/**
	 * @return A string representing the hidden layers, each number is the
	 *         number of nodes on a hidden layer.
	 */
	public String getHiddenLayers() {
		return bp.getHiddenLayers();
	}

	/**
	 * This will set whether A GUI is brought up to allow interaction by the
	 * user with the neural network during training.
	 * 
	 * @param a
	 *            True if gui should be created.
	 */
	public void setGUI(boolean a) {
		bp.setGUI(a);
	}

	/**
	 * @return The true if should show gui.
	 */
	public boolean getGUI() {
		return bp.getGUI();
	}

	/**
	 * This will set the size of the validation set.
	 * 
	 * @param a
	 *            The size of the validation set, as a percentage of the whole.
	 */
	public void setValidationSetSize(int a) {
		bp.setValidationSetSize(a);
	}

	/**
	 * @return The percentage size of the validation set.
	 */
	public int getValidationSetSize() {
		return bp.getValidationSetSize();
	}

	/**
	 * Set the number of training epochs to perform. Must be greater than 0.
	 * 
	 * @param n
	 *            The number of epochs to train through.
	 */
	public void setTrainingTime(int n) {
		bp.setTrainingTime(n);
	}

	/**
	 * @return The number of epochs to train through.
	 */
	public int getTrainingTime() {
		return bp.getTrainingTime();
	}

	@Override
	public String toString() {
		return bp.toString();
	}

	public static void main(String[] argv) {
		runClassifier(new ABCMLP(), argv);
	}
}
