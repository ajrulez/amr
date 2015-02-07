package edu.stanford.nlp.experiments.greedy;

import edu.stanford.nlp.experiments.ConstrainedSequence;
import edu.stanford.nlp.experiments.LinearPipe;
import edu.stanford.nlp.stamr.AMR;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.Pair;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * Created by keenon on 2/7/15.
 */
public class TrainableOracle extends Oracle {
    LinearPipe<Pair<GreedyState,Integer>, String> classifier;
    List<String> arcTypes = new ArrayList<>();

    int[] maxClassCounts;
    Map<String,Integer> specialClassExemptions = new HashMap<String,Integer>(){{
        put("NONE", -1);
        put("mod", 2);
    }};

    public TrainableOracle(AMR[] bank, List<Function<Pair<GreedyState,Integer>,Object>> features) {
        List<Pair<Pair<GreedyState,Integer>, String>> trainingExamples = new ArrayList<>();
        for (AMR amr : bank) {
            // Run the gold oracle over the AMR
            List<Pair<GreedyState,String[]>> derivation = TransitionRunner.run(
                    new GreedyState(GoldOracle.prepareForParse(amr.nodes), amr.sourceText, null),
                    new GoldOracle(amr));

            // Generate the training instances by examining the derivation
            for (Pair<GreedyState,String[]> pair : derivation) {
                if (pair.first.finished) continue;
                for (int i = 1; i < pair.second.length; i++) {
                    trainingExamples.add(new Pair<>(new Pair<>(pair.first, i), pair.second[i]));
                    // Keep a list of all the arcTypes in the training data, for interning later
                    if (!arcTypes.contains(pair.second[i])) arcTypes.add(pair.second[i]);
                }
            }
        }

        classifier = new LinearPipe<>(features, null);
        classifier.train(trainingExamples);

        maxClassCounts = new int[arcTypes.size()];
        for (int i = 0; i < arcTypes.size(); i++) {
            if (specialClassExemptions.containsKey(arcTypes.get(i))) {
                maxClassCounts[i] = specialClassExemptions.get(arcTypes.get(i));
            }
            else {
                maxClassCounts[i] = 1;
            }
        }
    }

    @Override
    public String[] predictArcs(GreedyState state) {

        // For the Boolean LP solver, we trim the ROOT element off the probability matrix, so that it doesn't get to
        // receive any arcs, which could screw up the constraints
        double[][] probs = new double[state.nodes.length-1][arcTypes.size()];
        for (int i = 0; i < state.nodes.length-1; i++) {
            Counter<String> arcTypeUnnormalizedLogLikelihoods = classifier.predictSoft(new Pair<>(state, i+1));

            // We still want everything in log space, but we want it normalized, so we do that
            double sum = 0.0;
            for (int j = 0; j < arcTypes.size(); j++) {
                probs[i][j] = Math.exp(arcTypeUnnormalizedLogLikelihoods.getCount(arcTypes.get(j)));
                sum += probs[i][j];
            }
            for (int j = 0; j < arcTypes.size(); j++) {
                probs[i][j] = Math.log(probs[i][j]/sum);
            }
        }

        int[] forcedClasses = new int[state.nodes.length-1];
        for (int i = 0; i < state.nodes.length-1; i++) {
            if (state.forcedArcs[state.head][i+1] != null) {
                forcedClasses[i] = arcTypes.indexOf(state.forcedArcs[state.head][i+1]);
            }
            else {
                forcedClasses[i] = -1;
            }
        }

        // Run a Boolean LP to figure out the constrained maximization problem :)

        int[] solvedClasses = ConstrainedSequence.solve(probs, maxClassCounts, forcedClasses);

        String[] arcs = new String[state.nodes.length];
        for (int i = 0; i < solvedClasses.length; i++) {
            arcs[i+1] = arcTypes.get(solvedClasses[i]);
        }
        return arcs;
    }
}
