package edu.stanford.nlp.experiments.greedy;

import com.github.keenon.minimalml.cache.CoreNLPCache;
import edu.stanford.nlp.experiments.ConstrainedSequence;
import edu.stanford.nlp.experiments.LinearPipe;
import edu.stanford.nlp.stamr.AMR;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.Pair;

import java.io.BufferedWriter;
import java.io.IOException;
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
    int[] minClassCounts;

    int[] rootMaxClassCounts;
    int[] rootMinClassCounts;

    Map<String,Integer> specialClassExemptions = new HashMap<String,Integer>(){{
        put("NONE", -1);
    }};

    private static List<Pair<Pair<GreedyState,Integer>, String>> toTrainingExamples(AMR[] bank, CoreNLPCache cache) {
        List<Pair<Pair<GreedyState,Integer>, String>> trainingExamples = new ArrayList<>();
        for (int i = 0; i < bank.length; i++) {
            AMR amr = bank[i];
            // Run the gold oracle over the AMR
            List<Pair<GreedyState, String[]>> derivation = TransitionRunner.run(
                    new GreedyState(GoldOracle.prepareForParse(amr.nodes),
                            amr.sourceText,
                            cache == null ? null : cache.getAnnotation(i)),
                    new GoldOracle(amr));

            // Generate the training instances by examining the derivation
            for (Pair<GreedyState,String[]> pair : derivation) {
                if (pair.first.finished) continue;
                for (int j = 1; j < pair.second.length; j++) {
                    trainingExamples.add(new Pair<>(new Pair<>(pair.first, j), pair.second[j]));
                }
            }
        }

        return trainingExamples;
    }


    public TrainableOracle(AMR[] bank, List<Function<Pair<GreedyState,Integer>,Object>> features, CoreNLPCache cache) {
        // Keep a list of all the arcTypes in the training data, for interning later
        arcTypes.add("NONE");
        arcTypes.add("ROOT");
        for (AMR amr : bank) {
            for (AMR.Arc arc : amr.arcs) {
                if (!arcTypes.contains(arc.title)) arcTypes.add(arc.title);
            }
        }

        maxClassCounts = new int[arcTypes.size()];
        minClassCounts = new int[arcTypes.size()];
        rootMaxClassCounts = new int[arcTypes.size()];
        rootMinClassCounts = new int[arcTypes.size()];
        for (int i = 0; i < arcTypes.size(); i++) {
            minClassCounts[i] = -1;

            if (specialClassExemptions.containsKey(arcTypes.get(i))) {
                maxClassCounts[i] = specialClassExemptions.get(arcTypes.get(i));
            }
            else {
                maxClassCounts[i] = 1;
            }

            if (arcTypes.get(i).equals("ROOT")) {
                rootMaxClassCounts[i] = 1;
                rootMinClassCounts[i] = 1;
            }
            else if (arcTypes.get(i).equals("NONE")) {
                rootMaxClassCounts[i] = -1;
                rootMinClassCounts[i] = -1;
            }
            else {
                rootMaxClassCounts[i] = 0;
                rootMinClassCounts[i] = 0;
            }
        }

        // Go through and count all the arcTypes, so we don't accidentally clip too aggressively
        for (int i = 0; i < bank.length; i++) {
            AMR amr = bank[i];
            // Run the gold oracle over the AMR
            List<Pair<GreedyState, String[]>> derivation = TransitionRunner.run(
                    new GreedyState(GoldOracle.prepareForParse(amr.nodes),
                            amr.sourceText,
                            cache == null ? null : cache.getAnnotation(i)),
                    new GoldOracle(amr));

            for (Pair<GreedyState,String[]> pair : derivation) {
                if (pair.first.finished) continue;
                Counter<String> arcTypeCounter = new ClassicCounter<>();
                for (String arc : pair.second) {
                    arcTypeCounter.incrementCount(arc);
                }
                for (String arc : arcTypeCounter.keySet()) {
                    int j = arcTypes.indexOf(arc);
                    if (maxClassCounts[j] > -1 && maxClassCounts[j] < arcTypeCounter.getCount(arc)) {
                        maxClassCounts[j] = (int) arcTypeCounter.getCount(arc);
                    }
                }
            }
        }

        List<Pair<Pair<GreedyState,Integer>, String>> trainingExamples = toTrainingExamples(bank, cache);

        classifier = new LinearPipe<>(features, TrainableOracle::debugOracleState);
        classifier.automaticallyReweightTrainingData = true;
        classifier.train(trainingExamples);
    }

    public static void debugOracleState(Pair<GreedyState,Integer> pair, BufferedWriter bw) {
        try {
            GreedyState state = pair.first;
            AMR.Node headNode = state.nodes[state.head];
            bw.write("Head Node: ");
            if (headNode == null) bw.write("ROOT");
            else bw.write(headNode.toString());
            bw.write("\n");

            AMR.Node tailNode = state.nodes[pair.second];
            bw.write("Tail Node: "+tailNode.toString()+"\n");

            int focus = tailNode.alignment;
            int headFocus = -1;
            if (headNode != null) headFocus = headNode.alignment;

            for (int i = 0; i < state.tokens.length; i++) {
                if (i == focus) bw.write(">TAIL>");
                if (i == headFocus) bw.write(">HEAD>");
                bw.write(state.tokens[i]);
                if (i == headFocus) bw.write("<HEAD<");
                if (i == focus) bw.write("<TAIL<");
                if (i != state.tokens.length-1) bw.write(" ");
            }
            bw.write("\n");
            bw.write("Stack: [");
            for (int i : state.q) {
                bw.write(" ");
                bw.write("\""+state.nodes[i].toString()+"\"");
            }
            bw.write(" ]\n");
            bw.write("Partial AMR:\n");
            AMR partial = Generator.generateAMR(state);
            bw.write(partial.toString(AMR.AlignmentPrinting.ALL));
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void analyze(AMR[] train,
                        CoreNLPCache trainCache,
                        AMR[] test,
                        CoreNLPCache testCache,
                        String directory) throws IOException {
        classifier.analyze(toTrainingExamples(train, trainCache), toTrainingExamples(test, testCache), directory);
    }

    public AMR cheat = null;

    @Override
    public String[] predictArcs(GreedyState state) {

        // For the Boolean LP solver, we trim the ROOT element off the probability matrix, so that it doesn't get to
        // receive any arcs, which could screw up the constraints
        double[][] probs = new double[state.nodes.length-1][arcTypes.size()];
        for (int i = 0; i < state.nodes.length-1; i++) {
            if (state.nodes[i+1] == null) continue;
            Counter<String> arcTypeUnnormalizedLogLikelihoods = classifier.predictSoft(new Pair<>(state, i+1));

            // We don't want to get nasty numerical errors with the log-roundoff, so we adjust all the logs to the mean
            double sum = 0;
            for (int j = 0; j < arcTypes.size(); j++) {
                probs[i][j] = Math.exp(arcTypeUnnormalizedLogLikelihoods.getCount(arcTypes.get(j)));
                sum += probs[i][j];
            }
            for (int j = 0; j < arcTypes.size(); j++) {
                probs[i][j] = Math.log(probs[i][j] / sum);
            }
        }

        /*
        if (cheat != null) {
            String[] gold = new GoldOracle(cheat).predictArcs(state);
            for (int i = 0; i < state.nodes.length-1; i++) {
                for (int j = 0; j < arcTypes.size(); j++) {
                    if (gold[i+1].equals(arcTypes.get(j))) {
                        probs[i][j] = 0;
                    }
                    else {
                        probs[i][j] = -1000;
                    }
                }
            }
        }
        */

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

        int[] solvedClasses;
        if (state.head == 0) {
            solvedClasses = ConstrainedSequence.solve(probs, rootMaxClassCounts, rootMinClassCounts, forcedClasses);
        }
        else {
            solvedClasses = ConstrainedSequence.solve(probs, maxClassCounts, minClassCounts, forcedClasses);
        }

        String[] arcs = new String[state.nodes.length];
        arcs[0] = "NONE";
        for (int i = 0; i < solvedClasses.length; i++) {
            arcs[i+1] = arcTypes.get(solvedClasses[i]);
        }

        if (cheat != null) {
            String[] gold = new GoldOracle(cheat).predictArcs(state);
            for (int i = 1; i < arcs.length; i++) {
                if (!gold[i].equals(arcs[i])) {
                    System.out.println("Breaking");
                }
            }
        }

        return arcs;
    }
}
