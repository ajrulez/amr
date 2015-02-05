package edu.stanford.nlp.experiments.greedy;

import edu.stanford.nlp.experiments.ConstrainedSequence;
import edu.stanford.nlp.experiments.LinearPipe;
import edu.stanford.nlp.stamr.AMR;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.Triple;

import java.util.*;
import java.util.function.Function;

/**
 * Created by keenon on 2/4/15.
 *
 * Just does the node connection portion of the AMR parsing.
 */
public class NodeConnector {

    List<String> classes = new ArrayList<>();

    LinearPipe<Pair<GreedyState,Integer>,String> arcTypePrediction = new LinearPipe<>(
            new ArrayList<Function<Pair<GreedyState,Integer>,Object>>(){{
                add((pair) -> {
                    GreedyState state = pair.first;
                    int cursor = pair.second;
                    StringBuilder sb = new StringBuilder();
                    while (cursor != 0) {
                        sb.append(state.nodes[cursor].toString());
                        cursor = state.originalParent[cursor];
                    }
                    return sb.toString();
                });
            }}, null);

    public void train(List<Pair<GreedyState,String[][]>> trainingData) {
        List<Pair<Pair<GreedyState,Integer>, String>> trainingExamples = new ArrayList<>();

        for (Pair<GreedyState,String[][]> pair : trainingData) {
            AMR.Node[] nodes = pair.first.nodes;
            String[][] arcs = pair.second;

            Queue<Integer> q = new ArrayDeque<>();
            Set<Integer> visited = new HashSet<>();
            q.add(0);

            GreedyState state = pair.first.deepClone();
            state.arcs = new String[arcs.length][arcs[0].length];
            state.originalParent = new int[nodes.length];

            GreedyState nextState = state.deepClone();

            while (!q.isEmpty()) {
                int head = q.poll();
                visited.add(head);
                state.head = head;
                for (int i = 1; i < nodes.length; i++) {
                    if (i == head) continue;
                    if (arcs[head][i] != null && !visited.contains(i)) q.add(i);
                    if (arcs[head][i] != null && state.originalParent[i] == 0) {
                        state.originalParent[i] = head;
                    }

                    String arcName = arcs[head][i] == null ? "NONE" : arcs[head][i];
                    if (!classes.contains(arcName)) classes.add(arcName);
                    nextState.arcs[head][i] = arcName;
                    trainingExamples.add(new Pair<>(new Pair<>(state, i), arcName));
                }
                state = nextState;
                nextState = state.deepClone();
            }
        }

        arcTypePrediction.train(trainingExamples);
    }

    public String[][] connect(AMR.Node[] nodes) {
        Queue<Integer> q = new ArrayDeque<>();
        Set<Integer> visited = new HashSet<>();
        q.add(0);

        GreedyState state = new GreedyState();
        state.nodes = nodes;
        state.arcs = new String[nodes.length][nodes.length];

        while (!q.isEmpty()) {
            int head = q.poll();
            visited.add(head);
            state.head = head;

            double[][] probs = new double[nodes.length][classes.size()];
            int[] maxCounts = new int[classes.size()];

            for (int i = 1; i < nodes.length; i++) {
                if (i == head) continue;
                // Fill in probs for each transition type
                Counter<String> counter = arcTypePrediction.predictSoft(new Pair<>(state, i));
                for (int j = 0; j < classes.size(); j++) {
                    probs[i][j] = counter.getCount(classes.get(j));
                }
            }

            int[] solvedClasses = ConstrainedSequence.solve(probs, maxCounts);
            for (int i = 1; i < nodes.length; i++) {
                state.arcs[state.head][i] = classes.get(solvedClasses[i]);
                if (!state.arcs[state.head][i].equals("NONE") && !visited.contains(i)) {
                    q.add(i);
                }
            }
        }

        return state.arcs;
    }

}