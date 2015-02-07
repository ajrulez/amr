package edu.stanford.nlp.experiments.greedy;

import edu.stanford.nlp.util.Pair;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by keenon on 2/7/15.
 */
public class TransitionRunner {
    public static List<Pair<GreedyState,String[]>> run(GreedyState state, Oracle oracle) {
        List<Pair<GreedyState,String[]>> decisions = new ArrayList<>();

        // Add state progressions and oracle decisions
        while (!state.finished) {
            String[] decision = oracle.predictArcs(state);
            decisions.add(new Pair<>(state, decision));
            state = state.transition(decision);
        }

        // Add final state
        decisions.add(new Pair<>(state, null));

        return decisions;
    }
}
