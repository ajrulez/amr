package edu.stanford.nlp.experiments.tests;

import edu.stanford.nlp.experiments.AMRPipelineStateBased;
import edu.stanford.nlp.experiments.greedy.GreedyState;
import edu.stanford.nlp.util.Pair;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * Created by keenon on 2/8/15.
 */
public class Test0 {
    static List<Function<Pair<GreedyState,Integer>,Object>> bfsOracleFeatures =
            new ArrayList<Function<Pair<GreedyState,Integer>,Object>>(){{
                add(pair -> {
                    GreedyState state = pair.first;
                    StringBuilder sb = new StringBuilder();
                    sb.append(state.nodes[pair.second].toString());
                    sb.append(state.nodes[pair.second].alignment);
                    int cursor = state.head;
                    while (cursor > 0) {
                        if (state.nodes[cursor] == null) {
                            System.out.println("Got a NULL");
                        }
                        sb.append(state.nodes[cursor].toString());
                        cursor = state.originalParent[cursor];
                    }
                    sb.append("(ROOT)");
                    sb.append(":");
                    sb.append(pair.first.tokens[0] + ":" + pair.first.tokens[1]);
                    return sb.toString();
                });
            }};

    public static void main(String[] args) throws IOException, InterruptedException {
        AMRPipelineStateBased.testPipeline("test0", bfsOracleFeatures);
    }
}
