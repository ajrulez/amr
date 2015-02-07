package edu.stanford.nlp.experiments.greedy;

import com.pholser.junit.quickcheck.generator.GenerationStatus;
import com.pholser.junit.quickcheck.generator.Generator;
import com.pholser.junit.quickcheck.random.SourceOfRandomness;
import edu.stanford.nlp.stamr.AMR;

import java.util.ArrayDeque;
import java.util.Queue;

/**
 * Created by keenon on 2/7/15.
 */
public class GreedyStateGen extends Generator<GreedyState> {
    public GreedyStateGen(Class<GreedyState> type) {
        super(type);
    }

    @Override
    public GreedyState generate(SourceOfRandomness sourceOfRandomness, GenerationStatus generationStatus) {
        int numTokens = sourceOfRandomness.nextInt(2, 10);
        String[] tokens = new String[numTokens];
        for (int i = 0; i < numTokens; i++) {
            tokens[i] = "blah";
        }

        int numNodes = sourceOfRandomness.nextInt(2, 10);
        AMR.Node[] nodes = new AMR.Node[numNodes];
        for (int i = 1; i < nodes.length; i++) {
            nodes[i] = new AMR.Node("b"+i, "blah"+i, AMR.NodeType.ENTITY);
            nodes[i].alignment = sourceOfRandomness.nextInt(numTokens);
        }

        GreedyState state = new GreedyState(nodes, tokens, null);

        while (!state.finished) {

            // Sometimes we stop early
            if (sourceOfRandomness.nextBoolean()) break;

            // Run a random oracle policy
            String[] headArcs = new String[nodes.length];
            for (int i = 1; i < nodes.length; i++) {
                if (sourceOfRandomness.nextDouble() > 0.8) {
                    headArcs[i] = "ARG"+sourceOfRandomness.nextInt(4);
                }
                else {
                    headArcs[i] = "NONE";
                }
            }
            state = state.transition(headArcs);
        }

        return state;
    }
}
