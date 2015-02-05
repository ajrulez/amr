package edu.stanford.nlp.experiments.greedy;

import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.stamr.AMR;

/**
 * Created by keenon on 2/4/15.
 *
 * Holds the state necessary for a greedy linking of AMR nodes.
 */
public class GreedyState {
    int head;
    AMR.Node[] nodes;
    String[][] arcs;
    int[] originalParent;

    Annotation annotation;
    String[] tokens;

    public GreedyState deepClone() {
        GreedyState clone = new GreedyState();
        clone.nodes = nodes;
        clone.arcs = new String[arcs.length][arcs[0].length];
        for (int i = 0; i < arcs.length; i++) {
            System.arraycopy(arcs[i], 0, clone.arcs[i], 0, arcs[i].length);
        }
        clone.originalParent = new int[nodes.length];
        System.arraycopy(originalParent, 0, clone.originalParent, 0, originalParent.length);

        clone.annotation = annotation;
        clone.tokens = tokens;

        return clone;
    }
}
