package edu.stanford.nlp.experiments.greedy;

/**
 * Created by keenon on 2/7/15.
 */
public abstract class Oracle {
    public abstract String[] predictArcs(GreedyState state);
}
