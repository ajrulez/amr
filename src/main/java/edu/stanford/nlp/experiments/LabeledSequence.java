package edu.stanford.nlp.experiments;

import edu.stanford.nlp.pipeline.Annotation;

/**
 * Created by keenon on 1/27/15.
 *
 * Holds all the interesting information associated with a labeled sequence
 */
public class LabeledSequence {
    public String[] tokens;
    public String[] labels;
    public Annotation annotation;
}
