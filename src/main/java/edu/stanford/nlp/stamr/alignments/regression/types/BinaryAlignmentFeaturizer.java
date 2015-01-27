package edu.stanford.nlp.stamr.alignments.regression.types;

import edu.stanford.nlp.stamr.AMR;
import edu.stanford.nlp.stamr.ontonotes.SRL;

/**
 * Holds a feature for binary alignment probabilities
 */
public abstract class BinaryAlignmentFeaturizer {
    public abstract String featurize(SRL srl, SRL.Arc arc);
    public abstract String featurize(AMR amr, AMR.Node node, int token, int parentToken, AMR.Arc parentArc);

    public boolean equals(Object o) {
        return getClass().equals(o.getClass());
    }
}
