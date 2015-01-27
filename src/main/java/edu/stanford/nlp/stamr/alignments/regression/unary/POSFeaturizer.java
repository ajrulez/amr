package edu.stanford.nlp.stamr.alignments.regression.unary;

import edu.stanford.nlp.stamr.AMR;
import edu.stanford.nlp.stamr.alignments.regression.types.UnaryAlignmentFeaturizer;
import edu.stanford.nlp.stamr.ontonotes.SRL;

/**
 * Creates simple unary features for POS tags
 */
public class POSFeaturizer extends UnaryAlignmentFeaturizer {
    @Override
    public String featurize(SRL srl) {
        return null;
    }

    @Override
    public String featurize(AMR amr, AMR.Node node, int token) {
        return node.title + ":" + amr.annotationWrapper.getPOSTagAtIndex(token);
    }
}
