package edu.stanford.nlp.stamr.alignments.regression.unary;

import edu.stanford.nlp.stamr.AMR;
import edu.stanford.nlp.stamr.alignments.regression.types.UnaryAlignmentFeaturizer;
import edu.stanford.nlp.stamr.ontonotes.SRL;

/**
 * Handles AMR-english co-occurrence
 */
public class LexicalFeaturizer extends UnaryAlignmentFeaturizer {
    @Override
    public String featurize(SRL srl) {
        return srl.sense.toLowerCase()+":"+srl.sourceToken.toLowerCase();
    }

    @Override
    public String featurize(AMR amr, AMR.Node node, int token) {
        return node.title.toLowerCase()+":"+amr.getSourceToken(token).toLowerCase();
    }
}
