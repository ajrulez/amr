package edu.stanford.nlp.stamr.alignments.regression.unary;

import edu.stanford.nlp.stamr.AMR;
import edu.stanford.nlp.stamr.AMRConstants;
import edu.stanford.nlp.stamr.alignments.regression.types.UnaryAlignmentFeaturizer;
import edu.stanford.nlp.stamr.ontonotes.SRL;

/**
 * Tries to capture the intuition that in the DFS order of tokens,
 * the second and so-on matches are more likely to be pronouns, supposedly.
 */
public class DFSOrderPronounFeaturizer extends UnaryAlignmentFeaturizer {
    @Override
    public String featurize(SRL srl) {
        return null;
    }

    @Override
    public String featurize(AMR amr, AMR.Node node, int token) {
        if (!node.isFirstRef) {
            if (AMRConstants.pronouns.contains(amr.getSourceToken(token))) {
                return "DFSPRONOUN:TRUE";
            }
            else {
                return "DFSPRONOUN:FALSE";
            }
        }
        return null;
    }
}
