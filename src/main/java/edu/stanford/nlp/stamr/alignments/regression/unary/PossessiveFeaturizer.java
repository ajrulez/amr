package edu.stanford.nlp.stamr.alignments.regression.unary;

import edu.stanford.nlp.stamr.AMR;
import edu.stanford.nlp.stamr.alignments.regression.types.UnaryAlignmentFeaturizer;
import edu.stanford.nlp.stamr.ontonotes.SRL;

/**
 * Fires if POS tag contains "$", and AMR node is linked to by a ":poss"
 */
public class PossessiveFeaturizer extends UnaryAlignmentFeaturizer {

    @Override
    public String featurize(SRL srl) {
        return null;
    }

    @Override
    public String featurize(AMR amr, AMR.Node node, int token) {
        if (amr.getParentArc(node).title.equals("poss")) {
            if (amr.annotationWrapper.getPOSTagAtIndex(token).contains("$"))
                return "POSSPOS:TRUE";
            else
                return "POSSPOS:FALSE";
        }
        return null;
    }
}
