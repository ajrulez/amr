package edu.stanford.nlp.stamr.alignments.jacobsandbox;

import edu.stanford.nlp.stamr.AMR;

/**
 * Created by jacob on 4/5/15.
 */
public class AMRRuleNode extends MatchNode {
    AMR amr;
    public AMRRuleNode(AMR amr){
        super("AMRRule");
        this.amr = amr;
    }

    @Override
    double score(AMR.Node match, Model.SoftCountDict dict) {
        if(amr == null) return 0.0;
        // TODO check that neighbors match too
        if(amr.nodeWithName(match.title) != null) return 1.0;
        else return 0.0;
    }
}
