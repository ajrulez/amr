package edu.stanford.nlp.stamr.alignments.jacobsandbox;

import edu.stanford.nlp.stamr.AMR;

/**
 * Created by jacob on 4/5/15.
 */
/* AugmentedToken should keep track of following:
 *   0. string value of token
 *   1. srl sense
 *   2. lemmatized value (stem)
 *   3. whether "blocked"
 *   We can start as a baseline by making sense = "NONE" and stem = token
 */
class AugmentedToken {
    public final int index;
    public final String value, sense, stem, ner;
    AMR amr = null;
    public final boolean blocked;
    public final JointEM.Action forcedAction;

    public AugmentedToken(int index, String value, String sense, String stem, String ner, boolean blocked) {
        this.index = index;
        this.value = value;
        this.sense = sense;
        this.stem = stem;
        this.ner = ner;
        this.blocked = blocked;
        forcedAction = null;
        //if(blocked && !value.toLowerCase().equals("end")) forcedAction = Action.NONE;
        //else forcedAction = null;
    }

    @Override
    public String toString() {
        return value + "[" + index + "]";
    }
}
