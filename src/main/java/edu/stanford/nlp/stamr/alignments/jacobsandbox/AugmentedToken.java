package edu.stanford.nlp.stamr.alignments.jacobsandbox;

import edu.stanford.nlp.stamr.AMR;

import java.util.List;

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
    public final boolean blocked;
    public final List<String> pathFromRoot;
    JointEM.Action forcedAction;
    AMR amr = null;

    public AugmentedToken(int index, String value, String sense, String stem, String ner,
                          List<String> pathFromRoot, boolean blocked) {
        this.index = index;
        this.value = value;
        this.sense = sense;
        this.stem = stem;
        this.ner = ner;
        this.pathFromRoot = pathFromRoot;
        this.blocked = blocked;
        forcedAction = null;
        //if(blocked && !value.toLowerCase().equals("end")) forcedAction = Action.NONE;
        //else forcedAction = null;
    }
    /** A heuristic to determine if this is likely to be a punctuation token */
    public boolean isPunctuation() {
        char firstChar = value.charAt(0);
        return value.length() == 1 && firstChar != 'a' && firstChar != 'A' &&
                firstChar != 'i' && firstChar != 'I' &&
                !(firstChar >= '0' && firstChar <= '9');
    }
    /** The incoming edge. Note that dangling nodes (e.g., punctation) will look like ROOT */
    public String incomingEdge() {
        if (pathFromRoot.size() == 0) {
            return "root";
        } else {
            return pathFromRoot.get(pathFromRoot.size() - 1);
        }
    }

    @Override
    public String toString() {
        return value + "[" + index + "]";
    }
}
