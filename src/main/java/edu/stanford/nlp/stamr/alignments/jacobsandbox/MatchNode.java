package edu.stanford.nlp.stamr.alignments.jacobsandbox;

import edu.stanford.nlp.stamr.AMR;

/**
 * Created by jacob on 4/5/15.
 */
class MatchNode {
    boolean isDict;
    String name;
    String quoteType;

    public MatchNode(String name) {
        this.name = name;
        this.isDict = false;
        this.quoteType = null;
    }

    public MatchNode(String name, boolean isDict) {
        this.name = name;
        this.isDict = isDict;
        this.quoteType = null;
    }

    public MatchNode(String name, String quoteType) {
        this.name = name;
        this.isDict = false;
        this.quoteType = quoteType;
    }

    double score(AMR.Node match, Model.SoftCountDict dict) {
        if (match instanceof NoneNode) return 0.0;
        if (quoteType != null) {
            if (quoteType.equals(match.title) && match.op1.equals(name)) return 1.0;
            if (match.type == AMR.NodeType.QUOTE && name.equals(match.title)) return 1.0;
            return 0.0;
        } else if (!isDict) {
            if (name.equals(match.title)) return 1.0;
            else return 0.0;
        } else {
            return dict.getProb(name, match.title);
        }
    }

    @Override
    public String toString() {
        if (isDict) return "DICT(" + name + ")";
        else return name;
    }
}
