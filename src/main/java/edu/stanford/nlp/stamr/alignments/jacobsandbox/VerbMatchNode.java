package edu.stanford.nlp.stamr.alignments.jacobsandbox;

import edu.stanford.nlp.stamr.AMR;

/**
 * TODO(gabor) JavaDoc
 * TODO(gabor) create subclasses of MatchNode for all actions, overriding score()
 *
 * @author Gabor Angeli
 */
public class VerbMatchNode extends MatchNode {
    public VerbMatchNode(String name) {
        super(name);
    }

    double score(AMR.Node match, Model.SoftCountDict dict) {
        if (name.length() >= 2 && match.title.length() >= 2) {
            if (name.substring(0, name.length() - 2).equals(match.title.substring(0, match.title.length() - 2))) {
                return 1.0;
            }
        }
        return 0.0;

    }
}
