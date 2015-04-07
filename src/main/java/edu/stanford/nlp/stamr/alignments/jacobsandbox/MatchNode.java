package edu.stanford.nlp.stamr.alignments.jacobsandbox;

import edu.stanford.nlp.stamr.AMR;
import edu.stanford.nlp.stats.Counter;

/**
 * Created by jacob on 4/5/15.
 */
interface MatchNode {

    /**
     * The interface method. Provide a score for this node compared to
     * the given AMR node.
     *
     * @param match
     * @param dict
     * @return
     */
    public double score(AMR.Node match, Model.SoftCountDict dict);



    public static class DictMatchNode implements MatchNode {
        public final String name;
        public DictMatchNode(String name) {
            this.name = name;
        }
        public double score(AMR.Node match, Model.SoftCountDict dict) {
            return dict.getProb(name, match.title);
        }
    }

    public static class NoneMatchNode implements MatchNode {
        public NoneMatchNode() { }
        public double score(AMR.Node match, Model.SoftCountDict dict) {
            if(match instanceof NoneNode){
                return 1.0;
            } else {
                return 0.0;
            }
        }
    }

    public static class ExactMatchNode implements MatchNode {
        public final String name;
        public ExactMatchNode(String name) {
            this.name = name;
        }
        public double score(AMR.Node match, Model.SoftCountDict dict) {
            return this.name.equalsIgnoreCase(match.title) ? 1.0 : 0.0;
        }
    }

    public static class VerbMatchNode implements MatchNode {
        public final String verbName;
        public VerbMatchNode(String name) {
            this.verbName = name;
        }

        public double score(AMR.Node match, Model.SoftCountDict dict) {
            if (verbName.length() >= 2 && match.title.length() >= 2) {
                if (verbName.substring(0, verbName.length() - 2).equals(match.title.substring(0, match.title.length() - 2))) {
                    return 1.0;
                }
            }
            return 0.0;

        }
    }

    public static class NamedEntityMatchNode implements MatchNode {
        public final String name;
        public final String neTag;
        public NamedEntityMatchNode(String name, String neTag) {
            this.name = name;
            this.neTag = neTag;
        }
        public double score(AMR.Node match, Model.SoftCountDict dict) {
            if (neTag.equals(match.title) && match.op1 != null && match.op1.equals(name)) return 1.0;
            if (match.type == AMR.NodeType.QUOTE && name.equals(match.title)) return 1.0;
            return 0.0;
        }
    }

    public static class LemmaMatchNode implements MatchNode {
        public final Counter<String> candidates;
        public LemmaMatchNode(Counter<String> lemmas) {
            this.candidates = lemmas;
        }
        public double score(AMR.Node match, Model.SoftCountDict dict) {
            return candidates.getCount(match.title.toLowerCase());
        }
    }

    /*

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
            if (quoteType.equals(match.title) && match.op1 != null && match.op1.equals(name)) return 1.0;
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

    */
}
