package edu.stanford.nlp.stamr.alignments.jacobsandbox;

import edu.stanford.nlp.stamr.AMR;
import edu.stanford.nlp.stats.Counter;

import java.util.HashSet;

/**
 * Created by jacob on 4/5/15.
 * Heavily modified by Gabor 2015-04-06
 */
interface MatchNode {

    /**
     * The interface method. Provide a score for this node compared to
     * the given AMR node.
     *
     * @param match
     * @param dict
     * @param nerDict
     * @return
     */
    public double score(AMR.Node match, Model.SoftCountDict dict, Model.SoftCountDict nerDict);



    public static class DictMatchNode implements MatchNode {
        public final String name;
        public DictMatchNode(String name) {
            this.name = name;
        }
        public double score(AMR.Node match, Model.SoftCountDict dict, Model.SoftCountDict nerDict) {
            return dict.getProb(name, match.title);
        }

        @Override
        public String toString() {
            return "DictMatchNode{" +
                    "name='" + name + '\'' +
                    '}';
        }
    }

    public static class NoneMatchNode implements MatchNode {
        public NoneMatchNode() { }
        public double score(AMR.Node match, Model.SoftCountDict dict, Model.SoftCountDict nerDict) {
            if(match instanceof NoneNode){
                return 1.0;
            } else {
                return 0.0;
            }
        }

        @Override
        public String toString() {
            return "NoneMatchNode{}";
        }
    }

    public static class ExactMatchNode implements MatchNode {
        public final String name;
        public ExactMatchNode(String name) {
            this.name = name;
        }
        public double score(AMR.Node match, Model.SoftCountDict dict, Model.SoftCountDict nerDict) {
            return this.name.equalsIgnoreCase(match.title) ? 1.0 : 0.0;
        }

        @Override
        public String toString() {
            return "ExactMatchNode{" +
                    "name='" + name + '\'' +
                    '}';
        }
    }

    static boolean verbMatch(String lhs, String rhs){
        return lhs.length() >= 2 && rhs.length() >= 2
                && lhs.substring(0, lhs.length()-2).equals(rhs.substring(0, rhs.length()-2));
    }

    public static class VerbMatchNode implements MatchNode {
        public final String verbName;
        public VerbMatchNode(String name) {
            this.verbName = name;
        }

        public double score(AMR.Node match, Model.SoftCountDict dict, Model.SoftCountDict nerDict) {
            if(verbMatch(verbName, match.title)) return 1.0;
            else return 0.0;
//            if (verbName.length() >= 2 && match.title.length() >= 2) {
//                if (verbName.substring(0, verbName.length() - 2).equals(match.title.substring(0, match.title.length() - 2))) {
//                    return 1.0;
//                }
//            }
//            return 0.0;

        }

        @Override
        public String toString() {
            return "VerbMatchNode{" +
                    "verbName='" + verbName + '\'' +
                    '}';
        }
    }

    public static class NamedEntityMatchNode implements MatchNode {
        public final String name;
        public final String ner;
        public NamedEntityMatchNode(String name, String ner) {
            this.name = name;
            this.ner = ner;
        }
        public double score(AMR.Node match, Model.SoftCountDict dict, Model.SoftCountDict nerDict) {
            if(match instanceof NoneNode) return 0.0;
            // case 1: part of the quote
            if(match.type == AMR.NodeType.QUOTE
                    && match.neighborSet.contains("name") && match.title.equals(name)) return 1.0;
            // case 2: name node
            if(match.title.equals("name") && match.op1 != null && match.op1.equals(name)) return 1.0;
            // case 3: ner node
            if(match.type != AMR.NodeType.ENTITY) return 0.0;
            if(match.neighborSet.contains("name")){
                //System.out.println("match " + match + " ref " + match.ref + " keys " + match.amr.adjacencySet.keySet());
                HashSet<AMR.CorefGroup> neighbors = match.amr.adjacencySet.get(new AMR.CorefGroup(match.ref));
                AMR.Node nameNeighbor = null;
                outer:
                for(AMR.CorefGroup group : neighbors){
                    if(group == null) continue;
                    for(AMR.Node nodeInGroup : group.nodes){
                        if(nodeInGroup.title.equals("name")){
                            nameNeighbor = nodeInGroup;
                            break outer;
                        }
                    }
                }
                if(nameNeighbor == null || nameNeighbor.op1 == null || !nameNeighbor.op1.equals(name)) return 0.0;
                return nerDict.getProb(ner, match.title);
            }
//            if (neTag.equals(match.title) && match.op1 != null && match.op1.equals(name)) return 1.0;
//            if (match.type == AMR.NodeType.QUOTE && name.equals(match.title)) return 1.0;
            return 0.0;
        }

        @Override
        public String toString() {
            return "NamedEntityMatchNode{" +
                    "name='" + name + '\'' +
                    ", ner='" + ner + '\'' +
                    '}';
        }
    }

    public static class XerMatchNode implements MatchNode {
        public final String verb;
        public XerMatchNode(String verb){
            this.verb = verb;
        }
        public double score(AMR.Node match, Model.SoftCountDict dict, Model.SoftCountDict nerDict){
            if(verb != null){
                if ("person".equals(match.title)){
                    for(String title : match.neighborSet){
                        if(verbMatch(verb, title)) return 1.0;
                    }
                    return 0.0;
                }
                if (verbMatch(verb, match.title)) return 1.0;
            }
            return 0.0;
        }
    }

    public static class LemmaMatchNode implements MatchNode {
        public final String stanfordLemma;
        public final Counter<String> candidates;
        public LemmaMatchNode(String stanfordLemma, Counter<String> lemmas) {
            this.stanfordLemma = stanfordLemma;
            this.candidates = lemmas;
        }
        public double score(AMR.Node match, Model.SoftCountDict dict, Model.SoftCountDict nerDict) {
//            return candidates.containsKey(match.title.toLowerCase()) ? 1.0 : 0.0;
            if(match.type != AMR.NodeType.ENTITY) return 0.0;
            if (stanfordLemma.equalsIgnoreCase(match.title)) {
                return 1.0;
            } else {
                return candidates.getCount(match.title.toLowerCase());
            }
        }

        @Override
        public String toString() {
            return "LemmaMatchNode{" +
                    "candidates=" + candidates +
                    '}';
        }
    }

}
