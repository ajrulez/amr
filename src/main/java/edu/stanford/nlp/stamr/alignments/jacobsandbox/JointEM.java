package edu.stanford.nlp.stamr.alignments.jacobsandbox;

import edu.stanford.nlp.cache.BatchCoreNLPCache;
import edu.stanford.nlp.cache.CoreNLPCache;
import edu.stanford.nlp.curator.CuratorAnnotations;
import edu.stanford.nlp.curator.PredicateArgumentAnnotation;
import edu.stanford.nlp.experiments.AMRNodeSet;
import edu.stanford.nlp.experiments.FrameManager;
import edu.stanford.nlp.experiments.LabeledSequence;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.stamr.AMR;
import edu.stanford.nlp.stamr.AMRSlurp;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.word2vec.Word2VecLoader;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Created by jacob on 3/26/15.
 * Purpose: test out new EM algorithm, which works as follows:
 *   1. We do the alignment and sequence tagging as a single model, by
 *      associating each alignment with an action (VERB, LEMMA, DICT, etc.)
 *      and only aligning to words that match that action.
 *   2. Because it's been way too long since I've done anything Bayesian,
 *      and I want to be a cool kid, we also handle the "DICT" feature using
 *      a Dirichlet process.
 */
public class JointEM {
    static FrameManager frameManager;
    static {
        try {
            frameManager = new FrameManager("data/frames");
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    void doEM(String path){
        // first, grab all the data
        AMR[] bank = null;
        try {
            bank = AMRSlurp.slurp(path, AMRSlurp.Format.LDC);
        } catch(IOException e){
            e.printStackTrace();
        }

        String[] sentences = new String[bank.length];
        for (int i = 0; i < bank.length; i++) {
            sentences[i] = bank[i].formatSourceTokens();
        }
        CoreNLPCache cache = new BatchCoreNLPCache(path, sentences);
        AugmentedToken[][] tokens = new AugmentedToken[bank.length][];
        AMR.Node[][] nodes = new AMR.Node[bank.length][];
        for(int i = 0; i < bank.length; i++){
            Annotation annotation = cache.getAnnotation(i);
            tokens[i] = augmentedTokens(bank[i].sourceText, annotation);
            nodes[i] = bank[i].nodes.toArray(new AMR.Node[0]);
        }

        // now tokens[i] has all the required token info
        // and nodes[i] has all the required node info
        // and we can just solve the alignment problem

    }

    enum Action { IDENTITY, NONE, VERB, LEMMA, DICT }
    /* AugmentedToken should keep track of following:
     *   0. string value of token
     *   1. srl sense
     *   2. lemmatized value (stem)
     *   3. whether "blocked"
     *   We can start as a baseline by making sense = "NONE" and stem = token
     */
    static class AugmentedToken {
        String value, sense, stem;
        boolean blocked;
        public AugmentedToken(String value, String sense, String stem, boolean blocked){
            this.value = value;
            this.sense = sense;
            this.stem = stem;
            this.blocked = blocked;
        }
    }

    static class MatchNode {
        boolean isDict;
        String name;
        public MatchNode(String name){
            this.name = name;
            this.isDict = false;
        }
        public MatchNode(String name, boolean isDict){
            this.name = name;
            this.isDict = true;
        }
        double score(AMR.Node match){
            if(!isDict){
                if(name.equals(match.title)) return 1.0;
                else return 0.0;
            } else {
                // be lazy for now, just return a low-ish number
                return 0.2;
            }
        }
    }

    MatchNode getNode(AugmentedToken token, Action action){
        switch(action){
            case IDENTITY:
                return new MatchNode(token.value.toLowerCase());
            case NONE:
                return null;
            case VERB:
                String srlSense = token.sense;
                if (srlSense.equals("-") || srlSense.equals("NONE")) {
                    String stem = token.stem;
                    return new MatchNode(frameManager.getClosestFrame(stem));
                } else {
                    return new MatchNode(srlSense);
                }
            case LEMMA:
                return new MatchNode(token.stem.toLowerCase());
            case DICT:
                return new MatchNode(token.value, true);
            default:
                throw new RuntimeException("invalid action");
        }
    }

    private AugmentedToken[] augmentedTokens(String[] tokens, Annotation annotation) {
        AugmentedToken[] output = new AugmentedToken[tokens.length];
        Set<Integer> blocked = new HashSet<>();

        LabeledSequence labeledSequence = new LabeledSequence();
        labeledSequence.tokens = tokens;
        labeledSequence.annotation = annotation;

        // Force anything inside an -LRB- -RRB- to be NONE, which is how AMR generally handles it

        boolean inBrace = false;
        for (int i = 0; i < tokens.length; i++) {
            if (tokens[i].equals("-LRB-")) {
                inBrace = true;
                blocked.add(i);
            }
            if (tokens[i].equals("-RRB-")) {
                inBrace = false;
                blocked.add(i);
            }
            if (inBrace) {
                blocked.add(i);
            }
        }

        for (int i = 0; i < tokens.length; i++) {
            String srlSense = getSRLSenseAtIndex(labeledSequence.annotation, i);
            String stem = labeledSequence.annotation.get(CoreAnnotations.TokensAnnotation.class).
                            get(i).get(CoreAnnotations.LemmaAnnotation.class).toLowerCase();
            output[i] = new AugmentedToken(tokens[i], srlSense, stem, blocked.contains(i));
        }

        return output;
    }

    private String getSRLSenseAtIndex(Annotation annotation, int index) {
        PredicateArgumentAnnotation srl = annotation.get(CuratorAnnotations.PropBankSRLAnnotation.class);
        if (srl == null) return "-"; // If we have no SRL on this example, then oh well
        for (PredicateArgumentAnnotation.AnnotationSpan span : srl.getPredicates()) {
            if ((span.startToken <= index) && (span.endToken >= index + 1)) {
                String sense = span.getAttribute("predicate") + "-" + span.getAttribute("sense");
                return sense;
            }
        }
        return "NONE";
    }



}
