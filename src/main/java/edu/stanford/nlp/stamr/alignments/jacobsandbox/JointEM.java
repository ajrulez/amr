package edu.stanford.nlp.stamr.alignments.jacobsandbox;

import edu.stanford.nlp.cache.BatchCoreNLPCache;
import edu.stanford.nlp.cache.CoreNLPCache;
import edu.stanford.nlp.curator.CuratorAnnotations;
import edu.stanford.nlp.curator.PredicateArgumentAnnotation;
import edu.stanford.nlp.experiments.FrameManager;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.stamr.AMR;
import edu.stanford.nlp.stamr.AMRSlurp;
import edu.stanford.nlp.util.concurrent.AtomicDouble;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.*;

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

    public static void main(String[] args) throws Exception {
//        doEM("data/train-3-subset.txt");
        doEM("data/training-500-subset.txt");
    }
    static void doEM(String path) throws Exception {
        int maxSize = 20;
        // first, grab all the data
        AMRSlurp.doingAlignments = false;
        AMR[] bank = AMRSlurp.slurp(path, AMRSlurp.Format.LDC);
        int bankSize = Math.min(bank.length, maxSize);
        AugmentedToken[][] tokens = new AugmentedToken[bankSize][];
        AMR.Node[][] nodes = new AMR.Node[bankSize][];
        read(path, bank, tokens, nodes, bankSize);

        // now tokens[i] has all the required token info
        // and nodes[i] has all the required node info
        // and we can just solve the alignment problem
        theta = new ConcurrentHashMap<String, AGPair>();
        double eta = 0.3;
        for(int iter = 0; iter < 5; iter++){
            final AtomicDouble logZTot = new AtomicDouble(0.0);
            ExecutorService threadPool = Executors.newFixedThreadPool(4);
            ArrayList<Future<Void>> threads = new ArrayList<>();
            for(int n = 0; n < bankSize; n++){
                final AugmentedToken[] curTokens = tokens[n];
                final AMR.Node[] curNodes = nodes[n];
                final int curN = n;
                Callable<Void> thread = new Callable<Void>() {
                    @Override
                    public Void call() throws Exception {
                        // do IBM model 1
                        // print current best predictions (for debugging)
                        for (AugmentedToken token : curTokens) {
                            Action bestAction = null;
                            double bestScore = 0.0;
                            for (Action action : Action.values()) {
                                List<String> features = extractFeatures(token, action);
                                double curScore = score(features);
                                if (bestAction == null || curScore > bestScore) {
                                    bestAction = action;
                                    bestScore = curScore;
                                }
                            }
                            if (curN == 0)
                                System.out.println("OPT(" + token + ") : " + bestAction + " => " + getNode(token, bestAction));
                        }


                        // loop over output
                        for (AMR.Node node : curNodes) {
                            double Zsrc = 0.0, Ztar = 0.0;
                            Map<String, Double> gradientSrc = new HashMap<String, Double>(),
                                    gradientTar = new HashMap<String, Double>();
                            for (AugmentedToken token : curTokens) {
                                double logZ2 = Double.NEGATIVE_INFINITY;
                                for (Action action : Action.values()) {
                                    List<String> features = extractFeatures(token, action);
                                    logZ2 = lse(logZ2, score(features));
                                }
                                for (Action action : Action.values()) {
                                    List<String> features = extractFeatures(token, action);
                                    double probSrc = Math.exp(score(features) - logZ2);
                                    double likelihood = getNode(token, action).score(node, curNodes.length);
                                    if (curN == 0 && likelihood > 0.5)
                                        System.out.println("likelihood(" + token + "," + action + "," + node + ") = " + likelihood);
                                    double probTar = probSrc * likelihood;
                                    Zsrc += probSrc;
                                    Ztar += probTar;
                                    incr(gradientSrc, features, probSrc);
                                    incr(gradientTar, features, probTar);
                                }
                            }
                            Map<String, Double> gradient = new HashMap<String, Double>();
                            incr(gradient, gradientSrc, -1.0 / Zsrc);
                            incr(gradient, gradientTar, 1.0 / Ztar);
                            adagrad(gradient, eta);
                            logZTot.addAndGet(Math.log(Zsrc / Ztar));
                        }
                        return null;
                    }
                };
                threads.add(threadPool.submit(thread));
            }
            threadPool.shutdown();
            threadPool.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
            System.out.println("cost after iteration " + iter + ": " + logZTot.doubleValue());
        }

        /* commented out because the next stage does the same thing (uncomment if path != lpPath)
        // get hard alignments at end
        for(int n = 0; n < bankSize; n++){
            for(AMR.Node node : nodes[n]){
                getBestToken(node, tokens[n], nodes[n].length);
            }
        }
        */

        // get cost on labeled dev set
        // first, grab all the data
        AMRSlurp.doingAlignments = false;
        String lpPath = "data/training-500-subset.txt";
        AMR[] lpBank = AMRSlurp.slurp(lpPath, AMRSlurp.Format.LDC);
        int lpBankSize = Math.min(lpBank.length, maxSize);
        AugmentedToken[][] lpTokens = new AugmentedToken[lpBankSize][];
        AMR.Node[][] lpNodes = new AMR.Node[lpBankSize][];
        read(lpPath, lpBank, lpTokens, lpNodes, lpBankSize);

        int numCorrect = 0, numTotal = 0;
        for(int n = 0; n < lpBankSize; n++){
            for(AMR.Node node : lpNodes[n]){
                TokenWithAction tokenWithAction = getBestToken(node, lpTokens[n], lpNodes[n].length);
                AugmentedToken token = tokenWithAction.token;
                Action action = tokenWithAction.action;
                System.out.println(node + "[" + node.alignment + " = " + lpTokens[n][node.alignment].value
                        + " / " + token.index + " = " + token.value + "/" + action + "]");
                if(token.index == node.alignment) numCorrect++;
                numTotal++;
            }
        }
        double percent = numCorrect * 100.0 / numTotal;
        System.out.println(numCorrect + "/" + numTotal + " correct (" + percent + ")");


        // TODO:
        // DONE 1. Print out hard alignments at end
        // DONE 2. Get cost on labeled development set
        // 3. Modify cost function based on Dirichlet prior
        // 4. Handle NAME construct
        // DONE 5. Print correct alignment ID
        // DONE 6. Make things faster

    }

    static void read(String path, AMR[] bank, AugmentedToken[][] tokens, AMR.Node[][] nodes, int bankSize){
        String[] sentences = new String[bankSize];
        for (int i = 0; i < bankSize; i++) {
            sentences[i] = bank[i].formatSourceTokens();
        }
        CoreNLPCache cache = new BatchCoreNLPCache(path, sentences);
        //AnnotationManager manager = new AnnotationManager();
        for(int i = 0; i < bankSize; i++){
            //Annotation annotation = manager.annotate(bank[i].formatSourceTokens()).annotation;
            Annotation annotation = cache.getAnnotation(i);
            tokens[i] = augmentedTokens(bank[i].sourceText, annotation);
            nodes[i] = bank[i].nodes.toArray(new AMR.Node[0]);
            System.out.println("Data for example " + i + ":");
            System.out.println("\ttokens:");
            for(AugmentedToken token : tokens[i]){
                System.out.println("\t\t" + token.value + " / " + token.sense + " / " + token.stem);
            }
            System.out.println("\tnodes:");
            for(AMR.Node node : nodes[i]){
                System.out.println("\t\t" + node.title);
            }
            System.out.println("-------------\n");
        }
    }

    static class TokenWithAction {
        AugmentedToken token;
        Action action;
        public TokenWithAction(AugmentedToken token, Action action){
            this.token = token;
            this.action = action;
        }
    }
    static TokenWithAction getBestToken(AMR.Node node, AugmentedToken[] tokens, int length){
        AugmentedToken bestToken = null;
        Action bestAction = null;
        double bestScore = 0.0;
        for(AugmentedToken token : tokens){
            double logZ2 = Double.NEGATIVE_INFINITY;
            for(Action action : Action.values()) {
                List<String> features = extractFeatures(token, action);
                logZ2 = lse(logZ2, score(features));
            }
            for(Action action : Action.values()){
                List<String> features = extractFeatures(token, action);
                double probSrc = Math.exp(score(features) - logZ2);
                double likelihood = getNode(token, action).score(node, length);
                double probTar = probSrc * likelihood;
                if(bestToken == null || probTar > bestScore) {
                    bestToken = token;
                    bestAction = action;
                    bestScore = probTar;
                }
            }
        }
        //System.out.println(node + " [" + bestToken + "|" + bestAction + "]");
        return new TokenWithAction(bestToken, bestAction);
    }

    static List<String> extractFeatures(AugmentedToken token, Action action){
        List<String> ret = new ArrayList<String>();
        //ret.add("ACT|"+action.toString());
        ret.add("VAL|"+token.value + "|" + action.toString());
        //ret.add("BLK|"+(token.blocked ? "1" : "0") + "|" + action.toString());
        if(token.value.length() > 4) {
            ret.add("PRE|" + token.value.substring(0, 4) + "|" + action.toString());
        }
        return ret;
    }

    static Map<String, AGPair> theta;
    static void adagrad(Map<String, Double> gradient, double step){
        for(Map.Entry<String, Double> e : gradient.entrySet()){
            AGPair p = theta.get(e.getKey());
            if(p == null) theta.put(e.getKey(), new AGPair(e.getValue(), step));
            else p.incr(e.getValue(), step);
        }
    }
    static double score(List<String> features){
        double ret = 0.0;
        for(String key : features){
            AGPair p = theta.get(key);
            if(p != null) ret += p.v;
        }
        return ret;
    }
    static void incr(Map<String, Double> map, List<String> keys, double x){
        for(String key : keys) {
            Double v = map.get(key);
            if (v == null) map.put(key, x);
            else map.put(key, v + x);
        }
    }
    static void incr(Map<String, Double> map, Map<String, Double> rhs, double x){
        for(Map.Entry<String, Double> e : rhs.entrySet()){
            String key = e.getKey();
            double dv = x * e.getValue();
            Double v = map.get(key);
            if(v == null) map.put(key, dv);
            else map.put(key, v + dv);
        }
    }
    static class AGPair {
        private static final double DELTA = 1e-4;
        double v, s;
        public AGPair(double val, double step){
            s = val*val + DELTA;
            v = step * val / Math.sqrt(s);
        }
        void incr(double val, double step){
            s += val * val;
            v += step * val / Math.sqrt(s);
        }
    }
    static double lse(double x, double y){
        if(x == Double.NEGATIVE_INFINITY) return y;
        else if(y == Double.NEGATIVE_INFINITY) return x;
        else if(x < y) return y + Math.log(1 + Math.exp(x-y));
        else return x + Math.log(1 + Math.exp(y-x));
    }

    enum Action { IDENTITY, NONE, VERB, LEMMA, DICT, NAME, PERSON }
    /* AugmentedToken should keep track of following:
     *   0. string value of token
     *   1. srl sense
     *   2. lemmatized value (stem)
     *   3. whether "blocked"
     *   We can start as a baseline by making sense = "NONE" and stem = token
     */
    static class AugmentedToken {
        int index;
        String value, sense, stem;
        boolean blocked;
        public AugmentedToken(int index, String value, String sense, String stem, boolean blocked){
            this.index = index;
            this.value = value;
            this.sense = sense;
            this.stem = stem;
            this.blocked = blocked;
        }
        @Override
        public String toString(){
            return value + "[" + index + "]";
        }
    }

    static class MatchNode {
        boolean isDict;
        String name;
        String quoteType;
        public MatchNode(String name){
            this.name = name;
            this.isDict = false;
            this.quoteType = null;
        }
        public MatchNode(String name, boolean isDict){
            this.name = name;
            this.isDict = isDict;
            this.quoteType = null;
        }
        public MatchNode(String name, String quoteType){
            this.name = name;
            this.isDict = false;
            this.quoteType = quoteType;
        }
        double score(AMR.Node match, int len){
            if(quoteType != null) {
                if(quoteType.equals(match.title)) return 1.0;
                if(match.type == AMR.NodeType.QUOTE && name.equals(match.title)) return 1.0;
                return 0.0;
            } else if(!isDict){
                if(name.equals(match.title)) return 1.0;
                else return 0.0;
            } else {
                // be lazy for now, just return a low-ish number
                return 0.01 / len;
            }
        }
        @Override
        public String toString(){
            if(isDict) return "DICT("+name+")";
            else return name;
        }
    }

    static MatchNode getNode(AugmentedToken token, Action action){
        switch(action){
            case IDENTITY:
                return new MatchNode(token.value.toLowerCase());
            case NONE:
                return new MatchNode("IMPOSSIBLE_TO_MATCH");
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
            case NAME:
                return new MatchNode(token.value, "name");
            case PERSON:
                return new MatchNode(token.value, "person");
            default:
                throw new RuntimeException("invalid action");
        }
    }

    private static AugmentedToken[] augmentedTokens(String[] tokens, Annotation annotation) {
        AugmentedToken[] output = new AugmentedToken[tokens.length];
        Set<Integer> blocked = new HashSet<>();

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
            String srlSense = getSRLSenseAtIndex(annotation, i);
            String stem = annotation.get(CoreAnnotations.TokensAnnotation.class).
                            get(i).get(CoreAnnotations.LemmaAnnotation.class).toLowerCase();
            output[i] = new AugmentedToken(i, tokens[i], srlSense, stem, blocked.contains(i));
        }

        return output;
    }

    private static String getSRLSenseAtIndex(Annotation annotation, int index) {
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
