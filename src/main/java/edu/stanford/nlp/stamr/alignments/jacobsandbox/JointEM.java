package edu.stanford.nlp.stamr.alignments.jacobsandbox;

import edu.stanford.nlp.cache.BatchCoreNLPCache;
import edu.stanford.nlp.cache.CoreNLPCache;
import edu.stanford.nlp.curator.CuratorAnnotations;
import edu.stanford.nlp.curator.PredicateArgumentAnnotation;
import edu.stanford.nlp.experiments.FrameManager;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.stamr.AMR;
import edu.stanford.nlp.stamr.AMRSlurp;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.util.Execution;
import edu.stanford.nlp.util.concurrent.AtomicDouble;
import edu.stanford.nlp.util.logging.StanfordRedwoodConfiguration;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;
import java.util.concurrent.*;

import static edu.stanford.nlp.util.logging.Redwood.Util.endTrack;
import static edu.stanford.nlp.util.logging.Redwood.Util.forceTrack;
import static edu.stanford.nlp.util.logging.Redwood.Util.startTrack;

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
@SuppressWarnings({"FieldCanBeLocal"})
public class JointEM {

    @Execution.Option(name="train.data", gloss="The path to the training data")
    private static String TRAIN_DATA = "realdata/train-aligned.txt";
    @Execution.Option(name="train.count", gloss="The number of examples to train on")
    private static int TRAIN_COUNT = Integer.MAX_VALUE;
    @Execution.Option(name="train.iters", gloss="The number of iterations to run EM for")
    private static int TRAIN_ITERS = 40;

    @Execution.Option(name="test.data", gloss="The path to the test data")
    private static String TEST_DATA = "data/training-500-subset.txt";
    @Execution.Option(name="test.count", gloss="The number of examples to test on")
    private static int TEST_COUNT = Integer.MAX_VALUE;

    @Execution.Option(name="model.dir", gloss="The path to a directory with saved models")
    private static File MODEL_DIR = new File("models/");
    @Execution.Option(name="model.clobber", gloss="If true, clobber any existing saved model")
    private static boolean MODEL_CLOBBER = false;

    private static final Set<String> namedEntityTypes = Collections.unmodifiableSet(new HashSet<String>() {{
        add("name");
        add("person");
        add("country");
        add("company");
        add("monetary-quantity");
        add("organization");
        add("government-organization");
        add("university");
        add("date-entity");
        add("aircraft-type");
        add("temporal-quantity");
        add("criminal-organization");
        add("political-party");
        add("spaceship");
        add("world-region");
        add("aircraft-type");
        add("game");
        add("city");
        add("event");
        add("book");
        add("earthquake");
        add("mass-quantity");
        add("continent");
    }});

    static FrameManager frameManager;
    static {
        try {
            frameManager = new FrameManager("data/frames");
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public static void main(String[] args) throws Exception {
        // Some setup
        Execution.fillOptions(JointEM.class, args);
        StanfordRedwoodConfiguration.setup();
        ProblemCache cache = new ProblemCache();

        startTrack("main");

        // Training
        startTrack("Creating Model");
        // 1. Try to load the model
        // (create the model directory)
        Model model;
        if (!MODEL_DIR.exists()) {
            if (!MODEL_DIR.mkdirs()) {
                throw new IllegalStateException("Could not create model dir: " + MODEL_DIR);
            }
        }
        // (get the model path)
        String modelSpecifier = new File(TRAIN_DATA).getName();
        modelSpecifier = modelSpecifier.substring(0, modelSpecifier.indexOf('.') < 0 ? modelSpecifier.length() : modelSpecifier.indexOf('.'));
        File modelPath = new File(MODEL_DIR + File.separator + modelSpecifier + ".model.ser.gz");
        // (try to load the model)
        if (!MODEL_CLOBBER && modelPath.exists() && modelPath.canRead()) {
            model = IOUtils.readObjectFromFile(modelPath);
        } else {
            // 2. Can't load the model: train a new one
            model = doEM(TRAIN_DATA, TRAIN_COUNT, TRAIN_ITERS, cache);
            startTrack("Training");
            IOUtils.writeObjectToFile(model, modelPath);
            endTrack("Training");
        }
        endTrack("Creating Model");

        // Testing
        startTrack("Evaluating");
        evaluate(TEST_DATA, TEST_COUNT, model, cache);
        endTrack("Evaluating");

        endTrack("main");
    }


    static Model doEM(String path, int maxSize, int trainingIters, ProblemCache cache) throws Exception {
        Model model = new Model();

        // first, grab all the data
        forceTrack("Reading data");
        AMRSlurp.doingAlignments = false;
        AMR[] bank = AMRSlurp.slurp(path, AMRSlurp.Format.LDC);
        int bankSize = Math.min(bank.length, maxSize);
        AugmentedToken[][] tokens = new AugmentedToken[bankSize][];
        AMR.Node[][] nodes = new AMR.Node[bankSize][];
        read(path, bank, tokens, nodes, bankSize);
        endTrack("Reading data");

        // initialize frequencies
        forceTrack("Initializing frequencies");
        HashMap<String, Integer> freqs = new HashMap<>();
        for(int n = 0; n < bankSize; n++){
            for(AMR.Node node : nodes[n]){
                if (namedEntityTypes.contains(node.title)) { continue; }
                Integer x = freqs.get(node.title);
                if(x == null) freqs.put(node.title, 1);
                else freqs.put(node.title, x+1);
            }
        }
        endTrack("Initializing frequencies");

        // now tokens[i] has all the required token info
        // and nodes[i] has all the required node info
        // and we can just solve the alignment problem
        forceTrack("EM");
        double eta = 0.3;
        Model.SoftCountDict oldDict = new Model.SoftCountDict(freqs, 2.0);
        for(int iter = 0; iter < trainingIters; iter++){
            forceTrack("Iteration " + (iter + 1) + " / " + trainingIters);
            final Model.SoftCountDict curDict = oldDict;
            final Model.SoftCountDict nextDict = new Model.SoftCountDict(freqs, 2.0);
            final AtomicDouble logZTot = new AtomicDouble(0.0);
            ExecutorService threadPool = Executors.newFixedThreadPool(Execution.threads);
            for(int n = 0; n < bankSize; n++){
                final AugmentedToken[] curTokens = tokens[n];
                final List<AMR.Node> curNodes = new ArrayList<>(Arrays.asList(nodes[n]));
                for(int count = 0; count < (curTokens.length + 1) / 3; count++) {
                    curNodes.add(new NoneNode());
                }
                final int curN = n;
                Callable<Void> thread = () -> {
                    // do IBM model 1
                    // print current best predictions (for debugging)
                    for (AugmentedToken token : curTokens) {
                        Action bestAction = null;
                        double bestScore = 0.0;
                        for (Action action : Action.values()) {
                            List<String> features = extractFeatures(token, action, cache);
                            double curScore = model.score(features);
                            if (bestAction == null || curScore > bestScore) {
                                bestAction = action;
                                bestScore = curScore;
                            }
                        }
                        if (curN == 0)
                            System.out.println("OPT(" + token + ") : " + bestAction + " => " + getNode(token, bestAction, cache));
                    }


                    // loop over output
                    // For each node...
                    for (AMR.Node node : curNodes) {
                        if (namedEntityTypes.contains(node.title)) { continue; }
                        double Zsrc = 0.0, Ztar = 0.0;
                        HashMap<String, Double> counts = new HashMap<>();
                        Map<String, Double> gradientSrc = new HashMap<>(),
                                gradientTar = new HashMap<>();
                        // For each token...
                        for (AugmentedToken token : curTokens) {
                            double logZ2 = Double.NEGATIVE_INFINITY;
                            for (Action action : Action.validValues(token, node)) {
                                List<String> features = extractFeatures(token, action, cache);
                                logZ2 = Util.lse(logZ2, model.score(features));
                            }
                            // For each action...
                            for (Action action : Action.validValues(token, node)) {
                                List<String> features = extractFeatures(token, action, cache);
                                double probSrc = Math.exp(model.score(features) - logZ2);
                                double likelihood = getNode(token, action, cache).score(node, curDict);
                                //if (curN == 0 && likelihood > 0.01)
                                //    System.out.println("likelihood(" + token + "," + action + "," + node + ") = " + likelihood);
                                double probTar = probSrc * likelihood;
                                Zsrc += probSrc;
                                Ztar += probTar;
                                Util.incr(gradientSrc, features, probSrc);
                                Util.incr(gradientTar, features, probTar);
                                if(action == Action.DICT){
                                    counts.put(token.value, probTar);
                                }
                            }
                        }
                        for(Map.Entry<String, Double> e : counts.entrySet()){
                            nextDict.addCount(e.getKey(), node.title, e.getValue() / Ztar);
                        }
                        Map<String, Double> gradient = new HashMap<>();
                        Util.incr(gradient, gradientSrc, -1.0 / Zsrc);
                        Util.incr(gradient, gradientTar, 1.0 / Ztar);
                        model.adagrad(gradient, eta);
                        logZTot.addAndGet(Math.log(Zsrc / Ztar));
                    }
                    return null;
                };
                threadPool.submit(thread);
            }
            if(iter >= 8 && iter % 1 == 0) {
                oldDict = nextDict;
            }
            threadPool.shutdown();
            threadPool.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
            System.out.println("cost after iteration " + (iter + 1) + ": " + logZTot.doubleValue());
            endTrack("Iteration " + (iter + 1) + " / " + trainingIters);
        }
        endTrack("EM");

        model.dict = oldDict;

        return model;

        /* commented out because the next stage does the same thing (uncomment if path != lpPath)
        // get hard alignments at end
        for(int n = 0; n < bankSize; n++){
            for(AMR.Node node : nodes[n]){
                getBestToken(node, tokens[n], nodes[n].length);
            }
        }
        */


        // TODO:
        // DONE 1. Print out hard alignments at end
        // DONE 2. Get cost on labeled development set
        // 3. Modify cost function based on Dirichlet prior
        // DONE 4. Handle NAME construct
        // DONE 5. Print correct alignment ID
        // DONE 6. Make things faster

    }


    private static void evaluate(String testData, int testSize, Model model, ProblemCache cache) throws IOException {
        startTrack("Evaluating");

        // get cost on labeled dev set
        // first, grab all the data
        forceTrack("Reading data");
        AMRSlurp.doingAlignments = false;
        AMR[] lpBank = AMRSlurp.slurp(testData, AMRSlurp.Format.LDC);
        int lpBankSize = Math.min(lpBank.length, testSize);
        AugmentedToken[][] lpTokens = new AugmentedToken[lpBankSize][];
        AMR.Node[][] lpNodes = new AMR.Node[lpBankSize][];
        read(testData, lpBank, lpTokens, lpNodes, lpBankSize);
        endTrack("Reading data");

        int numCorrect = 0, numTotal = 0;

        // Get final accuracy
        File debugFile = File.createTempFile("amr", ".txt");
        PrintWriter debugWriter = IOUtils.getPrintWriter(debugFile);
        for(int n = 0; n < lpBankSize; n++){
            // Get the sentence and AMR graph
            AugmentedToken[] sentence = lpTokens[n];
            AMR.Node[] amr = lpNodes[n];
            // Find the best alignments
            for(AMR.Node node : amr){
                if (namedEntityTypes.contains(node.title)) { continue; }
                // Get the best token for the AMR node
                TokenWithAction tokenWithAction = getBestToken(node, sentence, model, cache);
                AugmentedToken token = tokenWithAction.token;
                Action action = tokenWithAction.action;
                // Find the gold alignment
                Set<Integer> goldNodes = new HashSet<>();
                AMR.Node representativeGold = null;
                for (AMR.Node candidate : amr) {
                    if (namedEntityTypes.contains(node.title)) { continue; }
                    if (candidate.alignment == token.index) {
                        goldNodes.add(candidate.alignment);
                        representativeGold = candidate;
                    }
                }
                // Register the accuracy data point
                if(goldNodes.contains(node.alignment)) {
                    numCorrect += 1;
                }
                numTotal++;
                // Debug print the alignment
                String prefix = "✓";
                if(!goldNodes.contains(node.alignment)) {
                    prefix = "x";
                }
                String msg = prefix + " " + node + "[" + node.alignment + " = " + lpTokens[n][node.alignment].value
                        + " / " + token.index + " = " + token.value + "/" + action + "]";
                System.out.println(msg);
                debugWriter.println(action + "\t" + prefix + "\t" + token.value + "\t" + node + "\t" + representativeGold);
            }
        }

        // Print the final accuracies
        double percent = numCorrect * 100.0 / numTotal;
        System.out.println(numCorrect + "/" + numTotal + " correct (" + percent + ")");
        System.out.println("Debug data at " + debugFile.getPath());
        System.out.println("  format: action\tcorrect\tword\tguessed_node\tgold_node");
        debugWriter.close();

        endTrack("Evaluating");
    }

    /**
     * <p>
     *   Determine if a token could plausibly produce the given AMR node, given a perfect
     *   action execution module. So, for example, we assume perfect WSD, lemmatization, dictionary
     *   lookup, etc.
     * </p>
     *
     * <p>
     *   Note that this is a very forgiving function.
     * </p>
     *
     * @param token The token in the sentence.
     * @param node The AMR node we are proposing to align it with.
     * @param action The action we are proposing to take.
     *
     * @return True if the action could plausibly produce the node from the token. False otherwise.
     */
    static boolean plausiblyCompatible(AugmentedToken token, AMR.Node node, Action action) {
        return true;
        /*
        String normalizedTitle = node.title.toLowerCase().trim();
        String normalizedToken = token.value.toLowerCase().trim();
        String normalizedLemma = token.stem.toLowerCase().trim();
        // (check if this node looks like a verb)
        boolean nodeIsVerb = true;  // takes too long: normalizedTitle.matches(".*-[0-9]+");
        for (int i = normalizedTitle.length() - 1; i >= 0; --i) {
            if (normalizedTitle.charAt(i) == '-') { break; }
            if (normalizedTitle.charAt(i) < '0' || normalizedTitle.charAt(i) > '9') {
                nodeIsVerb = false;
                break;
            }
        }
        // Run action co-occurence
        switch (action) {
            case IDENTITY:
                // The title and the token must be the same.
                return normalizedTitle.equals(normalizedToken);
            case NONE:
                // You can always "generate" none (this should never fire, I think?)
                return true;
            case VERB:
                // The node is a verb
                return nodeIsVerb;
            case LEMMA:
                // The node is not a verb, and the token lemma is close to the title of the node.
                return !nodeIsVerb;// && JaroWinklerDistance.distance(normalizedLemma, normalizedTitle) > 0.5;
            case DICT:
                // This can always be generated.
                // If nothing else, this must be true so that at least _some_ action fires always.
                return true;
            case NAME:
                // Not a verb, and is quoted.
                return !nodeIsVerb;// && normalizedTitle.charAt(0) == '"' && normalizedTitle.charAt(normalizedTitle.length() - 1) == '"';
            case PERSON:
                // Not a verb, and is quoted.
                return !nodeIsVerb;// && normalizedTitle.charAt(0) == '"' && normalizedTitle.charAt(normalizedTitle.length() - 1) == '"';
            default:
                throw new IllegalStateException("Unknown action: " + action);
        }
        */
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
            nodes[i] = bank[i].nodes.toArray(new AMR.Node[bank[i].nodes.size()]);
//            System.out.println("Data for example " + i + ":");
//            System.out.println("\ttokens:");
//            for(AugmentedToken token : tokens[i]){
//                System.out.println("\t\t" + token.value + " / " + token.sense + " / " + token.stem);
//            }
//            System.out.println("\tnodes:");
//            for(AMR.Node node : nodes[i]){
//                System.out.println("\t\t" + node.title);
//            }
//            System.out.println("-------------\n");
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
    static TokenWithAction getBestToken(AMR.Node node, AugmentedToken[] tokens, Model model, ProblemCache cache){
        AugmentedToken bestToken = null;
        Action bestAction = null;
        double bestScore = 0.0;
        for(AugmentedToken token : tokens){
            double logZ2 = Double.NEGATIVE_INFINITY;
            for(Action action : Action.validValues(token, node)) {
                List<String> features = extractFeatures(token, action, cache);
                logZ2 = Util.lse(logZ2, model.score(features));
            }
            for(Action action : Action.validValues(token, node)){
                List<String> features = extractFeatures(token, action, cache);
                double probSrc = Math.exp(model.score(features) - logZ2);
                double likelihood = getNode(token, action, cache).score(node, model.dict);
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

    static List<String> extractFeatures(AugmentedToken token, Action action, ProblemCache cache){
        List<String> ret = new ArrayList<>();
        //ret.add("ACT|"+action.toString());
        ret.add("VAL|"+token.value + "|" + action.toString());
        //ret.add("BLK|"+(token.blocked ? "1" : "0") + "|" + action.toString());
        if(token.value.length() > 4) {
            ret.add("PRE|" + token.value.substring(0, 4) + "|" + action.toString());
        }
        // Word shape
//        ret.add("WORD_SHAPE|" + cache.getWordShape(token.value) + "|" + action.toString());
        // Incoming edge
        if (token.isPunctuation()) {
            ret.add("IS_PUNCT|" + action.toString());
        } else {
            ret.add("INCOMING_EDGE|" + token.incomingEdge() + "|" + action.toString());

        }
        return ret;
    }

    enum Action {
        IDENTITY, NONE, VERB, LEMMA, DICT, NAME, PERSON ;
        public static List<Action> validValues(AugmentedToken token, AMR.Node node) {
            List<Action> rtn = new ArrayList<>();
            for (Action candidate : Action.values()) {
                if (plausiblyCompatible(token, node, candidate)) {
                    rtn.add(candidate);
                }
            }
            return rtn;
        }
    }
    /* AugmentedToken should keep track of following:
     *   0. string value of token
     *   1. srl sense
     *   2. lemmatized value (stem)
     *   3. whether "blocked"
     *   We can start as a baseline by making sense = "NONE" and stem = token
     */
    static class AugmentedToken {
        public final int index;
        public final String value, sense, stem;
        public final List<String> pathFromRoot;
        boolean blocked;
        public AugmentedToken(int index, String value, String sense, String stem,
                              List<String> pathFromRoot, boolean blocked){
            this.index = index;
            this.value = value;
            this.sense = sense;
            this.stem = stem;
            this.pathFromRoot = pathFromRoot;
            this.blocked = blocked;
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
        double score(AMR.Node match, Model.SoftCountDict dict){
            if(match instanceof NoneNode) return 0.0;
            if(quoteType != null) {
                if(quoteType.equals(match.title)) return 1.0;
                if(match.type == AMR.NodeType.QUOTE && name.equals(match.title)) return 1.0;
                return 0.0;
            } else if(!isDict){
                if(name.equals(match.title)) return 1.0;
                else return 0.0;
            } else {
                return dict.getProb(name, match.title);
            }
        }
        @Override
        public String toString(){
            if(isDict) return "DICT("+name+")";
            else return name;
        }
    }

    static class MatchNoneNode extends MatchNode {
        public MatchNoneNode(){
            super("NONE");
        }
        @Override
        double score(AMR.Node match, Model.SoftCountDict dict) {
            if(match instanceof NoneNode){
                return 1.0;
            } else {
                return 0.0;
            }
        }
    }

    static MatchNode getNode(AugmentedToken token, Action action, ProblemCache cache){
        switch(action){
            case IDENTITY:
                return new MatchNode(token.value.toLowerCase());
            case NONE:
                return new MatchNoneNode();
            case VERB:
                String srlSense = token.sense;
                if (srlSense.equals("-") || srlSense.equals("NONE")) {
                    String stem = token.stem;
                    return cache.getClosestFrame(frameManager, stem);
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

        List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);
        List<CoreLabel> corenlpTokens = annotation.get(CoreAnnotations.TokensAnnotation.class);
        for (int i = 0; i < tokens.length; i++) {
            // (get token info)
            String srlSense = getSRLSenseAtIndex(annotation, i);
            String stem = corenlpTokens.get(i).lemma().toLowerCase();
            // (get dependency path from the root)
            SemanticGraph tree = sentences.get(corenlpTokens.get(i).sentIndex()).get(SemanticGraphCoreAnnotations.BasicDependenciesAnnotation.class);
            LinkedList<String> pathFromRoot = new LinkedList<>();
            IndexedWord node = tree.getNodeByIndexSafe(corenlpTokens.get(i).index());
            while (node != null) {
                Iterator<SemanticGraphEdge> parents = tree.incomingEdgeIterator(node);
                if (parents.hasNext()) {
                    SemanticGraphEdge edge = parents.next();
                    pathFromRoot.addFirst(edge.getRelation().toString());
                    node = edge.getGovernor();
                } else {
                    node = null;
                }
            }
            // (create augmented token
            output[i] = new AugmentedToken(i, tokens[i], srlSense, stem, pathFromRoot, blocked.contains(i));
        }

        return output;
    }

    private static String getSRLSenseAtIndex(Annotation annotation, int index) {
        PredicateArgumentAnnotation srl = annotation.get(CuratorAnnotations.PropBankSRLAnnotation.class);
        if (srl == null) return "-"; // If we have no SRL on this example, then oh well
        for (PredicateArgumentAnnotation.AnnotationSpan span : srl.getPredicates()) {
            if ((span.startToken <= index) && (span.endToken >= index + 1)) {
                return span.getAttribute("predicate") + "-" + span.getAttribute("sense");
            }
        }
        return "NONE";
    }
}
