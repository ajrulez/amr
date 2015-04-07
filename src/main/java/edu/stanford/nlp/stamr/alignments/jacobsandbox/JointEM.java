package edu.stanford.nlp.stamr.alignments.jacobsandbox;

import edu.berkeley.nlp.util.StringUtils;
import edu.stanford.nlp.cache.BatchCoreNLPCache;
import edu.stanford.nlp.cache.CoreNLPCache;
import edu.stanford.nlp.curator.CuratorAnnotations;
import edu.stanford.nlp.curator.PredicateArgumentAnnotation;
import edu.stanford.nlp.experiments.FrameManager;
import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.stamr.AMR;
import edu.stanford.nlp.stamr.AMRSlurp;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.Execution;
import edu.stanford.nlp.util.concurrent.AtomicDouble;
import edu.stanford.nlp.util.logging.StanfordRedwoodConfiguration;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;

import static edu.stanford.nlp.util.logging.Redwood.Util.*;

// TODO
// Initial accuracy: 82.9 (early stopping)
// 1. Add in some of Keenon's rule-based alignments
//    (83.1) -force NONE alignments inside -LRB- -RRB-
//    (83.3) -compute op1 for name and person
// 2. Use the hand-labeled alignments to help with training
//    (87.7, on half of the data) [note: sometimes 87.6, i guess due to nondeterminism with multithreading]
// 3. Run on more data

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

    @Execution.Option(name="train.data", gloss="The path to the training data")
    private static String TRAIN_DATA = "data/training-500-subset.txt";

    @Execution.Option(name="train.count", gloss="The number of examples to train on")
    private static int TRAIN_COUNT = Integer.MAX_VALUE;

    @Execution.Option(name="train.iters", gloss="The number of iterations to run EM for")
    private static int TRAIN_ITERS = 20;
    @Execution.Option(name="train.eta", gloss="The learning rate for EM.")
    private static double TRAIN_ETA = 0.3;
    @Execution.Option(name="train.gamma", gloss="The initial soft count of the dict (dirichlet gamma)")
    private static double TRAIN_GAMMA = 1.0;

    @Execution.Option(name="test.data", gloss="The path to the test data")
    private static String TEST_DATA = "data/training-500-subset.txt";
    @Execution.Option(name="test.count", gloss="The number of examples to test on")
    private static int TEST_COUNT = Integer.MAX_VALUE;

    @Execution.Option(name="model.dir", gloss="The path to a directory with saved models")
    private static File MODEL_DIR = new File("models/");
    @Execution.Option(name="model.clobber", gloss="If true, clobber any existing saved model")
    private static boolean MODEL_CLOBBER = true;

    private static final Set<String> namedEntityTypes = Collections.unmodifiableSet(new HashSet<String>() {{
        add("name");
        /*
        add("person");
        add("country");
        add("city");
        add("date-entity");
        add("continent");
        add("organization");
        add("monetary-quantity");
        add("temporal-quantity");
        add("company");
        add("government-organization");
        add("university");
        add("aircraft-type");
        add("criminal-organization");
        add("political-party");
        add("spaceship");
        add("world-region");
        add("aircraft-type");
        add("game");
        add("event");
        add("book");
        add("earthquake");
        add("mass-quantity");
        */
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
            model = doEM(cache);
            startTrack("Training");
            IOUtils.writeObjectToFile(model, modelPath);
            endTrack("Training");
        }
        endTrack("Creating Model");

        // Testing
        startTrack("Evaluating");
        evaluate(model, cache);
        endTrack("Evaluating");

        endTrack("main");
    }


    static Model doEM(ProblemCache cache) throws Exception {
        Model model = new Model();
        // first, grab all the data
        AMRSlurp.doingAlignments = false;
        AMR[] bank = AMRSlurp.slurp(TRAIN_DATA, AMRSlurp.Format.LDC);
        int bankSize = Math.min(bank.length, TRAIN_COUNT);
        AugmentedToken[][] tokens = new AugmentedToken[bankSize][];
        AMR.Node[][] nodes = new AMR.Node[bankSize][];
        read(TRAIN_DATA, bank, tokens, nodes, bankSize);

        // Create candidate lemma dictionary
        LemmaAction lemmaDict = LemmaAction.initialize(tokens, nodes, 10);

        // initialize frequencies
        HashMap<String, Integer> freqs = new HashMap<String, Integer>();
        for(int n = 0; n < bankSize; n++){
            for(AMR.Node node : nodes[n]){
                if (namedEntityTypes.contains(node.title)) { continue; }
                Integer x = freqs.get(node.title);
                if(x == null) freqs.put(node.title, 1);
                else freqs.put(node.title, x+1);
            }
        }

        // now tokens[i] has all the required token info
        // and nodes[i] has all the required node info
        // and we can just solve the alignment problem
        Model.SoftCountDict oldDict = new Model.SoftCountDict(freqs, TRAIN_GAMMA);
        for(int iter = 0; iter < TRAIN_ITERS; iter++){
            forceTrack("Iteration " + (iter + 1) + " / " + TRAIN_ITERS);
            final Model.SoftCountDict curDict = oldDict;
            final Model.SoftCountDict nextDict = new Model.SoftCountDict(freqs, TRAIN_GAMMA);
            final AtomicDouble logZTot = new AtomicDouble(0.0);
            ExecutorService threadPool = Executors.newFixedThreadPool(Execution.threads);
            ArrayList<Future<Void>> threads = new ArrayList<>();
            for(int n = 0; n < bankSize; n++){
                final AugmentedToken[] curTokens = tokens[n];
                final List<AMR.Node> curNodes = new ArrayList<>(Arrays.asList(nodes[n]));
                final boolean supervised = (n % 2 == 0);
                if(!supervised){
                    for(int count = 0; count < (curTokens.length + 1) / 3; count++) {
                        curNodes.add(new NoneNode());
                    }
                } else {
                    /*Set<Integer> indices = new HashSet<Integer>();
                    for(AMR.Node node : curNodes) indices.remove(node.alignment);
                    for(Integer index : indices){
                        AMR.Node newNode = new NoneNode();
                        newNode.alignment = index;
                        //curNodes.add(newNode);
                    }*/
                }
                final int curN = n;
                Callable<Void> thread = () -> {
                    try {
                        // do IBM model 1
                        // print current best predictions (for debugging)
                        if (curN == 0) {
                            for (AugmentedToken token : curTokens) {
                                Action bestAction = null;
                                double bestScore = 0.0;
                                if (token.forcedAction != null) {
                                    bestAction = token.forcedAction;
                                } else {
                                    for (Action action : Action.values()) {
                                        List<String> features = extractFeatures(token, action);
                                        double curScore = model.score(features);
                                        if (bestAction == null || curScore > bestScore) {
                                            bestAction = action;
                                            bestScore = curScore;
                                        }
                                    }
                                }
                                System.out.println("OPT(" + token + ") : " + bestAction + " => " + getNode(token, bestAction, lemmaDict, cache));
                            }
                        }


                        // loop over output
                        // For each node...
                        for (AMR.Node node : curNodes) {
                            if (namedEntityTypes.contains(node.title)) {
                                continue;
                            }
                            double Zsrc = 0.0, Ztar = 0.0;
                            HashMap<String, Double> counts = new HashMap<String, Double>();
                            Map<String, Double> gradientSrc = new HashMap<String, Double>(),
                                    gradientTar = new HashMap<String, Double>();
                            // For each token...
                            AugmentedToken[] theTokens = null;
                            if (supervised) {
                                theTokens = new AugmentedToken[1];
                                theTokens[0] = curTokens[node.alignment];
                                if (theTokens[0].forcedAction != null) {
                                    System.out.println("uh oh " + theTokens[0].forcedAction + " " + node.title);
                                }
                            } else {
                                theTokens = curTokens;
                            }
                            for (AugmentedToken token : theTokens) {
                                double logZ2 = Double.NEGATIVE_INFINITY;
                                int numValid = Action.validValues(token, node).size();
                                for (Action action : Action.validValues(token, node)) {
                                    List<String> features = extractFeatures(token, action);
                                    logZ2 = Util.lse(logZ2, model.score(features));
                                }
                                // For each action...
                                for (Action action : Action.validValues(token, node)) {
                                    List<String> features = extractFeatures(token, action);
                                    double probSrc = Math.exp(model.score(features) - logZ2);
                                    double likelihood = getNode(token, action, lemmaDict, cache).score(node, curDict);
                                    //if (curN == 0 && likelihood > 0.01)
                                    //    System.out.println("likelihood(" + token + "," + action + "," + node + ") = " + likelihood);
                                    double probTar = probSrc * likelihood;
                                    Zsrc += probSrc;
                                    Ztar += probTar;
                                    Util.incr(gradientSrc, features, probSrc);
                                    Util.incr(gradientTar, features, probTar);
                                    if (action == Action.DICT) {
                                        counts.put(token.value, probTar);
                                    }
                                }
                            }
                            for (Map.Entry<String, Double> e : counts.entrySet()) {
                                nextDict.addCount(e.getKey(), node.title, e.getValue() / Ztar);
                            }
                            Map<String, Double> gradient = new HashMap<String, Double>();
                            Util.incr(gradient, gradientSrc, -1.0 / Zsrc);
                            Util.incr(gradient, gradientTar, 1.0 / Ztar);
                            model.adagrad(gradient, TRAIN_ETA);
                            logZTot.addAndGet(Math.log(Zsrc / Ztar));
                        }
                    } catch (Throwable t) {
                        t.printStackTrace();
                        System.exit(1);
                    }
                    return null;
                };
                threads.add(threadPool.submit(thread));
            }
            if(iter >= 8 && iter % 1 == 0) {
                oldDict = nextDict;
            }
            threadPool.shutdown();
            threadPool.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
            System.out.println("cost after iteration " + (iter + 1) + ": " + logZTot.doubleValue());
            endTrack("Iteration " + (iter + 1) + " / " + TRAIN_ITERS);
        }

        /* commented out because the next stage does the same thing (uncomment if path != lpPath)
        // get hard alignments at end
        for(int n = 0; n < bankSize; n++){
            for(AMR.Node node : nodes[n]){
                getBestToken(node, tokens[n], nodes[n].length);
            }
        }
        */

        model.dict = oldDict;
        model.lemmaDict = lemmaDict;

        return model;
    }

    public static void evaluate(Model model, ProblemCache cache) throws IOException {
        // get cost on labeled dev set
        // first, grab all the data
        AMRSlurp.doingAlignments = false;
        String lpPath = "data/training-500-subset.txt";
        AMR[] lpBank = AMRSlurp.slurp(TEST_DATA, AMRSlurp.Format.LDC);
        int lpBankSize = Math.min(lpBank.length, TRAIN_COUNT);
        AugmentedToken[][] lpTokens = new AugmentedToken[lpBankSize][];
        AMR.Node[][] lpNodes = new AMR.Node[lpBankSize][];
        read(lpPath, lpBank, lpTokens, lpNodes, lpBankSize);

        int numCorrect = 0, numTotal = 0;

        // Get final accuracy
        File debugFile = File.createTempFile("amr", ".txt");
        PrintWriter debugWriter = IOUtils.getPrintWriter(debugFile);
        for(int n = 0; n < lpBankSize; n++){
            final boolean supervised = (n % 2 == 0);
            if(supervised) continue;
            // Get the sentence and AMR graph
            AugmentedToken[] sentence = lpTokens[n];
            AMR.Node[] amr = lpNodes[n];
            // Find the best alignments
            debugWriter.println(StringUtils.join(Arrays.asList(sentence).stream().map(x -> x.value).collect(Collectors.toList()), " "));
            for(AMR.Node node : amr){
                if (namedEntityTypes.contains(node.title)) { continue; }
                // Get the best token for the AMR node
                TokenWithAction tokenWithAction = getBestToken(node, sentence, model, cache);
                AugmentedToken token = tokenWithAction.token;
                Action action = tokenWithAction.action;
                // Find the gold alignment
                Set<AMR.Node> goldNodes = new HashSet<>();
                for (AMR.Node candidate : amr) {
                    if (namedEntityTypes.contains(node.title)) { continue; }
                    if (candidate.alignment == token.index) {
                        goldNodes.add(candidate);
                    }
                }
                // Register the accuracy data point
                if(goldNodes.contains(node)) {
                    numCorrect += 1;
                }
                numTotal++;
                // Debug print the alignment
                String prefix = "✓";
                if(!goldNodes.contains(node)) {
                    prefix = "x";
                }
                String msg = prefix + " " + node + "[" + node.alignment + " = " + lpTokens[n][node.alignment].value
                        + " / " + token.index + " = " + token.value + "/" + action + "]";
                System.out.println(msg);
                debugWriter.println(action + "\t" + prefix + "\t" + token.value + "\t" + node + "\t" + (goldNodes.isEmpty() ? "none" : goldNodes.iterator().next()));
            }
        }

        // Print the final accuracies
        double percent = numCorrect * 100.0 / numTotal;
        System.out.println(numCorrect + "/" + numTotal + " correct (" + percent + ")");
        System.out.println("Debug data at " + debugFile.getPath());
        System.out.println("  format: action\tcorrect\tword\tguessed_node\tgold_node");
        debugWriter.close();

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
        boolean nodeIsVerb = normalizedTitle.matches(".*-[0-9]+");
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
                return !nodeIsVerb && JaroWinklerDistance.distance(normalizedLemma, normalizedTitle) > 0.5;
            case DICT:
                // This can always be generated.
                // If nothing else, this must be true so that at least _some_ action fires always.
                return true;
            case NAME:
                // Not a verb, and is quoted.
                return !nodeIsVerb && normalizedTitle.charAt(0) == '"' && normalizedTitle.charAt(normalizedTitle.length() - 1) == '"';
            case PERSON:
                // Not a verb, and is quoted.
                return !nodeIsVerb && normalizedTitle.charAt(0) == '"' && normalizedTitle.charAt(normalizedTitle.length() - 1) == '"';
            default:
                throw new IllegalStateException("Unknown action: " + action);
        }
        */
    }


    private static boolean isLikelyRef(String str){
        return str.length() == 2 && str.charAt(0) >= 'a' && str.charAt(0) <= 'z'
                                 && str.charAt(1) >= '0' && str.charAt(1) <= '9';
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
            for(int j = 0; j < nodes[i].length; j++){
                if(isLikelyRef(nodes[i][j].title) && nodes[i][j].ref.length() == 0){
                    for(int k = 0; k < nodes[i].length; k++){
                        if(nodes[i][j].title.equals(nodes[i][k].ref)){
                            nodes[i][j].ref = nodes[i][k].ref;
                            nodes[i][j].title = nodes[i][k].title;
                            break;
                        }
                    }
                }
            }
//            System.out.println("Data for example " + i + ":");
//            System.out.println("\ttokens:");
//            for(AugmentedToken token : tokens[i]){
//                System.out.println("\t\t" + token.value + " / " + token.sense + " / " + token.stem);
//            }
//            System.out.println("\tnodes:");
//            for(AMR.Node node : nodes[i]){
//                System.out.println("\t\t" + node.title + "\t|" + node.ref + "|\t" + node.isFirstRef);
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
                List<String> features = extractFeatures(token, action);
                logZ2 = Util.lse(logZ2, model.score(features));
            }
            for(Action action : Action.validValues(token, node)){
                List<String> features = extractFeatures(token, action);
                double probSrc = Math.exp(model.score(features) - logZ2);
                double likelihood = getNode(token, action, model.lemmaDict, cache).score(node, model.dict);
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
        /*if(token.value.length() > 3) {
            ret.add("PRE|" + token.value.substring(0, 3) + "|" + action.toString());
            ret.add("POST|" + token.value.substring(token.value.length()-3) + "|" + action.toString());
        }*/
        ret.add("NER|" + token.ner + "|" + action.toString());
        ret.add("AMR|" + (token.amr == null ? "0" : "1") + "|" + action.toString());
        //ret.add("BLK|" + (token.blocked ? "1" : "0") + "|" + action.toString());
        return ret;
    }

    enum Action {
        IDENTITY, NONE, VERB, LEMMA, DICT, NAME, PERSON, /*VERB02, VERB03, VERB41, AMRRULE*/;
        public static List<Action> validValues(AugmentedToken token, AMR.Node node) {
            List<Action> rtn = new ArrayList<>();
            if(token.forcedAction != null){
                rtn.add(token.forcedAction);
                return rtn;
            }
            for (Action candidate : Action.values()) {
                if (plausiblyCompatible(token, node, candidate)) {
                    rtn.add(candidate);
                }
            }
            return rtn;
        }
    }

    private static String stemize(String stem, String end){
        return stem.toLowerCase().substring(0, stem.length() - 2) + end;
    }

    static MatchNode getNode(AugmentedToken token, Action action, LemmaAction lemmaDict, ProblemCache cache){
        String verb;
        switch(action){
            case IDENTITY:
                return new MatchNode.ExactMatchNode(token.value);
            case NONE:
                return new MatchNode.NoneMatchNode();
            case VERB:
                String srlSense = token.sense;
                if (srlSense.equals("-") || srlSense.equals("NONE")) {
                    // TODO(gabor) try me
//                    String stem = Counters.argmax(lemmaDict.lemmasFor(token.value.toLowerCase()));
                    String stem = token.stem;
                    return cache.getClosestFrame(frameManager, stem);
                } else {
                    return new MatchNode.VerbMatchNode(srlSense);
                }
//            case VERB02:
//                verb = cache.getClosestFrame(frameManager, token.stem).name;
//                return new MatchNode(stemize(verb, "02"));
//            case VERB03:
//                verb = cache.getClosestFrame(frameManager, token.stem).name;
//                return new MatchNode(stemize(verb, "03"));
//            case VERB41:
//                verb = cache.getClosestFrame(frameManager, token.stem).name;
//                return new MatchNode(stemize(verb, "41"));
            case LEMMA:
                // TODO(gabor) try me
//                return new MatchNode.LemmaMatchNode(lemmaDict.lemmasFor(token.value.toLowerCase()));
                  return new MatchNode.ExactMatchNode(token.stem);
            case DICT:
                return new MatchNode.DictMatchNode(token.value);
            case NAME:
                return new MatchNode.NamedEntityMatchNode(token.value, "name");
            case PERSON:
                return new MatchNode.NamedEntityMatchNode(token.value, "person");
//            case AMRRULE:
//                return new AMRRuleNode(token.amr);
            default:
                throw new RuntimeException("invalid action");
        }
    }

    private static AugmentedToken[] augmentedTokens(String[] tokens, Annotation annotation) {
        //AugmentedToken[] output = new AugmentedToken[tokens.length];
        ArrayList<AugmentedToken> output0 = new ArrayList<AugmentedToken>();
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

        HashSet<String> tagSet = new HashSet<String>();
        tagSet.add("date");
        tagSet.add("number");
        tagSet.add("money");
        AugmentedToken headToken = null;
        ArrayList<Integer> nerList = new ArrayList<Integer>();
        for (int i = 0; i < tokens.length; i++) {
            String srlSense = getSRLSenseAtIndex(annotation, i);
            String stem = annotation.get(CoreAnnotations.TokensAnnotation.class).
                            get(i).get(CoreAnnotations.LemmaAnnotation.class).toLowerCase();
            String nerTag = annotation.get(CoreAnnotations.TokensAnnotation.class).get(i).get(CoreAnnotations.NamedEntityTagAnnotation.class);
            nerTag = nerTag.toLowerCase();
            /*if(tokens[i].equals("million")){
                System.out.println("NERTAG " + tokens[i-1] + " " + nerTag);
            }*/
            output0.add(new AugmentedToken(i, tokens[i], srlSense, stem, nerTag, blocked.contains(i)));
            if(!tagSet.contains(nerTag)) {
                nerList.clear();
                continue;
            }
            // we currently don't handle the rules below correctly, so commented out to improve accuracy
            /*if(nerList.size() == 0){
                headToken = output0.get(output0.size()-1);
            }
            nerList.add(i);
            String nextTag = "o";
            if(i+1 < tokens.length){
                nextTag = annotation.get(CoreAnnotations.TokensAnnotation.class).get(i+1).get(CoreAnnotations.NamedEntityTagAnnotation.class);
                nextTag = nextTag.toLowerCase();
            }
            if(!nextTag.equals(nerTag)){
                System.out.println("Constructing amr...");
                if(nerTag.equals("number") || nerTag.equals("money")){
                    headToken.amr = RuleBased.constructNumberCluster(annotation, nerList);
                } else {
                    headToken.amr = RuleBased.constructDateCluster(annotation, nerList);
                }
                System.out.println(headToken.amr);
                nerList.clear();
            }*/
        }

        return output0.toArray(new AugmentedToken[0]);
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
