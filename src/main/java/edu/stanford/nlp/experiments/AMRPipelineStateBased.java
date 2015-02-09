package edu.stanford.nlp.experiments;

import com.github.keenon.minimalml.cache.BatchCoreNLPCache;
import com.github.keenon.minimalml.cache.CoreNLPCache;
import com.github.keenon.minimalml.cache.LazyCoreNLPCache;
import com.github.keenon.minimalml.word2vec.Word2VecLoader;
import edu.stanford.nlp.experiments.greedy.*;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.stamr.AMR;
import edu.stanford.nlp.stamr.AMRSlurp;
import edu.stanford.nlp.stamr.evaluation.Smatch;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.Triple;

import java.io.*;
import java.util.*;
import java.util.function.Function;

/**
 * Created by keenon on 1/27/15.
 *
 * Holds and trains several pipes, which can be analyzed separately or together to give results about AMR.
 */
public class AMRPipelineStateBased {

    static boolean REAL_DATA = false;

    /////////////////////////////////////////////////////
    // FEATURE SPECS
    //
    // Features can return Double, double[], or Strings
    // (or anything that handles .toString() without
    // collisions, String is default case)
    /////////////////////////////////////////////////////

    static Map<String,double[]> embeddings;

    static {
        try {
            embeddings = Word2VecLoader.loadData("data/google-300-trimmed.ser.gz");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    FrameManager frameManager;

    @SuppressWarnings("unchecked")
    LinearPipe<Pair<LabeledSequence,Integer>, String> nerPlusPlus = new LinearPipe<>(
            new ArrayList<Function<Pair<LabeledSequence,Integer>,Object>>(){{

                // Input pair is (Seq, index into Seq for current token)

                add((pair) -> pair.first.tokens[pair.second]);
                add((pair) -> embeddings.get(pair.first.tokens[pair.second]));
            }},
            AMRPipelineStateBased::writeNerPlusPlusContext
    );

    @SuppressWarnings("unchecked")
    LinearPipe<Triple<LabeledSequence,Integer,Integer>, String> dictionaryLookup = new LinearPipe<>(
            new ArrayList<Function<Triple<LabeledSequence,Integer,Integer>,Object>>(){{

                // Input triple is (Seq, index into Seq for start of expression, index into Seq for end of expression)

                add((triple) -> triple.first.tokens[triple.second]);
            }},
            AMRPipelineStateBased::writeDictionaryContext
    );

    TrainableOracle bfsOracle;

    public static String headOrRoot(GreedyState state) {
        if (state.head == 0) return "ROOT";
        else return state.nodes[state.head].toString();
    }

    public static String getDependencyPath(GreedyState state, int head, int tail) {
        if (head == 0 || tail == 0) { // if tail == 0, then something else went fairly wrong
            return "ROOT:NOPATH";
        }
        if (head == tail) {
            return "IDENTITY";
        }
        int headToken = state.nodes[head].alignment;
        int tailToken = state.nodes[tail].alignment;
        SemanticGraph graph = state.annotation.get(CoreAnnotations.SentencesAnnotation.class).get(0)
                .get(SemanticGraphCoreAnnotations.BasicDependenciesAnnotation.class);
        IndexedWord headIndexedWord = graph.getNodeByIndexSafe(headToken);
        IndexedWord tailIndexedWord = graph.getNodeByIndexSafe(tailToken);
        if (headIndexedWord == null || tailIndexedWord == null) {
            return "NOTOKENS:NOPATH";
        }
        List<SemanticGraphEdge> edges = graph.getShortestUndirectedPathEdges(headIndexedWord, tailIndexedWord);

        StringBuilder sb = new StringBuilder();
        IndexedWord currentWord = headIndexedWord;
        for (SemanticGraphEdge edge : edges) {
            if (edge.getDependent().equals(currentWord)) {
                sb.append(">");
                currentWord = edge.getGovernor();
            }
            else {
                if (!edge.getGovernor().equals(currentWord)) {
                    throw new IllegalStateException("Edges not in order");
                }
                sb.append("<");
                currentWord = edge.getDependent();
            }
            sb.append(edge.getRelation().getShortName());
            if (currentWord != headIndexedWord) {
                sb.append(":");
                sb.append(currentWord.get(CoreAnnotations.PartOfSpeechAnnotation.class));
            }
        }

        return sb.toString();
    }

    public static List<Integer> getParents(GreedyState state, int i) {
        List<Integer> parents = new ArrayList<>();
        int cursor = i;
        while (cursor > 0) {
            parents.add(cursor);
            cursor = state.originalParent[cursor];
        }
        return parents;
    }

    public static String getAMRPath(GreedyState state, int head, int tail) {
        if (state.originalParent[tail] == 0) return "NOPATH";
        StringBuilder sb = new StringBuilder();

        List<Integer> headParents = getParents(state, head);
        List<Integer> tailParents = getParents(state, tail);

        int cursor = tail;
        for (int i = 0; i < tailParents.size(); i++) {
            sb.append("<").append(state.arcs[tailParents.get(i)][cursor]);
            if (headParents.contains(tailParents.get(i))) break;
            cursor = tailParents.get(i);
        }
        for (int i = headParents.indexOf(cursor)-1; i >= 0; i--) {
            sb.append(">").append(state.arcs[cursor][headParents.get(i)]);
            cursor = headParents.get(i);
        }

        return sb.toString();
    }

    static List<Function<Pair<GreedyState,Integer>,Object>> bfsOracleFeatures =
            new ArrayList<Function<Pair<GreedyState,Integer>,Object>>(){{

                /**
                 * CMU features
                 */

                /**
                 Here's the list of CMU Features as seen in the ACL '14 paper:

                 - Self edge: 1 if edge is between two nodes in the same fragment
                 - Tail fragment root: 1 if the edge's tail is the root of its graph fragment
                 - Head fragment root: 1 if the edge's head is the root of its graph fragment
                 - Path: Dependency edge labels and parts of speech on the shortest syntactic path between any two words of the two spans
                 - Distance: Number of tokens (plus one) between the concept's spans
                 - Distance indicators: A feature for each distance value
                 - Log distance: Log of the distance feature + 1

                 Combos:

                 - Path & Head concept
                 - Path & Tail concept
                 - Path & Head word
                 - Path & Tail word
                 - Distance & Path

                 */

                // Tail fragment root
                add(pair -> {
                    GreedyState state = pair.first;
                    for (int i = 1; i < state.nodes.length; i++) {
                        if (state.forcedArcs[i][pair.second] != null) return 0.0;
                    }
                    return 1.0;
                });
                // Head fragment root
                add(pair -> {
                    GreedyState state = pair.first;
                    for (int i = 1; i < state.nodes.length; i++) {
                        if (state.forcedArcs[i][state.head] != null) return 0.0;
                    }
                    return 1.0;
                });
                // Path
                add(pair -> {
                    GreedyState state = pair.first;
                    return getDependencyPath(state, state.head, pair.second);
                });
                // Distance
                add(pair -> {
                    GreedyState state = pair.first;
                    if (state.head == 0) return 0.0;
                    return Math.abs(state.nodes[state.head].alignment - state.nodes[pair.second].alignment)+1.0;
                });
                // Log Distance
                add(pair -> {
                    GreedyState state = pair.first;
                    if (state.head == 0) return 0.0;
                    return Math.log(Math.abs(state.nodes[state.head].alignment - state.nodes[pair.second].alignment)+1.0);
                });
                // Distance Indicator
                add(pair -> {
                    GreedyState state = pair.first;
                    if (state.head == 0) return Integer.toString(0);
                    return Integer.toString(Math.abs(state.nodes[state.head].alignment - state.nodes[pair.second].alignment) + 1);
                });
                // Path + Head concept
                add(pair -> {
                    GreedyState state = pair.first;
                    String headConcept;
                    if (state.head == 0) headConcept = "ROOT";
                    else headConcept = state.nodes[state.head].toString();
                    return getDependencyPath(state, state.head, pair.second)+":"+headConcept;
                });
                // Path + Tail concept
                add(pair -> {
                    GreedyState state = pair.first;
                    String tailConcept;
                    return getDependencyPath(state, state.head, pair.second)+":"+state.nodes[pair.second].toString();
                });
                // Path + Head word
                add(pair -> {
                    GreedyState state = pair.first;
                    String headWord;
                    if (state.head == 0) headWord = "ROOT";
                    else headWord = state.tokens[state.nodes[state.head].alignment];
                    return getDependencyPath(state, state.head, pair.second)+":"+headWord;
                });
                // Path + Tail word
                add(pair -> {
                    GreedyState state = pair.first;
                    return getDependencyPath(state, state.head, pair.second)+":"+state.tokens[state.nodes[pair.second].alignment];
                });
                // Path + Distance Indicator
                add(pair -> {
                    GreedyState state = pair.first;
                    String dist;
                    if (state.head == 0) dist = "0";
                    else dist = Integer.toString(Math.abs(state.nodes[state.head].alignment - state.nodes[pair.second].alignment) + 1);
                    return getDependencyPath(state, state.head, pair.second)+":"+dist;
                });

                /**
                 * New features, because we have tons of context info available
                 */

                // Depth into partial AMR tree
                add(pair -> {
                    GreedyState state = pair.first;
                    return (double)getParents(state, state.head).size()+1;
                });
                // Log depth into partial AMR tree
                add(pair -> {
                    GreedyState state = pair.first;
                    return Math.log((double) getParents(state, state.head).size() + 1);
                });
                // Depth into partial AMR tree Indicator
                add(pair -> {
                    GreedyState state = pair.first;
                    return Integer.toString(getParents(state, state.head).size()+1);
                });
                // Indicator for the node we're linking to already having a parent somewhere in the tree
                add(pair -> {
                    GreedyState state = pair.first;
                    if (state.originalParent[pair.second] == 0) {
                        return 0.0;
                    }
                    return 1.0;
                });
                // Path from the head to the child through the tree, if it exists
                add(pair -> {
                    GreedyState state = pair.first;
                    return getAMRPath(state, state.head, pair.second);
                });
                // Parents of the current head token
                add(pair -> {
                    GreedyState state = pair.first;
                    return getAMRPath(state, 0, state.head);
                });
                // Parents of the current tail token
                add(pair -> {
                    GreedyState state = pair.first;
                    return getAMRPath(state, 0, pair.second);
                });
            }};

    /////////////////////////////////////////////////////
    // PIPELINE
    /////////////////////////////////////////////////////

    public void trainStages(List<Function<Pair<GreedyState,Integer>,Object>> bfsOracleFeatures) throws IOException {
        System.out.println("Loading frames");
        frameManager = new FrameManager("data/frames");

        System.out.println("Loading training data");
        Pair<List<LabeledSequence>,CoreNLPCache> pair = loadSequenceData(REAL_DATA ? "data/train-400-seq.txt" : "data/train-3-seq.txt");
        List<LabeledSequence> nerPlusPlusData = pair.first;
        List<LabeledSequence> dictionaryData = loadManygenData(REAL_DATA ? "data/train-400-manygen.txt" : "data/train-3-manygen.txt");
        AMR[] bank = AMRSlurp.slurp(REAL_DATA ? "data/train-400-subset.txt" : "data/train-3-subset.txt", AMRSlurp.Format.LDC);

        System.out.println("Training");
        nerPlusPlus.train(getNERPlusPlusForClassifier(nerPlusPlusData));
        dictionaryLookup.train(getDictionaryForClassifier(dictionaryData));

        bfsOracle = new TrainableOracle(bank, bfsOracleFeatures, pair.second);
    }

    private Pair<AMR.Node[],String[][]> predictNodes(String[] tokens, Annotation annotation) {
        LabeledSequence labeledSequence = new LabeledSequence();
        labeledSequence.tokens = tokens;
        labeledSequence.annotation = annotation;

        String[] labels = new String[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            labels[i] = nerPlusPlus.predict(new Pair<>(labeledSequence, i));
        }

        // First we get all adjacent DICT entries together...

        List<List<Integer>> adjacentDicts = new ArrayList<>();

        List<Integer> currentDict = null;
        for (int i = 0; i < tokens.length; i++) {
            if (labels[i].equals("DICT")) {
                if (currentDict == null) {
                    currentDict = new ArrayList<>();
                }
                currentDict.add(i);
            }
            else if (currentDict != null) {
                adjacentDicts.add(currentDict);
                currentDict = null;
            }
        }
        if (currentDict != null) {
            adjacentDicts.add(currentDict);
        }

        // Now we can go through and start to generate components as a list of AMR's

        List<AMR> gen = new ArrayList<>();

        // First do all non-dict elements

        for (int i = 0; i < tokens.length; i++) {
            AMR amr = null;
            if (labels[i].equals("VERB")) {
                String stem = labeledSequence.annotation.get(CoreAnnotations.TokensAnnotation.class).
                        get(i).get(CoreAnnotations.LemmaAnnotation.class).toLowerCase();
                amr = createAMRSingleton(frameManager.getClosestFrame(stem));
            }
            else if (labels[i].equals("IDENTITY")) {
                amr = createAMRSingleton(tokens[i].toLowerCase());
            }
            else if (labels[i].equals("LEMMA")) {
                amr = createAMRSingleton(labeledSequence.annotation.get(CoreAnnotations.TokensAnnotation.class).
                        get(i).get(CoreAnnotations.LemmaAnnotation.class).toLowerCase());
            }
            if (amr != null) {
                for (AMR.Node node : amr.nodes) {
                    node.alignment = i;
                }
                gen.add(amr);
            }
            //
            // else do nothing, no other tags generate any nodes, purely for MST
            //
        }

        // Add all the dict elements

        for (List<Integer> dict : adjacentDicts) {
            int first = dict.get(0);
            int last = dict.get(dict.size() - 1);
            String amrString = dictionaryLookup.predict(new Triple<>(labeledSequence, first, last));

            AMR amr;
            if (amrString.startsWith("(")) {
                assert(amrString.endsWith(")"));
                amr = AMRSlurp.parseAMRTree(amrString);
            }
            else if (amrString.startsWith("\"")) {
                amr = createAMRSingleton(amrString.substring(1, amrString.length()-1), AMR.NodeType.QUOTE);
            }
            else {
                amr = createAMRSingleton(amrString, AMR.NodeType.VALUE);
            }

            for (AMR.Node node : amr.nodes) {
                node.alignment = Math.min(first + node.alignment, tokens.length-1);
            }
            gen.add(amr);
        }

        // Go through and pick out all the nodes

        List<AMR.Node> genNodes = new ArrayList<>();
        for (AMR amr : gen) {
            genNodes.addAll(amr.nodes);
        }

        AMR.Node[] nodes = new AMR.Node[genNodes.size()+1];
        for (int i = 0; i < genNodes.size(); i++) {
            nodes[i+1] = genNodes.get(i);
        }

        int length = nodes.length-1;

        String[][] forcedArcs = new String[length+1][length+1];

        // Get back all the forced arcs

        for (AMR amr : gen) {
            for (AMR.Arc arc : amr.arcs) {
                for (int i = 1; i <= length; i++) {
                    for (int j = 1; j <= length; j++) {
                        if (arc.head == nodes[i] && arc.tail == nodes[j]) {
                            forcedArcs[i][j] = arc.title;
                        }
                    }
                }
            }
        }

        return new Pair<>(nodes, forcedArcs);
    }

    private AMR greedilyConstruct(AMRNodeStateBased state) {
        Set<AMR.Node> nodes = new HashSet<>();
        Collections.addAll(nodes, state.nodes);

        assert(state.tokens != null);
        // GoldOracle.prepareForParse(nodes)
        GreedyState startState = new GreedyState(state.nodes, state.tokens, state.annotation);

        List<Pair<GreedyState,String[]>> derivation = TransitionRunner.run(startState, bfsOracle);
        return Generator.generateAMR(derivation.get(derivation.size()-1).first);
    }

    public AMR runPipeline(String[] tokens, Annotation annotation) {
        Pair<AMR.Node[], String[][]> nodesAndArcs = predictNodes(tokens, annotation);

        AMRNodeStateBased nodeSet = new AMRNodeStateBased(nodesAndArcs.first.length-1, annotation);
        nodeSet.nodes = nodesAndArcs.first;
        nodeSet.forcedArcs = nodesAndArcs.second;
        nodeSet.tokens = tokens;

        return greedilyConstruct(nodeSet);
    }

    private AMR createAMRSingleton(String title) {
        return createAMRSingleton(title, AMR.NodeType.ENTITY);
    }

    private AMR createAMRSingleton(String title, AMR.NodeType type) {
        AMR amr = new AMR();
        if (type == AMR.NodeType.ENTITY) {
            amr.addNode("" + title.charAt(0), title);
        }
        else {
            amr.addNode(title, type);
        }
        return amr;
    }

    public void analyzeStages(String testPrefix) throws IOException {
        System.out.println("Loading training data");
        Pair<List<LabeledSequence>, CoreNLPCache> pairTrain =
                loadSequenceData(REAL_DATA ? "data/train-400-seq.txt" : "data/train-3-seq.txt");
        List<LabeledSequence> nerPlusPlusDataTrain = pairTrain.first;
        List<LabeledSequence> dictionaryDataTrain =
                loadManygenData(REAL_DATA ? "data/train-400-manygen.txt" : "data/train-3-manygen.txt");

        System.out.println("Loading testing data");
        Pair<List<LabeledSequence>, CoreNLPCache> pairTest =
                loadSequenceData(REAL_DATA ? "data/test-100-seq.txt" : "data/train-3-seq.txt");
        List<LabeledSequence> nerPlusPlusDataTest = pairTest.first;
        List<LabeledSequence> dictionaryDataTest =
                loadManygenData(REAL_DATA ? "data/test-100-manygen.txt" : "data/train-3-manygen.txt");

        System.out.println("Running analysis");
        nerPlusPlus.analyze(getNERPlusPlusForClassifier(nerPlusPlusDataTrain),
                getNERPlusPlusForClassifier(nerPlusPlusDataTest),
                "data/"+testPrefix+"/ner-plus-plus-analysis");

        dictionaryLookup.analyze(getDictionaryForClassifier(dictionaryDataTrain),
                getDictionaryForClassifier(dictionaryDataTest),
                "data/"+testPrefix+"/dictionary-lookup-analysis");

        AMR[] trainBank = AMRSlurp.slurp(REAL_DATA ? "data/train-400-subset.txt" : "data/train-3-subset.txt", AMRSlurp.Format.LDC);
        AMR[] testBank = AMRSlurp.slurp(REAL_DATA ? "data/test-100-subset.txt" : "data/train-3-subset.txt", AMRSlurp.Format.LDC);

        bfsOracle.analyze(trainBank, pairTrain.second, testBank, pairTest.second, "data/"+testPrefix+"/bfs-oracle-analysis");
    }

    public void testCompletePipeline(String testPrefix) throws IOException, InterruptedException {
        if (REAL_DATA) {
            analyzeAMRSubset("data/train-400-subset.txt", "data/train-400-conll.txt", "data/"+testPrefix+"/amr-train-analysis");
            analyzeAMRSubset("data/test-100-subset.txt", "data/test-100-conll.txt", "data/"+testPrefix+"/amr-test-analysis");
        }
        else {
            analyzeAMRSubset("data/train-3-subset.txt", "data/train-3-conll.txt", "data/"+testPrefix+"/amr-train-micro-overfit");
        }
    }

    private void analyzeAMRSubset(String path, String coNLLPath, String output) throws IOException, InterruptedException {
        AMR[] bank = AMRSlurp.slurp(path, AMRSlurp.Format.LDC);
        List<AMRNodeSet> mstDataTest = AMRPipeline.loadCoNLLData(coNLLPath);

        String[] sentences = new String[bank.length];
        for (int i = 0; i < bank.length; i++) {
            sentences[i] = bank[i].formatSourceTokens();
        }
        CoreNLPCache cache = new LazyCoreNLPCache(path, sentences);

        AMR[] recovered = new AMR[bank.length];
        AMR[] recoveredPerfectDict = new AMR[bank.length];

        for (int i = 0; i < bank.length; i++) {
            Annotation annotation = cache.getAnnotation(i);
            recovered[i] = runPipeline(bank[i].sourceText, annotation);

            AMRNodeSet set = mstDataTest.get(i);
            AMRNodeStateBased nodeSet = new AMRNodeStateBased(set.nodes.length-1, annotation);
            nodeSet.nodes = set.nodes;
            nodeSet.forcedArcs = set.forcedArcs;
            nodeSet.tokens = set.tokens;
            set.annotation = annotation;
            recoveredPerfectDict[i] = greedilyConstruct(nodeSet);
        }
        cache.close();

        System.out.println("Finished analyzing");

        File out = new File(output);
        if (out.exists()) out.delete();
        out.mkdirs();

        AMRSlurp.burp(output+"/gold.txt", AMRSlurp.Format.LDC, bank, AMR.AlignmentPrinting.ALL, false);
        AMRSlurp.burp(output+"/recovered.txt", AMRSlurp.Format.LDC, recovered, AMR.AlignmentPrinting.ALL, false);
        AMRSlurp.burp(output+"/recovered-perfect-dict.txt", AMRSlurp.Format.LDC, recoveredPerfectDict, AMR.AlignmentPrinting.ALL, false);

        double smatch = Smatch.smatch(bank, recovered);
        System.out.println("SMATCH for overall "+path+" = "+smatch);
        BufferedWriter bw = new BufferedWriter(new FileWriter(output+"/smatch.txt"));
        bw.write("Smatch: "+smatch);
        bw.close();

        double smatchPerfectDict = Smatch.smatch(bank, recoveredPerfectDict);
        System.out.println("SMATCH for perfect dict "+path+" = "+smatchPerfectDict);
        bw = new BufferedWriter(new FileWriter(output+"/smatch-perfect-dict.txt"));
        bw.write("Smatch: "+smatchPerfectDict);
        bw.close();
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        testPipeline("mainTest", bfsOracleFeatures);
    }

    public static void testPipeline(String testPrefix,
                                    List<Function<Pair<GreedyState,Integer>,Object>> bfsOracleFeatures)
            throws IOException, InterruptedException {
        AMRPipelineStateBased pipeline = new AMRPipelineStateBased();
        pipeline.trainStages(bfsOracleFeatures);
        pipeline.analyzeStages(testPrefix);
        pipeline.testCompletePipeline(testPrefix);
    }

    /////////////////////////////////////////////////////
    // LOADERS
    /////////////////////////////////////////////////////

    private Pair<List<LabeledSequence>,CoreNLPCache> loadSequenceData(String path) throws IOException {
        List<LabeledSequence> seqList = new ArrayList<>();

        BufferedReader br = new BufferedReader(new FileReader(path));

        List<String> tokens = new ArrayList<>();
        List<String> labels = new ArrayList<>();

        // Read the TSV file

        String line;
        while ((line = br.readLine()) != null) {
            if (line.length() == 0) {
                LabeledSequence seq = new LabeledSequence();
                seq.tokens = tokens.toArray(new String[tokens.size()]);
                seq.labels = labels.toArray(new String[labels.size()]);
                tokens.clear();
                labels.clear();
                seqList.add(seq);
            }
            else {
                String[] parts = line.split("\t");
                assert(parts.length == 2);
                tokens.add(parts[0]);
                labels.add(parts[1]);
            }
        }
        if (tokens.size() > 0) {
            LabeledSequence seq = new LabeledSequence();
            seq.tokens = tokens.toArray(new String[tokens.size()]);
            seq.labels = labels.toArray(new String[labels.size()]);
            tokens.clear();
            labels.clear();
            seqList.add(seq);
        }

        // Do or load the annotations

        String[] sentences = new String[seqList.size()];
        for (int i = 0; i < seqList.size(); i++) {
            sentences[i] = seqList.get(i).formatTokens();
        }
        CoreNLPCache coreNLPCache = new BatchCoreNLPCache(path, sentences);
        for (int i = 0; i < seqList.size(); i++) {
            seqList.get(i).annotation = coreNLPCache.getAnnotation(i);
        }

        br.close();

        return new Pair<>(seqList, coreNLPCache);
    }

    private List<LabeledSequence> loadManygenData(String path) throws IOException {
        List<LabeledSequence> seqList = new ArrayList<>();

        BufferedReader br = new BufferedReader(new FileReader(path));

        // Read the TSV file

        LabeledSequence currentSeq = null;
        int stage = 0;

        String line;
        while ((line = br.readLine()) != null) {

            stage ++;

            if (line.length() == 0) {
                if (currentSeq != null) {
                    seqList.add(currentSeq);
                }
                currentSeq = null;
                stage = 0;
            }

            else if (stage == 1) {
                assert(currentSeq == null);

                currentSeq = new LabeledSequence();
                currentSeq.labels = new String[0];
                currentSeq.tokens = line.split(" ");
                currentSeq.labels = new String[currentSeq.tokens.length];
            }
            else {
                String[] parts = line.split("\t");
                assert(parts.length == 4);
                assert(currentSeq != null);

                // Format layout:
                // AMR, START_INDEX, END_INDEX

                String amr = parts[0];

                int startIndex = Integer.parseInt(parts[1]);
                int endIndex = Integer.parseInt(parts[2]);
                for (int i = startIndex; i <= endIndex; i++) {
                    currentSeq.labels[i] = amr;
                }
            }
        }
        if (currentSeq != null) {
            seqList.add(currentSeq);
        }

        // Do or load the annotations

        String[] sentences = new String[seqList.size()];
        for (int i = 0; i < seqList.size(); i++) {
            sentences[i] = seqList.get(i).formatTokens();
        }
        CoreNLPCache coreNLPCache = new BatchCoreNLPCache(path, sentences);
        for (int i = 0; i < seqList.size(); i++) {
            seqList.get(i).annotation = coreNLPCache.getAnnotation(i);
        }

        br.close();

        return seqList;
    }

    /////////////////////////////////////////////////////
    // DATA TRANSFORMS
    /////////////////////////////////////////////////////

    public static List<Pair<Pair<LabeledSequence,Integer>,String>> getNERPlusPlusForClassifier(
            List<LabeledSequence> nerPlusPlusData) {
        List<Pair<Pair<LabeledSequence,Integer>,String>> nerPlusPlusTrainingData = new ArrayList<>();
        for (LabeledSequence seq : nerPlusPlusData) {
            for (int i = 0; i < seq.tokens.length; i++) {
                Pair<Pair<LabeledSequence,Integer>,String> pair = new Pair<>();
                pair.first = new Pair<>(seq, i);
                pair.second = seq.labels[i];
                nerPlusPlusTrainingData.add(pair);
            }
        }
        return nerPlusPlusTrainingData;
    }

    public static List<Pair<Triple<LabeledSequence,Integer,Integer>,String>> getDictionaryForClassifier(
        List<LabeledSequence> dictionaryData) {
        List<Pair<Triple<LabeledSequence,Integer,Integer>,String>> dictionaryTrainingData = new ArrayList<>();
        for (LabeledSequence seq : dictionaryData) {
            int min = seq.tokens.length;
            int max = 0;
            for (int i = 0; i < seq.tokens.length; i++) {
                if (seq.labels[i] != null) {
                    if (i < min) min = i;
                    if (i > max) max = i;
                }
            }
            assert(min <= max);

            Pair<Triple<LabeledSequence,Integer,Integer>,String> pair = new Pair<>();
            pair.first = new Triple<>(seq, min, max);
            pair.second = seq.labels[min];
            dictionaryTrainingData.add(pair);
        }
        return dictionaryTrainingData;
    }

    /////////////////////////////////////////////////////
    // ERROR ANALYSIS
    /////////////////////////////////////////////////////

    public static void writeNerPlusPlusContext(Pair<LabeledSequence,Integer> pair, BufferedWriter bw) {
        LabeledSequence seq = pair.first;
        for (int i = 0; i < seq.tokens.length; i++) {
            try {
                if (i != 0) bw.write(" ");
                if (i == pair.second) bw.write("[");
                bw.write(seq.tokens[i]);
                if (i == pair.second) bw.write("]");
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public static void writeDictionaryContext(Triple<LabeledSequence,Integer,Integer> triple, BufferedWriter bw) {
        LabeledSequence seq = triple.first;
        for (int i = 0; i < seq.tokens.length; i++) {
            try {
                if (i != 0) bw.write(" ");
                if (i == triple.second) bw.write("[");
                bw.write(seq.tokens[i]);
                if (i == triple.third) bw.write("]");
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public static void writeArcContext(Triple<AMRNodeStateBased, Integer, Integer> triple, BufferedWriter bw) {
        AMRNodeStateBased set = triple.first;

        AMR.Node source = set.nodes[triple.second];
        AMR.Node sink = set.nodes[triple.third];

        if (source == null || sink == null) return;

        try {
            if (triple.second == 0) {
                bw.write("SOURCE:ROOT");
            }
            else {
                bw.write("SOURCE: " + source.toString() + "=\"" + set.tokens[source.alignment] + "\"\n");
            }
            bw.write("SINK: "+sink.toString()+"=\""+set.tokens[sink.alignment]+"\"\n");
        } catch (IOException e) {
            e.printStackTrace();
        }

        for (int i = 0; i < set.tokens.length; i++) {
            try {
                if (i != 0) bw.write(" ");
                if (i == set.nodes[triple.second].alignment) bw.write("<<");
                if (i == set.nodes[triple.third].alignment) bw.write(">>");
                bw.write(set.tokens[i]);
                if (i == set.nodes[triple.second].alignment) bw.write(">>");
                if (i == set.nodes[triple.third].alignment) bw.write("<<");
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        try {
            bw.write("\n");
        } catch (IOException e) {
            e.printStackTrace();
        }


        for (int i = 1; i < set.nodes.length; i++) {
            if (i == triple.second || i == triple.third) continue;
            try {
                AMR.Node node = set.nodes[i];
                bw.write(node.toString()+"=\""+set.tokens[node.alignment]+"\"\n");
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
