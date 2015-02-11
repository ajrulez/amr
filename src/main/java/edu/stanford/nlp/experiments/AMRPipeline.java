package edu.stanford.nlp.experiments;

import edu.stanford.nlp.cache.BatchCoreNLPCache;
import edu.stanford.nlp.cache.CoreNLPCache;
import edu.stanford.nlp.cache.LazyCoreNLPCache;
import edu.stanford.nlp.experiments.greedy.Generator;
import edu.stanford.nlp.experiments.greedy.GreedyState;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.stamr.AMR;
import edu.stanford.nlp.stamr.AMRParser;
import edu.stanford.nlp.stamr.AMRSlurp;
import edu.stanford.nlp.stamr.datagen.DumpSequence;
import edu.stanford.nlp.stamr.evaluation.Smatch;
import edu.stanford.nlp.stamr.utils.MSTGraph;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.Triple;
import edu.stanford.nlp.word2vec.Word2VecLoader;
import edu.stanford.nlp.wsd.WordNet;

import java.io.*;
import java.util.*;
import java.util.function.Function;

/**
 * Created by keenon on 1/27/15.
 *
 * Holds and trains several pipes, which can be analyzed separately or together to give results about AMR.
 */
public class AMRPipeline {

    public static boolean FULL_DATA = false;
    public static boolean TINY_DATA = true;

    /////////////////////////////////////////////////////
    // FEATURE SPECS
    //
    // Features can return Double, double[], or Strings
    // (or anything that handles .toString() without
    // collisions, String is default case)
    /////////////////////////////////////////////////////

    static Map<String,double[]> embeddings;
    static FrameManager frameManager;

    static {
        try {
            embeddings = Word2VecLoader.loadData("data/google-300-trimmed.ser.gz");
            frameManager = new FrameManager("data/frames");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @SuppressWarnings("unchecked")
    LinearPipe<Pair<LabeledSequence,Integer>, String> nerPlusPlus = new LinearPipe<>(
            new ArrayList<Function<Pair<LabeledSequence,Integer>,Object>>(){{

                // Input pair is (Seq, index into Seq for current token)

                add((pair) -> pair.first.tokens[pair.second]);
                add((pair) -> embeddings.get(pair.first.tokens[pair.second]));

                // Left context
                add((pair) -> {
                    if (pair.second == 0) return "^";
                    else return pair.first.tokens[pair.second-1];
                });
                // Right context
                add((pair) -> {
                    if (pair.second == pair.first.tokens.length-1) return "$";
                    else return pair.first.tokens[pair.second+1];
                });

                // POS
                add((pair) -> pair.first.annotation.get(CoreAnnotations.TokensAnnotation.class)
                        .get(pair.second).get(CoreAnnotations.PartOfSpeechAnnotation.class));
                // Left POS
                add((pair) -> {
                    if (pair.second == 0) return "^";
                    else return pair.first.annotation.get(CoreAnnotations.TokensAnnotation.class)
                            .get(pair.second-1).get(CoreAnnotations.PartOfSpeechAnnotation.class);
                });
                // Right POS
                add((pair) -> {
                    if (pair.second == pair.first.tokens.length-1) return "$";
                    else return pair.first.annotation.get(CoreAnnotations.TokensAnnotation.class)
                            .get(pair.second+1).get(CoreAnnotations.PartOfSpeechAnnotation.class);
                });

                // Token dependency parent
                add((pair) -> {
                    SemanticGraph graph = pair.first.annotation.get(CoreAnnotations.SentencesAnnotation.class).get(0)
                            .get(SemanticGraphCoreAnnotations.BasicDependenciesAnnotation.class);
                    IndexedWord indexedWord = graph.getNodeByIndexSafe(pair.second);
                    if (indexedWord == null) return "NON-DEP";
                    List<IndexedWord> l = graph.getPathToRoot(indexedWord);
                    if (l.size() > 0) {
                        return l.get(0).word();
                    }
                    else return "ROOT";
                });
                // Token dependency parent POS
                add((pair) -> {
                    SemanticGraph graph = pair.first.annotation.get(CoreAnnotations.SentencesAnnotation.class).get(0)
                            .get(SemanticGraphCoreAnnotations.BasicDependenciesAnnotation.class);
                    IndexedWord indexedWord = graph.getNodeByIndexSafe(pair.second);
                    if (indexedWord == null) return "NON-DEP";
                    List<IndexedWord> l = graph.getPathToRoot(indexedWord);
                    if (l.size() > 0) {
                        return l.get(0).get(CoreAnnotations.PartOfSpeechAnnotation.class);
                    }
                    else return "ROOT";
                });
                // Dependency parent arc
                add((pair) -> {
                    SemanticGraph graph = pair.first.annotation.get(CoreAnnotations.SentencesAnnotation.class).get(0)
                            .get(SemanticGraphCoreAnnotations.BasicDependenciesAnnotation.class);
                    IndexedWord indexedWord = graph.getNodeByIndexSafe(pair.second);
                    if (indexedWord == null) return "NON-DEP";
                    IndexedWord parent = graph.getParent(indexedWord);
                    if (parent == null) return "ROOT";
                    return graph.getAllEdges(parent, indexedWord).get(0).getRelation().getShortName();
                });
                // Dependency parent arc + Parent POS
                add((pair) -> {
                    SemanticGraph graph = pair.first.annotation.get(CoreAnnotations.SentencesAnnotation.class).get(0)
                            .get(SemanticGraphCoreAnnotations.BasicDependenciesAnnotation.class);
                    IndexedWord indexedWord = graph.getNodeByIndexSafe(pair.second);
                    if (indexedWord == null) return "NON-DEP";
                    IndexedWord parent = graph.getParent(indexedWord);
                    if (parent == null) return "ROOT";
                    return graph.getAllEdges(parent, indexedWord).get(0).getRelation().getShortName()+
                            parent.get(CoreAnnotations.PartOfSpeechAnnotation.class);
                });

                // Closest frame similarity
                add((pair) -> frameManager.getMaxSimilarity(pair.first.tokens[pair.second].toLowerCase()));
                // Closest frame token
                add((pair) -> frameManager.getClosestFrame(pair.first.tokens[pair.second].toLowerCase()));
            }},
            AMRPipeline::writeNerPlusPlusContext
    );

    @SuppressWarnings("unchecked")
    LinearPipe<Triple<LabeledSequence,Integer,Integer>, String> dictionaryLookup = new LinearPipe<>(
            new ArrayList<Function<Triple<LabeledSequence,Integer,Integer>,Object>>(){{

                // Input triple is (Seq, index into Seq for start of expression, index into Seq for end of expression)

                add((triple) -> triple.first.tokens[triple.second]);
            }},
            AMRPipeline::writeDictionaryContext
    );

    private String getPath(AMRNodeSet set, int head, int tail) {
        if (head == 0 || tail == 0) { // if tail == 0, then something else went fairly wrong
            return "ROOT:NOPATH";
        }
        if (head == tail) {
            return "IDENTITY";
        }
        int headToken = set.nodes[head].alignment;
        int tailToken = set.nodes[tail].alignment;
        SemanticGraph graph = set.annotation.get(CoreAnnotations.SentencesAnnotation.class).get(0)
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

    @SuppressWarnings("unchecked")
    ArrayList<Function<Triple<AMRNodeSet,Integer,Integer>,Object>> cmuFeatures =
            new ArrayList<Function<Triple<AMRNodeSet,Integer,Integer>,Object>>(){{
                // Input triple is (Set, array offset for first node, array offset for second node)

                // Self edge
                add((triple) -> {
                    if (triple.first.forcedArcs[triple.second][triple.third] != null) {
                        return 1.0;
                    }
                    else return 0.0;
                });
                // Head fragment root
                add((triple) -> {
                    for (int i = 1; i < triple.first.nodes.length; i++) {
                        // Any non-ROOT forced parents mean that this is not the head of the fragment
                        if (triple.first.forcedArcs[i][triple.second] != null) {
                            return 0.0;
                        }
                    }
                    return 1.0;
                });
                // Tail fragment root
                add((triple) -> {
                    for (int i = 1; i < triple.first.nodes.length; i++) {
                        // Any non-ROOT forced parents mean that this is not the head of the fragment
                        if (triple.first.forcedArcs[i][triple.third] != null) {
                            return 0.0;
                        }
                    }
                    return 1.0;
                });
                // Path
                add((triple) -> getPath(triple.first, triple.second, triple.third));
                // Distance
                add((triple) -> {
                    if (triple.second != 0) {
                        int headAlignment = triple.first.nodes[triple.second].alignment;
                        int tailAlignment = triple.first.nodes[triple.third].alignment;
                        return Math.abs(headAlignment - tailAlignment) + 1.0;
                    }
                    return 0.0;
                });
                // Distance Indicator
                add((triple) -> {
                    if (triple.second != 0) {
                        int headAlignment = triple.first.nodes[triple.second].alignment;
                        int tailAlignment = triple.first.nodes[triple.third].alignment;
                        return "D:"+Math.abs(headAlignment - tailAlignment) + 1.0;
                    }
                    return "D:ROOT";
                });
                // Log distance
                add((triple) -> {
                    if (triple.second != 0) {
                        int headAlignment = triple.first.nodes[triple.second].alignment;
                        int tailAlignment = triple.first.nodes[triple.third].alignment;
                        return Math.log(Math.abs(headAlignment - tailAlignment) + 1.0);
                    }
                    return 0.0;
                });
                // Path + Head concept
                add((triple) -> {
                    String headConcept;
                    if (triple.second == 0) {
                        headConcept = "ROOT";
                    }
                    else {
                        headConcept = triple.first.nodes[triple.second].toString();
                    }
                    return headConcept + ":" + getPath(triple.first, triple.second, triple.third);
                });
                // Path + Tail concept
                add((triple) -> {
                    String tailConcept = triple.first.nodes[triple.third].toString();
                    return tailConcept + ":" + getPath(triple.first, triple.second, triple.third);
                });
                // Path + Head word
                add((triple) -> {
                    String headWord;
                    if (triple.second == 0) {
                        headWord = "ROOT";
                    }
                    else {
                        headWord = triple.first.tokens[triple.first.nodes[triple.second].alignment];
                    }
                    return headWord + ":" + getPath(triple.first, triple.second, triple.third);
                });
                // Path + Tail word
                add((triple) -> {
                    String tailWord = triple.first.tokens[triple.first.nodes[triple.third].alignment];
                    return tailWord + ":" + getPath(triple.first, triple.second, triple.third);
                });
                // Path + Distance
                add((triple) -> {
                    String distanceIndicator;
                    if (triple.second != 0) {
                        int headAlignment = triple.first.nodes[triple.second].alignment;
                        int tailAlignment = triple.first.nodes[triple.third].alignment;
                        distanceIndicator = Integer.toString(Math.abs(headAlignment - tailAlignment));
                    }
                    else {
                        distanceIndicator = "ROOT";
                    }
                    return distanceIndicator + ":" + getPath(triple.first, triple.second, triple.third);
                });
    }};

    LinearPipe<Triple<AMRNodeSet,Integer,Integer>, Boolean> arcExistence = new LinearPipe<>(
            cmuFeatures,
            AMRPipeline::writeArcContext
    );

    @SuppressWarnings("unchecked")
    LinearPipe<Triple<AMRNodeSet,Integer,Integer>, String> arcType = new LinearPipe<>(
            cmuFeatures,
            AMRPipeline::writeArcContext
    );

    /////////////////////////////////////////////////////
    // PIPELINE
    /////////////////////////////////////////////////////

    public void trainStages() throws IOException {
        System.out.println("Loading training data");
        List<LabeledSequence> nerPlusPlusData = loadSequenceData(FULL_DATA ? "realdata/train-seq.txt" : ( TINY_DATA ? "data/train-3-seq.txt" : "data/train-400-seq.txt"));
        List<LabeledSequence> dictionaryData = loadManygenData(FULL_DATA ? "realdata/train-manygen.txt" : ( TINY_DATA ? "data/train-3-manygen.txt" : "data/train-400-manygen.txt"));
        List<AMRNodeSet> mstData = loadCoNLLData(FULL_DATA ? "realdata/train-conll.txt" : ( TINY_DATA ? "data/train-3-conll.txt" : "data/train-400-manygen.txt"));

        System.out.println("Training");
        nerPlusPlus.train(getNERPlusPlusForClassifier(nerPlusPlusData));
        dictionaryLookup.train(getDictionaryForClassifier(dictionaryData));
        arcExistence.train(getArcExistenceForClassifier(mstData));
        arcType.train(getArcTypeForClassifier(mstData));
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
                // TODO: Train a simple sense-tagger, or just use DICT for everything
                amr = createAMRSingleton(labeledSequence.annotation.get(CoreAnnotations.TokensAnnotation.class).
                        get(i).get(CoreAnnotations.LemmaAnnotation.class).toLowerCase()+"-01");
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

            for (String amrString : getBestAMRChunks(labeledSequence, first, last)) {
                AMR amr;
                if (amrString.startsWith("(")) {
                    assert (amrString.endsWith(")"));
                    amr = AMRSlurp.parseAMRTree(amrString);
                } else if (amrString.startsWith("\"")) {
                    amr = createAMRSingleton(amrString.substring(1, amrString.length() - 1), AMR.NodeType.QUOTE);
                } else {
                    amr = createAMRSingleton(amrString, AMR.NodeType.VALUE);
                }

                for (AMR.Node node : amr.nodes) {
                    node.alignment = Math.min(first + node.alignment, tokens.length - 1);
                }
                gen.add(amr);
            }
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

    private List<String> getBestAMRChunks(LabeledSequence labeledSequence, int first, int last) {
        List<String> bestChunks = new ArrayList<>();

        Counter<String> amrStrings = dictionaryLookup.predictSoft(new Triple<>(labeledSequence, first, last));

        String amrString = null;
        double bestCount = 0.0;
        for (String s : amrStrings.keySet()) {
            double count = amrStrings.getCount(s);
            if (count > bestCount) {
                amrString = s;
                bestCount = count;
            }
        }

        if (bestCount > 0) {
            bestChunks.add(amrString);
            return bestChunks;
        }
        else {
            // Recursion base case, we didn't find anything
            if (first == last) {
                return new ArrayList<>();
            }

            for (int pivot = first; pivot < last; pivot++) {
                List<String> part1 = getBestAMRChunks(labeledSequence, first, pivot);
                List<String> part2 = getBestAMRChunks(labeledSequence, pivot+1, last);

                if (part1.size() + part2.size() > bestChunks.size()) {
                    bestChunks = new ArrayList<>();
                    bestChunks.addAll(part1);
                    bestChunks.addAll(part2);
                }
            }

            return bestChunks;
        }
    }

    public AMR runMSTPipeline(String[] tokens, Annotation annotation, AMRNodeSet nodeSet) {

        // Run MST on the arcs we've got

        MSTGraph mstGraph = new MSTGraph();

        int length = nodeSet.nodes.length-1;

        for (int i = 0; i <= length; i++) {
            if (nodeSet.nodes[i] == null) continue;
            for (int j = 1; j <= length; j++) {
                if (nodeSet.nodes[j] == null) continue;
                if (i == j) continue;
                int forcedParent = -1;
                for (int k = 0; k <= length; k++) {
                    if (nodeSet.forcedArcs[k][j] != null) {
                        // Add this with such a high weight that it has to be included in the final MST
                        forcedParent = k;
                        break;
                    }
                }

                if (forcedParent != -1) {
                    if (forcedParent == i)
                        mstGraph.addArc(i, j, nodeSet.forcedArcs[i][j], 1);
                    else
                        mstGraph.addArc(i, j, "NO-ARC", Double.NEGATIVE_INFINITY);
                }
                else {
                    Counter<Boolean> counter = arcExistence.predictSoft(new Triple<>(nodeSet, i, j));
                    Counters.logNormalizeInPlace(counter);
                    mstGraph.addArc(i, j, "NO-LABEL", counter.getCount(true));
                }
            }
        }

        // Stitch based on the MST we got

        Map<Integer,Set<Pair<String,Integer>>> arcMap = mstGraph.getMST(false);

        // Just a validity check on the graph
        Set<Integer> seenNodes = new HashSet<>();
        Queue<Integer> visitQueue = new ArrayDeque<>();
        visitQueue.add(-1);

        while (!visitQueue.isEmpty()) {
            int i = visitQueue.poll();
            seenNodes.add(i);
            if (arcMap.containsKey(i)) {
                for (Pair<String, Integer> arc : arcMap.get(i)) {
                    if (!seenNodes.contains(arc.second)) {
                        visitQueue.add(arc.second);
                    }
                }
            }
        }

        for (int i = 0; i < nodeSet.nodes.length; i++) {
            if (nodeSet.nodes[i] != null && !seenNodes.contains(i)) {
                throw new IllegalStateException("Must contain all relevant nodes!");
            }
        }

        GreedyState state = new GreedyState(nodeSet.nodes, tokens, annotation);
        for (int i : arcMap.keySet()) {
            for (Pair<String,Integer> arc : arcMap.get(i)) {
                int head = i == -1 ? 0 : i;
                if (arc.first.equals("NO-LABEL")) {
                    arc.first = arcType.predict(new Triple<>(nodeSet, head, arc.second));
                }
                state.arcs[head][arc.second] = arc.first;
            }
        }

        // Verify that the arcs are all present that need to be
        for (int i = 0; i < nodeSet.nodes.length; i++) {
            if (nodeSet.nodes[i] != null) {
                boolean hasIncoming = false;
                for (int j = 0; j < nodeSet.nodes.length; j++) {
                    if (state.arcs[i][j] != null && !state.arcs[i][j].equals("NONE")) {
                        hasIncoming = true;
                        break;
                    }
                }
                if (!hasIncoming) {
                    throw new IllegalStateException("Must have incoming arcs for all nodes. Missing: "+nodeSet.nodes[i]);
                }
            }
        }

        AMR generated = Generator.generateAMR(state);

        // Verify the generated AMR has all relevant nodes
        for (int i = 0; i < nodeSet.nodes.length; i++) {
            if (nodeSet.nodes[i] != null) {
                boolean containsAnEqual = false;
                for (AMR.Node node : generated.nodes) {
                    if (nodeSet.nodes[i].title.equals(node.title)) containsAnEqual = true;
                }
                if (!containsAnEqual) {
                    throw new IllegalStateException("Must contain all nodes we arc'd for. Missing: "+nodeSet.nodes[i]);
                }
            }
        }

        return generated;
    }

    public AMR runPipeline(String[] tokens, Annotation annotation) {
        AMRNodeSet nodeSet = new AMRNodeSet();
        nodeSet.annotation = annotation;
        nodeSet.tokens = tokens;
        Pair<AMR.Node[], String[][]> nodesAndArcs = predictNodes(tokens, annotation);
        nodeSet.nodes = nodesAndArcs.first;
        nodeSet.correctArcs = new String[nodeSet.nodes.length][nodeSet.nodes.length];
        nodeSet.forcedArcs = nodesAndArcs.second;

        return runMSTPipeline(tokens, annotation, nodeSet);
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

    public void analyzeStages() throws IOException {
        System.out.println("Loading training data");
        List<LabeledSequence> nerPlusPlusDataTrain = loadSequenceData(FULL_DATA ? "realdata/train-seq.txt" : ( TINY_DATA ? "data/train-3-seq.txt" : "data/train-400-seq.txt"));
        List<LabeledSequence> dictionaryDataTrain = loadManygenData(FULL_DATA ? "realdata/train-manygen.txt" : ( TINY_DATA ? "data/train-3-manygen.txt" : "data/train-400-manygen.txt"));
        List<AMRNodeSet> mstDataTrain = loadCoNLLData(FULL_DATA ? "realdata/train-conll.txt" : ( TINY_DATA ? "data/train-3-conll.txt" : "data/train-400-conll.txt"));

        System.out.println("Loading testing data");
        List<LabeledSequence> nerPlusPlusDataTest = loadSequenceData(FULL_DATA ? "realdata/test-seq.txt" : (TINY_DATA ? "data/train-3-seq.txt" : "data/test-100-seq.txt"));
        List<LabeledSequence> dictionaryDataTest = loadManygenData(FULL_DATA ? "realdata/test-manygen.txt" : (TINY_DATA ? "data/train-3-manygen.txt" : "data/test-100-manygen.txt"));
        List<AMRNodeSet> mstDataTest = loadCoNLLData(FULL_DATA ? "realdata/test-conll.txt" : (TINY_DATA ? "data/train-3-conll.txt" : "data/test-100-conll.txt"));

        System.out.println("Running analysis");
        nerPlusPlus.analyze(getNERPlusPlusForClassifier(nerPlusPlusDataTrain),
                getNERPlusPlusForClassifier(nerPlusPlusDataTest),
                FULL_DATA ? "realdata/ner-plus-plus-analysis" : "data/ner-plus-plus-analysis");

        dictionaryLookup.analyze(getDictionaryForClassifier(dictionaryDataTrain),
                getDictionaryForClassifier(dictionaryDataTest),
                FULL_DATA ? "realdata/dictionary-lookup-analysis" : "data/dictionary-lookup-analysis");

        arcExistence.analyze(getArcExistenceForClassifier(mstDataTrain),
                getArcExistenceForClassifier(mstDataTest),
                FULL_DATA ? "realdata/arc-existence-analysis" : "data/arc-existence-analysis");

        arcType.analyze(getArcTypeForClassifier(mstDataTrain),
                getArcTypeForClassifier(mstDataTest),
                FULL_DATA ? "realdata/arc-type-analysis" : "data/arc-type-analysis");
    }

    public void testCompletePipeline() throws IOException, InterruptedException {
        if (FULL_DATA) {
            analyzeAMRSubset("realdata/train-subset.txt", "dabta/train-conll.txt", "realdata/amr-train-analysis");
            analyzeAMRSubset("realdata/test-subset.txt", "realdata/test-conll.txt", "realdata/amr-test-analysis");
        }
        else {
            if (TINY_DATA) {
                analyzeAMRSubset("data/train-3-subset.txt", "data/train-3-conll.txt", "data/amr-train-analysis");
            }
            else {
                analyzeAMRSubset("data/train-400-subset.txt", "data/train-400-conll.txt", "data/amr-train-analysis");
                analyzeAMRSubset("data/test-100-subset.txt", "data/test-100-conll.txt", "data/amr-test-analysis");
            }
        }
    }

    private void analyzeAMRSubset(String path, String coNLLPath, String output) throws IOException, InterruptedException {
        AMR[] bank = AMRSlurp.slurp(path, AMRSlurp.Format.LDC);
        List<AMRNodeSet> mstDataTest = loadCoNLLData(coNLLPath);

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
            recoveredPerfectDict[i] = runMSTPipeline(bank[i].sourceText, annotation, mstDataTest.get(i));
        }
        cache.close();

        System.out.println("Finished analyzing");

        File out = new File(output);
        if (out.exists()) out.delete();
        out.mkdirs();

        AMRSlurp.burp(output+"/gold.txt", AMRSlurp.Format.LDC, bank, AMR.AlignmentPrinting.ALL, false);
        AMRSlurp.burp(output+"/recovered.txt", AMRSlurp.Format.LDC, recovered, AMR.AlignmentPrinting.ALL, false);
        AMRSlurp.burp(output+"/recovered-perfect-dict.txt", AMRSlurp.Format.LDC, recoveredPerfectDict, AMR.AlignmentPrinting.ALL, false);

        DumpSequence.dumpCONLL(recovered, output+"/recovered-conll.txt");
        DumpSequence.dumpCONLL(recoveredPerfectDict, output + "/recovered-perfect-dict-conll.txt");

        double smatch = Smatch.smatch(bank, recovered);
        System.out.println("SMATCH for overall "+path+" = "+smatch);
        BufferedWriter bw = new BufferedWriter(new FileWriter(output+"/smatch.txt"));
        bw.write("Smatch: "+smatch);
        bw.close();

        double smatchPerfectDict = Smatch.smatch(bank, recoveredPerfectDict);
        System.out.println("SMATCH for perfect dict "+path+" = "+smatchPerfectDict);
        bw = new BufferedWriter(new FileWriter(output+"/smatch-perfect-dict.txt"));
        bw.write("Smatch: "+smatch);
        bw.close();
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        AMRPipeline pipeline = new AMRPipeline();
        pipeline.trainStages();
        pipeline.analyzeStages();
        pipeline.testCompletePipeline();
    }

    /////////////////////////////////////////////////////
    // LOADERS
    /////////////////////////////////////////////////////

    private List<LabeledSequence> loadSequenceData(String path) throws IOException {
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

        return seqList;
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

    public static List<AMRNodeSet> loadCoNLLData(String path) throws IOException {
        List<AMRNodeSet> setList = new ArrayList<>();

        BufferedReader br = new BufferedReader(new FileReader(path));

        // Read the TSV file

        int stage = 0;

        AMRNodeSet currentNodeSet = null;

        String line;
        while ((line = br.readLine()) != null) {

            stage++;

            if (line.length() == 0) {
                // a newline to flush out old stuff
                stage = 0;
                if (currentNodeSet != null) {
                    for (int i = 0; i < currentNodeSet.nodes.length; i++) {
                        for (int j = 0; j < currentNodeSet.nodes.length; j++) {
                            if (i == j) continue;
                            if (currentNodeSet.nodes[i] == null || currentNodeSet.nodes[j] == null) continue;
                            if (currentNodeSet.nodes[i].alignment == currentNodeSet.nodes[j].alignment) {
                                currentNodeSet.forcedArcs[i][j] = currentNodeSet.correctArcs[i][j];
                                currentNodeSet.correctArcs[i][j] = "NONE";
                            }
                        }
                    }
                    setList.add(currentNodeSet);
                }
                currentNodeSet = null;
            }

            else if (stage == 1) {
                // read a sentence
                assert(currentNodeSet == null);
                currentNodeSet = new AMRNodeSet();

                String[] parts = line.split("\t");
                int length = Integer.parseInt(parts[0]);

                currentNodeSet.tokens = parts[1].split(" ");

                currentNodeSet.correctArcs = new String[length+1][length+1];
                currentNodeSet.forcedArcs = new String[length+1][length+1];
                currentNodeSet.nodes = new AMR.Node[length+1];
            }
            else {
                String[] parts = line.split("\t");
                assert(parts.length == 5);

                // Format layout:
                // NODE_INDEX, NODE, PARENT_INDEX, ARC_NAME, ALIGNMENT

                int index = Integer.parseInt(parts[0]);
                String node = parts[1];
                int parentIndex = Integer.parseInt(parts[2]);
                String arcName = parts[3];
                int alignment = Integer.parseInt(parts[4]);

                // Parse out the node

                AMR.Node parsedNode;
                if (node.startsWith("\"")) {
                    String dequoted = node.substring(1, node.length()-1);
                    parsedNode = new AMR.Node(""+dequoted.toLowerCase().charAt(0), dequoted, AMR.NodeType.QUOTE);
                }
                else if (node.startsWith("(")) {
                    String[] debracedParts = node.substring(1, node.length()-1).split("/");
                    parsedNode = new AMR.Node(debracedParts[0], debracedParts[1], AMR.NodeType.ENTITY);
                }
                else {
                    parsedNode = new AMR.Node("x", node, AMR.NodeType.VALUE);
                }
                parsedNode.alignment = alignment;

                assert(currentNodeSet != null);

                currentNodeSet.nodes[index] = parsedNode;
                currentNodeSet.correctArcs[parentIndex][index] = arcName;
            }
        }
        if (currentNodeSet != null) {
            setList.add(currentNodeSet);
        }

        String[] sentences = new String[setList.size()];
        for (int i = 0; i < setList.size(); i++) {
            sentences[i] = setList.get(i).formatTokens();
        }
        CoreNLPCache coreNLPCache = new BatchCoreNLPCache(path, sentences);
        for (int i = 0; i < setList.size(); i++) {
            setList.get(i).annotation = coreNLPCache.getAnnotation(i);
        }

        return setList;
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

    public static List<Pair<Triple<AMRNodeSet,Integer,Integer>,Boolean>> getArcExistenceForClassifier(
            List<AMRNodeSet> mstData) {

        List<Pair<Triple<AMRNodeSet,Integer,Integer>,Boolean>> arcExistenceTrue = new ArrayList<>();
        List<Pair<Triple<AMRNodeSet,Integer,Integer>,Boolean>> arcExistenceFalse = new ArrayList<>();

        for (AMRNodeSet set : mstData) {
            for (int i = 0; i < set.correctArcs.length; i++) {
                if (i != 0 && set.nodes[i] == null) continue;
                for (int j = 1; j < set.correctArcs[i].length; j++) {
                    if (set.nodes[j] == null) continue;
                    boolean arcExists = set.correctArcs[i][j] != null;
                    if (arcExists) {
                        arcExistenceTrue.add(new Pair<>(new Triple<>(set, i, j), true));
                    }
                    else {
                        arcExistenceFalse.add(new Pair<>(new Triple<>(set, i, j), false));
                    }
                }
            }
        }

        int numTrueClones = Math.max(1, arcExistenceFalse.size() / arcExistenceTrue.size());

        List<Pair<Triple<AMRNodeSet,Integer,Integer>,Boolean>> arcExistenceData = new ArrayList<>();
        arcExistenceData.addAll(arcExistenceFalse);
        for (int i = 0; i < numTrueClones; i++) {
            arcExistenceData.addAll(arcExistenceTrue);
        }

        return arcExistenceData;
    }

    public static List<Pair<Triple<AMRNodeSet,Integer,Integer>,String>> getArcTypeForClassifier(
            List<AMRNodeSet> mstData) {
        List<Pair<Triple<AMRNodeSet,Integer,Integer>,String>> arcTypeData = new ArrayList<>();
        for (AMRNodeSet set : mstData) {
            for (int i = 0; i < set.correctArcs.length; i++) {
                if (i != 0 && set.nodes[i] == null) continue;
                for (int j = 1; j < set.correctArcs[i].length; j++) {
                    if (set.nodes[j] == null) continue;
                    boolean arcExists = set.correctArcs[i][j] != null;
                    if (arcExists) {
                        arcTypeData.add(new Pair<>(new Triple<>(set, i, j), set.correctArcs[i][j]));
                    }
                }
            }
        }
        return arcTypeData;
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

    public static void writeArcContext(Triple<AMRNodeSet, Integer, Integer> triple, BufferedWriter bw) {
        AMRNodeSet set = triple.first;

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
