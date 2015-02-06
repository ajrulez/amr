package edu.stanford.nlp.experiments;

import com.github.keenon.minimalml.cache.BatchCoreNLPCache;
import com.github.keenon.minimalml.cache.CoreNLPCache;
import com.github.keenon.minimalml.cache.LazyCoreNLPCache;
import com.github.keenon.minimalml.word2vec.Word2VecLoader;
import edu.stanford.nlp.experiments.greedy.GreedyState;
import edu.stanford.nlp.experiments.greedy.NodeConnector;
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

    NodeConnector nodeConnector = new NodeConnector();

    /////////////////////////////////////////////////////
    // PIPELINE
    /////////////////////////////////////////////////////

    public void trainStages() throws IOException {
        System.out.println("Loading training data");
        List<LabeledSequence> nerPlusPlusData = loadSequenceData("data/train-400-seq.txt");
        List<LabeledSequence> dictionaryData = loadManygenData("data/train-400-manygen.txt");
        AMR[] bank = AMRSlurp.slurp("data/train-400-subset.txt", AMRSlurp.Format.LDC);

        System.out.println("Training");
        nerPlusPlus.train(getNERPlusPlusForClassifier(nerPlusPlusData));
        dictionaryLookup.train(getDictionaryForClassifier(dictionaryData));

        nodeConnector.train(getArcStitchingForNodeConnect(bank, nerPlusPlusData));
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

        // Now we can go through and start to generate components as a list of AMR's

        List<AMR> gen = new ArrayList<>();

        // First do all non-dict elements

        for (int i = 0; i < tokens.length; i++) {
            AMR amr = null;
            if (labels[i].equals("VERB")) {
                // TODO: Train a simple sense-tagger, or just use DICT for everything
                amr = createAMRSingleton(tokens[i].toLowerCase()+"-01");
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

        String[][] arcs = nodeConnector.connect(state.nodes, state.forcedArcs);

        Map<Integer,Set<Pair<String,Integer>>> arcMap = new HashMap<>();

        for (int i = 0; i < arcs.length; i++) {
            arcMap.put(i, new HashSet<>());
            for (int j = 0; j < arcs[i].length; j++) {
                if (arcs[i][j] != null) {
                    arcMap.get(i).add(new Pair<>(arcs[i][j], j));
                }
            }
        }

        Map<Integer,AMR.Node> oldToNew = new HashMap<>();
        AMR result = new AMR();
        result.sourceText = state.tokens;

        int root = arcMap.get(0).iterator().next().second;
        AMR.Node rootNode = state.nodes[root];
        if (rootNode.type == AMR.NodeType.ENTITY) {
            oldToNew.put(root, result.addNode(rootNode.ref, rootNode.title, rootNode.alignment));
        }
        else {
            oldToNew.put(root, result.addNode(rootNode.title, rootNode.type));
        }

        recursivelyAttach(arcMap, result, oldToNew, root, state);

        for (AMR.Node node : result.nodes) {
            result.giveNodeUniqueRef(node);
        }

        // Simple coref solution, match based on identical source tokens

        for (AMR.Node node : result.nodes) {
            for (AMR.Node node2 : result.nodes) {
                if (node == node2) continue;
                String token1 = state.tokens[node.alignment];
                String token2 = state.tokens[node2.alignment];
                if (token1.equalsIgnoreCase(token2)) {
                    node.ref = node2.ref;
                }
            }
        }

        return result;
    }

    private void recursivelyAttach(Map<Integer,Set<Pair<String,Integer>>> arcMap,
                                   AMR result,
                                   Map<Integer,AMR.Node> oldToNew,
                                   int parent,
                                   AMRNodeStateBased nodeSet) {
        if (!arcMap.containsKey(parent)) return;

        Set<Pair<String,Integer>> outgoingArcs = arcMap.get(parent);

        for (Pair<String,Integer> arc : outgoingArcs) {
            int child = arc.second;

            AMR.Node childNode = nodeSet.nodes[child];
            AMR.Node childNodeNew;

            if (oldToNew.containsKey(child)) {
                childNodeNew = oldToNew.get(child);
            }
            else {
                if (childNode.type == AMR.NodeType.ENTITY) {
                    childNodeNew = result.addNode(childNode.ref, childNode.title, childNode.alignment);
                } else {
                    childNodeNew = result.addNode(childNode.title, childNode.type);
                }
                oldToNew.put(child, childNodeNew);
            }

            AMR.Node parentNodeNew = oldToNew.get(parent);

            result.addArc(parentNodeNew, childNodeNew, arc.first);

            recursivelyAttach(arcMap, result, oldToNew, child, nodeSet);
        }
    }

    public AMR runPipeline(String[] tokens, Annotation annotation) {
        Pair<AMR.Node[], String[][]> nodesAndArcs = predictNodes(tokens, annotation);
        AMRNodeStateBased nodeSet = new AMRNodeStateBased(nodesAndArcs.first.length-1, annotation);
        nodeSet.nodes = nodesAndArcs.first;
        nodeSet.forcedArcs = nodesAndArcs.second;

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

    public void analyzeStages() throws IOException {
        System.out.println("Loading training data");
        List<LabeledSequence> nerPlusPlusDataTrain = loadSequenceData("data/train-400-seq.txt");
        List<LabeledSequence> dictionaryDataTrain = loadManygenData("data/train-400-manygen.txt");

        System.out.println("Loading testing data");
        List<LabeledSequence> nerPlusPlusDataTest = loadSequenceData("data/test-100-seq.txt");
        List<LabeledSequence> dictionaryDataTest = loadManygenData("data/test-100-manygen.txt");

        System.out.println("Running analysis");
        nerPlusPlus.analyze(getNERPlusPlusForClassifier(nerPlusPlusDataTrain),
                getNERPlusPlusForClassifier(nerPlusPlusDataTest),
                "data/ner-plus-plus-analysis");

        dictionaryLookup.analyze(getDictionaryForClassifier(dictionaryDataTrain),
                getDictionaryForClassifier(dictionaryDataTest),
                "data/dictionary-lookup-analysis");

        /*
        AMR[] trainBank = AMRSlurp.slurp("data/train-400-subset.txt", AMRSlurp.Format.LDC);
        AMR[] testBank = AMRSlurp.slurp("data/test-100-subset.txt", AMRSlurp.Format.LDC);
        */
    }

    public void testCompletePipeline() throws IOException, InterruptedException {
        analyzeAMRSubset("data/train-400-subset.txt", "data/train-400-conll.txt", "data/amr-train-analysis");
        analyzeAMRSubset("data/test-100-subset.txt", "data/test-100-conll.txt", "data/amr-test-analysis");
    }

    private void analyzeAMRSubset(String path, String coNLLPath, String output) throws IOException, InterruptedException {
        AMR[] bank = AMRSlurp.slurp(path, AMRSlurp.Format.LDC);

        String[] sentences = new String[bank.length];
        for (int i = 0; i < bank.length; i++) {
            sentences[i] = bank[i].formatSourceTokens();
        }
        CoreNLPCache cache = new LazyCoreNLPCache(path, sentences);

        AMR[] recovered = new AMR[bank.length];
        for (int i = 0; i < bank.length; i++) {
            Annotation annotation = cache.getAnnotation(i);
            recovered[i] = runPipeline(bank[i].sourceText, annotation);
        }
        cache.close();

        System.out.println("Finished analyzing");

        File out = new File(output);
        if (out.exists()) out.delete();
        out.mkdirs();

        AMRSlurp.burp(output+"/gold.txt", AMRSlurp.Format.LDC, bank, AMR.AlignmentPrinting.ALL, false);
        AMRSlurp.burp(output+"/recovered.txt", AMRSlurp.Format.LDC, recovered, AMR.AlignmentPrinting.ALL, false);

        double smatch = Smatch.smatch(bank, recovered);
        System.out.println("SMATCH for overall "+path+" = "+smatch);
        BufferedWriter bw = new BufferedWriter(new FileWriter(output+"/smatch.txt"));
        bw.write("Smatch: "+smatch);
        bw.close();
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        AMRPipelineStateBased pipeline = new AMRPipelineStateBased();
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

    public static List<Pair<GreedyState, String[][]>> getArcStitchingForNodeConnect(AMR[] bank, List<LabeledSequence> nerPlusPlusData) {
        List<Pair<GreedyState, String[][]>> list = new ArrayList<>();

        for (int i = 0; i < bank.length; i++) {
            AMR amr = bank[i];

            Pair<GreedyState, String[][]> pair = NodeConnector.amrToContextAndArcs(amr);
            pair.first.annotation = nerPlusPlusData.get(i).annotation;
            list.add(pair);
        }

        return list;
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

    public static List<Pair<Triple<AMRNodeStateBased,Integer,Integer>,String>> getArcTypeForClassifier(AMR[] bank,
                                                                                                       List<LabeledSequence> genSequences,
                                                                                                       List<LabeledSequence> dictSequences) {
        List<Pair<Triple<AMRNodeStateBased,Integer,Integer>,String>> trainingData = new ArrayList<>();
        for (int i = 0; i < bank.length; i++) {
            AMR amr = bank[i];
            AMRNodeStateBased state = new AMRNodeStateBased(amr.nodes.size(), dictSequences.get(i).annotation);
            state.tokens = amr.sourceText;

            // We reserve index 0 for ROOT
            int j = 1;
            List<AMR.Node> nodeList = new ArrayList<>();
            for (AMR.Node node : amr.nodes) {
                state.nodes[j] = node;
                nodeList.add(node);
                j++;
            }

            state.correctArcs[0][nodeList.indexOf(amr.head)+1] = "ROOT";
            for (AMR.Arc arc : amr.arcs) {
                state.correctArcs[nodeList.indexOf(arc.head)+1][nodeList.indexOf(arc.tail)+1] = arc.title;
            }

            Queue<Integer> visitQueue = new ArrayDeque<>();
            Set<Integer> visited = new HashSet<>();
            visitQueue.add(0);

            while (!visitQueue.isEmpty()) {
                state = new AMRNodeStateBased(state);
                state.currentParent = visitQueue.poll();
                visited.add(state.currentParent);
                for (int k = 1; k < state.correctArcs[state.currentParent].length; k++) {
                    if (k == state.currentParent) continue;
                    String arc = state.correctArcs[state.currentParent][k];
                    if (arc == null) {
                        arc = "NONE";
                    }
                    else if (!visited.contains(k)) {
                        visitQueue.add(k);
                    }
                    trainingData.add(new Pair<>(new Triple<>(state, state.currentParent, k), arc));
                }
            }
        }

        return trainingData;
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
