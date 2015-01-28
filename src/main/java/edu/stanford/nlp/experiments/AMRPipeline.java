package edu.stanford.nlp.experiments;

import com.github.keenon.minimalml.cache.BatchCoreNLPCache;
import com.github.keenon.minimalml.cache.CoreNLPCache;
import com.github.keenon.minimalml.word2vec.Word2VecLoader;
import edu.stanford.nlp.stamr.AMR;
import edu.stanford.nlp.stamr.AMRParser;
import edu.stanford.nlp.stamr.AMRSlurp;
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
public class AMRPipeline {

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

    @SuppressWarnings("unchecked")
    LinearPipe<Triple<AMRNodeSet,Integer,Integer>, Boolean> arcExistence = new LinearPipe<>(
            new ArrayList<Function<Triple<AMRNodeSet,Integer,Integer>,Object>>(){{

                // Input triple is (Set, array offset for first node, array offset for second node)

                add((triple) -> triple.second);
            }},
            AMRPipeline::writeArcContext
    );

    @SuppressWarnings("unchecked")
    LinearPipe<Triple<AMRNodeSet,Integer,Integer>, String> arcType = new LinearPipe<>(
            new ArrayList<Function<Triple<AMRNodeSet,Integer,Integer>,Object>>(){{

                // Input triple is (Set, array offset for first node, array offset for second node)

                add((triple) -> triple.second);
            }},
            AMRPipeline::writeArcContext
    );

    /////////////////////////////////////////////////////
    // PIPELINE
    /////////////////////////////////////////////////////

    public void trainStages() throws IOException {
        System.out.println("Loading training data");
        List<LabeledSequence> nerPlusPlusData = loadSequenceData("data/train-400-seq.txt");
        List<LabeledSequence> dictionaryData = loadManygenData("data/train-400-manygen.txt");
        List<AMRNodeSet> mstData = loadCoNLLData("data/train-400-conll.txt");

        System.out.println("Training");
        nerPlusPlus.train(getNERPlusPlusForClassifier(nerPlusPlusData));
        dictionaryLookup.train(getDictionaryForClassifier(dictionaryData));
        arcExistence.train(getArcExistenceForClassifier(mstData));
        arcType.train(getArcTypeForClassifier(mstData));
    }

    public AMR runPipeline(String[] parts) {
        // TODO, actually stitch together outputs
        return new AMR();
    }

    public void analyzeStages() throws IOException {
        System.out.println("Loading training data");
        List<LabeledSequence> nerPlusPlusDataTrain = loadSequenceData("data/train-400-seq.txt");
        List<LabeledSequence> dictionaryDataTrain = loadManygenData("data/train-400-manygen.txt");
        List<AMRNodeSet> mstDataTrain = loadCoNLLData("data/train-400-conll.txt");

        System.out.println("Loading testing data");
        List<LabeledSequence> nerPlusPlusDataTest = loadSequenceData("data/test-100-seq.txt");
        List<LabeledSequence> dictionaryDataTest = loadManygenData("data/test-100-manygen.txt");
        List<AMRNodeSet> mstDataTest = loadCoNLLData("data/test-100-conll.txt");

        System.out.println("Running analysis");
        nerPlusPlus.analyze(getNERPlusPlusForClassifier(nerPlusPlusDataTrain),
                getNERPlusPlusForClassifier(nerPlusPlusDataTest),
                "data/ner-plus-plus-analysis");

        dictionaryLookup.analyze(getDictionaryForClassifier(dictionaryDataTrain),
                getDictionaryForClassifier(dictionaryDataTest),
                "data/dictionary-lookup-analysis");

        arcExistence.analyze(getArcExistenceForClassifier(mstDataTrain),
                getArcExistenceForClassifier(mstDataTest),
                "data/arc-existence-analysis");

        arcType.analyze(getArcTypeForClassifier(mstDataTrain),
                getArcTypeForClassifier(mstDataTest),
                "data/arc-type-analysis");
    }

    public static void main(String[] args) throws IOException {
        AMRPipeline pipeline = new AMRPipeline();
        pipeline.trainStages();
        pipeline.analyzeStages();
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

    private List<AMRNodeSet> loadCoNLLData(String path) throws IOException {
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
                setList.add(currentNodeSet);
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
        List<Pair<Triple<AMRNodeSet,Integer,Integer>,Boolean>> arcExistenceData = new ArrayList<>();
        for (AMRNodeSet set : mstData) {
            for (int i = 0; i < set.correctArcs.length; i++) {
                for (int j = 0; j < set.correctArcs[i].length; j++) {
                    boolean arcExists = set.correctArcs[i][j] != null;
                    arcExistenceData.add(new Pair<>(new Triple<>(set, i, j), arcExists));
                }
            }
        }
        return arcExistenceData;
    }

    public static List<Pair<Triple<AMRNodeSet,Integer,Integer>,String>> getArcTypeForClassifier(
            List<AMRNodeSet> mstData) {
        List<Pair<Triple<AMRNodeSet,Integer,Integer>,String>> arcTypeData = new ArrayList<>();
        for (AMRNodeSet set : mstData) {
            for (int i = 0; i < set.correctArcs.length; i++) {
                for (int j = 0; j < set.correctArcs[i].length; j++) {
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
