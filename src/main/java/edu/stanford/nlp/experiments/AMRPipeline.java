package edu.stanford.nlp.experiments;

import com.github.keenon.minimalml.cache.BatchCoreNLPCache;
import com.github.keenon.minimalml.cache.CoreNLPCache;
import edu.stanford.nlp.stamr.AMR;
import edu.stanford.nlp.stamr.AMRParser;
import edu.stanford.nlp.stamr.AMRSlurp;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.Triple;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Function;

/**
 * Created by keenon on 1/27/15.
 *
 * Holds and trains several pipes, which can be analyzed separately or together to give results about AMR.
 */
public class AMRPipeline {

    /////////////////////////////////////////////////////
    // FEATURE SPECS
    /////////////////////////////////////////////////////

    @SuppressWarnings("unchecked")
    LinearPipe<Pair<LabeledSequence,Integer>, String> nerPlusPlus = new LinearPipe<>(
            new ArrayList<Function<Pair<LabeledSequence,Integer>,Object>>(){{
                add((pair) -> pair.second);
            }}
    );

    @SuppressWarnings("unchecked")
    LinearPipe<Triple<LabeledSequence,Integer,Integer>, String> dictionaryLookup = new LinearPipe<>(
            new ArrayList<Function<Triple<LabeledSequence,Integer,Integer>,Object>>(){{
                add((triple) -> triple.second);
            }}
    );

    @SuppressWarnings("unchecked")
    LinearPipe<Triple<AMRNodeSet,Integer,Integer>, Boolean> arcExistence = new LinearPipe<>(
            new ArrayList<Function<Triple<AMRNodeSet,Integer,Integer>,Object>>(){{
                add((triple) -> triple.second);
            }}
    );

    @SuppressWarnings("unchecked")
    LinearPipe<Triple<AMRNodeSet,Integer,Integer>, String> arcType = new LinearPipe<>(
            new ArrayList<Function<Triple<AMRNodeSet,Integer,Integer>,Object>>(){{
                add((triple) -> triple.second);
            }}
    );

    /////////////////////////////////////////////////////
    // PIPELINE
    /////////////////////////////////////////////////////

    public void trainStages() throws IOException {
        List<LabeledSequence> nerPlusPlusData = loadSequenceData("data/training-500-seq.txt");
        List<LabeledSequence> dictionaryData = loadManygenData("data/training-500-manygen.txt");
        List<AMRNodeSet> mstData = loadCoNLLData("data/training-500-conll.txt");

        nerPlusPlus.train(getNERPlusPlusForClassifier(nerPlusPlusData));
        dictionaryLookup.train(getDictionaryForClassifier(dictionaryData));
        arcExistence.train(getArcExistenceForClassifier(mstData));
        arcType.train(getArcTypeForClassifier(mstData));
    }

    public void analyzeStages() throws IOException {
        List<LabeledSequence> nerPlusPlusData = loadSequenceData("data/training-500-seq.txt");
        List<LabeledSequence> dictionaryData = loadManygenData("data/training-500-manygen.txt");
        List<AMRNodeSet> mstData = loadCoNLLData("data/training-500-conll.txt");

        nerPlusPlus.analyze(getNERPlusPlusForClassifier(nerPlusPlusData));
        dictionaryLookup.analyze(getDictionaryForClassifier(dictionaryData));
        arcExistence.analyze(getArcExistenceForClassifier(mstData));
        arcType.analyze(getArcTypeForClassifier(mstData));
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
                currentSeq = new LabeledSequence();
            }

            else if (stage == 1) {
                assert(currentSeq != null);
                currentSeq.tokens = line.split(" ");
                currentSeq.labels = new String[currentSeq.tokens.length];
            }
            else {
                String[] parts = line.split("\t");
                assert(parts.length == 4);
                assert(currentSeq != null);

                // Format layout:
                // TOKEN, AMR, START_INDEX, END_INDEX

                String token = parts[0];
                String amr = parts[1];

                int startIndex = Integer.parseInt(parts[2]);
                int endIndex = Integer.parseInt(parts[3]);
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

                currentNodeSet.tokens = line.split(" ");

                int length = currentNodeSet.tokens.length;
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

    public static Set<Pair<Pair<LabeledSequence,Integer>,String>> getNERPlusPlusForClassifier(
            List<LabeledSequence> nerPlusPlusData) {
        Set<Pair<Pair<LabeledSequence,Integer>,String>> nerPlusPlusTrainingData = new HashSet<>();
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

    public static Set<Pair<Triple<LabeledSequence,Integer,Integer>,String>> getDictionaryForClassifier(
        List<LabeledSequence> dictionaryData) {
        Set<Pair<Triple<LabeledSequence,Integer,Integer>,String>> dictionaryTrainingData = new HashSet<>();
        for (LabeledSequence seq : dictionaryData) {
            int min = seq.tokens.length;
            int max = 0;
            for (int i = 0; i < seq.tokens.length; i++) {
                if (seq.labels[i].length() > 0) {
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

    public static Set<Pair<Triple<AMRNodeSet,Integer,Integer>,Boolean>> getArcExistenceForClassifier(
            List<AMRNodeSet> mstData) {
        Set<Pair<Triple<AMRNodeSet,Integer,Integer>,Boolean>> arcExistenceData = new HashSet<>();
        for (AMRNodeSet set : mstData) {
            for (int i = 0; i < set.correctArcs.length; i++) {
                for (int j = 0; j < set.correctArcs[i].length; j++) {
                    boolean arcExists = set.correctArcs[i][j].length() == 1;
                    arcExistenceData.add(new Pair<>(new Triple<>(set, i, j), arcExists));
                }
            }
        }
        return arcExistenceData;
    }

    public static Set<Pair<Triple<AMRNodeSet,Integer,Integer>,String>> getArcTypeForClassifier(
            List<AMRNodeSet> mstData) {
        Set<Pair<Triple<AMRNodeSet,Integer,Integer>,String>> arcTypeData = new HashSet<>();
        for (AMRNodeSet set : mstData) {
            for (int i = 0; i < set.correctArcs.length; i++) {
                for (int j = 0; j < set.correctArcs[i].length; j++) {
                    boolean arcExists = set.correctArcs[i][j].length() == 1;
                    if (arcExists) {
                        arcTypeData.add(new Pair<>(new Triple<>(set, i, j), set.correctArcs[i][j]));
                    }
                }
            }
        }
        return arcTypeData;
    }
}
