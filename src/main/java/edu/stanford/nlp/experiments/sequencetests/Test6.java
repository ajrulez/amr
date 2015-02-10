package edu.stanford.nlp.experiments.sequencetests;

import com.github.keenon.minimalml.word2vec.Word2VecLoader;
import edu.stanford.nlp.experiments.AMRPipelineStateBased;
import edu.stanford.nlp.experiments.FrameManager;
import edu.stanford.nlp.experiments.LabeledSequence;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.process.WordShapeClassifier;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.util.Pair;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * Created by keenon on 2/9/15.
 */
public class Test6 {

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

    static List<Function<Pair<LabeledSequence,Integer>,Object>> seqFeatures =
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

                add((pair) -> {
                    int shapeType = WordShapeClassifier.WORDSHAPECHRIS4;
                    return WordShapeClassifier.wordShape(pair.first.tokens[pair.second], shapeType);
                });
            }};

    public static void main(String[] args) throws IOException {
        AMRPipelineStateBased.testSequenceTagger("seq-test-6", seqFeatures);
    }
}
