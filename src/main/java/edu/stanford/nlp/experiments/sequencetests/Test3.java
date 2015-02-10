package edu.stanford.nlp.experiments.sequencetests;

import com.github.keenon.minimalml.word2vec.Word2VecLoader;
import edu.stanford.nlp.experiments.AMRPipelineStateBased;
import edu.stanford.nlp.experiments.LabeledSequence;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.util.Pair;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * Created by keenon on 2/9/15.
 */
public class Test3 {

    static Map<String,double[]> embeddings;

    static {
        try {
            embeddings = Word2VecLoader.loadData("data/google-300-trimmed.ser.gz");
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
            }};

    public static void main(String[] args) throws IOException {
        AMRPipelineStateBased.testSequenceTagger("seq-test-3", seqFeatures);
    }
}
