package edu.stanford.nlp.experiments;

import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.Triple;

import java.util.function.Function;

/**
 * Created by keenon on 1/27/15.
 *
 * Holds and trains several pipes, which can be analyzed separately or together to give results about AMR.
 */
public class AMRPipeline {
    @SuppressWarnings("unchecked")
    LinearPipe<Pair<LabeledSequence,Integer>, String> nerPlusPlus = new LinearPipe<>(new Function[]{
            (pair) -> 0
    });

    @SuppressWarnings("unchecked")
    LinearPipe<Pair<LabeledSequence,Integer>, String> dictionaryLookup = new LinearPipe<>(new Function[]{
            (pair) -> 0
    });

    @SuppressWarnings("unchecked")
    LinearPipe<Triple<AMRNodeSet,Integer,Integer>, Boolean> arcExistence = new LinearPipe<>(new Function[]{
            (pair) -> 0
    });

    @SuppressWarnings("unchecked")
    LinearPipe<Triple<AMRNodeSet,Integer,Integer>, String> arcType = new LinearPipe<>(new Function[]{
            (pair) -> 0
    });

    public void trainStages() {

    }
}
