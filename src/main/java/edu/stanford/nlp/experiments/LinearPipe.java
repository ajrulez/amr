package edu.stanford.nlp.experiments;

import edu.stanford.nlp.classify.LinearClassifier;
import edu.stanford.nlp.classify.LinearClassifierFactory;
import edu.stanford.nlp.classify.RVFDataset;
import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.Triple;

import java.util.List;
import java.util.Set;
import java.util.function.Function;

/**
 * Created by keenon on 1/27/15.
 *
 * Implements a simple linear ML pipe, which takes in some sort of information, and pipes out a new sort.
 *
 * Useful for both standalone testing and full-runs of real stuff.
 */
public class LinearPipe<IN,OUT> {

    Function<IN,Object>[] features;
    LinearClassifier<OUT,String> classifier;

    @SuppressWarnings("unchecked")
    public LinearPipe(List<Function<IN,Object>> features) {
        this.features = features.toArray(new Function[features.size()]);
    }

    private Counter<String> featurize(IN in) {
        Counter<String> featureCounts = new ClassicCounter<>();

        for (int i = 0; i < features.length; i++) {
            Function<IN,Object> feature = features[i];

            Object obj = feature.apply(in);
            if (obj instanceof double[]) {
                double[] arr = (double[])obj;
                for (int j = 0; j < arr.length; j++) {
                    featureCounts.setCount(i + ":" + j, arr[j]);
                }
            }
            else if (obj instanceof Double) {
                featureCounts.setCount(Integer.toString(i), (double)obj);
            }
            else {
                featureCounts.setCount(obj.toString(), 1.0);
            }
        }

        return featureCounts;
    }

    private RVFDatum<OUT, String> toDatum(IN in, OUT out) {
        return new RVFDatum<>(featurize(in), out);
    }

    public void train(Set<Pair<IN,OUT>> data) {
        LinearClassifierFactory<OUT,String> factory = new LinearClassifierFactory<>();
        factory.setSigma(2.0);  // higher -> less regularization (default=1)
        RVFDataset<OUT, String> dataset = new RVFDataset<>();
        for (Pair<IN,OUT> pair : data) {
            dataset.add(toDatum(pair.first, pair.second));
        }
        classifier = factory.trainClassifier(dataset);
    }

    public OUT predict(IN in) {
        return classifier.classOf(new RVFDatum<OUT, String>(featurize(in)));
    }

    public void analyze(Set<Pair<IN,OUT>> test) {

    }
}
