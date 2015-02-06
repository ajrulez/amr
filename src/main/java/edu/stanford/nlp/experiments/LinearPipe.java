package edu.stanford.nlp.experiments;

import edu.stanford.nlp.classify.LinearClassifier;
import edu.stanford.nlp.classify.LinearClassifierFactory;
import edu.stanford.nlp.classify.RVFDataset;
import edu.stanford.nlp.ling.RVFDatum;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.Triple;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.function.BiConsumer;
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
    public BiConsumer<IN, BufferedWriter> debugErrorContext;

    @SuppressWarnings("unchecked")
    public LinearPipe(List<Function<IN,Object>> features, BiConsumer<IN, BufferedWriter> debugErrorContext) {
        this.features = features.toArray(new Function[features.size()]);
        this.debugErrorContext = debugErrorContext;
    }

    public void debugFeatures(IN in) {
        Counter<String> features = featurize(in);
        System.out.println("Debugging features for "+in.toString());
        for (String s : features.keySet()) {
            System.out.println(s+":"+features.getCount(s));
        }
    }

    private Counter<String> featurize(IN in) {
        Counter<String> featureCounts = new ClassicCounter<>();

        for (int i = 0; i < features.length; i++) {
            Function<IN,Object> feature = features[i];

            Object obj = feature.apply(in);

            if (obj == null) continue;

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

    public void train(List<Pair<IN,OUT>> data) {
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

    public Counter<OUT> predictSoft(IN in) {
        return classifier.scoresOf(new RVFDatum<OUT, String>(featurize(in)));
    }

    public void analyze(List<Pair<IN,OUT>> train, List<Pair<IN,OUT>> test, String directory) throws IOException {
        File dir = new File(directory);
        if (dir.exists()) dir.delete();
        dir.mkdirs();

        analyze(train, directory+"/train");
        analyze(test, directory+"/test");
    }

    private void analyze(List<Pair<IN,OUT>> data, String directory) throws IOException {
        List<Triple<IN,OUT,OUT>> predictions = new ArrayList<>();
        for (Pair<IN,OUT> pair : data) {
            predictions.add(new Triple<>(pair.first, pair.second, predict(pair.first)));
        }

        File dir = new File(directory);
        if (dir.exists()) dir.delete();
        dir.mkdirs();

        writeAccuracy(predictions, directory+"/accuracy.txt");
        writeConfusionMatrix(predictions, directory + "/confusion.csv");
        writeErrors(predictions, directory + "/errors.txt");
    }

    private void writeAccuracy(List<Triple<IN,OUT,OUT>> predictions, String path) throws IOException {
        BufferedWriter bw = new BufferedWriter(new FileWriter(path));
        int correct = 0;
        for (Triple<IN,OUT,OUT> prediction : predictions) {
            if (prediction.second.equals(prediction.third)) correct++;
        }
        bw.write("Accuracy: "+((double)correct / predictions.size()));
        bw.close();
    }

    private Pair<List<OUT>, int[][]> getConfusionMatrix(List<Triple<IN,OUT,OUT>> predictions) {
        List<OUT> tagTypes = new ArrayList<>();
        for (Triple<IN,OUT,OUT> prediction : predictions) {
            if (!tagTypes.contains(prediction.second)) tagTypes.add(prediction.second);
            if (!tagTypes.contains(prediction.third)) tagTypes.add(prediction.third);
        }

        int[][] counts = new int[tagTypes.size()][tagTypes.size()];

        for (Triple<IN,OUT,OUT> prediction : predictions) {
            int target = tagTypes.indexOf(prediction.second);
            int guess = tagTypes.indexOf(prediction.third);

            counts[target][guess]++;
        }

        return new Pair<>(tagTypes, counts);
    }

    private void writeConfusionMatrix(List<Triple<IN,OUT,OUT>> predictions, String path) throws IOException {
        Pair<List<OUT>, int[][]> confusion = getConfusionMatrix(predictions);

        BufferedWriter bw = new BufferedWriter(new FileWriter(path));

        bw.write("ROW=TARGET:COL=GUESS");
        for (OUT out : confusion.first) {
            bw.write(","+out.toString());
        }
        bw.write("\n");

        int[][] counts = confusion.second;
        for (int i = 0; i < counts.length; i++) {
            bw.write(confusion.first.get(i).toString());
            for (int j = 0; j < counts[i].length; j++) {
                bw.write(","+counts[i][j]);
            }
            if (i != counts.length-1) {
                bw.write("\n");
            }
        }

        bw.close();
    }

    private List<Triple<IN,OUT,OUT>> sortByConfusion(List<Triple<IN,OUT,OUT>> predictions) {
        List<Triple<IN,OUT,OUT>> sortClone = new ArrayList<>();
        sortClone.addAll(predictions);

        final Pair<List<OUT>, int[][]> confusion = getConfusionMatrix(predictions);

        sortClone.sort(new Comparator<Triple<IN, OUT, OUT>>() {
            @Override
            public int compare(Triple<IN, OUT, OUT> o1, Triple<IN, OUT, OUT> o2) {
                int firstTarget = confusion.first.indexOf(o1.second);
                int firstGuess = confusion.first.indexOf(o1.third);
                int secondTarget = confusion.first.indexOf(o2.second);
                int secondGuess = confusion.first.indexOf(o2.third);

                return confusion.second[secondTarget][secondGuess] - confusion.second[firstTarget][firstGuess];
            }
        });

        return sortClone;
    }

    private void writeErrors(List<Triple<IN,OUT,OUT>> predictions, String path) throws IOException {
        List<Triple<IN,OUT,OUT>> sorted = sortByConfusion(predictions);

        BufferedWriter bw = new BufferedWriter(new FileWriter(path));

        for (Triple<IN,OUT,OUT> example : sorted) {
            if (!example.second.equals(example.third)) {
                bw.write("TARGET: " + example.second.toString() + "\n");
                bw.write("GUESS: " + example.third.toString() + "\n");
                if (debugErrorContext != null) {
                    bw.write("CONTEXT:\n");
                    debugErrorContext.accept(example.first, bw);
                }
                bw.write("\n");
            }
        }

        bw.close();
    }
}
