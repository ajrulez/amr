package edu.stanford.nlp.stamr.alignments.jacobsandbox;

import edu.stanford.nlp.keenonutils.JaroWinklerDistance;
import edu.stanford.nlp.stamr.AMR;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

import static edu.stanford.nlp.util.logging.Redwood.Util.endTrack;
import static edu.stanford.nlp.util.logging.Redwood.Util.forceTrack;

/**
 * TODO(gabor) JavaDoc
 *
 * @author Gabor Angeli
 */
public class LemmaAction implements Serializable {
    private static final double serialVersionUID = 1l;

    private final Map<String, Counter<String>> lemmas;

    private LemmaAction(Map<String, Counter<String>> lemmas) {
        this.lemmas = lemmas;
    }

    public Counter<String> lemmasFor(String token) {
        return lemmas.getOrDefault(token.toLowerCase(), new ClassicCounter<String>(){{setCount(token, 1.0);}});
    }

    public static LemmaAction initialize(AugmentedToken[][] tokens, AMR.Node[][] nodes, int candidatesPerWord) {
        forceTrack("Creating lemma dictionary");
        if (tokens.length != nodes.length) {
            throw new IllegalArgumentException("Dataset is malformed");
        }

        // Variables
        Counter<String> globalCounts = new ClassicCounter<>();
        Map<String, Counter<String>> lemmas = new HashMap<>();

        // Collect counts
        for (int dataI = 0; dataI < tokens.length; ++dataI) {
            for (int tokenI = 0; tokenI < tokens[dataI].length; ++tokenI) {
                // Get the closest matching word in the AMR graph
                String token = tokens[dataI][tokenI].value.toLowerCase();
                String lemma = tokens[dataI][tokenI].stem.toLowerCase();
                String bestLemma = null;
                double bestJaroWinkler = Double.NEGATIVE_INFINITY;
                for (int nodeI = 0; nodeI < nodes[dataI].length; ++nodeI) {
                    String node = nodes[dataI][nodeI].title.replaceAll("-[0-9]*", "");
                    double jaroWinkler = JaroWinklerDistance.distance(lemma, node);
                    if (node.contains(lemma)) {
                        jaroWinkler = 1.0;
                    }
                    if (jaroWinkler > bestJaroWinkler) {
                        bestJaroWinkler = jaroWinkler;
                        bestLemma = node;
                    }
                }
                // Register the mapping
                globalCounts.incrementCount(bestLemma, bestJaroWinkler);
                Counter<String> counts = lemmas.get(token);
                if (counts == null) {
                    counts = new ClassicCounter<>();
                    lemmas.put(token, counts);
                }
                counts.incrementCount(bestLemma, bestJaroWinkler);
            }
        }

        // Get lemma candidates
        for (Map.Entry<String, Counter<String>> entry : lemmas.entrySet()) {
            Counter<String> counts = entry.getValue();
            for (String lemma : counts.keySet()) {
                assert globalCounts.getCount(lemma) > 0.0;
                counts.setCount(lemma, counts.getCount(lemma) / globalCounts.getCount(lemma));
            }
            Counters.retainTop(counts, candidatesPerWord);
            Counters.normalize(counts);
        }

        // Return
        endTrack("Creating lemma dictionary");
        return new LemmaAction(lemmas);

    }
}
