package edu.stanford.nlp.stamr.alignments.jacobsandbox;

import edu.stanford.nlp.keenonutils.JaroWinklerDistance;
import edu.stanford.nlp.stamr.AMR;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.stats.Counters;
import edu.stanford.nlp.util.StringUtils;

import java.io.PrintWriter;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Optional;

import static edu.stanford.nlp.util.logging.Redwood.Util.endTrack;
import static edu.stanford.nlp.util.logging.Redwood.Util.forceTrack;

/**
 * <p>
 * A class learning how AMR lemmatizes strings. At a high level, this class runs a greedy
 * edit-distance-based aligner to find the most likely alignment for each word in each sentence, and
 * counts these statistics. The affinity for a word to its candidate lemma is then the PMI^2 of the
 * joint counts, considering the marginal counts of the word and the AMR node it wants to align to.
 * The whole thing is smoothed with one count for every word to its Stanford lemma.
 * </p>
 *
 * Notable points:
 * <ul>
 *     <li>Not every word has a lemma in here.</li>
 *     <li>Lemma scores are initially normalized to be between 0 and 1, but are not a distribution
 *         -- rather, I just divide by the max.</li>
 *     <li>The identity mapping is not in here -- if a word and its lemma are the same, then use the
 *         Identity action instead.</li>
 * </ul>
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

    public Optional<String> lemmatize(String token) {
        return lemmas.containsKey(token) ? Optional.of(Counters.argmax(lemmas.get(token))) : Optional.empty();
    }

    public void print(PrintWriter writer) {
        for (Map.Entry<String, Counter<String>> entry : lemmas.entrySet()) {
            writer.println(entry.getKey() + "\t" + Counters.toSortedString(entry.getValue(), 10, "%1$s -> %2$f", "\t"));
        }
    }

    public static LemmaAction initialize(AugmentedToken[][] tokens, AMR.Node[][] nodes, int candidatesPerWord) {
        forceTrack("Creating lemma dictionary");
        if (tokens.length != nodes.length) {
            throw new IllegalArgumentException("Dataset is malformed");
        }

        // Variables
        Counter<String> lemmaCounts = new ClassicCounter<>();
        Counter<String> wordCounts = new ClassicCounter<>();
        Map<String, Counter<String>> lemmas = new HashMap<>();

        Map<String, String> naiveLemmaMap = new HashMap<>();

        // Collect counts
        for (int dataI = 0; dataI < tokens.length; ++dataI) {
            for (int tokenI = 0; tokenI < tokens[dataI].length; ++tokenI) {
                // Get the token
                String token = tokens[dataI][tokenI].value.toLowerCase();
                String lemma = tokens[dataI][tokenI].stem.toLowerCase();
                String lemmaEssence = lemma.replace("ll", "l").replace("tt", "t").replace("pp", "p").replace("ss", "s");
                if (token.matches("[0-9\\-\\.]+")) { continue; }
                // Register the lemma
                if (!token.equals(lemma)) {
                    naiveLemmaMap.put(token, lemma);
                }
                // Get the closest matching word in the AMR graph
                String bestLemma = null;
                double bestScore = Double.NEGATIVE_INFINITY;
                for (int nodeI = 0; nodeI < nodes[dataI].length; ++nodeI) {
                    String node = nodes[dataI][nodeI].title.replaceAll("-[0-9]+", "").replace("\"", "").toLowerCase();
                    double score;
                    if (node.equals(lemma)) {
                        score = 1.0 + 3e-10;
                    } else if (node.contains(lemma)) {
                        score = 1.0 + 2e-10;
                    } else if (node.contains(lemmaEssence)) {
                        score = 1.0 + 1e-10;
                    } else {
                        double jw = JaroWinklerDistance.distance(lemma, node);
                        double ed = 1.0 - ((double) StringUtils.editDistance(lemma, node)) / Math.max(lemma.length(), node.length());
                        score = (jw + ed) / 2.0;
                    }
                    if (score > bestScore) {
                        bestScore = score;
                        bestLemma = node;
                    }
                }
                // Register the mapping
                if (bestScore > 0.1 && !bestLemma.equals(token)) {
                    lemmaCounts.incrementCount(bestLemma, bestScore);
                    wordCounts.incrementCount(token, 1.0);
                    Counter<String> counts = lemmas.get(token);
                    if (counts == null) {
                        counts = new ClassicCounter<>();
                        lemmas.put(token, counts);
                    }
                    counts.incrementCount(bestLemma, bestScore);
                }
            }
        }

        // Register Stanford lemma counts
        for (Map.Entry<String, String> entry : naiveLemmaMap.entrySet()) {
            wordCounts.incrementCount(entry.getKey());
            lemmaCounts.incrementCount(entry.getValue());
            if (!lemmas.containsKey(entry.getKey())) {
                lemmas.put(entry.getKey(), new ClassicCounter<>());
            }
            lemmas.get(entry.getKey()).incrementCount(entry.getValue());

        }


        // Get lemma candidates
        for (Map.Entry<String, Counter<String>> entry : lemmas.entrySet()) {
            Counter<String> counts = entry.getValue();
            // (calculate PMI^2)
            for (String lemma : counts.keySet()) {
                double jointCount = counts.getCount(lemma);
                assert jointCount > 0.0;
                double wordCount = wordCounts.getCount(entry.getKey());
                assert wordCount > 0.0;
                double lemmaCount = lemmaCounts.getCount(lemma);
                assert lemmaCount > 0.0;
                counts.setCount(lemma, jointCount * jointCount / (wordCount * lemmaCount)); //counts.getCount(lemma) / lemmaCounts.getCount(lemma));
            }
            // (threshold and normalize to between 0 and 1)
            Counters.retainAbove(counts, 0.1);
            Counters.retainTop(counts, candidatesPerWord);
            if (counts.size() > 0) {
                assert counts.totalCount() > 0.0;
                Counters.divideInPlace(counts, Counters.max(counts));
            }
        }

        // Remove empty lemma dictionary entries
        Iterator<Map.Entry<String, Counter<String>>> iter = lemmas.entrySet().iterator();
        while (iter.hasNext()) {
            if (iter.next().getValue().size() == 0) {
                iter.remove();
            }
        }

        // Return
        endTrack("Creating lemma dictionary");
        return new LemmaAction(lemmas);

    }
}
