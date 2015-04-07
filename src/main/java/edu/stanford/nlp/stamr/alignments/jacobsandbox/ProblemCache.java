package edu.stanford.nlp.stamr.alignments.jacobsandbox;

import edu.stanford.nlp.experiments.FrameManager;
import edu.stanford.nlp.process.WordShapeClassifier;
import edu.stanford.nlp.stats.Counters;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

/**
 * A class to encapsulate cached computation. This should be discarded when the EM aligner is
 * done, but speeds things up significantly while it's running.
 *
 * Feel free to instantiate me with new ProblemCache() and just pass me along.
 *
 * @author Gabor Angeli
 */
public class ProblemCache {

    public ProblemCache() {

    }

    private final Map<String, String> getClosestFrameCache = new HashMap<>();
    private FrameManager getClosestFrameCacheCond = null;

    /**
     * A cached version of frameManager.getClosestFrame(stem);
     */
    public MatchNode getClosestFrame(FrameManager frameManager, String value, LemmaAction lemmaDict) {
        if (getClosestFrameCacheCond == null) {
            getClosestFrameCacheCond = frameManager;
        }
        if (getClosestFrameCacheCond != frameManager) {
            throw new IllegalArgumentException("FrameManager changed between cache calls!");
        }
        String rtn = getClosestFrameCache.get(value);
        if (rtn == null) {
            for (String stem : Counters.toSortedList(lemmaDict.lemmasFor(value.toLowerCase()))) {
                if (frameManager.containsFrameWithLemma(stem)) {
                    rtn = frameManager.getClosestFrame(stem);
                    break;
                }
            }
            if (rtn == null) {
                rtn = frameManager.getClosestFrame(value);
            }
            getClosestFrameCache.put(value, rtn);
        }
        return new MatchNode.VerbMatchNode(rtn);
    }


    private final Map<String, String> getWordShapeCache = new HashMap<>();

    /**
     * A cache for the word shape classifier.
     */
    public String getWordShape(String input) {
        String shape;
        if ( (shape = getWordShapeCache.get(input)) != null) {
            return shape;
        }
        shape = WordShapeClassifier.wordShape(input, WordShapeClassifier.WORDSHAPECHRIS4);
        getWordShapeCache.put(input, shape);
        return shape;
    }

}
