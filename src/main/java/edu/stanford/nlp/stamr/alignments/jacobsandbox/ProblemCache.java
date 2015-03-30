package edu.stanford.nlp.stamr.alignments.jacobsandbox;

import edu.stanford.nlp.experiments.FrameManager;

import java.util.HashMap;
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
    public JointEM.MatchNode getClosestFrame(FrameManager frameManager, String stem) {
        if (getClosestFrameCacheCond == null) {
            getClosestFrameCacheCond = frameManager;
        }
        if (getClosestFrameCacheCond != frameManager) {
            throw new IllegalArgumentException("FrameManager changed between cache calls!");
        }
        String rtn = getClosestFrameCache.get(stem);
        if (rtn == null) {
            rtn = frameManager.getClosestFrame(stem);
            getClosestFrameCache.put(stem, rtn);
        }
        return new JointEM.MatchNode(rtn);
    }


}
