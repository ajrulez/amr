package edu.stanford.nlp.stamr.alignments.jacobsandbox;

import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Created by jacob on 3/28/15.
 */
public class Model implements Serializable {
    private static final double serialVersionUID = 1l;

    private  Map<String, AGPair> theta = new ConcurrentHashMap<>();

    // TODO(gabor) handle me better than setting me after doEM()
    public SoftCountDict dict;
    // TODO(gabor) handle me better than setting me after doEM()
    public SoftCountDict nerDict;
    // TODO(gabor) handle me better than setting me after doEM()
    public LemmaAction lemmaDict;

    public void adagrad(Map<String, Double> gradient, double step){
        for(Map.Entry<String, Double> e : gradient.entrySet()){
            AGPair p = theta.get(e.getKey());
            if(p == null) theta.put(e.getKey(), new AGPair(e.getValue(), step));
            else p.incr(e.getValue(), step);
        }
    }

    public double score(List<String> features){
        double ret = 0.0;
        for(String key : features){
            AGPair p = theta.get(key);
            if(p != null) ret += p.v;
        }
        return ret;
    }

    private static class AGPair implements Serializable {
        private static final double serialVersionUID = 1l;
        private static final double DELTA = 1e-4;
        double v, s;
        public AGPair(double val, double step){
            s = val*val + DELTA;
            v = step * val / Math.sqrt(s);
        }
        void incr(double val, double step){
            s += val * val;
            v += step * val / Math.sqrt(s);
        }
    }

    public static class SoftCountDict implements Serializable {
        private static final double serialVersionUID = 1l;
        ConcurrentHashMap<String, ConcurrentHashMap<String, Double>> softCounts = new ConcurrentHashMap<>();
        ConcurrentHashMap<String, Double> totals = new ConcurrentHashMap<>();
        ConcurrentHashMap<String, Double> initCounts = new ConcurrentHashMap<>();
        ConcurrentHashMap<String, Double> uniqueCounts = new ConcurrentHashMap<>();
        double gamma, alpha;
        public SoftCountDict(Map<String, Integer> freqs, double gamma, double alpha){
            int total = 0;
            for(Integer i : freqs.values()) total += i;
            for(Map.Entry<String, Integer> e : freqs.entrySet()){
                initCounts.put(e.getKey(), e.getValue() / (double) total);
            }
            this.gamma = gamma;
            this.alpha = alpha;
        }
        void addCount(String x, String y, double dv){
            ConcurrentHashMap<String, Double> softCountsX = Util.getMapSafe(softCounts, x);
            double v = Util.getDoubleSafe(softCountsX, y);
            softCountsX.put(y, v + dv);
            double totalX = Util.getDoubleSafe(totals, x);
            totals.put(x, totalX + dv);
            double du = Math.min(v + dv, 1.0) - Math.min(v, 1.0);
            double u = Util.getDoubleSafe(uniqueCounts, x);
            uniqueCounts.put(x, u + du);
        }
        double getProb(String x, String y){
            ConcurrentHashMap<String, Double> softCountsX = Util.getMapSafe(softCounts, x);
            double v = Util.getDoubleSafe(softCountsX, y);
            double v0 = Util.getDoubleSafe(initCounts, y);
            double totalX = Util.getDoubleSafe(totals, x);
            double u = Util.getDoubleSafe(uniqueCounts, x);
            return (Math.max(v - 1.0, 0.0) + Math.min(v, 1.0) * (1-alpha) + (gamma + u * alpha) * v0) / (totalX + gamma);
        }
    }
}
