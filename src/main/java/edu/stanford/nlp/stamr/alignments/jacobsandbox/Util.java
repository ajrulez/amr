package edu.stanford.nlp.stamr.alignments.jacobsandbox;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Created by jacob on 3/28/15.
 */
public class Util {

    static void incr(Map<String, Double> map, List<String> keys, double x){
        for(String key : keys) {
            Double v = map.get(key);
            if (v == null) map.put(key, x);
            else map.put(key, v + x);
        }
    }

    static void incr(Map<String, Double> map, Map<String, Double> rhs, double x){
        for(Map.Entry<String, Double> e : rhs.entrySet()){
            String key = e.getKey();
            double dv = x * e.getValue();
            Double v = map.get(key);
            if(v == null) map.put(key, dv);
            else map.put(key, v + dv);
        }
    }

    static double lse(double x, double y){
        if(x == Double.NEGATIVE_INFINITY) return y;
        else if(y == Double.NEGATIVE_INFINITY) return x;
        else if(x < y) return y + Math.log(1 + Math.exp(x-y));
        else return x + Math.log(1 + Math.exp(y-x));
    }

    static double getDoubleSafe(ConcurrentHashMap<String, Double> map, String key){
        Double v = map.get(key);
        if(v == null) return 0.0;
        else return v;
    }
    static ConcurrentHashMap<String, Double> getMapSafe(ConcurrentHashMap<String, ConcurrentHashMap<String, Double>> map, String key){
        ConcurrentHashMap<String, Double> map2 = map.get(key);
        if(map2 == null){
            map2 = new ConcurrentHashMap<String, Double>();
            map.put(key, map2);
        }
        return map2;
    }
}
