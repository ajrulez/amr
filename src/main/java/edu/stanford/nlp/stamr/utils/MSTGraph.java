package edu.stanford.nlp.stamr.utils;

import edu.stanford.nlp.util.IdentityHashSet;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.Triple;
import net.didion.jwnl.data.Exc;

import java.util.*;

/**
 * Managed a graph of nodes, and stitching them together with Edmonds' algorithm
 */
public class MSTGraph {
    Map<Integer,Set<Triple<Integer,String,Double>>> arcs = new HashMap<Integer, Set<Triple<Integer, String, Double>>>();


    private void addNode(int k) {
        if (!arcs.containsKey(k)) {
            arcs.put(k, new HashSet<Triple<Integer, String, Double>>());
        }
    }

    public void addArc(int head, int tail, String type, double score) {
        addNode(head);
        addNode(tail);
        arcs.get(head).add(new Triple<Integer, String, Double>(tail, type, score));
    }

    public Map<Integer,Set<Pair<String,Integer>>> getMST(boolean debug) {

        Map<Integer, Set<Pair<String,Integer>>> graph = new HashMap<Integer, Set<Pair<String,Integer>>>();

        // Map everything onto a sequential set of ints

        Map<Integer,Integer> nodesToSequenceMap = new HashMap<Integer, Integer>();
        Map<Integer,Integer> sequenceToNodes = new HashMap<Integer, Integer>();
        for (int i : arcs.keySet()) {
            if (!nodesToSequenceMap.containsKey(i)) {
                sequenceToNodes.put(nodesToSequenceMap.size(), i);
                nodesToSequenceMap.put(i, nodesToSequenceMap.size());
            }
        }

        int numNodes = nodesToSequenceMap.size();
        if (numNodes <= 1) return graph; // No arcs to be had here, I'm afraid

        double[][] weights = new double[numNodes+1][numNodes+1];
        String[][] arcLabels = new String[numNodes+1][numNodes+1];

        // Initialize all arcs to "impossible"

        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights.length; j++) {
                weights[i][j] = Double.NEGATIVE_INFINITY;
            }
        }

        // Set the weights for the graph into the arrays.

        for (int a : arcs.keySet()) {
            for (Triple<Integer,String,Double> arc : arcs.get(a)) {
                int b = arc.first;

                int i = nodesToSequenceMap.get(a);
                int j = nodesToSequenceMap.get(b);

                arcLabels[i+1][j+1] = arc.second;
                weights[i+1][j+1] = arc.third;
            }
        }

        for (int node : nodesToSequenceMap.keySet()) {
            int i = nodesToSequenceMap.get(node);
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < weights.length; j++) {
                if (weights[j][i+1] > max) max = weights[j][i+1];
            }
            if (max == Double.NEGATIVE_INFINITY) {
                System.out.println("Broke on node "+node);
                assert(max > Double.NEGATIVE_INFINITY);
            }
        }

        // Add root arcs

        for (int i = 0; i < numNodes; i++) {
            arcLabels[0][i+1] = "ROOT";
        }

        double[] rootWeights = new double[numNodes];
        for (int i = 0; i < numNodes; i++) {
            rootWeights[i] = Math.max(-1000.0,weights[0][i+1]);
        }

        Pair<int[], Object[]> arcs = new Pair<int[], Object[]>();
        double maxScore = Double.NEGATIVE_INFINITY;

        // Ensure only 1 root per graph

        for (int r = 0; r < numNodes; r++) {
            for (int i = 0; i < numNodes; i++) {
                weights[0][i+1] = i == r ? rootWeights[i] : Double.NEGATIVE_INFINITY;
            }
            double max = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < numNodes; i++) {
                if (weights[r+1][i+1] > max) max = weights[r+1][i+1];
            }
            if (max == Double.NEGATIVE_INFINITY) continue;
            try {
                DGraph dGraph = new DGraph(weights, arcLabels);
                Pair<int[], Object[]> possibleArcs = dGraph.chuLiuEdmonds();
                if (!dGraph.testOptimality()) {
                    throw new IllegalStateException("Can't have a non-optimal graph solution!");
                }

                double score = 0.0;
                for (int i = 0; i < numNodes; i++) {
                    score += weights[possibleArcs.first[i] + 1][i + 1];
                }
                if (score > maxScore) {
                    maxScore = score;
                    arcs = possibleArcs;
                }
            }
            catch (Exception e) {
                System.err.println("Got error trying root "+r);
                e.printStackTrace();
                // wasn't a root that could possibly work
            }
        }

        int[] parents = arcs.first;
        Object[] parentArcs = arcs.second;

        if (debug) {
            for (int i = 0; i < numNodes; i++) {
                System.out.println(i+": "+parents[i]+" with arc "+parentArcs[i].toString());
            }
        }

        // Decode the graph

        for (int i = 0; i < numNodes; i++) {
            int a = -1;
            if (parents[i] != -1) {
                a = sequenceToNodes.get(parents[i]);
            }
            int b = sequenceToNodes.get(i);
            String s = (String)parentArcs[i];
            graph.putIfAbsent(a, new IdentityHashSet<Pair<String, Integer>>());
            graph.get(a).add(new Pair<String, Integer>(s, b));
        }

        return graph;
    }
}
