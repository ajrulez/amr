package edu.stanford.nlp.stamr.utils;

import edu.stanford.nlp.util.IdentityHashSet;
import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.Triple;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

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

        // Add root arcs

        for (int i = 0; i < numNodes; i++) {
            arcLabels[0][i+1] = "ROOT";
        }

        double[] rootWeights = new double[numNodes];
        for (int i = 0; i < numNodes; i++) {
            rootWeights[i] = weights[0][i+1];
        }

        Pair<int[], Object[]> arcs = new Pair<int[], Object[]>();
        double maxScore = Double.NEGATIVE_INFINITY;

        // Ensure only 1 root per graph

        for (int r = 0; r < numNodes; r++) {
            for (int i = 0; i < numNodes; i++) {
                weights[0][i+1] = i == r ? rootWeights[i] : -10000;
            }
            DGraph dGraph = new DGraph(weights, arcLabels);
            Pair<int[], Object[]> possibleArcs = dGraph.chuLiuEdmonds();

            double score = 0.0;
            for (int i = 0; i < numNodes; i++) {
                score += weights[possibleArcs.first[i]+1][i+1];
            }
            if (score > maxScore) {
                maxScore = score;
                arcs = possibleArcs;
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
