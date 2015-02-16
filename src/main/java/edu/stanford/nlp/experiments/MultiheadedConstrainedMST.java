package edu.stanford.nlp.experiments;

import edu.stanford.nlp.util.Pair;
import edu.stanford.nlp.util.Triple;
import gurobi.*;

import java.util.*;

/**
 * Created by keenon on 2/15/15.
 *
 * Implements a couple of constraints over a multi-headed MST as an ILP, and then prays for a solution.
 */
public class MultiheadedConstrainedMST {
    public static Map<Integer,Set<Pair<String,Integer>>> solve(Map<Integer,Set<Triple<Integer,String,Double>>> arcs,
                                                               int maxNumberHeads) {
        int numNodes = arcs.keySet().size();
        List<String> arcTypes = new ArrayList<>();
        for (Set<Triple<Integer,String,Double>> arcSet : arcs.values()) {
            for (Triple<Integer,String,Double> arc : arcSet) {
                if (!arcTypes.contains(arc.second)) arcTypes.add(arc.second);
            }
        }
        int numArcs = arcTypes.size();

        // Set up the final graph to return

        Map<Integer,Set<Pair<String,Integer>>> graph = new HashMap<>();

        try {
            GRBEnv env = new GRBEnv();
            env.set(GRB.IntParam.OutputFlag, 0);

            GRBModel model = new GRBModel(env);

            GRBVar[][][] arc = new GRBVar[numNodes][numNodes][numArcs];
            for (int i = 0; i < numNodes; i++) {
                for (int j = 1; j < numNodes; j++) {
                    for (int k = 0; k < numArcs; k++) {
                        arc[i][j][k] = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "arc" + i + "," + j + "," + k);
                    }
                }
            }
            model.update();

            // Make sure all arc-sets can only have one type

            for (int i = 0; i < numNodes; i++) {
                for (int j = 1; j < numNodes; j++) {
                    GRBLinExpr expr = new GRBLinExpr();
                    for (int k = 0; k < numArcs; k++) {
                        expr.addTerm(1.0, arc[i][j][k]);
                    }
                    model.addConstr(expr, GRB.LESS_EQUAL, 1.0, "type" + i + "," + j);
                }
            }

            // Make sure we disallow ROOT anywhere but at the ROOT

            if (arcTypes.contains("ROOT")) {
                int root = arcTypes.indexOf("ROOT");
                for (int i = 1; i < numNodes; i++) {
                    for (int j = 1; j < numNodes; j++) {
                        GRBLinExpr expr = new GRBLinExpr();
                        expr.addTerm(1.0, arc[i][j][root]);
                        model.addConstr(expr, GRB.EQUAL, 0.0, "nonROOT" + i + "," + j);
                    }
                }
            }

            // Make sure all nodes can't have the same arc-type more than once outgoing

            for (int i = 0; i < numNodes; i++) {
                for (int k = 0; k < numArcs; k++) {
                    GRBLinExpr expr = new GRBLinExpr();
                    for (int j = 1; j < numNodes; j++) {
                        expr.addTerm(1.0, arc[i][j][k]);
                    }
                    model.addConstr(expr, GRB.LESS_EQUAL, 1.0, "arcTypeOutgoing" + i + "," + k);
                }
            }
            // Limit the number of heads a given node can have

            for (int j = 1; j < numNodes; j++) {
                GRBLinExpr expr = new GRBLinExpr();
                for (int i = 0; i < numNodes; i++) {
                    for (int k = 0; k < numArcs; k++) {
                        expr.addTerm(1.0, arc[i][j][k]);
                    }
                }
                model.addConstr(expr, GRB.LESS_EQUAL, maxNumberHeads, "maxHeads" + j);
                model.addConstr(expr, GRB.GREATER_EQUAL, 1.0, "minHeads" + j);
            }

            // Limit the root to a single outgoing arc

            GRBLinExpr rootChildren = new GRBLinExpr();
            for (int j = 1; j < numNodes; j++) {
                for (int k = 0; k < numArcs; k++) {
                    rootChildren.addTerm(1.0, arc[0][j][k]);
                }
            }
            model.addConstr(rootChildren, GRB.EQUAL, 1.0, "rootSingleChild");

            // Add the goal

            GRBLinExpr goal = new GRBLinExpr();
            for (int i = 0; i < numNodes; i++) {
                Set<Triple<Integer,String,Double>> outArcs = arcs.get(i);
                double[][] probs = new double[numNodes][numArcs];

                // We want to minimize the objective, so we multiply all log probs by *-1

                for (int j = 1; j < numNodes; j++) {
                    for (int k = 0; k < numArcs; k++) {
                        probs[j][k] = 10000;
                    }
                }
                for (Triple<Integer,String,Double> outArc : outArcs) {
                    probs[outArc.first][arcTypes.indexOf(outArc.second)] = -outArc.third;
                }

                for (int j = 1; j < numNodes; j++) {
                    for (int k = 0; k < numArcs; k++) {
                        goal.addTerm(probs[j][k], arc[i][j][k]);
                    }
                }
            }
            model.setObjective(goal);

            model.optimize();

            // Get the values of the model

            for (int i = 0; i < numNodes; i++) {
                graph.put(i, new HashSet<>());
                for (int j = 1; j < numNodes; j++) {
                    int arcType = -1;
                    for (int k = 0; k < numArcs; k++) {
                        if (arc[i][j][k].get(GRB.DoubleAttr.X) == 1) {
                            arcType = k;
                            break;
                        }
                    }
                    if (arcType != -1) {
                        graph.get(i).add(new Pair<>(arcTypes.get(arcType), j));
                    }
                }
            }

            // Dispose of model and environment
            model.dispose();
            env.dispose();

            while (true) {
                Set<Integer> visited = new HashSet<>();
                Queue<Integer> q = new ArrayDeque<>();
                q.add(0);
                while (!q.isEmpty()) {
                    int head = q.poll();
                    visited.add(head);
                    if (!graph.containsKey(head)) graph.put(head, new HashSet<>());
                    for (Pair<String, Integer> a : graph.get(head)) {
                        if (!visited.contains(a.second) && !q.contains(a.second))
                            q.add(a.second);
                    }
                }

                // Do a simple linking of everything that wasn't included in the graph yet

                double bestArcScore = Double.NEGATIVE_INFINITY;
                Triple<Integer,String,Integer> bestArc = null;

                for (int j : visited) {
                    if (j == 0) continue;
                    for (Triple<Integer,String,Double> childArc : arcs.get(j)) {
                        if (!visited.contains(childArc.first)) {
                            if (childArc.third > bestArcScore) {
                                boolean alreadyHaveArcWithName = false;
                                for (Triple<Integer,String,Double> otherChildArc : arcs.get(j)) {
                                    if (otherChildArc == childArc) continue;
                                    if (otherChildArc.second.equals(childArc.second)) {
                                        alreadyHaveArcWithName = true;
                                        break;
                                    }
                                }
                                if (!alreadyHaveArcWithName) {
                                    bestArcScore = childArc.third;
                                    bestArc = new Triple<>(j, childArc.second, childArc.first);
                                }
                            }
                        }
                    }
                }

                if (bestArc == null) {
                    /*
                    for (int i = 0; i < numNodes; i++) {
                        if (!visited.contains(i)) throw new IllegalStateException("Have no arc to node "+i+" still");
                    }
                    */
                    break;
                }
                graph.get(bestArc.first).add(new Pair<>(bestArc.second, bestArc.third));
            }
        } catch (GRBException e) {
            System.out.println("Error code: " + e.getErrorCode() + ". " +
                    e.getMessage());
        }

        return graph;
    }
}
