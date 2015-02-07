package edu.stanford.nlp.experiments.greedy;

import edu.stanford.nlp.stamr.AMR;

import java.util.*;

/**
 * Created by keenon on 2/7/15.
 */
public class Generator {

    private static AMR.Node lazyGenNode(AMR amr, Map<Integer,AMR.Node> newNodes, GreedyState state, int i) {
        if (!newNodes.containsKey(i)) {
            AMR.Node n = state.nodes[i];
            if (n.type == AMR.NodeType.ENTITY) {
                newNodes.put(i, amr.addNode(n.ref, n.title, n.alignment));
            }
            else {
                newNodes.put(i, amr.addNode(n.title, n.type, n.alignment));
            }
        }
        return newNodes.get(i);
    }

    public static AMR generateAMR(GreedyState state) {
        Map<Integer,AMR.Node> newNodes = new HashMap<>();
        AMR amr = new AMR();
        amr.sourceText = state.tokens;

        Queue<Integer> q = new ArrayDeque<>();
        Set<Integer> visited = new HashSet<>();

        q.add(0);
        while (!q.isEmpty()) {
            int head = q.poll();
            visited.add(head);
            for (int i = 0; i < state.nodes.length; i++) {
                if (head == 0) {
                    if (state.arcs[head][i] != null && !state.arcs[head][i].equals("NONE")) {
                        // Generate node, becomes root
                        lazyGenNode(amr, newNodes, state, i);
                        q.add(i);
                    }
                }
                else {
                    if (state.arcs[head][i] != null && !state.arcs[head][i].equals("NONE")) {
                        AMR.Node newHead = lazyGenNode(amr, newNodes, state, head);
                        AMR.Node newTail = lazyGenNode(amr, newNodes, state, i);
                        amr.addArc(newHead, newTail, state.arcs[head][i]);
                        if (!visited.contains(i)) {
                            q.add(i);
                        }
                    }
                }
            }
        }

        return amr;
    }
}
