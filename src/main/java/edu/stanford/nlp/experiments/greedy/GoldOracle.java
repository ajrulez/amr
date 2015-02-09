package edu.stanford.nlp.experiments.greedy;

import edu.stanford.nlp.stamr.AMR;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Created by keenon on 2/7/15.
 */
public class GoldOracle extends Oracle{
    AMR amr;

    public GoldOracle(AMR amr) {
        this.amr = amr;
    }

    public static boolean nodesConsideredEqual(AMR.Node n1, AMR.Node n2) {
        if (n1 == null && n2 == null) return true;
        else if (n1 == null || n2 == null) return false;
        else if (n1.type == n2.type) {
            return (n1.alignment == n2.alignment) && n1.title.equals(n2.title);
        }
        return false;
    }

    private int indexOfNode(AMR.Node node, AMR.Node[] nodes) {
        for (int i = 0; i < nodes.length; i++) {
            if (nodesConsideredEqual(nodes[i], node)) return i;
        }
        return -1;
    }

    public static AMR.Node[] prepareForParse(Set<AMR.Node> nodes) {
        List<AMR.Node> deduped = new ArrayList<>();
        outer: for (AMR.Node node : nodes) {
            for (AMR.Node dupe : deduped) {
                if (nodesConsideredEqual(node, dupe)) continue outer;
            }
            deduped.add(node);
        }

        AMR.Node[] arr = new AMR.Node[deduped.size()+1];
        for (int i = 0; i < deduped.size(); i++) {
            arr[i+1] = deduped.get(i);
        }

        return arr;
    }

    @Override
    public String[] predictArcs(GreedyState state) {
        String[] arcs = new String[state.nodes.length];

        // Put in a blank slate
        for (int i = 0; i < arcs.length; i++) {
            arcs[i] = "NONE";
        }

        if (state.head == 0) {
            // Just note the ROOT node
            arcs[indexOfNode(amr.head, state.nodes)] = "ROOT";
        }
        else {
            // Put all the AMR arcs emanating from the AMR head node
            for (AMR.Arc arc : amr.arcs) {
                int headIndex = indexOfNode(arc.head, state.nodes);
                if (headIndex == state.head) {
                    int tailIndex = indexOfNode(arc.tail, state.nodes);
                    arcs[tailIndex] = arc.title;
                }
            }
        }

        return arcs;
    }
}
