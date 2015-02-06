package edu.stanford.nlp.experiments.greedy;

import edu.stanford.nlp.stamr.AMR;
import edu.stanford.nlp.stamr.AMRSlurp;
import edu.stanford.nlp.util.Pair;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

public class NodeConnectorTest {

    @Test
    public void testTrain() throws Exception {
        AMR[] bank = AMRSlurp.slurp("data/test-100-subset.txt", AMRSlurp.Format.LDC);
        NodeConnector nodeConnector = new NodeConnector();

        List<Pair<GreedyState,String[][]>> list = new ArrayList<>();

        list.add(NodeConnector.amrToContextAndArcs(bank[0]));

        nodeConnector.train(list);

        AMR amr = bank[0];

        AMR.Node[] nodes = new AMR.Node[amr.nodes.size()+1];
        List<AMR.Node> nodeList = new ArrayList<>();
        int i = 1;
        for (AMR.Node node : amr.nodes) {
            nodes[i] = node;
            nodeList.add(node);
            i++;
        }

        GreedyState state = new GreedyState();
        state.nodes = nodes;

        String[][] arcs = new String[nodes.length][nodes.length];
        state.arcs = new String[nodes.length][nodes.length];
        state.originalParent = new int[nodes.length];

        for (AMR.Arc arc : amr.arcs) {
            arcs[nodeList.indexOf(arc.head)+1][nodeList.indexOf(arc.tail)+1] = arc.title;
        }
        arcs[0][nodeList.indexOf(amr.head)+1] = "ROOT";

        String[][] forcedArcs = new String[nodes.length][nodes.length];

        String[][] recoveredArcs = nodeConnector.connect(nodes,
                forcedArcs,
                amr.multiSentenceAnnotationWrapper.sentences.get(0).annotation,
                amr.sourceText);
        for (int j = 0; j < recoveredArcs.length; j++) {
            for (int k = 0; k < recoveredArcs[j].length; k++) {
                if (arcs[j][k] == null) arcs[j][k] = "NONE";
                if (recoveredArcs[j][k] == null) recoveredArcs[j][k] = "NONE";
                assertEquals(arcs[j][k], recoveredArcs[j][k]);
            }
        }
    }
}