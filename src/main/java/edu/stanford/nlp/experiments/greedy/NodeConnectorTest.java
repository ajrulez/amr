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

        AMR amr = bank[8];

        Pair<GreedyState,String[][]> train = NodeConnector.amrToContextAndArcs(amr);
        list.add(train);

        System.out.println("Trying to get:");
        System.out.println(amr.toString());

        /**
         * (f / foolish
             :mode interrogative
             :domain (i / i)
             :condition (d / do-02
                 :ARG0 i
                 :ARG1 (t / this)))
         */

        nodeConnector.train(list);

        AMR.Node[] nodes = NodeConnector.filterNodes(amr.nodes);
        List<AMR.Node> nodeList = new ArrayList<>();
        for (AMR.Node node : nodes) {
            if (node != null) nodeList.add(node);
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

        nodeConnector.testValid = arcs;

        for (int j = 0; j < arcs.length; j++) {
            for (int k = 0; k < arcs[j].length; k++) {
                if (arcs[j][k] == null) arcs[j][k] = "NONE";
                if (train.second[j][k] == null) train.second[j][k] = "NONE";
                assertEquals(train.second[j][k], arcs[j][k]);
            }
        }

        String[][] recoveredArcs = nodeConnector.connect(nodes,
                forcedArcs,
                null, //amr.multiSentenceAnnotationWrapper.sentences.get(0).annotation,
                amr.sourceText);

        System.out.println("Original arcs: "+Arrays.deepToString(arcs));
        System.out.println("Recovered arcs: "+Arrays.deepToString(recoveredArcs));

        for (int j = 0; j < recoveredArcs.length; j++) {
            for (int k = 0; k < recoveredArcs[j].length; k++) {
                if (arcs[j][k] == null) arcs[j][k] = "NONE";
                if (recoveredArcs[j][k] == null) recoveredArcs[j][k] = "NONE";
                assertEquals(arcs[j][k], recoveredArcs[j][k]);
            }
        }
    }
}