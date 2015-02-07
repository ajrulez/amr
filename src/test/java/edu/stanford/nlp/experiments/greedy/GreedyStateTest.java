package edu.stanford.nlp.experiments.greedy;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import com.github.keenon.minimalml.kernels.LinearKernel;
import com.pholser.junit.quickcheck.From;
import com.pholser.junit.quickcheck.generator.InRange;
import edu.stanford.nlp.stamr.AMR;
import org.junit.contrib.theories.Theories;
import org.junit.contrib.theories.Theory;
import org.junit.runner.RunWith;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.function.Function;

import static org.junit.Assert.*;
import static org.junit.Assume.*;

import com.pholser.junit.quickcheck.ForAll;

@RunWith(Theories.class)
public class GreedyStateTest {

    @Theory
    public void testCloneAndEquals(@ForAll @From(GreedyStateGen.class) GreedyState state) {
        assertTrue(state.equals(state));

        GreedyState clone = state.deepClone();
        assertTrue(clone.equals(state));
        assertTrue(state.equals(clone));
    }

    @Theory
    public void testNoneClearsQueue(@ForAll @From(GreedyStateGen.class) GreedyState state) {
        while (!state.finished) {
            String[] arcs = new String[state.nodes.length];
            for (int i = 0; i < arcs.length; i++) {
                arcs[i] = "NONE";
            }
            GreedyState nextState = state.transition(arcs);
            if (state.q.size() > 0) {
                assertTrue(nextState.q.size() == state.q.size() - 1);
            }
            else {
                assertTrue(nextState.finished);
            }
            state = nextState;
        }
        assertTrue(state.q.isEmpty());
    }

    @Theory
    public void testLoopingBaseline() {
        AMR.Node[] nodes = new AMR.Node[3];
        for (int i = 0; i < nodes.length; i++) {
            nodes[i] = new AMR.Node("r", "blah", AMR.NodeType.ENTITY);
        }
        String[] tokens = new String[0];

        GreedyState state = new GreedyState(nodes, tokens, null);

        state = state.transition(new String[]{null, "ROOT", "NONE"});
        state = state.transition(new String[]{null, "NONE", "ARG1"});
        state = state.transition(new String[]{null, "ARG2", "NONE"});

        assertTrue(state.finished);
    }

    @Theory
    public void testRightBranchingBaseline(@ForAll @InRange(minInt = 2, maxInt = 20) int size) {
        AMR.Node[] nodes = new AMR.Node[size];
        for (int i = 0; i < nodes.length; i++) {
            nodes[i] = new AMR.Node("r", "blah", AMR.NodeType.ENTITY);
        }
        String[] tokens = new String[0];

        GreedyState state = new GreedyState(nodes, tokens, null);

        for (int i = 0; i < nodes.length; i++) {
            assertEquals(i, state.head);
            String[] arcs = new String[nodes.length];
            for (int j = 0; j < arcs.length; j++) {
                arcs[j] = "NONE";
            }
            if (i < nodes.length - 1) {
                arcs[i + 1] = "RBB";
            }
            state = state.transition(arcs);
            if (i < nodes.length - 1) {
                assertEquals(0, state.q.size());
                assertFalse(state.finished);
            }
            else {
                assertEquals(0, state.q.size());
                assertTrue(state.finished);
            }
        }
        assertTrue(state.q.size() == 0);
        assertTrue(state.finished);

        for (int i = 0; i < nodes.length; i++) {
            for (int j = 1; j < nodes.length; j++) {
                if (i == j-1) {
                    assertEquals("RBB", state.arcs[i][j]);
                    assertEquals(i, state.originalParent[j]);
                }
                else {
                    assertEquals("NONE", state.arcs[i][j]);
                }
            }
        }
    }
}