package edu.stanford.nlp.experiments.greedy;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import com.github.keenon.minimalml.kernels.LinearKernel;
import com.pholser.junit.quickcheck.From;
import com.pholser.junit.quickcheck.generator.InRange;
import edu.stanford.nlp.stamr.AMR;
import edu.stanford.nlp.stamr.AMRSlurp;
import edu.stanford.nlp.util.Pair;
import org.junit.contrib.theories.Theories;
import org.junit.contrib.theories.Theory;
import org.junit.runner.RunWith;

import java.io.*;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Function;

import static org.junit.Assert.*;
import static org.junit.Assume.*;

import com.pholser.junit.quickcheck.ForAll;

@RunWith(Theories.class)
public class GoldOracleTest {

    @Theory
    public void testPrepareForParse() throws IOException {
        AMR[] bank = AMRSlurp.slurp("data/test-100-subset.txt", AMRSlurp.Format.LDC);
        for (AMR amr : bank) {
            AMR.Node[] forParse = GoldOracle.prepareForParse(amr.nodes);
            assertNull(forParse[0]);
            for (int i = 1; i < forParse.length; i++) {
                for (int j = 1; j < forParse.length; j++) {
                    if (i == j) continue;
                    assertFalse(GoldOracle.nodesConsideredEqual(forParse[i],forParse[j]));
                }
            }
        }
    }

    @Theory
    public void testReversibleTransform() throws IOException {
        AMR[] bank = AMRSlurp.slurp("data/test-100-subset.txt", AMRSlurp.Format.LDC);
        for (AMR amr : bank) {

            List<Pair<GreedyState,String[]>> derivation = TransitionRunner.run(
                    new GreedyState(GoldOracle.prepareForParse(amr.nodes), amr.sourceText, null),
                    new GoldOracle(amr));

            Set<Integer> visitedHead = new HashSet<>();

            for (Pair<GreedyState,String[]> pair : derivation) {
                if (pair.first.finished) continue;
                if (visitedHead.contains(pair.first.head)) {
                    throw new IllegalStateException("Visited "+pair.first.head+" twice!");
                }
                visitedHead.add(pair.first.head);
            }

            // assertEquals(amr.nodes.size() + 1, derivation.size());

            GreedyState finalState = derivation.get(derivation.size()-1).first;
            AMR recovered = Generator.generateAMR(finalState);

            assertEquals(amr, recovered);
        }
    }
}