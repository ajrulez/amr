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
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Function;

import static org.junit.Assert.*;
import static org.junit.Assume.*;

import com.pholser.junit.quickcheck.ForAll;

@RunWith(Theories.class)
public class TrainableOracleTest {

    @Theory
    public void testWildlyOverfit() throws IOException {
        AMR[] bank = AMRSlurp.slurp("data/test-100-subset.txt", AMRSlurp.Format.LDC);

        TrainableOracle oracle = new TrainableOracle(bank, new ArrayList<Function<Pair<GreedyState,Integer>,Object>>(){{
            add(pair -> {
                GreedyState state = pair.first;
                StringBuilder sb = new StringBuilder();
                sb.append(state.nodes[pair.second].toString());
                sb.append(state.nodes[pair.second].alignment);
                int cursor = state.head;
                while (cursor != 0) {
                    sb.append(state.nodes[cursor].toString());
                    cursor = state.originalParent[cursor];
                }
                sb.append("(ROOT)");
                sb.append(":");
                sb.append(pair.first.tokens[0] + ":" + pair.first.tokens[1]);
                return sb.toString();
            });
        }});

        for (int i = 0; i < bank.length; i++) {
            AMR amr = bank[i];

            List<Pair<GreedyState,String[]>> derivation = TransitionRunner.run(
                    new GreedyState(GoldOracle.prepareForParse(amr.nodes), amr.sourceText, null),
                    oracle);

            Set<Integer> visitedHead = new HashSet<>();

            for (Pair<GreedyState,String[]> pair : derivation) {
                if (pair.first.finished) continue;
                if (visitedHead.contains(pair.first.head)) {
                    throw new IllegalStateException("Visited "+pair.first.head+" twice!");
                }
                visitedHead.add(pair.first.head);
            }

            List<Pair<GreedyState,String[]>> goldDerivation = TransitionRunner.run(
                    new GreedyState(GoldOracle.prepareForParse(amr.nodes), amr.sourceText, null),
                    new GoldOracle(amr));

            assertEquals(goldDerivation.size(), derivation.size());

            for (int j = 0; j < derivation.size(); j++) {
                assertEquals(goldDerivation.get(j).first, derivation.get(j).first);
            }
        }
    }
}