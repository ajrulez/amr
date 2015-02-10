package edu.stanford.nlp.experiments.pipelinetests;

import com.github.keenon.minimalml.word2vec.Word2VecLoader;
import edu.stanford.nlp.experiments.AMRPipelineStateBased;
import edu.stanford.nlp.experiments.greedy.GreedyState;
import edu.stanford.nlp.stamr.AMR;
import edu.stanford.nlp.util.Pair;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * Created by keenon on 2/8/15.
 */
public class Test2 {
    static Map<String,double[]> embeddings;

    static {
        try {
            embeddings = Word2VecLoader.loadData("data/google-300-trimmed.ser.gz");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    static List<Function<Pair<GreedyState,Integer>,Object>> bfsOracleFeatures =
            new ArrayList<Function<Pair<GreedyState,Integer>,Object>>(){{

                /**
                 * CMU features
                 */

                /**
                 Here's the list of CMU Features as seen in the ACL '14 paper:

                 - Self edge: 1 if edge is between two nodes in the same fragment
                 - Tail fragment root: 1 if the edge's tail is the root of its graph fragment
                 - Head fragment root: 1 if the edge's head is the root of its graph fragment
                 - Path: Dependency edge labels and parts of speech on the shortest syntactic path between any two words of the two spans
                 - Distance: Number of tokens (plus one) between the concept's spans
                 - Distance indicators: A feature for each distance value
                 - Log distance: Log of the distance feature + 1

                 Combos:

                 - Path & Head concept
                 - Path & Tail concept
                 - Path & Head word
                 - Path & Tail word
                 - Distance & Path

                 */

                // Tail fragment root
                add(pair -> {
                    GreedyState state = pair.first;
                    for (int i = 1; i < state.nodes.length; i++) {
                        if (state.forcedArcs[i][pair.second] != null) return 0.0;
                    }
                    return 1.0;
                });
                // Head fragment root
                add(pair -> {
                    GreedyState state = pair.first;
                    for (int i = 1; i < state.nodes.length; i++) {
                        if (state.forcedArcs[i][state.head] != null) return 0.0;
                    }
                    return 1.0;
                });
                // Path
                add(pair -> {
                    GreedyState state = pair.first;
                    return AMRPipelineStateBased.getDependencyPath(state, state.head, pair.second);
                });
                // Distance
                add(pair -> {
                    GreedyState state = pair.first;
                    if (state.head == 0) return 0.0;
                    return Math.abs(state.nodes[state.head].alignment - state.nodes[pair.second].alignment)+1.0;
                });
                // Log Distance
                add(pair -> {
                    GreedyState state = pair.first;
                    if (state.head == 0) return 0.0;
                    return Math.log(Math.abs(state.nodes[state.head].alignment - state.nodes[pair.second].alignment)+1.0);
                });
                // Distance Indicator
                add(pair -> {
                    GreedyState state = pair.first;
                    if (state.head == 0) return Integer.toString(0);
                    return Integer.toString(Math.abs(state.nodes[state.head].alignment - state.nodes[pair.second].alignment) + 1);
                });
                // Path + Head concept
                add(pair -> {
                    GreedyState state = pair.first;
                    String headConcept;
                    if (state.head == 0) headConcept = "ROOT";
                    else headConcept = state.nodes[state.head].title;
                    return AMRPipelineStateBased.getDependencyPath(state, state.head, pair.second)+":"+headConcept;
                });
                // Path + Tail concept
                add(pair -> {
                    GreedyState state = pair.first;
                    return AMRPipelineStateBased.getDependencyPath(state, state.head, pair.second)+":"+state.nodes[pair.second].title;
                });
                // Path + Head word
                add(pair -> {
                    GreedyState state = pair.first;
                    String headWord;
                    if (state.head == 0) headWord = "ROOT";
                    else headWord = state.tokens[state.nodes[state.head].alignment];
                    return AMRPipelineStateBased.getDependencyPath(state, state.head, pair.second)+":"+headWord;
                });
                // Path + Tail word
                add(pair -> {
                    GreedyState state = pair.first;
                    return AMRPipelineStateBased.getDependencyPath(state, state.head, pair.second)+":"+state.tokens[state.nodes[pair.second].alignment];
                });
                // Path + Distance Indicator
                add(pair -> {
                    GreedyState state = pair.first;
                    String dist;
                    if (state.head == 0) dist = "0";
                    else dist = Integer.toString(Math.abs(state.nodes[state.head].alignment - state.nodes[pair.second].alignment) + 1);
                    return AMRPipelineStateBased.getDependencyPath(state, state.head, pair.second)+":"+dist;
                });

                /**
                 * New features, silly additions
                 */
                // name -> QUOTE indicator
                add(pair -> {
                    GreedyState state = pair.first;
                    if (state.head != 0) {
                        if (state.nodes[state.head].title.equals("name") && state.nodes[pair.second].type == AMR.NodeType.QUOTE) return 1.0;
                    }
                    return 0.0;
                });
                // name -> NON-QUOTE indicator
                add(pair -> {
                    GreedyState state = pair.first;
                    if (state.head != 0) {
                        if (state.nodes[state.head].title.equals("name") && state.nodes[pair.second].type != AMR.NodeType.QUOTE) return 1.0;
                    }
                    return 0.0;
                });
                // head embedding
                add(pair -> {
                    GreedyState state = pair.first;
                    if (state.head == 0) return new double[300];
                    return embeddings.get(state.tokens[state.nodes[state.head].alignment]);
                });
                // tail embedding
                add(pair -> {
                    GreedyState state = pair.first;
                    return embeddings.get(state.tokens[state.nodes[pair.second].alignment]);
                });

                /**
                 * New features only possible for context, because we have tons of context info available
                 */

                // Head-head embedding
                add(pair -> {
                    GreedyState state = pair.first;
                    if (state.head == 0 || state.originalParent[state.head] == 0) return new double[300];
                    return embeddings.get(state.tokens[state.nodes[state.originalParent[state.head]].alignment]);
                });
                // Depth into partial AMR tree
                add(pair -> {
                    GreedyState state = pair.first;
                    return (double)AMRPipelineStateBased.getParents(state, state.head).size()+1;
                });
                // Log depth into partial AMR tree
                add(pair -> {
                    GreedyState state = pair.first;
                    return Math.log((double) AMRPipelineStateBased.getParents(state, state.head).size() + 1);
                });
                // Depth into partial AMR tree Indicator
                add(pair -> {
                    GreedyState state = pair.first;
                    return Integer.toString(AMRPipelineStateBased.getParents(state, state.head).size()+1);
                });
                // Indicator for the node we're linking to already having a parent somewhere in the tree
                add(pair -> {
                    GreedyState state = pair.first;
                    if (state.originalParent[pair.second] == 0) {
                        return 0.0;
                    }
                    return 1.0;
                });
                // Path from the head to the child through the tree, if it exists
                add(pair -> {
                    GreedyState state = pair.first;
                    return AMRPipelineStateBased.getAMRPath(state, state.head, pair.second);
                });
                // Parents of the current head token
                add(pair -> {
                    GreedyState state = pair.first;
                    return AMRPipelineStateBased.getAMRPath(state, 0, state.head);
                });
                // Parents of the current tail token
                add(pair -> {
                    GreedyState state = pair.first;
                    return AMRPipelineStateBased.getAMRPath(state, 0, pair.second);
                });
            }};

    public static void main(String[] args) throws IOException, InterruptedException {
        AMRPipelineStateBased.testPipeline("test2", bfsOracleFeatures);
    }
}
