package edu.stanford.nlp.experiments.greedy;

import edu.stanford.nlp.experiments.ConstrainedSequence;
import edu.stanford.nlp.experiments.LinearPipe;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.stamr.AMR;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.Pair;

import java.util.*;
import java.util.function.Function;

/**
 * Created by keenon on 2/4/15.
 *
 * Just does the node connection portion of the AMR parsing.
 */
public class NodeConnector {

    List<String> classes = new ArrayList<>();
    Map<String, Integer> classSizeOverride = new HashMap<String,Integer>(){{
        put("NONE",-1);
    }};

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

    private int getRootNode(GreedyState state) {
        SemanticGraph graph = state.annotation.get(CoreAnnotations.SentencesAnnotation.class).get(0)
                .get(SemanticGraphCoreAnnotations.BasicDependenciesAnnotation.class);

        int rootToken = graph.getFirstRoot().index();

        System.out.println("Root token: "+rootToken);

        for (int i = 0; i < state.nodes.length; i++) {
            if (state.nodes[i] != null && state.nodes[i].alignment == rootToken) return i;
        }
        return -1;
    }

    private String getPath(GreedyState state, int head, int tail) {
        if (head == 0) {
            return "ROOT:NOPATH";
        }
        if (head == tail) {
            return "IDENTITY";
        }
        if (head == -1) {
            return "OOB:NOPATH";
        }
        int headToken = state.nodes[head].alignment;
        int tailToken = state.nodes[tail].alignment;
        if (state.annotation == null) {
            throw new IllegalStateException("Can't have a null annotation for our greedy state!");
        }
        SemanticGraph graph = state.annotation.get(CoreAnnotations.SentencesAnnotation.class).get(0)
                .get(SemanticGraphCoreAnnotations.BasicDependenciesAnnotation.class);
        IndexedWord headIndexedWord = graph.getNodeByIndexSafe(headToken);
        IndexedWord tailIndexedWord = graph.getNodeByIndexSafe(tailToken);
        if (headIndexedWord == null || tailIndexedWord == null) {
            return "NOTOKENS:NOPATH";
        }
        List<SemanticGraphEdge> edges = graph.getShortestUndirectedPathEdges(headIndexedWord, tailIndexedWord);

        StringBuilder sb = new StringBuilder();
        IndexedWord currentWord = headIndexedWord;
        for (SemanticGraphEdge edge : edges) {
            if (edge.getDependent().equals(currentWord)) {
                sb.append(">");
                currentWord = edge.getGovernor();
            }
            else {
                if (!edge.getGovernor().equals(currentWord)) {
                    throw new IllegalStateException("Edges not in order");
                }
                sb.append("<");
                currentWord = edge.getDependent();
            }
            sb.append(edge.getRelation().getShortName());
            if (currentWord != headIndexedWord) {
                sb.append(":");
                sb.append(currentWord.get(CoreAnnotations.PartOfSpeechAnnotation.class));
            }
        }

        return sb.toString();
    }

    LinearPipe<Pair<GreedyState,Integer>,String> arcTypePrediction = new LinearPipe<>(
            new ArrayList<Function<Pair<GreedyState,Integer>,Object>>(){{

                // Builds a completely lexicalized (on nodes) path to the root of the AMR tree so far

                add((pair) -> {
                    GreedyState state = pair.first;
                    StringBuilder sb = new StringBuilder();
                    sb.append(state.nodes[pair.second].toString());
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

                /*
                // Builds a simple dependency path between the two nodes

                add((pair) -> getPath(pair.first, pair.first.head, pair.second));

                // Path to root for the target node

                add((pair) -> getPath(pair.first, getRootNode(pair.first), pair.second));

                // Path to root and lexicalize

                add((pair) -> getPath(pair.first, getRootNode(pair.first), pair.second) + pair.first.nodes[pair.second].toString());

                // Gets the fraction of the way you are into the sentence, into a couple of buckets

                add((pair) -> {
                    double frac = (double)pair.second / pair.first.nodes.length;

                    int buckets = 8;

                    return "BUCKET "+Math.round(frac * buckets);
                });
                */
            }}, null);

    private static int identityIndexOf(Object[] arr, Object o) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == o) return i;
        }
        return -1;
    }

    public static Pair<GreedyState,String[][]> amrToContextAndArcs(AMR amr) {
        AMR.Node[] nodes = new AMR.Node[amr.nodes.size()+1];
        int i = 1;
        for (AMR.Node node : amr.nodes) {
            nodes[i] = node;
            i++;
        }

        GreedyState state = new GreedyState();
        state.nodes = nodes;
        state.tokens = amr.sourceText;

        String[][] arcs = new String[nodes.length][nodes.length];
        state.arcs = new String[nodes.length][nodes.length];
        state.originalParent = new int[nodes.length];

        for (AMR.Arc arc : amr.arcs) {
            arcs[identityIndexOf(nodes, arc.head)][identityIndexOf(nodes, arc.tail)] = arc.title;
        }
        arcs[0][identityIndexOf(nodes, amr.head)] = "ROOT";

        return new Pair<>(state, arcs);
    }

    public void train(List<Pair<GreedyState,String[][]>> trainingData) {
        List<Pair<Pair<GreedyState,Integer>, String>> trainingExamples = new ArrayList<>();

        for (Pair<GreedyState,String[][]> pair : trainingData) {
            AMR.Node[] nodes = pair.first.nodes;
            String[][] arcs = pair.second;

            Queue<Integer> q = new ArrayDeque<>();
            Set<Integer> visited = new HashSet<>();
            q.add(0);

            GreedyState state = pair.first.deepClone();
            state.arcs = new String[arcs.length][arcs[0].length];
            state.originalParent = new int[nodes.length];

            GreedyState nextState = state.deepClone();

            while (!q.isEmpty()) {
                int head = q.poll();
                visited.add(head);
                state.head = head;
                System.out.println("New state: " + head);
                System.out.println("Original parents: "+Arrays.toString(state.originalParent));
                for (int i = 1; i < nodes.length; i++) {
                    if (i == head) continue;
                    if (arcs[head][i] != null && !visited.contains(i)) q.add(i);
                    if (arcs[head][i] != null && nextState.originalParent[i] == 0) {
                        nextState.originalParent[i] = head;
                    }

                    String arcName = arcs[head][i] == null ? "NONE" : arcs[head][i];
                    if (!classes.contains(arcName)) classes.add(arcName);
                    // Add this arc to the context for future states
                    nextState.arcs[head][i] = arcName;
                    // Add this actual classification to the existing context
                    Pair<GreedyState,Integer> featurizable = new Pair<>(state, i);
                    trainingExamples.add(new Pair<>(featurizable, arcName));
                    if (!arcName.equals("NONE")) {
                        System.out.println("Adding training example: "+arcName);
                        arcTypePrediction.debugFeatures(featurizable);
                    }
                }
                state = nextState;
                nextState = state.deepClone();
            }
        }

        arcTypePrediction.train(trainingExamples);
    }

    public String[][] connect(AMR.Node[] nodes, String[][] forcedArcs, Annotation annotation, String[] tokens) {
        Queue<Integer> q = new ArrayDeque<>();
        Set<Integer> visited = new HashSet<>();
        q.add(0);

        GreedyState state = new GreedyState();
        state.tokens = tokens;
        state.nodes = nodes;
        state.arcs = new String[nodes.length][nodes.length];
        state.originalParent = new int[nodes.length];
        state.annotation = annotation;

        while (!q.isEmpty()) {
            int head = q.poll();
            visited.add(head);
            state.head = head;

            double[][] probs = new double[nodes.length][classes.size()];
            int[] maxCounts = new int[classes.size()];

            // Class count constraints at ROOT are special
            if (head == 0) {
                for (int i = 0; i < classes.size(); i++) {
                    if (classes.get(i).equals("NONE")) maxCounts[i] = nodes.length-1;
                    if (classes.get(i).equals("ROOT")) maxCounts[i] = 1;
                }
            }
            else {
                for (int i = 0; i < classes.size(); i++) {
                    if (classSizeOverride.containsKey(classes.get(i))) {
                        int count = classSizeOverride.get(classes.get(i));
                        if (count == -1) {
                            maxCounts[i] = nodes.length;
                        } else {
                            maxCounts[i] = count;
                        }
                    } else {
                        maxCounts[i] = 1;
                    }
                }
            }

            for (int i = 1; i < nodes.length; i++) {
                if (i == head) continue;

                Pair<GreedyState, Integer> featurizable = new Pair<>(state, i);
                System.out.println("Testing features: ");
                arcTypePrediction.debugFeatures(featurizable);

                // This comes out as unnormalized log-probability
                Counter<String> counter = arcTypePrediction.predictSoft(featurizable);

                double sum = 0.0;
                for (int j = 0; j < classes.size(); j++) {
                    probs[i][j] = Math.exp(counter.getCount(classes.get(j)));
                    sum += probs[i][j];
                }

                // We want to maximize the product of probabilities, so that's the sum of the logs

                for (int j = 0; j < classes.size(); j++) {
                    probs[i][j] = Math.log(probs[i][j] / sum);
                }
            }

            int[] forcedClasses = new int[nodes.length];
            forcedClasses[0] = classes.indexOf("NONE");
            for (int i = 1; i < nodes.length; i++) {
                if (forcedArcs[head][i] != null) {
                    forcedClasses[i] = classes.indexOf(forcedArcs[head][i]);
                }
                else {
                    forcedClasses[i] = -1;
                }
            }

            int[] solvedClasses = ConstrainedSequence.solve(probs, maxCounts, forcedClasses);
            for (int i = 1; i < nodes.length; i++) {
                state.arcs[state.head][i] = classes.get(solvedClasses[i]);
                if (!state.arcs[state.head][i].equals("NONE") && !visited.contains(i)) {
                    state.originalParent[i] = state.head;
                    q.add(i);
                }
            }
        }

        return state.arcs;
    }

}
