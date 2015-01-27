package edu.stanford.nlp.stamr.datagen;

import edu.stanford.nlp.stamr.AMR;
import edu.stanford.nlp.stamr.AMRSlurp;
import edu.stanford.nlp.util.IdentityHashSet;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * Created by keenon on 12/3/14.
 *
 * Prints out the sequence labellings for AMR sentences, after they've been aligned
 */
public class DumpSequence {
    public static void main(String[] args) throws IOException {
        dumpPreAligned();
    }

    public static void dumpPreAligned() throws IOException {
        AMR[] train = AMRSlurp.slurp("data/training-500-subset.txt", AMRSlurp.Format.LDC);
        dumpSequences(train, "data/training-500-seq.txt");
        dumpManygenDictionaries(train, "data/training-500-manygen.txt");
        dumpCONLL(train, "data/training-500-conll.txt");
    }

    private static String getType(AMR amr, int i) {
        Set<AMR.Node> nodes = amr.nodesWithAlignment(i);
        if (nodes.size() == 0) return "NONE";
        if (nodes.size() == 1) {
            AMR.Node node = nodes.iterator().next();

            if (node.type == AMR.NodeType.QUOTE) return "DICT";

            if (node.title.contains("-")) {
                String[] components = node.title.split("-");
                if (components.length == 2) {
                    String senseTag = components[1];
                    try {
                        int ignored = Integer.parseInt(senseTag);
                        return "VERB";
                    }
                    catch (Exception e) {
                        // do nothing
                    }
                }
            }

            if (node.title.equalsIgnoreCase(amr.sourceText[i])) return "IDENTITY";

            if (node.title.equalsIgnoreCase(amr.multiSentenceAnnotationWrapper.sentences.get(0).getLemmaAtIndex(i))) return "LEMMA";

            for (AMR.Node otherNode : amr.nodes) {
                if (otherNode != node) {
                    if (otherNode.ref.equals(node.ref)) {
                        return "COREF";
                    }
                }
            }

            return "DICT";
        }

        if (amr.nodeSetConnected(nodes)) {
            return "DICT";
        }
        else {
            return "DISJOINT";
        }
    }

    public static void dumpSequences(AMR[] bank, String path) throws IOException {
        BufferedWriter bw = new BufferedWriter(new FileWriter(path));
        for (AMR amr : bank) {
            for (int i = 0; i < amr.sourceText.length; i++) {
                String type = getType(amr, i);
                bw.append(amr.sourceText[i]).append("\t").append(type).append("\n");
            }
            bw.append("\n");
        }
        bw.close();
    }

    public static void dumpManygenDictionaries(AMR[] bank, String path) throws IOException {
        Map<String,List<String>> dictionaries = new HashMap<String, List<String>>();

        int tokens = 0;
        for (AMR amr : bank) {
            boolean lastDict = false;
            int dictStart = -1;

            for (int i = 0; i < amr.sourceText.length; i++) {
                String type = getType(amr, i);
                if (type.equals("DICT")) {
                    if (!lastDict) {
                        lastDict = true;
                        dictStart = i;
                    }
                }
                else if (lastDict) {
                    addToDict(dictStart, i-1, amr, dictionaries);
                    lastDict = false;
                }

                tokens++;
            }

            if (lastDict) {
                addToDict(dictStart, amr.sourceText.length-1, amr, dictionaries);
                lastDict = false;
            }
        }

        int numIdentical = 0;

        BufferedWriter bw = new BufferedWriter(new FileWriter(path));
        for (String s : dictionaries.keySet()) {
            Set<String> set = new HashSet<String>();

            for (int i = 0; i < dictionaries.get(s).size(); i++) {
                if (i != 0) bw.write("\n");
                bw.write(s+"\t");
                bw.write(dictionaries.get(s).get(i));

                set.add(dictionaries.get(s).get(i));
            }
            bw.write("\n");

            if (set.size() == 1) {
                numIdentical ++;
            }
        }
        bw.close();

        System.out.println("Dictionary entries with only 1 choice:");
        System.out.println(numIdentical+" / "+dictionaries.keySet().size()+" = "+
                ((double)numIdentical / dictionaries.keySet().size()));

        int multichoice = dictionaries.keySet().size() - numIdentical;
        System.out.println("Dictionaries with multiple choices:");
        System.out.println(multichoice+" / "+tokens+" = "+
                ((double)multichoice / tokens));
    }

    private static void addToDict(int start, int end, AMR amr, Map<String,List<String>> dictionaries) {
        Set<AMR.Node> nodes = new IdentityHashSet<AMR.Node>();

        String sourceTokens = "";
        for (int i = start; i <= end; i++) {
            nodes.addAll(amr.nodesWithAlignment(i));

            if (i != start) {
                sourceTokens += "_";
            }
            sourceTokens += amr.sourceText[i].toLowerCase();
        }

        if (amr.nodeSetConnected(nodes)) {
            AMR clone = amr.cloneConnectedSubset(nodes).first;
            for (AMR.Node node : clone.depthFirstSearch()) {
                clone.giveNodeUniqueRef(node);
            }
            String gen = clone.toString().replaceAll("\\n","").replaceAll("\\t","").replaceAll(" ", "");
            String context = amr.formatSourceTokens();

            if (!dictionaries.containsKey(sourceTokens)) {
                dictionaries.put(sourceTokens, new ArrayList<String>());
            }

            dictionaries.get(sourceTokens).add(gen+"\t"+start+"\t"+end+"\n"+context+"\n");
        }
    }

    // Dumps:
    //
    // node \t head \t dep_rel
    //
    // Where "head" is the index into the list of the node's parent arc. We don't do multi-headed structures.

    public static void dumpCONLL(AMR[] bank, String path) throws IOException {
        BufferedWriter bw = new BufferedWriter(new FileWriter(path));
        for (AMR amr : bank) {
            AMR.Node[] nodes = new AMR.Node[amr.nodes.size()];
            int i = 0;
            for (AMR.Node node : amr.nodes) {
                nodes[i++] = node;
            }
            assert(i == nodes.length);

            for (int j = 0; j < amr.sourceText.length; j++) {
                if (j != 0) bw.append(" ");
                bw.append(amr.sourceText[j]);
            }
            bw.append("\n");

            for (int j = 0; j < nodes.length; j++) {
                bw.append(""+(j+1)).append("\t");
                bw.append(nodes[j].toString().replaceAll(" ","")).append("\t");

                if (amr.incomingArcs.containsKey(nodes[j])) {
                    List<AMR.Arc> incoming = amr.incomingArcs.get(nodes[j]);

                    assert(incoming.size() > 0);

                    // Multiheaded (incoming.size() > 1) can just pick one and should still be a tree.

                    AMR.Arc parent = incoming.get(0);
                    int parentId = Arrays.asList(nodes).indexOf(parent.head)+1;
                    bw.append(""+parentId).append("\t").append(parent.title);
                }
                else {
                    bw.append("0\tROOT");
                }
                bw.append("\t"+nodes[j].alignment);
                bw.append("\n");
            }
            bw.append("\n");
        }
        bw.close();
    }
}
