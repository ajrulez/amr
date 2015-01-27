package edu.stanford.nlp.stamr.datagen;

import edu.stanford.nlp.stamr.AMR;
import edu.stanford.nlp.stamr.AMRSlurp;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Set;

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
        dumpManygen(train, "data/training-500-manygen.txt");
    }

    private static String getType(AMR amr, int i) {
        Set<AMR.Node> nodes = amr.nodesWithAlignment(i);
        if (nodes.size() == 0) return "NONE";
        if (nodes.size() == 1) {
            AMR.Node node = nodes.iterator().next();

            if (node.type == AMR.NodeType.QUOTE) return "QUOTE";

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

        for (AMR.Node node : nodes) {
            if (node.type == AMR.NodeType.QUOTE) return "QUOTE";
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

    public static void dumpManygen(AMR[] bank, String path) throws IOException {
        BufferedWriter bw = new BufferedWriter(new FileWriter(path));
        for (AMR amr : bank) {
            for (int i = 0; i < amr.sourceText.length; i++) {
                String type = getType(amr, i);
                bw.append(amr.sourceText[i]).append("\t");
                if (type.equals("NONE")) {
                    bw.append("NONE").append("\n");
                }
                else {
                    Set<AMR.Node> nodes = amr.nodesWithAlignment(i);
                    if (amr.nodeSetConnected(nodes)) {
                        AMR miniclone = amr.cloneConnectedSubset(nodes).first;
                        String s = miniclone.toString();
                        s = s.replaceAll("\\n","").replaceAll("\\t","").replaceAll(" ","");
                        bw.append(s).append("\n");
                    }
                    else {
                        AMR.Node head = amr.getSetHead(nodes);
                        bw.append(head.toString().replaceAll(" ","")).append("\n");
                    }
                }
            }
            bw.append("\n");
        }
        bw.close();
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
                nodes[i] = node;
            }
            bw.append("\n");
        }
        bw.close();
    }
}
