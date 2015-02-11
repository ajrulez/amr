package edu.stanford.nlp.stamr.datagen;

import com.mysql.jdbc.Buffer;
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
        // dumpMicrodata();
        // dumpPreAlignedSplit();
        dumpGiantdata();
    }

    public static void dumpPreAligned() throws IOException {
        AMR[] train = AMRSlurp.slurp("data/training-500-subset.txt", AMRSlurp.Format.LDC);
        dumpSequences(train, "data/training-500-seq.txt");
        dumpManygenDictionaries(train, "data/training-500-manygen.txt");
        dumpCONLL(train, "data/training-500-conll.txt");
    }

    public static void dumpGiantdata() throws IOException {
        AMR[] train = AMRSlurp.slurp("realdata/train-aligned.txt", AMRSlurp.Format.LDC);
        AMR[] test = AMRSlurp.slurp("realdata/dev-aligned.txt", AMRSlurp.Format.LDC);

        AMRSlurp.burp("realdata/train-subset.txt", AMRSlurp.Format.LDC, train, AMR.AlignmentPrinting.ALL, false);
        AMRSlurp.burp("realdata/test-subset.txt", AMRSlurp.Format.LDC, test, AMR.AlignmentPrinting.ALL, false);

        dumpSequences(train, "realdata/train-seq.txt");
        dumpManygenDictionaries(train, "realdata/train-manygen.txt");
        dumpCONLL(train, "realdata/train-conll.txt");

        dumpSequences(test, "realdata/test-seq.txt");
        dumpManygenDictionaries(test, "realdata/test-manygen.txt");
        dumpCONLL(test, "realdata/test-conll.txt");
    }

    public static void dumpMicrodata() throws IOException {
        AMR[] bank = AMRSlurp.slurp("data/training-500-subset.txt", AMRSlurp.Format.LDC);

        AMR[] train = new AMR[]{bank[0], bank[1], bank[2]};

        AMRSlurp.burp("data/train-"+train.length+"-subset.txt", AMRSlurp.Format.LDC, train, AMR.AlignmentPrinting.ALL, false);

        dumpSequences(train, "data/train-" + train.length + "-seq.txt");
        dumpManygenDictionaries(train, "data/train-"+train.length+"-manygen.txt");
        dumpCONLL(train, "data/train-"+train.length+"-conll.txt");
    }

    public static void dumpPreAlignedSplit() throws IOException {
        AMR[] bank = AMRSlurp.slurp("data/training-500-subset.txt", AMRSlurp.Format.LDC);
        List<AMR> trainList = new ArrayList<>();
        List<AMR> testList = new ArrayList<>();
        for (int i = 0; i < bank.length; i++) {
            if (i % 5 == 0) testList.add(bank[i]);
            else trainList.add(bank[i]);
        }
        AMR[] train = trainList.toArray(new AMR[trainList.size()]);
        AMR[] test = testList.toArray(new AMR[testList.size()]);

        AMRSlurp.burp("data/train-"+train.length+"-subset.txt", AMRSlurp.Format.LDC, train, AMR.AlignmentPrinting.ALL, false);
        AMRSlurp.burp("data/test-"+test.length+"-subset.txt", AMRSlurp.Format.LDC, test, AMR.AlignmentPrinting.ALL, false);

        dumpSequences(train, "data/train-" + train.length + "-seq.txt");
        dumpManygenDictionaries(train, "data/train-"+train.length+"-manygen.txt");
        dumpCONLL(train, "data/train-"+train.length+"-conll.txt");

        dumpSequences(test, "data/test-"+test.length+"-seq.txt");
        dumpManygenDictionaries(test, "data/test-"+test.length+"-manygen.txt");
        dumpCONLL(test, "data/test-"+test.length+"-conll.txt");
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

            if (amr.multiSentenceAnnotationWrapper != null) {
                if (node.title.equalsIgnoreCase(amr.multiSentenceAnnotationWrapper.sentences.get(0).getLemmaAtIndex(i)))
                    return "LEMMA";
            }

            for (AMR.Node otherNode : amr.nodes) {
                if (otherNode != node) {
                    if (otherNode.ref.equals(node.ref)) {
                        return "COREF";
                    }
                }
            }

            return "DICT";
        }

        return "DICT";
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
            int dictStart = -1;
            String lastType = "";

            for (int i = 0; i < amr.sourceText.length; i++) {
                String type = getType(amr, i)+":"+amr.multiSentenceAnnotationWrapper.sentences.get(0).getNERSenseAtIndex(i);

                if (!type.equals(lastType)) {
                    if (lastType.startsWith("DICT")) {
                        addToDict(dictStart, i-1, amr, dictionaries);
                    }
                    if (type.startsWith("DICT")) dictStart = i;
                }

                lastType = type;

                tokens++;
            }

            if (lastType.startsWith("DICT")) {
                addToDict(dictStart, amr.sourceText.length-1, amr, dictionaries);
            }
        }

        int numIdentical = 0;

        BufferedWriter bw = new BufferedWriter(new FileWriter(path));
        for (String s : dictionaries.keySet()) {
            Set<String> set = new HashSet<String>();

            for (int i = 0; i < dictionaries.get(s).size(); i++) {
                if (i != 0) bw.write("\n");
                // bw.write(s+"\t");
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

        nodes = amr.getLargestConnectedSet(nodes);

        AMR clone = amr.cloneConnectedSubset(nodes).first;
        int minAlignment = 1000;
        for (AMR.Node node : clone.nodes) {
            if (node.alignment < minAlignment) minAlignment = node.alignment;
        }

        for (AMR.Node node : clone.depthFirstSearch()) {
            clone.giveNodeUniqueRef(node);
            node.alignment = node.alignment - minAlignment;
        }

        clone.sourceText = new String[]{
                "A",
                "B",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "J",
                "K",
                "L",
                "M",
                "N",
                "O",
                "P",
        };

        String gen = clone.toString(AMR.AlignmentPrinting.ALL).replaceAll("\\n","").replaceAll("\\t","");
        String context = amr.formatSourceTokens();

        if (!dictionaries.containsKey(sourceTokens)) {
            dictionaries.put(sourceTokens, new ArrayList<>());
        }

        dictionaries.get(sourceTokens).add(context+"\n"+gen+"\t"+start+"\t"+end+"\n");
    }

    public static void dumpCoNLLAMR(AMR amr, BufferedWriter bw) throws IOException {
        AMR.Node[] nodes = new AMR.Node[amr.nodes.size()];

        int i = 0;
        for (AMR.Node node : amr.nodes) {
            nodes[i++] = node;
        }
        assert(i == nodes.length);

        bw.append(nodes.length+"\t");

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
                int parentId = 0;
                for (int k = 0; k < nodes.length; k++) {
                    if (parent.head == nodes[k]) parentId = k + 1;
                }
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

    // Dumps:
    //
    // node \t head \t dep_rel
    //
    // Where "head" is the index into the list of the node's parent arc. We don't do multi-headed structures.

    public static void dumpCONLL(AMR[] bank, String path) throws IOException {
        BufferedWriter bw = new BufferedWriter(new FileWriter(path));
        for (AMR amr : bank) {
            dumpCoNLLAMR(amr, bw);
        }
        bw.close();
    }
}
