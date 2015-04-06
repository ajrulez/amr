package edu.stanford.nlp.stamr.alignments.jacobsandbox;

import edu.stanford.nlp.ie.NumberNormalizer;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.stamr.AMR;
import edu.stanford.nlp.stamr.AMRConstants;
import edu.stanford.nlp.util.Pair;

import java.util.Collections;
import java.util.List;

/**
 * Created by jacob on 4/5/15.
 */
public class RuleBased {
    // Approach:
    // 1. Replace AMR nodes with AMR "fragments" (which are chunks we can align to)
    // 2. Replace tokens with spans (based on NER spans, basically)
    // 3. Spans can have "forced" rules

    /*private AMR constructNERCluster(Annotation annotation, List<Integer> nerList, String tag, boolean debug) {
        tag = tag.toLowerCase();
        if (tag.equals("date")) { // || tag.equals("time") || tag.equals("duration")) {
            return constructDateCluster(annotation, nerList, tag, debug);
        }
        else if (tag.equals("ordinal")) {
            return constructOrdinalCluster(annotation, nerList, debug);
        }
        else if (tag.equals("number")) {
            return constructNumberCluster(annotation, nerList, debug);
        }
        else if (tag.equals("person")) {
            return constructEntityCluster(annotation, nerList, tag, debug);
        }
        return null;
    }*/

    static AMR constructDateCluster(Annotation annotation, List<Integer> dateList) {
        boolean debug = false;
        String tag = "date";
        AMR dateChunk = new AMR();
        if (tag.equals("date")) {
            tag = "date-entity";
        }
        AMR.Node root = dateChunk.addNode("" + tag.charAt(0), tag, dateList.get(0));

        String time = annotation.get(CoreAnnotations.TokensAnnotation.class).get(dateList.get(0)).get(CoreAnnotations.NormalizedNamedEntityTagAnnotation.class);
        if (time == null) return dateChunk;
        if (debug) System.out.println("DATE: "+dateList.get(0)+" -> "+time);

        int year = -1;
        boolean isWeek = false;
        int week = -1;
        int month = -1;
        int day = -1;

        String[] parts = time.split("-");
        try {
            year = Integer.parseInt(parts[0]);
        }
        catch (Exception ignored) {}
        if (parts.length > 1) {
            if (parts[1].startsWith("W")) {
                isWeek = true;
                try {
                    week = Integer.parseInt(parts[1].substring(1));
                } catch (Exception ignored) {
                }
            } else {
                try {
                    month = Integer.parseInt(parts[1]);
                } catch (Exception ignored) {
                }
            }
        }
        if (parts.length > 2) {
            try {
                day = Integer.parseInt(parts[2]);
            } catch (Exception ignored) {
            }
        }

        if (year != -1) {
            AMR.Node yearN = dateChunk.addNode(""+year, AMR.NodeType.VALUE);
            dateChunk.addArc(root, yearN, "year");
        }
        if (month != -1) {
            AMR.Node monthN = dateChunk.addNode(""+month, AMR.NodeType.VALUE);
            dateChunk.addArc(root, monthN, "month");
        }
        if (week != -1) {
            AMR.Node weekN = dateChunk.addNode(""+week, AMR.NodeType.VALUE);
            dateChunk.addArc(root, weekN, "week");
        }
        if (day != -1) {
            if (isWeek) {
                String weekday = AMRConstants.weekdays.get(day).toLowerCase();
                AMR.Node dayN = dateChunk.addNode(""+weekday.charAt(0), weekday);
                dateChunk.addArc(root, dayN, "weekday");
            }
            else {
                AMR.Node dayN = dateChunk.addNode("" + day, AMR.NodeType.VALUE);
                dateChunk.addArc(root, dayN, "day");
            }
        }

        // Just append everything if we have nothing else
        if (year == -1 && month == -1 && week == -1 && day == -1) {
            for (int i : dateList) {
                String token = annotation.get(CoreAnnotations.TokensAnnotation.class).get(i).lemma();
                AMR.Node word = dateChunk.addNode(""+token.charAt(0), token);
                dateChunk.addArc(root, word, "time");
            }
        }

        return dateChunk;
    }

    private static String getSequentialTokens(Annotation annotation, List<Integer> list) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < list.size(); i++) {
            if (i != 0) sb.append(" ");
            sb.append(annotation.get(CoreAnnotations.TokensAnnotation.class).get(list.get(i)));
        }
        return sb.toString();
    }

    private AMR constructOrdinalCluster(Annotation annotation, List<Integer> ordinalList, boolean debug) {
        AMR chunk = new AMR();
        AMR.Node head = chunk.addNode("o", "ordinal-entity");
        Number n = null;
        try {
            n = NumberNormalizer.wordToNumber(getSequentialTokens(annotation, ordinalList));
        }
        catch (Exception ignored) {}
        if (n == null) n = 1;
        AMR.Node tail = chunk.addNode(n.toString(), AMR.NodeType.VALUE, Collections.min(ordinalList));
        chunk.addArc(head, tail, "value");
        return chunk;
    }

    static AMR constructNumberCluster(Annotation annotation, List<Integer> numberList) {
        boolean debug = false;
        AMR chunk = new AMR();
        Number n = null;
        try {
            NumberNormalizer.wordToNumber(getSequentialTokens(annotation, numberList));
        }
        catch (Exception ignored) {}
        if (n == null) n = 1;
        chunk.addNode(n.toString(), AMR.NodeType.VALUE);
        return chunk;
    }

    private AMR constructEntityCluster(Annotation annotation, List<Integer> nerList, String tag, boolean debug) {
        AMR nerChunk = new AMR();
        AMR.Node root = nerChunk.addNode("" + tag.charAt(0), tag, nerList.get(0));
        AMR.Node name = nerChunk.addNode("n", "name", nerList.get(0));
        nerChunk.addArc(root, name, "name");

        String quoteConjunction = "";
        for (int i : nerList) {
            if (quoteConjunction.length() > 0) quoteConjunction += " ";
            quoteConjunction += annotation.get(CoreAnnotations.TokensAnnotation.class).get(i).word().toLowerCase();
        }

        if (debug) System.out.println("Checking quote conjunction: \""+quoteConjunction+"\"");

        if (AMRConstants.commonNamedEntityConfusions.containsKey(quoteConjunction)) {
            Pair<String,String> pair = AMRConstants.commonNamedEntityConfusions.get(quoteConjunction);
            // Set the type
            root.title = pair.first;
            // Set the parts as children
            String[] parts = pair.second.split(" ");
            for (int i = 0; i < parts.length; i++) {
                AMR.Node opTag = nerChunk.addNode(parts[i], AMR.NodeType.QUOTE, root.alignment);
                nerChunk.addArc(name, opTag, "op" + (i + 1));
            }
        }
        else {
            for (int i = 0; i < nerList.size(); i++) {
                AMR.Node opTag = nerChunk.addNode(annotation.get(CoreAnnotations.TokensAnnotation.class).get(nerList.get(i)).word(), AMR.NodeType.QUOTE, nerList.get(i));
                nerChunk.addArc(name, opTag, "op" + (i + 1));
            }
        }
        if (debug) {
            System.out.println("Built NER cluster: ");
            System.out.println(nerChunk.toString());
        }

        int min = 10000;
        for (int i : nerList) {
            min = Math.min(min, i);
        }

        for (AMR.Node node : nerChunk.nodes) {
            if (node.alignment == 0) {
                node.alignment = min;
            }
        }

        return nerChunk;
    }

}
