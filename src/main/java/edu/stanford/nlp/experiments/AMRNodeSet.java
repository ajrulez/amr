package edu.stanford.nlp.experiments;

import edu.stanford.nlp.stamr.AMR;
import edu.stanford.nlp.util.Triple;

import java.util.HashSet;
import java.util.Set;

/**
 * Created by keenon on 1/27/15.
 *
 * Holds a bunch of AMR nodes, their source sentence and mappings, and any arcs that are forced by look-ups.
 */
public class AMRNodeSet {
    public AMR.Node[] nodes;
    public Set<Triple<Integer,Integer,String>> forcedArcs = new HashSet<>();
    public LabeledSequence sourceSequence;
}
