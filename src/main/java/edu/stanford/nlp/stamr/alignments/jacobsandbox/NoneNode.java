package edu.stanford.nlp.stamr.alignments.jacobsandbox;

import edu.stanford.nlp.stamr.AMR;

/**
 * Created by jacob on 4/3/15.
 */
public class NoneNode extends AMR.Node {
    public NoneNode(){
        super("NONE_NODE", "NONE_NODE", AMR.NodeType.VALUE, null);
    }
}
