package edu.stanford.nlp.stamr.alignments.conditional.unary;

import edu.stanford.nlp.stamr.AMR;
import edu.stanford.nlp.stamr.alignments.conditional.types.UnaryAlignmentFeature;
import edu.stanford.nlp.stamr.ontonotes.SRL;
import edu.stanford.nlp.stamr.utils.ConditionalDistribution;

/**
 * Handles scoring for basic lexical affinities
 */
public class LexicalUnaryFeature extends UnaryAlignmentFeature {
    ConditionalDistribution<String,String> lexicalDistribution = new ConditionalDistribution<String, String>();

    @Override
    public void observe(SRL srl) {
        // lexicalDistribution.observe(srl.sense,srl.sourceToken,1.0);
    }

    @Override
    public void observe(AMR amr, AMR.Node node, int token, double probability) {
        lexicalDistribution.observe(node.title,amr.multiSentenceAnnotationWrapper.sentences.get(0).getPOSTagAtIndex(token)+amr.getSourceToken(token),probability);
    }

    @Override
    public double score(AMR amr, AMR.Node node, int token) {
        return lexicalDistribution.probAGivenB(node.title,amr.multiSentenceAnnotationWrapper.sentences.get(0).getPOSTagAtIndex(token)+amr.getSourceToken(token));
    }

    @Override
    public void addAll(UnaryAlignmentFeature uf) {
        lexicalDistribution.addAll(((LexicalUnaryFeature)uf).lexicalDistribution);
    }

    @Override
    public void clear() {
        lexicalDistribution.clear();
    }

    @Override
    public void cook() {

    }
}
