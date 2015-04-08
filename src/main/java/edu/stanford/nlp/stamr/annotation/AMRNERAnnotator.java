package edu.stanford.nlp.stamr.annotation;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.tokensregex.TokenSequenceMatcher;
import edu.stanford.nlp.ling.tokensregex.TokenSequencePattern;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.Annotator;
import edu.stanford.nlp.pipeline.TokensRegexNERAnnotator;
import edu.stanford.nlp.util.CoreMap;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Gabor's hacks to tweak Stanford NER to conform to the AMR NER semantics.
 *
 * @author Gabor Angeli
 */
@SuppressWarnings("UnusedDeclaration")  // Loaded by reflection in the pipeline
public class AMRNERAnnotator implements Annotator {
  private final TokensRegexNERAnnotator regexner;

  @SuppressWarnings("UnusedDeclaration") // Constructor for custom annotation loading
  public AMRNERAnnotator(String prefix, Properties props) {
    regexner = new TokensRegexNERAnnotator(prefix, props);

  }

  private static String rewrite(String chunk, String ner) {
    if ("ORGANIZATION".equals(ner) && chunk.toUpperCase().equals(chunk)) {
      return ner; // acronyms tend to be organizations
    }
    switch (ner) {
      case "TRUE-ORGANIZATION":
        return "ORGANIZATION";
      case "ORGANIZATION":
        return "COMPANY";
      case "PERCENT":
        return "PERCENTAGE-ENTITY";
      case "MONEY":
        return "MONETARY-QUANTITY";
      case "SET":
        return "DATE";
      default:
        return ner;
    }
  }

  private static final Map<String, Set<TokenSequencePattern>> NER_REWRITES = new LinkedHashMap<String, Set<TokenSequencePattern>>() {{
    put("PUBLICATION", new HashSet<TokenSequencePattern>() {{
      add(TokenSequencePattern.compile("[{ner:ORGANIZATION}]* [{ner:ORGANIZATION, lemma:/[Nn]ews.*/}] [{ner:ORGANIZATION}]*"));
      add(TokenSequencePattern.compile("[{ner:ORGANIZATION}]* [{ner:ORGANIZATION, lemma:/[Bb]ook/}] [{ner:ORGANIZATION}]*"));
      add(TokenSequencePattern.compile("[{ner:ORGANIZATION}]* [{ner:ORGANIZATION, lemma:/[Pp]ublish.*/}] [{ner:ORGANIZATION}]*"));
    }});
    put("UNIVERSITY", new HashSet<TokenSequencePattern>() {{
      add(TokenSequencePattern.compile("[{ner:ORGANIZATION}]* [{ner:ORGANIZATION, lemma:/[Cc]ollege/}] [{ner:ORGANIZATION}]*"));
      add(TokenSequencePattern.compile("[{ner:ORGANIZATION}]* [{ner:ORGANIZATION, lemma:/[Uu]niversity/}] [{ner:ORGANIZATION}]*"));
      add(TokenSequencePattern.compile("[{ner:ORGANIZATION}]* [{ner:ORGANIZATION, lemma:/[Ss]chool.*/}] [{ner:ORGANIZATION}]*"));
      add(TokenSequencePattern.compile("[{ner:ORGANIZATION}]* [{ner:ORGANIZATION, lemma:/[Aa]cademy.*/}] [{ner:ORGANIZATION}]*"));
    }});
    put("GOVERNMENT-ORGANIZATION", new HashSet<TokenSequencePattern>() {{
      add(TokenSequencePattern.compile("[{ner:/ORGANIZATION|COUNTRY/}]* [{ner:ORGANIZATION, lemma:/[Dd]efen[cs]e/}] [{ner:ORGANIZATION}]*"));
      add(TokenSequencePattern.compile("[{ner:/ORGANIZATION|COUNTRY/}]* [{ner:ORGANIZATION, lemma:/[Ff]oreign/}] [{ner:ORGANIZATION}]*"));
      add(TokenSequencePattern.compile("[{ner:/ORGANIZATION|COUNTRY/}]* [{ner:ORGANIZATION, lemma:/[Dd]efense/}] [{ner:ORGANIZATION}]*"));
      add(TokenSequencePattern.compile("[{ner:/ORGANIZATION|COUNTRY/}]* [{ner:ORGANIZATION, lemma:/[Pp]arliament/}] [{ner:ORGANIZATION}]*"));
      add(TokenSequencePattern.compile("[{ner:/ORGANIZATION|COUNTRY/}]* [{ner:ORGANIZATION, lemma:/[Cc]ongress/}] [{ner:ORGANIZATION}]*"));
      add(TokenSequencePattern.compile("[{ner:/ORGANIZATION|COUNTRY/}]* [{ner:ORGANIZATION, lemma:/[Bb]ureau/}] [{ner:ORGANIZATION}]*"));
      add(TokenSequencePattern.compile("[{ner:/ORGANIZATION|COUNTRY/}]* [{ner:ORGANIZATION, lemma:/[Pp]eople/}] [{ner:ORGANIZATION, lemma:/'s/}] [{ner:ORGANIZATION}]*"));
      add(TokenSequencePattern.compile("[{ner:/ORGANIZATION|COUNTRY/}]* [{ner:ORGANIZATION, lemma:/[Oo]ffice/}] [{ner:ORGANIZATION}]*"));
      add(TokenSequencePattern.compile("[{ner:/ORGANIZATION|COUNTRY/}]* [{ner:ORGANIZATION, lemma:/[Hh]ouse/}] [{ner:ORGANIZATION}]*"));
      add(TokenSequencePattern.compile("[{ner:/ORGANIZATION|COUNTRY/}]* [{ner:ORGANIZATION, lemma:/[Ff]ederal/}] [{ner:ORGANIZATION}]*"));
      add(TokenSequencePattern.compile("[{ner:/ORGANIZATION|COUNTRY/}]* [{ner:ORGANIZATION, lemma:/[Cc]ommission/}] [{ner:ORGANIZATION}]*"));
      add(TokenSequencePattern.compile("[{ner:/ORGANIZATION|COUNTRY/}]* [{ner:ORGANIZATION, lemma:/[Nn]arcotics/}] [{ner:ORGANIZATION}]*"));
      add(TokenSequencePattern.compile("[{ner:/ORGANIZATION|COUNTRY/}]* [{ner:ORGANIZATION, lemma:/[Mm]inistry/}] [{ner:ORGANIZATION}]*"));
      add(TokenSequencePattern.compile("[{ner:/ORGANIZATION|COUNTRY/}]* [{ner:ORGANIZATION, lemma:/[Nn]ational/}] [{ner:ORGANIZATION}]*"));
      add(TokenSequencePattern.compile("[{ner:/ORGANIZATION|COUNTRY/}]* [{ner:ORGANIZATION, lemma:/[Dd]epartment/}] [{ner:ORGANIZATION}]*"));
      add(TokenSequencePattern.compile("[{ner:/ORGANIZATION|COUNTRY/}]* [{ner:ORGANIZATION, lemma:/[Mm]unicipal/}] [{ner:ORGANIZATION}]*"));
      add(TokenSequencePattern.compile("[{ner:/ORGANIZATION|COUNTRY/}]* [{ner:ORGANIZATION, lemma:/[Aa]gency/}] [{ner:ORGANIZATION}]*"));
      add(TokenSequencePattern.compile("[{ner:/ORGANIZATION|COUNTRY/}]* [{ner:ORGANIZATION, lemma:/[Cc]ouncil/}] [{ner:ORGANIZATION}]*"));
    }});
    put("POLITICAL-PARTY", new HashSet<TokenSequencePattern>() {{
        add(TokenSequencePattern.compile("[{ner:/ORGANIZATION|POLITICAL\\-PARTY|GOVERNMENT-ORGANIZATION/}]* [{ner:/ORGANIZATION|POLITICAL\\-PARTY|GOVERNMENT-ORGANIZATION/, lemma:/[Pp]arty/}]"));
    }});
    put("TRUE-ORGANIZATION", new HashSet<TokenSequencePattern>() {{
      add(TokenSequencePattern.compile("[{ner:ORGANIZATION}]* [{ner:ORGANIZATION, lemma:/[Tt]rust/}] [{ner:ORGANIZATION}]*"));
      add(TokenSequencePattern.compile("[{ner:ORGANIZATION}]* [{ner:ORGANIZATION, lemma:/[Ii]nternational/}] [{ner:ORGANIZATION}]*"));
      add(TokenSequencePattern.compile("[{ner:ORGANIZATION}]* [{ner:ORGANIZATION, lemma:/[Cc]ouncil/}] [{ner:ORGANIZATION}]*"));
      add(TokenSequencePattern.compile("[{ner:ORGANIZATION}]* [{ner:ORGANIZATION, lemma:/[Uu]nited/}] [{ner:ORGANIZATION}]*"));
      add(TokenSequencePattern.compile("[{ner:ORGANIZATION}]* [{ner:ORGANIZATION, lemma:/[Aa]ssociation/}] [{ner:ORGANIZATION}]*"));
      add(TokenSequencePattern.compile("[{ner:ORGANIZATION}]* [{ner:ORGANIZATION, lemma:/[Aa]lliance/}] [{ner:ORGANIZATION}]*"));
      add(TokenSequencePattern.compile("[{ner:ORGANIZATION}]* [{ner:ORGANIZATION, lemma:/[Oo]rganization/}] [{ner:ORGANIZATION}]*"));
      add(TokenSequencePattern.compile("[{ner:ORGANIZATION}]* [{ner:ORGANIZATION, lemma:/[Nn]orth.*/}] [{ner:ORGANIZATION}]*"));
      add(TokenSequencePattern.compile("[{ner:ORGANIZATION}]* [{ner:ORGANIZATION, lemma:/[Ss]outh.*/}] [{ner:ORGANIZATION}]*"));
      add(TokenSequencePattern.compile("[{ner:ORGANIZATION}]* [{ner:ORGANIZATION, lemma:/[Ee]ast.*/}] [{ner:ORGANIZATION}]*"));
      add(TokenSequencePattern.compile("[{ner:ORGANIZATION}]* [{ner:ORGANIZATION, lemma:/[Ww]est.*/}] [{ner:ORGANIZATION}]*"));
    }});
    put("MONETARY-QUANTITY", new HashSet<TokenSequencePattern>() {{
      add(TokenSequencePattern.compile("[{lemma:/.*[Dd]ollars?.*/}]"));
    }});
    put("PERCENT", new HashSet<TokenSequencePattern>() {{
      add(TokenSequencePattern.compile("[{ner:PERCENT}] [{lemma:/[Pp]ercent/}]"));
    }});
    put("NUMBER", new HashSet<TokenSequencePattern>() {{
      add(TokenSequencePattern.compile("[{lemma:/[Mm]any/}]"));
      add(TokenSequencePattern.compile("[{lemma:/[Ss]ome/}]"));
      add(TokenSequencePattern.compile("[{lemma:/[Ss]everal/}]"));
      add(TokenSequencePattern.compile("[{lemma:/[Nn]umerous/}]"));
      add(TokenSequencePattern.compile("[{lemma:/[Tt]ons/}]"));
      add(TokenSequencePattern.compile("[{lemma:/[Mm]ost/}]"));
    }});
    put("TIME", new HashSet<TokenSequencePattern>() {{
      add(TokenSequencePattern.compile("[{lemma:/[Aa]fter/}]"));
      add(TokenSequencePattern.compile("[{lemma:/[Uu]ntil/}]"));
      add(TokenSequencePattern.compile("[{lemma:/[Rr]ecent/}]"));
      add(TokenSequencePattern.compile("[{lemma:/[Bb]efore/}]"));
      add(TokenSequencePattern.compile("[{lemma:/[Ss]ince/}]"));
      add(TokenSequencePattern.compile("[{lemma:/[Dd]uring/}]"));
      add(TokenSequencePattern.compile("[{lemma:/[Tt]ime/}]"));
    }});
  }};

  private static final Map<String, String> KEYWORD_MAPPING = new HashMap<String, String>(){{
    put("spaceship", "SPACESHIP");
    put("peninsula", "PENINSULA");
    put("newspaper", "NEWSPAPER");
    put("project", "PROJECT");
    put("earthquake", "EARTHQUAKE");
    put("dissident", "DISSIDENT");
    put("county", "COUNTY");
//    put("war", "WAR");
//    put("tour", "TOUR");
//    put("river", "RIVER");
//    put("province", "PROVINCE");
//    put("party", "POLITICAL-PARTY");
//    put("platform", "PLATFORM");
//    put("peninsula", "PENINSULA");
//    put("newspaper", "NEWSPAPER");
//    put("network", "NETWORK");
//    put("mountain", "MOUNTAIN");
//    put("monopoly", "MONOPOLY");
//    put("module", "MODULE");
//    put("language", "LANGUAGE");
//    put("journalist", "JOURNALIST");
  }};

  private static void hackNER(List<CoreLabel> sentence) {
    // TokensRegex
    for (String newNER : NER_REWRITES.keySet()) {
      for (TokenSequencePattern pattern : NER_REWRITES.get(newNER)) {
        TokenSequenceMatcher matcher = pattern.getMatcher(sentence);
        while (matcher.find()) {
          for (CoreMap elem : matcher.groupNodes()) {
            elem.set(CoreAnnotations.NamedEntityTagAnnotation.class, newNER);
          }
        }
      }
    }

    // Rewrite Tags
    int entityStart = -1;
    for (int i = 0; i < sentence.size(); ++i) {
      if (!sentence.get(i).ner().equals("O") && entityStart < 0) {
        entityStart = i;
      } else if (sentence.get(i).ner().equals("O") && entityStart >= 0) {
        String gloss = String.join(" ", sentence.subList(entityStart, i).stream().map(CoreLabel::word).collect(Collectors.toList()));
        String ner = sentence.get(i - 1).ner();
        String newNER = rewrite(gloss, ner);
        for (int k = entityStart; k < i; ++k) {
          sentence.get(k).setNER(newNER);
        }
        entityStart = -1;
      } else if (!sentence.get(i).ner().equals("O") && i > 0 && !sentence.get(i - 1).ner().equals(sentence.get(i).ner())) {
        String gloss = String.join(" ", sentence.subList(entityStart, i).stream().map(CoreLabel::word).collect(Collectors.toList()));
        String ner = sentence.get(i - 1).ner();
        String newNER = rewrite(gloss, ner);
        for (int k = entityStart; k < i; ++k) {
          sentence.get(k).setNER(newNER);
        }
        entityStart = i;
      }
    }

    // Find one-off NER tags
    Set<String> invalidNERs = new HashSet<String>(){{
      add(null); add("O"); add("TIME"); add("DATE"); add("DURATION"); add("NUMBER"); add("PERCENTAGE-ENTITY"); add("MONETARY-QUANTITY");
      add("MASS-QUANTITY");
    }};
    int maxDistance = 1;
    for (int i = 0; i < sentence.size(); ++i) {
      String nerOrNull = KEYWORD_MAPPING.get(sentence.get(i).word().toLowerCase());
      if (nerOrNull != null) {
        // Set NER forward
        int offsetForward = 0;
        int k = i + 1;
        boolean seenNER = false;
        String ner = null;
        while (offsetForward < maxDistance && k < sentence.size()) {
          if (invalidNERs.contains(sentence.get(k).ner()) || (ner != null && !sentence.get(k).ner().equals(ner))) {
            if (seenNER) { break; }
            offsetForward += 1;
          } else {
            ner = sentence.get(k).ner();
            sentence.get(k).setNER(nerOrNull);
            seenNER = true;
          }
          k += 1;
        }
        // Set NER forward
        int offsetBack = 0;
        k = i - 1;
        seenNER = false;
        ner = null;
        while (offsetBack < maxDistance && k >= 0) {
          if (invalidNERs.contains(sentence.get(k).ner()) || (ner != null && !sentence.get(k).ner().equals(ner))) {
            if (seenNER) { break; }
            offsetBack += 1;
          } else {
            ner = sentence.get(k).ner();
            sentence.get(k).setNER(nerOrNull);
            seenNER = true;
          }
          k -= 1;
        }
      }
    }
  }

  @Override
  public void annotate(Annotation annotation) {
    regexner.annotate(annotation);
    List<CoreLabel> tokens = annotation.get(CoreAnnotations.TokensAnnotation.class);
    hackNER(tokens);
  }

  @SuppressWarnings("unchecked")
  @Override
  public Set<Requirement> requirementsSatisfied() {
    return Collections.EMPTY_SET;
  }

  @Override
  public Set<Requirement> requires() {
    return Annotator.TOKENIZE_SSPLIT_NER;
  }
}
