package edu.stanford.nlp.experiments;

import edu.stanford.nlp.keenonutils.JaroWinklerDistance;
import jdk.internal.org.xml.sax.XMLReader;
import org.w3c.dom.*;
import org.xml.sax.EntityResolver;
import org.xml.sax.InputSource;
import org.xml.sax.SAXException;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.stream.XMLReporter;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by keenon on 2/8/15.
 */
public class FrameManager {
    List<Frame> frames;

    public FrameManager(String path) throws IOException {
        frames = loadFrames(path);
    }

    public static List<Frame> loadFrames(String path) throws IOException {
        List<Frame> frames = new ArrayList<>();
        try {
            File dir = new File(path);
            if (!dir.isDirectory()) {
                throw new IllegalArgumentException("Must pass a folder to loadFrames()");
            }

            DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
            dbf.setValidating(false);
            dbf.setNamespaceAware(true);
            dbf.setFeature("http://xml.org/sax/features/namespaces", false);
            dbf.setFeature("http://xml.org/sax/features/validation", false);
            dbf.setFeature("http://apache.org/xml/features/nonvalidating/load-dtd-grammar", false);
            dbf.setFeature("http://apache.org/xml/features/nonvalidating/load-external-dtd", false);

            DocumentBuilder db = dbf.newDocumentBuilder();

            for (File frame : dir.listFiles()) {
                Document doc = db.parse(frame);
                NodeList nl = doc.getElementsByTagName("predicate");
                for (int i = 0; i < nl.getLength(); i++) {
                    Node n = nl.item(i);
                    String lemma = n.getAttributes().getNamedItem("lemma").getNodeValue();
                    NodeList rl = n.getChildNodes();
                    for (int j = 0; j < rl.getLength(); j++) {
                        Node c = rl.item(j);
                        if (c.getNodeName().equals("roleset")) {
                            String sense = c.getAttributes().getNamedItem("id").getNodeValue();
                            sense = sense.replaceAll("\\.","-");

                            Frame f = new Frame(lemma, sense);
                            frames.add(f);
                        }
                    }
                }
            }
        } catch (ParserConfigurationException e) {
            e.printStackTrace();
        } catch (SAXException e) {
            e.printStackTrace();
        }
        return frames;
    }

    public String getClosestFrame(String token) {
        return getClosestFrame(token, frames);
    }

    public static String getClosestFrame(String token, List<Frame> frames) {
        double maxSimilarity = 0;
        Frame closestFrame = null;
        for (Frame f : frames) {
            double dist = JaroWinklerDistance.distance(token.toLowerCase(), f.lemma.toLowerCase());
            if (dist > maxSimilarity) {
                maxSimilarity = dist;
                closestFrame = f;
            }
        }
        if (closestFrame != null) {
            return closestFrame.sense;
        }
        return token.toLowerCase()+"-01";
    }
}
