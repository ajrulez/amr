package edu.stanford.nlp.cache;

import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.CoreNLPProtos;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;

import java.io.*;
import java.util.Properties;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Created by keenon on 12/28/14.
 *
 * This is the baseline cache, just read into an array in memory, and write from an array in memory. No attempt at
 * laziness or minimizing memory footprint.
 */
public class BatchCoreNLPCache extends CoreNLPCache {
    Annotation[] annotations;

    public BatchCoreNLPCache(String sourcePath, String[] sentences) {
        String cachePath = sourcePath;
        if (cachePath.endsWith(".txt"))
            cachePath = cachePath.substring(0, cachePath.length()-".txt".length());
        cachePath += ".ser.gz";

        File cacheFile = new File(cachePath);
        try {
            if (cacheFile.exists()) {
                if (useProtobuf) {
                    InputStream is = new GZIPInputStream(new FileInputStream(cacheFile));
                    int len = BinaryUtils.readInt(is);
                    annotations = new Annotation[len];
                    for (int i = 0; i < len; i++) {
                        CoreNLPProtos.Document doc = CoreNLPProtos.Document.parseFrom(is);
                        annotations[i] = protobufAnnotationSerializer.fromProto(doc);
                    }
                    is.close();
                }
                else if (useKryo) {
                    Input input = new Input(new GZIPInputStream(new FileInputStream(cacheFile)));
                    int len = input.readInt();
                    annotations = new Annotation[len];
                    for (int i = 0; i < len; i++) {
                        annotations[i] = kryo.readObject(input, Annotation.class);
                    }
                    input.close();
                }
                else {
                    ObjectInputStream ois = new ObjectInputStream(new GZIPInputStream(new FileInputStream(cacheFile)));
                    annotations = (Annotation[])ois.readObject();
                    ois.close();
                }
            }
            else {
                Properties props = new Properties();
                props.put("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref, srl, nom, prep");
                StanfordCoreNLP coreNLP = new StanfordCoreNLP(props);
                Properties propsFallback = new Properties();
                propsFallback.put("annotators", "tokenize, ssplit, pos, lemma, ner, parse, dcoref");
                StanfordCoreNLP coreNLPFallback = new StanfordCoreNLP(propsFallback);

                int numThreads = Runtime.getRuntime().availableProcessors();

                annotations = new Annotation[sentences.length];
                Thread[] threads = new Thread[numThreads];
                for (int i = 0; i < threads.length; i++) {
                    threads[i] = new Thread(new AnnotatorRunnable(coreNLP,
                                                                    coreNLPFallback,
                                                                    numThreads,
                                                                    i,
                                                                    sentences,
                                                                    annotations));
                    threads[i].start();
                }

                // ... Multithreaded batched munging goes on here ...

                for (int i = 0; i < threads.length; i++) {
                    threads[i].join();
                }

                // Single threaded writing out of annotations

                if (useProtobuf) {
                    OutputStream os = new GZIPOutputStream(new FileOutputStream(cacheFile));
                    BinaryUtils.writeInt(os, annotations.length);
                    for (int i = 0; i < annotations.length; i++) {
                        CoreNLPProtos.Document serialized = protobufAnnotationSerializer.toProto(annotations[i]);
                        // serialized.writeTo(new CodedOutputStream(os));
                    }
                    os.close();
                }
                else if (useKryo) {
                    Output output = new Output(new GZIPOutputStream(new FileOutputStream(cacheFile)));
                    for (int i = 0; i < annotations.length; i++) {
                        kryo.writeObject(output, annotations[i]);
                    }
                    output.close();
                }
                else {
                    ObjectOutputStream oos =
                            new ObjectOutputStream(new GZIPOutputStream(new FileOutputStream(cacheFile)));
                    oos.writeObject(annotations);
                    oos.close();
                }
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
    }

    private static class AnnotatorRunnable implements Runnable {
        StanfordCoreNLP coreNLP;
        StanfordCoreNLP coreNLPFallback;
        int numThreads;
        int threadIdx;
        String[] sentences;
        Annotation[] annotations;

        public AnnotatorRunnable(StanfordCoreNLP coreNLP,
                                 StanfordCoreNLP coreNLPFallback,
                                 int numThreads,
                                 int threadIdx,
                                 String[] sentences,
                                 Annotation[] annotations) {
            this.coreNLP = coreNLP;
            this.coreNLPFallback = coreNLPFallback;
            this.numThreads = numThreads;
            this.threadIdx = threadIdx;
            this.sentences = sentences;
            this.annotations = annotations;
        }

        @Override
        public void run() {
            for (int i = threadIdx; i < sentences.length; i += numThreads) {
                Annotation annotation = new Annotation(sentences[i]);
                try {
                    coreNLP.annotate(annotation);
                }
                catch (Exception e) {
                    coreNLPFallback.annotate(annotation);
                }
                annotations[i] = annotation;
            }
        }
    }

    @Override
    public Annotation getAnnotation(int index) {
        return annotations[index];
    }

    @Override
    public void close() {
        // This is a no-op, there are no threads that might need to complete
    }
}
