/*
    @Positive
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
    @Positive
 */
    @Positive
package com.sun.org.apache.xerces.internal.jaxp.validation;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.lang.ref.Reference;
    @Positive
import java.lang.ref.ReferenceQueue;
    @Positive
import java.lang.ref.SoftReference;
    @Positive
import com.sun.org.apache.xerces.internal.xni.grammars.Grammar;
    @Positive
import com.sun.org.apache.xerces.internal.xni.grammars.XMLGrammarDescription;
    @Positive
import com.sun.org.apache.xerces.internal.xni.grammars.XMLSchemaDescription;
    @Positive
import com.sun.org.apache.xerces.internal.xni.grammars.XMLGrammarPool;

    @Positive
final class SoftReferenceGrammarPool implements XMLGrammarPool {

    @Positive
    protected static final int TABLE_SIZE;

    @Positive
    protected static final Grammar[] ZERO_LENGTH_GRAMMAR_ARRAY;

    @Positive
    protected Entry[] fGrammars;

    @Positive
    protected boolean fPoolIsLocked;

    @Positive
    protected int fGrammarCount;

    @Positive
    protected final ReferenceQueue<Grammar> fReferenceQueue;

    @Positive
    public SoftReferenceGrammarPool() {
    @Positive
    }

    @Positive
    public SoftReferenceGrammarPool(int initialCapacity) {
    @Positive
    }

    @Positive
    public Grammar[] retrieveInitialGrammarSet(String grammarType);

    @Positive
    public void cacheGrammars(String grammarType, Grammar[] grammars);

    @Positive
    public Grammar retrieveGrammar(XMLGrammarDescription desc);

    @Positive
    public void putGrammar(Grammar grammar);

    @Positive
    public Grammar getGrammar(XMLGrammarDescription desc);

    @Positive
    public Grammar removeGrammar(XMLGrammarDescription desc);

    @Positive
    @Pure
    @Positive
    public boolean containsGrammar(XMLGrammarDescription desc);

    @Positive
    public void lockPool();

    @Positive
    public void unlockPool();

    @Positive
    public void clear();

    @Positive
    public boolean equals(XMLGrammarDescription desc1, XMLGrammarDescription desc2);

    @Positive
    public int hashCode(XMLGrammarDescription desc);

    @Positive
    static final class Entry {

    @Positive
        public int hash;

    @Positive
        public int bucket;

    @Positive
        public Entry prev;

    @Positive
        public Entry next;

    @Positive
        public XMLGrammarDescription desc;

    @Positive
        public SoftGrammarReference grammar;

    @Positive
        protected Entry(int hash, int bucket, XMLGrammarDescription desc, Grammar grammar, Entry next, ReferenceQueue<Grammar> queue) {
    @Positive
        }

    @Positive
        protected void clear();
    @Positive
    }

    @Positive
    static final class SoftGrammarReference extends SoftReference<Grammar> {

    @Positive
        public Entry entry;

    @Positive
        protected SoftGrammarReference(Entry entry, Grammar grammar, ReferenceQueue<Grammar> queue) {
    @Positive
        }
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
