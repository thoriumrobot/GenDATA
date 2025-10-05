/*
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
 */
package com.sun.org.apache.xerces.internal.jaxp.validation;

import org.checkerframework.dataflow.qual.Pure;
import java.lang.ref.Reference;
import java.lang.ref.ReferenceQueue;
import java.lang.ref.SoftReference;
import com.sun.org.apache.xerces.internal.xni.grammars.Grammar;
import com.sun.org.apache.xerces.internal.xni.grammars.XMLGrammarDescription;
import com.sun.org.apache.xerces.internal.xni.grammars.XMLSchemaDescription;
import com.sun.org.apache.xerces.internal.xni.grammars.XMLGrammarPool;

final class SoftReferenceGrammarPool implements XMLGrammarPool {

    protected static final int TABLE_SIZE;

    protected static final Grammar[] ZERO_LENGTH_GRAMMAR_ARRAY;

    protected Entry[] fGrammars;

    protected boolean fPoolIsLocked;

    protected int fGrammarCount;

    protected final ReferenceQueue<Grammar> fReferenceQueue;

    public SoftReferenceGrammarPool() {
    }

    public SoftReferenceGrammarPool(int initialCapacity) {
    }

    public Grammar[] retrieveInitialGrammarSet(String grammarType);

    public void cacheGrammars(String grammarType, Grammar[] grammars);

    public Grammar retrieveGrammar(XMLGrammarDescription desc);

    public void putGrammar(Grammar grammar);

    public Grammar getGrammar(XMLGrammarDescription desc);

    public Grammar removeGrammar(XMLGrammarDescription desc);

    @Pure
    public boolean containsGrammar(XMLGrammarDescription desc);

    public void lockPool();

    public void unlockPool();

    public void clear();

    public boolean equals(XMLGrammarDescription desc1, XMLGrammarDescription desc2);

    public int hashCode(XMLGrammarDescription desc);

    static final class Entry {

        public int hash;

        public int bucket;

        public Entry prev;

        public Entry next;

        public XMLGrammarDescription desc;

        public SoftGrammarReference grammar;

        protected Entry(int hash, int bucket, XMLGrammarDescription desc, Grammar grammar, Entry next, ReferenceQueue<Grammar> queue) {
        }

        protected void clear();
    }

    static final class SoftGrammarReference extends SoftReference<Grammar> {

        public Entry entry;

        protected SoftGrammarReference(Entry entry, Grammar grammar, ReferenceQueue<Grammar> queue) {
        }
    }
}
