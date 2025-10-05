/*
 * reserved comment block
 * DO NOT REMOVE OR ALTER!
 */
package com.sun.org.apache.xerces.internal.util;

import org.checkerframework.dataflow.qual.Pure;
import com.sun.org.apache.xerces.internal.xni.grammars.Grammar;
import com.sun.org.apache.xerces.internal.xni.grammars.XMLGrammarDescription;
import com.sun.org.apache.xerces.internal.xni.grammars.XMLGrammarPool;

public class XMLGrammarPoolImpl implements XMLGrammarPool {

    protected static final int TABLE_SIZE;

    protected Entry[] fGrammars;

    protected boolean fPoolIsLocked;

    protected int fGrammarCount;

    public XMLGrammarPoolImpl() {
    }

    public XMLGrammarPoolImpl(int initialCapacity) {
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

    protected static final class Entry {

        public int hash;

        public XMLGrammarDescription desc;

        public Grammar grammar;

        public Entry next;

        protected Entry(int hash, XMLGrammarDescription desc, Grammar grammar, Entry next) {
        }

        protected void clear();
    }
}
