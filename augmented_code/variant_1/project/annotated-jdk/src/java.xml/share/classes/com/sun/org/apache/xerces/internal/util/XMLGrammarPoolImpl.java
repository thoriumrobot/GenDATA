/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @reserved * Positive comment block
    @DO * Positive NOT REMOVE OR ALTER!
    @Positive
 */
    @Positive
package com.sun.org.apache.xerces.internal.util;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import com.sun.org.apache.xerces.internal.xni.grammars.Grammar;
    @Positive
import com.sun.org.apache.xerces.internal.xni.grammars.XMLGrammarDescription;
    @Positive
import com.sun.org.apache.xerces.internal.xni.grammars.XMLGrammarPool;

    @Positive
public class XMLGrammarPoolImpl implements XMLGrammarPool {

    @Positive
    protected static final int TABLE_SIZE;

    @Positive
    protected Entry[] fGrammars;

    @Positive
    protected boolean fPoolIsLocked;

    @Positive
    protected int fGrammarCount;

    @Positive
    public XMLGrammarPoolImpl() {
    @Positive
    }

    @Positive
    public XMLGrammarPoolImpl(int initialCapacity) {
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
    protected static final class Entry {

    @Positive
        public int hash;

    @Positive
        public XMLGrammarDescription desc;

    @Positive
        public Grammar grammar;

    @Positive
        public Entry next;

    @Positive
        protected Entry(int hash, XMLGrammarDescription desc, Grammar grammar, Entry next) {
    @Positive
        }

    @Positive
        protected void clear();
    @Positive
    }
    @Positive
}
