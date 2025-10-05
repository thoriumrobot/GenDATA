/*
    @Positive
 * reserved comment block
    @Positive
 * DO NOT REMOVE OR ALTER!
    @Positive
 */
    @Positive
package com.sun.org.apache.xerces.internal.parsers;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import com.sun.org.apache.xerces.internal.xni.grammars.Grammar;
    @Positive
import com.sun.org.apache.xerces.internal.xni.grammars.XMLGrammarPool;
    @Positive
import com.sun.org.apache.xerces.internal.xni.grammars.XMLGrammarDescription;
    @Positive
import com.sun.org.apache.xerces.internal.util.XMLGrammarPoolImpl;
    @Positive
import com.sun.org.apache.xerces.internal.util.ShadowedSymbolTable;
    @Positive
import com.sun.org.apache.xerces.internal.util.SymbolTable;
    @Positive
import com.sun.org.apache.xerces.internal.util.SynchronizedSymbolTable;

    @Positive
public class CachingParserPool {

    @Positive
    public static final boolean DEFAULT_SHADOW_SYMBOL_TABLE;

    @Positive
    public static final boolean DEFAULT_SHADOW_GRAMMAR_POOL;

    @Positive
    protected SymbolTable fSynchronizedSymbolTable;

    @Positive
    protected XMLGrammarPool fSynchronizedGrammarPool;

    @Positive
    protected boolean fShadowSymbolTable;

    @Positive
    protected boolean fShadowGrammarPool;

    @Positive
    public CachingParserPool() {
    @Positive
    }

    @Positive
    public CachingParserPool(SymbolTable symbolTable, XMLGrammarPool grammarPool) {
    @Positive
    }

    @Positive
    public SymbolTable getSymbolTable();

    @Positive
    public XMLGrammarPool getXMLGrammarPool();

    @Positive
    public void setShadowSymbolTable(boolean shadow);

    @Positive
    public DOMParser createDOMParser();

    @Positive
    public SAXParser createSAXParser();

    @Positive
    public static final class SynchronizedGrammarPool implements XMLGrammarPool {

    @Positive
        public SynchronizedGrammarPool(XMLGrammarPool grammarPool) {
    @Positive
        }

    @Positive
        public Grammar[] retrieveInitialGrammarSet(String grammarType);

    @Positive
        public Grammar retrieveGrammar(XMLGrammarDescription gDesc);

    @Positive
        public void cacheGrammars(String grammarType, Grammar[] grammars);

    @Positive
        public void lockPool();

    @Positive
        public void clear();

    @Positive
        public void unlockPool();
    @Positive
    }

    @Positive
    public static final class ShadowedGrammarPool extends XMLGrammarPoolImpl {

    @Positive
        public ShadowedGrammarPool(XMLGrammarPool grammarPool) {
    @Positive
        }

    @Positive
        public Grammar[] retrieveInitialGrammarSet(String grammarType);

    @Positive
        public Grammar retrieveGrammar(XMLGrammarDescription gDesc);

    @Positive
        public void cacheGrammars(String grammarType, Grammar[] grammars);

    @Positive
        public Grammar getGrammar(XMLGrammarDescription desc);

    @Positive
        @Pure
    @Positive
        public boolean containsGrammar(XMLGrammarDescription desc);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
