/*
 * reserved comment block
 * DO NOT REMOVE OR ALTER!
 */
package com.sun.org.apache.xerces.internal.parsers;

import org.checkerframework.dataflow.qual.Pure;
import com.sun.org.apache.xerces.internal.xni.grammars.Grammar;
import com.sun.org.apache.xerces.internal.xni.grammars.XMLGrammarPool;
import com.sun.org.apache.xerces.internal.xni.grammars.XMLGrammarDescription;
import com.sun.org.apache.xerces.internal.util.XMLGrammarPoolImpl;
import com.sun.org.apache.xerces.internal.util.ShadowedSymbolTable;
import com.sun.org.apache.xerces.internal.util.SymbolTable;
import com.sun.org.apache.xerces.internal.util.SynchronizedSymbolTable;

public class CachingParserPool {

    public static final boolean DEFAULT_SHADOW_SYMBOL_TABLE;

    public static final boolean DEFAULT_SHADOW_GRAMMAR_POOL;

    protected SymbolTable fSynchronizedSymbolTable;

    protected XMLGrammarPool fSynchronizedGrammarPool;

    protected boolean fShadowSymbolTable;

    protected boolean fShadowGrammarPool;

    public CachingParserPool() {
    }

    public CachingParserPool(SymbolTable symbolTable, XMLGrammarPool grammarPool) {
    }

    public SymbolTable getSymbolTable();

    public XMLGrammarPool getXMLGrammarPool();

    public void setShadowSymbolTable(boolean shadow);

    public DOMParser createDOMParser();

    public SAXParser createSAXParser();

    public static final class SynchronizedGrammarPool implements XMLGrammarPool {

        public SynchronizedGrammarPool(XMLGrammarPool grammarPool) {
        }

        public Grammar[] retrieveInitialGrammarSet(String grammarType);

        public Grammar retrieveGrammar(XMLGrammarDescription gDesc);

        public void cacheGrammars(String grammarType, Grammar[] grammars);

        public void lockPool();

        public void clear();

        public void unlockPool();
    }

    public static final class ShadowedGrammarPool extends XMLGrammarPoolImpl {

        public ShadowedGrammarPool(XMLGrammarPool grammarPool) {
        }

        public Grammar[] retrieveInitialGrammarSet(String grammarType);

        public Grammar retrieveGrammar(XMLGrammarDescription gDesc);

        public void cacheGrammars(String grammarType, Grammar[] grammars);

        public Grammar getGrammar(XMLGrammarDescription desc);

        @Pure
        public boolean containsGrammar(XMLGrammarDescription desc);
    }
}
