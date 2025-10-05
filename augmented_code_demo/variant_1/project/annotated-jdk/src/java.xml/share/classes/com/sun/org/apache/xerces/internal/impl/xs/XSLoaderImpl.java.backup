/*
 * reserved comment block
 * DO NOT REMOVE OR ALTER!
 */
package com.sun.org.apache.xerces.internal.impl.xs;

import org.checkerframework.dataflow.qual.Pure;
import com.sun.org.apache.xerces.internal.impl.xs.util.XSGrammarPool;
import com.sun.org.apache.xerces.internal.xni.grammars.Grammar;
import com.sun.org.apache.xerces.internal.xni.grammars.XMLGrammarDescription;
import com.sun.org.apache.xerces.internal.xni.grammars.XSGrammar;
import com.sun.org.apache.xerces.internal.xni.parser.XMLInputSource;
import com.sun.org.apache.xerces.internal.xs.LSInputList;
import com.sun.org.apache.xerces.internal.xs.StringList;
import com.sun.org.apache.xerces.internal.xs.XSConstants;
import com.sun.org.apache.xerces.internal.xs.XSLoader;
import com.sun.org.apache.xerces.internal.xs.XSModel;
import com.sun.org.apache.xerces.internal.xs.XSNamedMap;
import com.sun.org.apache.xerces.internal.xs.XSObjectList;
import com.sun.org.apache.xerces.internal.xs.XSTypeDefinition;
import org.w3c.dom.DOMConfiguration;
import org.w3c.dom.DOMException;
import org.w3c.dom.DOMStringList;
import org.w3c.dom.ls.LSInput;

public final class XSLoaderImpl implements XSLoader, DOMConfiguration {

    public XSLoaderImpl() {
    }

    public DOMConfiguration getConfig();

    public XSModel loadURIList(StringList uriList);

    public XSModel loadInputList(LSInputList is);

    public XSModel loadURI(String uri);

    public XSModel load(LSInput is);

    public void setParameter(String name, Object value) throws DOMException;

    public Object getParameter(String name) throws DOMException;

    public boolean canSetParameter(String name, Object value);

    public DOMStringList getParameterNames();

    private static final class XSGrammarMerger extends XSGrammarPool {

        public XSGrammarMerger() {
        }

        public void putGrammar(Grammar grammar);

        @Pure
        public boolean containsGrammar(XMLGrammarDescription desc);

        public Grammar getGrammar(XMLGrammarDescription desc);

        public Grammar retrieveGrammar(XMLGrammarDescription desc);

        public Grammar[] retrieveInitialGrammarSet(String grammarType);
    }
}
