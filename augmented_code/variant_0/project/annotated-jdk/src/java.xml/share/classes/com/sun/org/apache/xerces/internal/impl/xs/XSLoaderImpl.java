/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * reserved comment block
    @Positive
 * DO NOT REMOVE OR ALTER!
    @Positive
 */
    @Positive
package com.sun.org.apache.xerces.internal.impl.xs;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import com.sun.org.apache.xerces.internal.impl.xs.util.XSGrammarPool;
    @Positive
import com.sun.org.apache.xerces.internal.xni.grammars.Grammar;
    @Positive
import com.sun.org.apache.xerces.internal.xni.grammars.XMLGrammarDescription;
    @Positive
import com.sun.org.apache.xerces.internal.xni.grammars.XSGrammar;
    @Positive
import com.sun.org.apache.xerces.internal.xni.parser.XMLInputSource;
    @Positive
import com.sun.org.apache.xerces.internal.xs.LSInputList;
    @Positive
import com.sun.org.apache.xerces.internal.xs.StringList;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSConstants;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSLoader;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSModel;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSNamedMap;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSObjectList;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSTypeDefinition;
    @Positive
import org.w3c.dom.DOMConfiguration;
    @Positive
import org.w3c.dom.DOMException;
    @Positive
import org.w3c.dom.DOMStringList;
    @Positive
import org.w3c.dom.ls.LSInput;

    @Positive
public final class XSLoaderImpl implements XSLoader, DOMConfiguration {

    @Positive
    public XSLoaderImpl() {
    @Positive
    }

    @Positive
    public DOMConfiguration getConfig();

    @Positive
    public XSModel loadURIList(StringList uriList);

    @Positive
    public XSModel loadInputList(LSInputList is);

    @Positive
    public XSModel loadURI(String uri);

    @Positive
    public XSModel load(LSInput is);

    @Positive
    public void setParameter(String name, Object value) throws DOMException;

    @Positive
    public Object getParameter(String name) throws DOMException;

    @Positive
    public boolean canSetParameter(String name, Object value);

    @Positive
    public DOMStringList getParameterNames();

    @Positive
    private static final class XSGrammarMerger extends XSGrammarPool {

    @Positive
        public XSGrammarMerger() {
    @Positive
        }

    @Positive
        public void putGrammar(Grammar grammar);

    @Positive
        @Pure
    @Positive
        public boolean containsGrammar(XMLGrammarDescription desc);

    @Positive
        public Grammar getGrammar(XMLGrammarDescription desc);

    @Positive
        public Grammar retrieveGrammar(XMLGrammarDescription desc);

    @Positive
        public Grammar[] retrieveInitialGrammarSet(String grammarType);
    @Positive
    }
    @Positive
}
