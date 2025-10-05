/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2005, 2017, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
    @Positive
 */
    @Positive
package com.sun.org.apache.xerces.internal.impl;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import com.sun.org.apache.xerces.internal.util.NamespaceContextWrapper;
    @Positive
import com.sun.org.apache.xerces.internal.util.NamespaceSupport;
    @Positive
import com.sun.org.apache.xerces.internal.util.SymbolTable;
    @Positive
import com.sun.org.apache.xerces.internal.util.XMLAttributesImpl;
    @Positive
import com.sun.org.apache.xerces.internal.util.XMLChar;
    @Positive
import com.sun.org.apache.xerces.internal.util.XMLStringBuffer;
    @Positive
import com.sun.org.apache.xerces.internal.xni.XNIException;
    @Positive
import com.sun.org.apache.xerces.internal.xni.parser.XMLInputSource;
    @Positive
import com.sun.xml.internal.stream.Entity;
    @Positive
import com.sun.xml.internal.stream.StaxErrorReporter;
    @Positive
import com.sun.xml.internal.stream.XMLEntityStorage;
    @Positive
import com.sun.xml.internal.stream.dtd.nonvalidating.DTDGrammar;
    @Positive
import com.sun.xml.internal.stream.dtd.nonvalidating.XMLNotationDecl;
    @Positive
import com.sun.xml.internal.stream.events.EntityDeclarationImpl;
    @Positive
import com.sun.xml.internal.stream.events.NotationDeclarationImpl;
    @Positive
import java.io.BufferedInputStream;
    @Positive
import java.io.BufferedReader;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.Reader;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import javax.xml.XMLConstants;
    @Positive
import javax.xml.namespace.NamespaceContext;
    @Positive
import javax.xml.namespace.QName;
    @Positive
import javax.xml.stream.Location;
    @Positive
import javax.xml.stream.XMLInputFactory;
    @Positive
import javax.xml.stream.XMLStreamConstants;
    @Positive
import javax.xml.stream.XMLStreamException;
    @Positive
import javax.xml.stream.events.EntityDeclaration;
    @Positive
import javax.xml.stream.events.NotationDeclaration;
    @Positive
import javax.xml.stream.events.XMLEvent;

    @Positive
public class XMLStreamReaderImpl implements javax.xml.stream.XMLStreamReader {

    @Positive
    protected static final String ENTITY_MANAGER;

    @Positive
    protected static final String ERROR_REPORTER;

    @Positive
    protected static final String SYMBOL_TABLE;

    @Positive
    protected static final String READER_IN_DEFINED_STATE;

    @Positive
    protected XMLDocumentScannerImpl fScanner;

    @Positive
    protected NamespaceContextWrapper fNamespaceContextWrapper;

    @Positive
    protected XMLEntityManager fEntityManager;

    @Positive
    protected StaxErrorReporter fErrorReporter;

    @Positive
    protected XMLEntityScanner fEntityScanner;

    @Positive
    protected XMLInputSource fInputSource;

    @Positive
    protected PropertyManager fPropertyManager;

    @Positive
    public XMLStreamReaderImpl(InputStream inputStream, PropertyManager props) throws XMLStreamException {
    @Positive
    }

    @Positive
    public XMLDocumentScannerImpl getScanner();

    @Positive
    public XMLStreamReaderImpl(String systemid, PropertyManager props) throws XMLStreamException {
    @Positive
    }

    @Positive
    public XMLStreamReaderImpl(InputStream inputStream, String encoding, PropertyManager props) throws XMLStreamException {
    @Positive
    }

    @Positive
    public XMLStreamReaderImpl(Reader reader, PropertyManager props) throws XMLStreamException {
    @Positive
    }

    @Positive
    public XMLStreamReaderImpl(XMLInputSource inputSource, PropertyManager props) throws XMLStreamException {
    @Positive
    }

    @Positive
    public final void setInputSource(XMLInputSource inputSource) throws XMLStreamException;

    @Positive
    final void init(PropertyManager propertyManager) throws XMLStreamException;

    @Positive
    public boolean canReuse();

    @Positive
    public void reset();

    @Positive
    public void close() throws XMLStreamException;

    @Positive
    public String getCharacterEncodingScheme();

    @Positive
    public int getColumnNumber();

    @Positive
    public String getEncoding();

    @Positive
    public int getEventType();

    @Positive
    public int getLineNumber();

    @Positive
    public String getLocalName();

    @Positive
    public String getNamespaceURI();

    @Positive
    public String getPIData();

    @Positive
    public String getPITarget();

    @Positive
    public String getPrefix();

    @Positive
    public char[] getTextCharacters();

    @Positive
    public int getTextLength();

    @Positive
    public int getTextStart();

    @Positive
    public String getValue();

    @Positive
    public String getVersion();

    @Positive
    public boolean hasAttributes();

    @Positive
    public boolean hasName();

    @Positive
    @Pure
    @Positive
    public boolean hasNext() throws XMLStreamException;

    @Positive
    public boolean hasValue();

    @Positive
    public boolean isEndElement();

    @Positive
    public boolean isStandalone();

    @Positive
    public boolean isStartElement();

    @Positive
    public boolean isWhiteSpace();

    @Positive
    public int next() throws XMLStreamException;

    @Positive
    final static String getEventTypeString(int eventType);

    @Positive
    public int getAttributeCount();

    @Positive
    public QName getAttributeName(int index);

    @Positive
    public String getAttributeLocalName(int index);

    @Positive
    public String getAttributeNamespace(int index);

    @Positive
    public String getAttributePrefix(int index);

    @Positive
    public javax.xml.namespace.QName getAttributeQName(int index);

    @Positive
    public String getAttributeType(int index);

    @Positive
    public String getAttributeValue(int index);

    @Positive
    public String getAttributeValue(String namespaceURI, String localName);

    @Positive
    public String getElementText() throws XMLStreamException;

    @Positive
    public Location getLocation();

    @Positive
    public javax.xml.namespace.QName getName();

    @Positive
    public NamespaceContext getNamespaceContext();

    @Positive
    public int getNamespaceCount();

    @Positive
    public String getNamespacePrefix(int index);

    @Positive
    public String getNamespaceURI(int index);

    @Positive
    public Object getProperty(java.lang.String name) throws java.lang.IllegalArgumentException;

    @Positive
    public String getText();

    @Positive
    public void require(int type, String namespaceURI, String localName) throws XMLStreamException;

    @Positive
    public int getTextCharacters(int sourceStart, char[] target, int targetStart, int length) throws XMLStreamException;

    @Positive
    public boolean hasText();

    @Positive
    public boolean isAttributeSpecified(int index);

    @Positive
    public boolean isCharacters();

    @Positive
    public int nextTag() throws XMLStreamException;

    @Positive
    public boolean standaloneSet();

    @Positive
    public javax.xml.namespace.QName convertXNIQNametoJavaxQName(com.sun.org.apache.xerces.internal.xni.QName qname);

    @Positive
    public String getNamespaceURI(String prefix);

    @Positive
    protected void setPropertyManager(PropertyManager propertyManager);

    @Positive
    protected PropertyManager getPropertyManager();

    @Positive
    static void pr(String str);

    @Positive
    protected List<EntityDeclaration> getEntityDecls();

    @Positive
    protected List<NotationDeclaration> getNotationDecls();
    @Positive
}
