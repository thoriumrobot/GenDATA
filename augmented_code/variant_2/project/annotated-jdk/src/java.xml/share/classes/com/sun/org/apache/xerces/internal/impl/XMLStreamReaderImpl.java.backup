/*
 * Copyright (c) 2005, 2017, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package com.sun.org.apache.xerces.internal.impl;

import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import com.sun.org.apache.xerces.internal.util.NamespaceContextWrapper;
import com.sun.org.apache.xerces.internal.util.NamespaceSupport;
import com.sun.org.apache.xerces.internal.util.SymbolTable;
import com.sun.org.apache.xerces.internal.util.XMLAttributesImpl;
import com.sun.org.apache.xerces.internal.util.XMLChar;
import com.sun.org.apache.xerces.internal.util.XMLStringBuffer;
import com.sun.org.apache.xerces.internal.xni.XNIException;
import com.sun.org.apache.xerces.internal.xni.parser.XMLInputSource;
import com.sun.xml.internal.stream.Entity;
import com.sun.xml.internal.stream.StaxErrorReporter;
import com.sun.xml.internal.stream.XMLEntityStorage;
import com.sun.xml.internal.stream.dtd.nonvalidating.DTDGrammar;
import com.sun.xml.internal.stream.dtd.nonvalidating.XMLNotationDecl;
import com.sun.xml.internal.stream.events.EntityDeclarationImpl;
import com.sun.xml.internal.stream.events.NotationDeclarationImpl;
import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import javax.xml.XMLConstants;
import javax.xml.namespace.NamespaceContext;
import javax.xml.namespace.QName;
import javax.xml.stream.Location;
import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamConstants;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.events.EntityDeclaration;
import javax.xml.stream.events.NotationDeclaration;
import javax.xml.stream.events.XMLEvent;

public class XMLStreamReaderImpl implements javax.xml.stream.XMLStreamReader {

    protected static final String ENTITY_MANAGER;

    protected static final String ERROR_REPORTER;

    protected static final String SYMBOL_TABLE;

    protected static final String READER_IN_DEFINED_STATE;

    protected XMLDocumentScannerImpl fScanner;

    protected NamespaceContextWrapper fNamespaceContextWrapper;

    protected XMLEntityManager fEntityManager;

    protected StaxErrorReporter fErrorReporter;

    protected XMLEntityScanner fEntityScanner;

    protected XMLInputSource fInputSource;

    protected PropertyManager fPropertyManager;

    public XMLStreamReaderImpl(InputStream inputStream, PropertyManager props) throws XMLStreamException {
    }

    public XMLDocumentScannerImpl getScanner();

    public XMLStreamReaderImpl(String systemid, PropertyManager props) throws XMLStreamException {
    }

    public XMLStreamReaderImpl(InputStream inputStream, String encoding, PropertyManager props) throws XMLStreamException {
    }

    public XMLStreamReaderImpl(Reader reader, PropertyManager props) throws XMLStreamException {
    }

    public XMLStreamReaderImpl(XMLInputSource inputSource, PropertyManager props) throws XMLStreamException {
    }

    public final void setInputSource(XMLInputSource inputSource) throws XMLStreamException;

    final void init(PropertyManager propertyManager) throws XMLStreamException;

    public boolean canReuse();

    public void reset();

    public void close() throws XMLStreamException;

    public String getCharacterEncodingScheme();

    public int getColumnNumber();

    public String getEncoding();

    public int getEventType();

    public int getLineNumber();

    public String getLocalName();

    public String getNamespaceURI();

    public String getPIData();

    public String getPITarget();

    public String getPrefix();

    public char[] getTextCharacters();

    public int getTextLength();

    public int getTextStart();

    public String getValue();

    public String getVersion();

    public boolean hasAttributes();

    public boolean hasName();

    @Pure
    public boolean hasNext() throws XMLStreamException;

    public boolean hasValue();

    public boolean isEndElement();

    public boolean isStandalone();

    public boolean isStartElement();

    public boolean isWhiteSpace();

    public int next() throws XMLStreamException;

    final static String getEventTypeString(int eventType);

    public int getAttributeCount();

    public QName getAttributeName(int index);

    public String getAttributeLocalName(int index);

    public String getAttributeNamespace(int index);

    public String getAttributePrefix(int index);

    public javax.xml.namespace.QName getAttributeQName(int index);

    public String getAttributeType(int index);

    public String getAttributeValue(int index);

    public String getAttributeValue(String namespaceURI, String localName);

    public String getElementText() throws XMLStreamException;

    public Location getLocation();

    public javax.xml.namespace.QName getName();

    public NamespaceContext getNamespaceContext();

    public int getNamespaceCount();

    public String getNamespacePrefix(int index);

    public String getNamespaceURI(int index);

    public Object getProperty(java.lang.String name) throws java.lang.IllegalArgumentException;

    public String getText();

    public void require(int type, String namespaceURI, String localName) throws XMLStreamException;

    public int getTextCharacters(int sourceStart, char[] target, int targetStart, int length) throws XMLStreamException;

    public boolean hasText();

    public boolean isAttributeSpecified(int index);

    public boolean isCharacters();

    public int nextTag() throws XMLStreamException;

    public boolean standaloneSet();

    public javax.xml.namespace.QName convertXNIQNametoJavaxQName(com.sun.org.apache.xerces.internal.xni.QName qname);

    public String getNamespaceURI(String prefix);

    protected void setPropertyManager(PropertyManager propertyManager);

    protected PropertyManager getPropertyManager();

    static void pr(String str);

    protected List<EntityDeclaration> getEntityDecls();

    protected List<NotationDeclaration> getNotationDecls();
}
