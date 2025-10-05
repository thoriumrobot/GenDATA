/*
 * Copyright (c) 2005, 2019, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.xml.internal.stream.writers;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import com.sun.org.apache.xerces.internal.impl.Constants;
import com.sun.org.apache.xerces.internal.impl.PropertyManager;
import com.sun.org.apache.xerces.internal.util.NamespaceSupport;
import com.sun.org.apache.xerces.internal.util.SymbolTable;
import com.sun.org.apache.xerces.internal.xni.QName;
import com.sun.xml.internal.stream.util.ReadOnlyIterator;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.Charset;
import java.nio.charset.CharsetEncoder;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
import javax.xml.XMLConstants;
import javax.xml.namespace.NamespaceContext;
import javax.xml.stream.XMLOutputFactory;
import javax.xml.stream.XMLStreamConstants;
import javax.xml.stream.XMLStreamException;
import javax.xml.transform.stream.StreamResult;
import jdk.xml.internal.SecuritySupport;

public final class XMLStreamWriterImpl extends AbstractMap<Object, Object> implements XMLStreamWriterBase {

    public static final String START_COMMENT;

    public static final String END_COMMENT;

    public static final String DEFAULT_ENCODING;

    public static final String DEFAULT_XMLDECL;

    public static final String DEFAULT_XML_VERSION;

    public static final char CLOSE_START_TAG;

    public static final char OPEN_START_TAG;

    public static final String OPEN_END_TAG;

    public static final char CLOSE_END_TAG;

    public static final String START_CDATA;

    public static final String END_CDATA;

    public static final String CLOSE_EMPTY_ELEMENT;

    public static final String SPACE;

    public static final String UTF_8;

    public static final String OUTPUTSTREAM_PROPERTY;

    public XMLStreamWriterImpl(OutputStream outputStream, PropertyManager props) throws IOException {
    }

    public XMLStreamWriterImpl(OutputStream outputStream, String encoding, PropertyManager props) throws java.io.IOException {
    }

    public XMLStreamWriterImpl(Writer writer, PropertyManager props) throws java.io.IOException {
    }

    public XMLStreamWriterImpl(StreamResult sr, String encoding, PropertyManager props) throws java.io.IOException {
    }

    public void reset();

    void reset(boolean resetProperties);

    public void setOutput(StreamResult sr, String encoding) throws IOException;

    public boolean canReuse();

    public void setEscapeCharacters(boolean escape);

    public boolean getEscapeCharacters();

    @Override
    public void close() throws XMLStreamException;

    @Override
    public void flush() throws XMLStreamException;

    @Override
    public NamespaceContext getNamespaceContext();

    @Override
    public String getPrefix(String uri) throws XMLStreamException;

    @Override
    public Object getProperty(String str) throws IllegalArgumentException;

    @Override
    public void setDefaultNamespace(String uri) throws XMLStreamException;

    @Override
    public void setNamespaceContext(NamespaceContext namespaceContext) throws XMLStreamException;

    @Override
    public void setPrefix(String prefix, String uri) throws XMLStreamException;

    @Override
    public void writeAttribute(String localName, String value) throws XMLStreamException;

    @Override
    public void writeAttribute(String namespaceURI, String localName, String value) throws XMLStreamException;

    @Override
    public void writeAttribute(String prefix, String namespaceURI, String localName, String value) throws XMLStreamException;

    @Override
    public void writeCData(String cdata) throws XMLStreamException;

    @Override
    public void writeCharacters(String data) throws XMLStreamException;

    @Override
    public void writeCharacters(char[] data, int start, int len) throws XMLStreamException;

    @Override
    public void writeComment(String comment) throws XMLStreamException;

    @Override
    public void writeDTD(String dtd) throws XMLStreamException;

    @Override
    public void writeDefaultNamespace(String namespaceURI) throws XMLStreamException;

    @Override
    public void writeEmptyElement(String localName) throws XMLStreamException;

    @Override
    public void writeEmptyElement(String namespaceURI, String localName) throws XMLStreamException;

    @Override
    public void writeEmptyElement(String prefix, String localName, String namespaceURI) throws XMLStreamException;

    @Override
    public void writeEndDocument() throws XMLStreamException;

    @Override
    public void writeEndElement() throws XMLStreamException;

    @Override
    public void writeEntityRef(String refName) throws XMLStreamException;

    @Override
    public void writeNamespace(String prefix, String namespaceURI) throws XMLStreamException;

    @Override
    public void writeProcessingInstruction(String target) throws XMLStreamException;

    @Override
    public void writeProcessingInstruction(String target, String data) throws XMLStreamException;

    @Override
    public void writeStartDocument() throws XMLStreamException;

    @Override
    public void writeStartDocument(String version) throws XMLStreamException;

    @Override
    public void writeStartDocument(String encoding, String version) throws XMLStreamException;

    public void writeStartDocument(String encoding, String version, boolean standalone, boolean standaloneSet) throws XMLStreamException;

    @Override
    public void writeStartElement(String localName) throws XMLStreamException;

    @Override
    public void writeStartElement(String namespaceURI, String localName) throws XMLStreamException;

    @Override
    public void writeStartElement(String prefix, String localName, String namespaceURI) throws XMLStreamException;

    protected void repair();

    void correctPrefix(QName attr1, QName attr2);

    void checkForNull(QName attr);

    void removeDuplicateDecls();

    void repairNamespaceDecl(QName attr);

    boolean isDeclared(QName attr);

    protected class ElementStack {

        protected ElementState[] fElements;

        protected short fDepth;

        public ElementStack() {
        }

        public ElementState push(ElementState element);

        public ElementState push(String prefix, String localpart, String rawname, String uri, boolean isEmpty);

        public ElementState pop();

        public void clear();

        public ElementState peek();

        public boolean empty();
    }

    class ElementState extends QName {

        public boolean isEmpty;

        public ElementState() {
        }

        public ElementState(String prefix, String localpart, String rawname, String uri) {
        }

        public void setValues(String prefix, String localpart, String rawname, String uri, boolean isEmpty);
    }

    class Attribute extends QName {
    }

    class NamespaceContextImpl implements NamespaceContext {

        public String getNamespaceURI(String prefix);

        public String getPrefix(String uri);

        public Iterator<String> getPrefixes(String uri);
    }

    @Override
    public int size();

    @Override
    public boolean isEmpty();

    @Override
    @Pure
    public boolean containsKey(Object key);

    @Override
    public Object get(Object key);

    @Override
    public Set<Entry<Object, Object>> entrySet();

    @Override
    public String toString();

    @Override
    public int hashCode();

    @Override
    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);
}
