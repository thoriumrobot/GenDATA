/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2005, 2019, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package com.sun.xml.internal.stream.writers;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import com.sun.org.apache.xerces.internal.impl.Constants;
    @Positive
import com.sun.org.apache.xerces.internal.impl.PropertyManager;
    @Positive
import com.sun.org.apache.xerces.internal.util.NamespaceSupport;
    @Positive
import com.sun.org.apache.xerces.internal.util.SymbolTable;
    @Positive
import com.sun.org.apache.xerces.internal.xni.QName;
    @Positive
import com.sun.xml.internal.stream.util.ReadOnlyIterator;
    @Positive
import java.io.FileOutputStream;
    @Positive
import java.io.IOException;
    @Positive
import java.io.OutputStream;
    @Positive
import java.io.OutputStreamWriter;
    @Positive
import java.io.Writer;
    @Positive
import java.nio.charset.Charset;
    @Positive
import java.nio.charset.CharsetEncoder;
    @Positive
import java.util.AbstractMap;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Random;
    @Positive
import java.util.Set;
    @Positive
import javax.xml.XMLConstants;
    @Positive
import javax.xml.namespace.NamespaceContext;
    @Positive
import javax.xml.stream.XMLOutputFactory;
    @Positive
import javax.xml.stream.XMLStreamConstants;
    @Positive
import javax.xml.stream.XMLStreamException;
    @Positive
import javax.xml.transform.stream.StreamResult;
    @Positive
import jdk.xml.internal.SecuritySupport;

    @Positive
public final class XMLStreamWriterImpl extends AbstractMap<Object, Object> implements XMLStreamWriterBase {

    @Positive
    public static final String START_COMMENT;

    @Positive
    public static final String END_COMMENT;

    @Positive
    public static final String DEFAULT_ENCODING;

    @Positive
    public static final String DEFAULT_XMLDECL;

    @Positive
    public static final String DEFAULT_XML_VERSION;

    @Positive
    public static final char CLOSE_START_TAG;

    @Positive
    public static final char OPEN_START_TAG;

    @Positive
    public static final String OPEN_END_TAG;

    @Positive
    public static final char CLOSE_END_TAG;

    @Positive
    public static final String START_CDATA;

    @Positive
    public static final String END_CDATA;

    @Positive
    public static final String CLOSE_EMPTY_ELEMENT;

    @Positive
    public static final String SPACE;

    @Positive
    public static final String UTF_8;

    @Positive
    public static final String OUTPUTSTREAM_PROPERTY;

    @Positive
    public XMLStreamWriterImpl(OutputStream outputStream, PropertyManager props) throws IOException {
    @Positive
    }

    @Positive
    public XMLStreamWriterImpl(OutputStream outputStream, String encoding, PropertyManager props) throws java.io.IOException {
    @Positive
    }

    @Positive
    public XMLStreamWriterImpl(Writer writer, PropertyManager props) throws java.io.IOException {
    @Positive
    }

    @Positive
    public XMLStreamWriterImpl(StreamResult sr, String encoding, PropertyManager props) throws java.io.IOException {
    @Positive
    }

    @Positive
    public void reset();

    @Positive
    void reset(boolean resetProperties);

    @Positive
    public void setOutput(StreamResult sr, String encoding) throws IOException;

    @Positive
    public boolean canReuse();

    @Positive
    public void setEscapeCharacters(boolean escape);

    @Positive
    public boolean getEscapeCharacters();

    @Positive
    @Override
    @Positive
    public void close() throws XMLStreamException;

    @Positive
    @Override
    @Positive
    public void flush() throws XMLStreamException;

    @Positive
    @Override
    @Positive
    public NamespaceContext getNamespaceContext();

    @Positive
    @Override
    @Positive
    public String getPrefix(String uri) throws XMLStreamException;

    @Positive
    @Override
    @Positive
    public Object getProperty(String str) throws IllegalArgumentException;

    @Positive
    @Override
    @Positive
    public void setDefaultNamespace(String uri) throws XMLStreamException;

    @Positive
    @Override
    @Positive
    public void setNamespaceContext(NamespaceContext namespaceContext) throws XMLStreamException;

    @Positive
    @Override
    @Positive
    public void setPrefix(String prefix, String uri) throws XMLStreamException;

    @Positive
    @Override
    @Positive
    public void writeAttribute(String localName, String value) throws XMLStreamException;

    @Positive
    @Override
    @Positive
    public void writeAttribute(String namespaceURI, String localName, String value) throws XMLStreamException;

    @Positive
    @Override
    @Positive
    public void writeAttribute(String prefix, String namespaceURI, String localName, String value) throws XMLStreamException;

    @Positive
    @Override
    @Positive
    public void writeCData(String cdata) throws XMLStreamException;

    @Positive
    @Override
    @Positive
    public void writeCharacters(String data) throws XMLStreamException;

    @Positive
    @Override
    @Positive
    public void writeCharacters(char[] data, int start, int len) throws XMLStreamException;

    @Positive
    @Override
    @Positive
    public void writeComment(String comment) throws XMLStreamException;

    @Positive
    @Override
    @Positive
    public void writeDTD(String dtd) throws XMLStreamException;

    @Positive
    @Override
    @Positive
    public void writeDefaultNamespace(String namespaceURI) throws XMLStreamException;

    @Positive
    @Override
    @Positive
    public void writeEmptyElement(String localName) throws XMLStreamException;

    @Positive
    @Override
    @Positive
    public void writeEmptyElement(String namespaceURI, String localName) throws XMLStreamException;

    @Positive
    @Override
    @Positive
    public void writeEmptyElement(String prefix, String localName, String namespaceURI) throws XMLStreamException;

    @Positive
    @Override
    @Positive
    public void writeEndDocument() throws XMLStreamException;

    @Positive
    @Override
    @Positive
    public void writeEndElement() throws XMLStreamException;

    @Positive
    @Override
    @Positive
    public void writeEntityRef(String refName) throws XMLStreamException;

    @Positive
    @Override
    @Positive
    public void writeNamespace(String prefix, String namespaceURI) throws XMLStreamException;

    @Positive
    @Override
    @Positive
    public void writeProcessingInstruction(String target) throws XMLStreamException;

    @Positive
    @Override
    @Positive
    public void writeProcessingInstruction(String target, String data) throws XMLStreamException;

    @Positive
    @Override
    @Positive
    public void writeStartDocument() throws XMLStreamException;

    @Positive
    @Override
    @Positive
    public void writeStartDocument(String version) throws XMLStreamException;

    @Positive
    @Override
    @Positive
    public void writeStartDocument(String encoding, String version) throws XMLStreamException;

    @Positive
    public void writeStartDocument(String encoding, String version, boolean standalone, boolean standaloneSet) throws XMLStreamException;

    @Positive
    @Override
    @Positive
    public void writeStartElement(String localName) throws XMLStreamException;

    @Positive
    @Override
    @Positive
    public void writeStartElement(String namespaceURI, String localName) throws XMLStreamException;

    @Positive
    @Override
    @Positive
    public void writeStartElement(String prefix, String localName, String namespaceURI) throws XMLStreamException;

    @Positive
    protected void repair();

    @Positive
    void correctPrefix(QName attr1, QName attr2);

    @Positive
    void checkForNull(QName attr);

    @Positive
    void removeDuplicateDecls();

    @Positive
    void repairNamespaceDecl(QName attr);

    @Positive
    boolean isDeclared(QName attr);

    @Positive
    protected class ElementStack {

    @Positive
        protected ElementState[] fElements;

    @Positive
        protected short fDepth;

    @Positive
        public ElementStack() {
    @Positive
        }

    @Positive
        public ElementState push(ElementState element);

    @Positive
        public ElementState push(String prefix, String localpart, String rawname, String uri, boolean isEmpty);

    @Positive
        public ElementState pop();

    @Positive
        public void clear();

    @Positive
        public ElementState peek();

    @Positive
        public boolean empty();
    @Positive
    }

    @Positive
    class ElementState extends QName {

    @Positive
        public boolean isEmpty;

    @Positive
        public ElementState() {
    @Positive
        }

    @Positive
        public ElementState(String prefix, String localpart, String rawname, String uri) {
    @Positive
        }

    @Positive
        public void setValues(String prefix, String localpart, String rawname, String uri, boolean isEmpty);
    @Positive
    }

    @Positive
    class Attribute extends QName {
    @Positive
    }

    @Positive
    class NamespaceContextImpl implements NamespaceContext {

    @Positive
        public String getNamespaceURI(String prefix);

    @Positive
        public String getPrefix(String uri);

    @Positive
        public Iterator<String> getPrefixes(String uri);
    @Positive
    }

    @Positive
    @Override
    @Positive
    public int size();

    @Positive
    @Override
    @Positive
    public boolean isEmpty();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    public boolean containsKey(Object key);

    @Positive
    @Override
    @Positive
    public Object get(Object key);

    @Positive
    @Override
    @Positive
    public Set<Entry<Object, Object>> entrySet();

    @Positive
    @Override
    @Positive
    public String toString();

    @Positive
    @Override
    @Positive
    public int hashCode();

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);
    @Positive
}
