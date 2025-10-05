/*
    @Positive
 * Copyright (c) 2005, 2006, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.org.apache.xerces.internal.impl;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import javax.xml.stream.XMLStreamReader;
    @Positive
import javax.xml.stream.StreamFilter;
    @Positive
import javax.xml.stream.XMLStreamException;
    @Positive
import javax.xml.namespace.QName;
    @Positive
import javax.xml.stream.events.XMLEvent;

    @Positive
public class XMLStreamFilterImpl implements javax.xml.stream.XMLStreamReader {

    @Positive
    public XMLStreamFilterImpl(XMLStreamReader reader, StreamFilter filter) throws XMLStreamException {
    @Positive
    }

    @Positive
    protected void setStreamFilter(StreamFilter sf);

    @Positive
    @SideEffectsOnly("this")
    @Positive
    public int next() throws XMLStreamException;

    @Positive
    public int nextTag() throws XMLStreamException;

    @Positive
    @Pure
    @Positive
    public boolean hasNext() throws XMLStreamException;

    @Positive
    public void close() throws XMLStreamException;

    @Positive
    public int getAttributeCount();

    @Positive
    public QName getAttributeName(int index);

    @Positive
    public String getAttributeNamespace(int index);

    @Positive
    public String getAttributePrefix(int index);

    @Positive
    public String getAttributeType(int index);

    @Positive
    public String getAttributeValue(int index);

    @Positive
    public String getAttributeValue(String namespaceURI, String localName);

    @Positive
    public String getCharacterEncodingScheme();

    @Positive
    public String getElementText() throws XMLStreamException;

    @Positive
    public String getEncoding();

    @Positive
    public int getEventType();

    @Positive
    public String getLocalName();

    @Positive
    public javax.xml.stream.Location getLocation();

    @Positive
    public javax.xml.namespace.QName getName();

    @Positive
    public javax.xml.namespace.NamespaceContext getNamespaceContext();

    @Positive
    public int getNamespaceCount();

    @Positive
    public String getNamespacePrefix(int index);

    @Positive
    public String getNamespaceURI();

    @Positive
    public String getNamespaceURI(int index);

    @Positive
    public String getNamespaceURI(String prefix);

    @Positive
    public String getPIData();

    @Positive
    public String getPITarget();

    @Positive
    public String getPrefix();

    @Positive
    public Object getProperty(java.lang.String name) throws java.lang.IllegalArgumentException;

    @Positive
    public String getText();

    @Positive
    public char[] getTextCharacters();

    @Positive
    public int getTextCharacters(int sourceStart, char[] target, int targetStart, int length) throws XMLStreamException;

    @Positive
    public int getTextLength();

    @Positive
    public int getTextStart();

    @Positive
    public String getVersion();

    @Positive
    public boolean hasName();

    @Positive
    public boolean hasText();

    @Positive
    public boolean isAttributeSpecified(int index);

    @Positive
    public boolean isCharacters();

    @Positive
    public boolean isEndElement();

    @Positive
    public boolean isStandalone();

    @Positive
    public boolean isStartElement();

    @Positive
    public boolean isWhiteSpace();

    @Positive
    public void require(int type, String namespaceURI, String localName) throws XMLStreamException;

    @Positive
    public boolean standaloneSet();

    @Positive
    public String getAttributeLocalName(int index);
    @Positive
}

// CFWR semantic augmentation - variant 0
