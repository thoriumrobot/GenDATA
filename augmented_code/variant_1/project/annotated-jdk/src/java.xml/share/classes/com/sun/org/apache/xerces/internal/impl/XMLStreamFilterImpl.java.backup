/*
 * Copyright (c) 2005, 2006, Oracle and/or its affiliates. All rights reserved.
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
import javax.xml.stream.XMLStreamReader;
import javax.xml.stream.StreamFilter;
import javax.xml.stream.XMLStreamException;
import javax.xml.namespace.QName;
import javax.xml.stream.events.XMLEvent;

public class XMLStreamFilterImpl implements javax.xml.stream.XMLStreamReader {

    public XMLStreamFilterImpl(XMLStreamReader reader, StreamFilter filter) throws XMLStreamException {
    }

    protected void setStreamFilter(StreamFilter sf);

    @SideEffectsOnly("this")
    public int next() throws XMLStreamException;

    public int nextTag() throws XMLStreamException;

    @Pure
    public boolean hasNext() throws XMLStreamException;

    public void close() throws XMLStreamException;

    public int getAttributeCount();

    public QName getAttributeName(int index);

    public String getAttributeNamespace(int index);

    public String getAttributePrefix(int index);

    public String getAttributeType(int index);

    public String getAttributeValue(int index);

    public String getAttributeValue(String namespaceURI, String localName);

    public String getCharacterEncodingScheme();

    public String getElementText() throws XMLStreamException;

    public String getEncoding();

    public int getEventType();

    public String getLocalName();

    public javax.xml.stream.Location getLocation();

    public javax.xml.namespace.QName getName();

    public javax.xml.namespace.NamespaceContext getNamespaceContext();

    public int getNamespaceCount();

    public String getNamespacePrefix(int index);

    public String getNamespaceURI();

    public String getNamespaceURI(int index);

    public String getNamespaceURI(String prefix);

    public String getPIData();

    public String getPITarget();

    public String getPrefix();

    public Object getProperty(java.lang.String name) throws java.lang.IllegalArgumentException;

    public String getText();

    public char[] getTextCharacters();

    public int getTextCharacters(int sourceStart, char[] target, int targetStart, int length) throws XMLStreamException;

    public int getTextLength();

    public int getTextStart();

    public String getVersion();

    public boolean hasName();

    public boolean hasText();

    public boolean isAttributeSpecified(int index);

    public boolean isCharacters();

    public boolean isEndElement();

    public boolean isStandalone();

    public boolean isStartElement();

    public boolean isWhiteSpace();

    public void require(int type, String namespaceURI, String localName) throws XMLStreamException;

    public boolean standaloneSet();

    public String getAttributeLocalName(int index);
}
