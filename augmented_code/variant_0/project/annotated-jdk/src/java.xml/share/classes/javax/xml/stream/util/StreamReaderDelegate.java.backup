/*
 * Copyright (c) 2009, Oracle and/or its affiliates. All rights reserved.
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
package javax.xml.stream.util;

import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import javax.xml.namespace.QName;
import javax.xml.namespace.NamespaceContext;
import javax.xml.stream.XMLStreamReader;
import javax.xml.stream.Location;
import javax.xml.stream.XMLStreamException;

public class StreamReaderDelegate implements XMLStreamReader {

    public StreamReaderDelegate() {
    }

    public StreamReaderDelegate(XMLStreamReader reader) {
    }

    public void setParent(XMLStreamReader reader);

    public XMLStreamReader getParent();

    public int next() throws XMLStreamException;

    public int nextTag() throws XMLStreamException;

    public String getElementText() throws XMLStreamException;

    public void require(int type, String namespaceURI, String localName) throws XMLStreamException;

    @Pure
    public boolean hasNext() throws XMLStreamException;

    public void close() throws XMLStreamException;

    public String getNamespaceURI(String prefix);

    public NamespaceContext getNamespaceContext();

    public boolean isStartElement();

    public boolean isEndElement();

    public boolean isCharacters();

    public boolean isWhiteSpace();

    public String getAttributeValue(String namespaceUri, String localName);

    public int getAttributeCount();

    public QName getAttributeName(int index);

    public String getAttributePrefix(int index);

    public String getAttributeNamespace(int index);

    public String getAttributeLocalName(int index);

    public String getAttributeType(int index);

    public String getAttributeValue(int index);

    public boolean isAttributeSpecified(int index);

    public int getNamespaceCount();

    public String getNamespacePrefix(int index);

    public String getNamespaceURI(int index);

    public int getEventType();

    public String getText();

    public int getTextCharacters(int sourceStart, char[] target, int targetStart, int length) throws XMLStreamException;

    public char[] getTextCharacters();

    public int getTextStart();

    public int getTextLength();

    public String getEncoding();

    public boolean hasText();

    public Location getLocation();

    public QName getName();

    public String getLocalName();

    public boolean hasName();

    public String getNamespaceURI();

    public String getPrefix();

    public String getVersion();

    public boolean isStandalone();

    public boolean standaloneSet();

    public String getCharacterEncodingScheme();

    public String getPITarget();

    public String getPIData();

    public Object getProperty(String name);
}
