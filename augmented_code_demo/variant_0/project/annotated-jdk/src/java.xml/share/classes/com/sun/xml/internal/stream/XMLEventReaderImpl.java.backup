/*
 * Copyright (c) 2005, 2018, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.xml.internal.stream;

import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import com.sun.xml.internal.stream.events.XMLEventAllocatorImpl;
import java.util.NoSuchElementException;
import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamConstants;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamReader;
import javax.xml.stream.events.EntityReference;
import javax.xml.stream.events.XMLEvent;
import javax.xml.stream.util.XMLEventAllocator;

public class XMLEventReaderImpl implements javax.xml.stream.XMLEventReader {

    protected XMLStreamReader fXMLReader;

    protected XMLEventAllocator fXMLEventAllocator;

    public XMLEventReaderImpl(XMLStreamReader reader) throws XMLStreamException {
    }

    @Pure
    public boolean hasNext();

    @SideEffectsOnly("this")
    public XMLEvent nextEvent() throws XMLStreamException;

    public void remove();

    public void close() throws XMLStreamException;

    public String getElementText() throws XMLStreamException;

    public Object getProperty(java.lang.String name) throws java.lang.IllegalArgumentException;

    public XMLEvent nextTag() throws XMLStreamException;

    public Object next();

    public XMLEvent peek() throws XMLStreamException;
}
