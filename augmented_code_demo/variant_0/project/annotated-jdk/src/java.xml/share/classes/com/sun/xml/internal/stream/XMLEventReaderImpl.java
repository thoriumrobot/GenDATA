/*
    @Positive
 * Copyright (c) 2005, 2018, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.xml.internal.stream;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import com.sun.xml.internal.stream.events.XMLEventAllocatorImpl;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import javax.xml.stream.XMLInputFactory;
    @Positive
import javax.xml.stream.XMLStreamConstants;
    @Positive
import javax.xml.stream.XMLStreamException;
    @Positive
import javax.xml.stream.XMLStreamReader;
    @Positive
import javax.xml.stream.events.EntityReference;
    @Positive
import javax.xml.stream.events.XMLEvent;
    @Positive
import javax.xml.stream.util.XMLEventAllocator;

    @Positive
public class XMLEventReaderImpl implements javax.xml.stream.XMLEventReader {

    @Positive
    protected XMLStreamReader fXMLReader;

    @Positive
    protected XMLEventAllocator fXMLEventAllocator;

    @Positive
    public XMLEventReaderImpl(XMLStreamReader reader) throws XMLStreamException {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    public boolean hasNext();

    @Positive
    @SideEffectsOnly("this")
    @Positive
    public XMLEvent nextEvent() throws XMLStreamException;

    @Positive
    public void remove();

    @Positive
    public void close() throws XMLStreamException;

    @Positive
    public String getElementText() throws XMLStreamException;

    @Positive
    public Object getProperty(java.lang.String name) throws java.lang.IllegalArgumentException;

    @Positive
    public XMLEvent nextTag() throws XMLStreamException;

    @Positive
    public Object next();

    @Positive
    public XMLEvent peek() throws XMLStreamException;
    @Positive
}

// CFWR semantic augmentation - variant 0
