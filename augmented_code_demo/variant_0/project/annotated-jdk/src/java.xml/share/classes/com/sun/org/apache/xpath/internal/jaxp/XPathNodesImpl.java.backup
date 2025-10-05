/*
 * Copyright (c) 2015, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.org.apache.xpath.internal.jaxp;

import java.util.Iterator;
import javax.xml.xpath.XPathException;
import javax.xml.xpath.XPathNodes;
import javax.xml.xpath.XPathEvaluationResult.XPathResultType;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

public class XPathNodesImpl implements XPathNodes {

    public XPathNodesImpl(NodeList nodeList, Class<Node> elementType) {
    }

    @Override
    public Iterator<Node> iterator();

    class NodeSetIterator<E> implements Iterator<E> {

        @Pure
        public boolean hasNext();

        @SideEffectsOnly("this")
        public E next();
    }

    @Override
    public int size();

    @Override
    public Node get(int index) throws XPathException;
}
