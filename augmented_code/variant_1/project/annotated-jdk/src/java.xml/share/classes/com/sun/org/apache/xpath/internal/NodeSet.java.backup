/*
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
 */
package com.sun.org.apache.xpath.internal;

import org.checkerframework.dataflow.qual.Pure;
import com.sun.org.apache.xalan.internal.res.XSLMessages;
import com.sun.org.apache.xml.internal.utils.DOM2Helper;
import com.sun.org.apache.xpath.internal.axes.ContextNodeList;
import com.sun.org.apache.xpath.internal.res.XPATHErrorResources;
import org.w3c.dom.DOMException;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.w3c.dom.traversal.NodeFilter;
import org.w3c.dom.traversal.NodeIterator;

public class NodeSet implements NodeList, NodeIterator, Cloneable, ContextNodeList {

    public NodeSet() {
    }

    public NodeSet(int blocksize) {
    }

    public NodeSet(NodeList nodelist) {
    }

    public NodeSet(NodeSet nodelist) {
    }

    public NodeSet(NodeIterator ni) {
    }

    public NodeSet(Node node) {
    }

    public Node getRoot();

    public NodeIterator cloneWithReset() throws CloneNotSupportedException;

    public void reset();

    public int getWhatToShow();

    public NodeFilter getFilter();

    public boolean getExpandEntityReferences();

    public Node nextNode() throws DOMException;

    public Node previousNode() throws DOMException;

    public void detach();

    public boolean isFresh();

    public void runTo(int index);

    public Node item(int index);

    public int getLength();

    public void addNode(Node n);

    public void insertNode(Node n, int pos);

    public void removeNode(Node n);

    public void addNodes(NodeList nodelist);

    public void addNodes(NodeSet ns);

    public void addNodes(NodeIterator iterator);

    public void addNodesInDocOrder(NodeList nodelist, XPathContext support);

    public void addNodesInDocOrder(NodeIterator iterator, XPathContext support);

    public int addNodeInDocOrder(Node node, boolean test, XPathContext support);

    public int addNodeInDocOrder(Node node, XPathContext support);

    transient protected int m_next;

    public int getCurrentPos();

    public void setCurrentPos(int i);

    public Node getCurrentNode();

    transient protected boolean m_mutable;

    transient protected boolean m_cacheNodes;

    public boolean getShouldCacheNodes();

    public void setShouldCacheNodes(boolean b);

    public int getLast();

    public void setLast(int last);

    protected int m_firstFree;

    public Object clone() throws CloneNotSupportedException;

    public int size();

    public void addElement(Node value);

    public final void push(Node value);

    public final Node pop();

    public final Node popAndTop();

    public final void popQuick();

    public final Node peepOrNull();

    public final void pushPair(Node v1, Node v2);

    public final void popPair();

    public final void setTail(Node n);

    public final void setTailSub1(Node n);

    public final Node peepTail();

    public final Node peepTailSub1();

    public void insertElementAt(Node value, int at);

    public void appendNodes(NodeSet nodes);

    public void removeAllElements();

    public boolean removeElement(Node s);

    public void removeElementAt(int i);

    public void setElementAt(Node node, int index);

    public Node elementAt(int i);

    @Pure
    public boolean contains(Node s);

    public int indexOf(Node elem, int index);

    public int indexOf(Node elem);
}
