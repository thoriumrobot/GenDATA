/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
    @Positive
 */
    @Positive
package com.sun.org.apache.xpath.internal;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import com.sun.org.apache.xalan.internal.res.XSLMessages;
    @Positive
import com.sun.org.apache.xml.internal.utils.DOM2Helper;
    @Positive
import com.sun.org.apache.xpath.internal.axes.ContextNodeList;
    @Positive
import com.sun.org.apache.xpath.internal.res.XPATHErrorResources;
    @Positive
import org.w3c.dom.DOMException;
    @Positive
import org.w3c.dom.Node;
    @Positive
import org.w3c.dom.NodeList;
    @Positive
import org.w3c.dom.traversal.NodeFilter;
    @Positive
import org.w3c.dom.traversal.NodeIterator;

    @Positive
public class NodeSet implements NodeList, NodeIterator, Cloneable, ContextNodeList {

    @Positive
    public NodeSet() {
    @Positive
    }

    @Positive
    public NodeSet(int blocksize) {
    @Positive
    }

    @Positive
    public NodeSet(NodeList nodelist) {
    @Positive
    }

    @Positive
    public NodeSet(NodeSet nodelist) {
    @Positive
    }

    @Positive
    public NodeSet(NodeIterator ni) {
    @Positive
    }

    @Positive
    public NodeSet(Node node) {
    @Positive
    }

    @Positive
    public Node getRoot();

    @Positive
    public NodeIterator cloneWithReset() throws CloneNotSupportedException;

    @Positive
    public void reset();

    @Positive
    public int getWhatToShow();

    @Positive
    public NodeFilter getFilter();

    @Positive
    public boolean getExpandEntityReferences();

    @Positive
    public Node nextNode() throws DOMException;

    @Positive
    public Node previousNode() throws DOMException;

    @Positive
    public void detach();

    @Positive
    public boolean isFresh();

    @Positive
    public void runTo(int index);

    @Positive
    public Node item(int index);

    @Positive
    public int getLength();

    @Positive
    public void addNode(Node n);

    @Positive
    public void insertNode(Node n, int pos);

    @Positive
    public void removeNode(Node n);

    @Positive
    public void addNodes(NodeList nodelist);

    @Positive
    public void addNodes(NodeSet ns);

    @Positive
    public void addNodes(NodeIterator iterator);

    @Positive
    public void addNodesInDocOrder(NodeList nodelist, XPathContext support);

    @Positive
    public void addNodesInDocOrder(NodeIterator iterator, XPathContext support);

    @Positive
    public int addNodeInDocOrder(Node node, boolean test, XPathContext support);

    @Positive
    public int addNodeInDocOrder(Node node, XPathContext support);

    @Positive
    transient protected int m_next;

    @Positive
    public int getCurrentPos();

    @Positive
    public void setCurrentPos(int i);

    @Positive
    public Node getCurrentNode();

    @Positive
    transient protected boolean m_mutable;

    @Positive
    transient protected boolean m_cacheNodes;

    @Positive
    public boolean getShouldCacheNodes();

    @Positive
    public void setShouldCacheNodes(boolean b);

    @Positive
    public int getLast();

    @Positive
    public void setLast(int last);

    @Positive
    protected int m_firstFree;

    @Positive
    public Object clone() throws CloneNotSupportedException;

    @Positive
    public int size();

    @Positive
    public void addElement(Node value);

    @Positive
    public final void push(Node value);

    @Positive
    public final Node pop();

    @Positive
    public final Node popAndTop();

    @Positive
    public final void popQuick();

    @Positive
    public final Node peepOrNull();

    @Positive
    public final void pushPair(Node v1, Node v2);

    @Positive
    public final void popPair();

    @Positive
    public final void setTail(Node n);

    @Positive
    public final void setTailSub1(Node n);

    @Positive
    public final Node peepTail();

    @Positive
    public final Node peepTailSub1();

    @Positive
    public void insertElementAt(Node value, int at);

    @Positive
    public void appendNodes(NodeSet nodes);

    @Positive
    public void removeAllElements();

    @Positive
    public boolean removeElement(Node s);

    @Positive
    public void removeElementAt(int i);

    @Positive
    public void setElementAt(Node node, int index);

    @Positive
    public Node elementAt(int i);

    @Positive
    @Pure
    @Positive
    public boolean contains(Node s);

    @Positive
    public int indexOf(Node elem, int index);

    @Positive
    public int indexOf(Node elem);
    @Positive
}
