/*
 * reserved comment block
 * DO NOT REMOVE OR ALTER!
 */
package com.sun.org.apache.xpath.internal;

import org.checkerframework.dataflow.qual.Pure;
import com.sun.org.apache.xalan.internal.res.XSLMessages;
import com.sun.org.apache.xml.internal.dtm.DTM;
import com.sun.org.apache.xml.internal.dtm.DTMFilter;
import com.sun.org.apache.xml.internal.dtm.DTMIterator;
import com.sun.org.apache.xml.internal.dtm.DTMManager;
import com.sun.org.apache.xml.internal.utils.NodeVector;
import com.sun.org.apache.xpath.internal.res.XPATHErrorResources;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;
import org.w3c.dom.traversal.NodeIterator;

public class NodeSetDTM extends NodeVector implements DTMIterator, Cloneable {

    public NodeSetDTM(DTMManager dtmManager) {
    }

    public NodeSetDTM(int blocksize, int dummy, DTMManager dtmManager) {
    }

    public NodeSetDTM(NodeSetDTM nodelist) {
    }

    public NodeSetDTM(DTMIterator ni) {
    }

    public NodeSetDTM(NodeIterator iterator, XPathContext xctxt) {
    }

    public NodeSetDTM(NodeList nodeList, XPathContext xctxt) {
    }

    public NodeSetDTM(int node, DTMManager dtmManager) {
    }

    public void setEnvironment(Object environment);

    public int getRoot();

    public void setRoot(int context, Object environment);

    public Object clone() throws CloneNotSupportedException;

    public DTMIterator cloneWithReset() throws CloneNotSupportedException;

    public void reset();

    public int getWhatToShow();

    public DTMFilter getFilter();

    public boolean getExpandEntityReferences();

    public DTM getDTM(int nodeHandle);

    public DTMManager getDTMManager();

    public int nextNode();

    public int previousNode();

    public void detach();

    public void allowDetachToRelease(boolean allowRelease);

    public boolean isFresh();

    public void runTo(int index);

    public int item(int index);

    public int getLength();

    public void addNode(int n);

    public void insertNode(int n, int pos);

    public void removeNode(int n);

    public void addNodes(DTMIterator iterator);

    public void addNodesInDocOrder(DTMIterator iterator, XPathContext support);

    public int addNodeInDocOrder(int node, boolean test, XPathContext support);

    public int addNodeInDocOrder(int node, XPathContext support);

    public int size();

    public void addElement(int value);

    public void insertElementAt(int value, int at);

    public void appendNodes(NodeVector nodes);

    public void removeAllElements();

    public boolean removeElement(int s);

    public void removeElementAt(int i);

    public void setElementAt(int node, int index);

    public void setItem(int node, int index);

    public int elementAt(int i);

    @Pure
    public boolean contains(int s);

    public int indexOf(int elem, int index);

    public int indexOf(int elem);

    transient protected int m_next;

    public int getCurrentPos();

    public void setCurrentPos(int i);

    public int getCurrentNode();

    transient protected boolean m_mutable;

    transient protected boolean m_cacheNodes;

    protected int m_root;

    public boolean getShouldCacheNodes();

    public void setShouldCacheNodes(boolean b);

    public boolean isMutable();

    public int getLast();

    public void setLast(int last);

    public boolean isDocOrdered();

    public int getAxis();
}
