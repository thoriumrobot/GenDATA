/*
    @Positive
 * reserved comment block
    @Positive
 * DO NOT REMOVE OR ALTER!
    @Positive
 */
    @Positive
package com.sun.org.apache.xpath.internal;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import com.sun.org.apache.xalan.internal.res.XSLMessages;
    @Positive
import com.sun.org.apache.xml.internal.dtm.DTM;
    @Positive
import com.sun.org.apache.xml.internal.dtm.DTMFilter;
    @Positive
import com.sun.org.apache.xml.internal.dtm.DTMIterator;
    @Positive
import com.sun.org.apache.xml.internal.dtm.DTMManager;
    @Positive
import com.sun.org.apache.xml.internal.utils.NodeVector;
    @Positive
import com.sun.org.apache.xpath.internal.res.XPATHErrorResources;
    @Positive
import org.w3c.dom.Node;
    @Positive
import org.w3c.dom.NodeList;
    @Positive
import org.w3c.dom.traversal.NodeIterator;

    @Positive
public class NodeSetDTM extends NodeVector implements DTMIterator, Cloneable {

    @Positive
    public NodeSetDTM(DTMManager dtmManager) {
    @Positive
    }

    @Positive
    public NodeSetDTM(int blocksize, int dummy, DTMManager dtmManager) {
    @Positive
    }

    @Positive
    public NodeSetDTM(NodeSetDTM nodelist) {
    @Positive
    }

    @Positive
    public NodeSetDTM(DTMIterator ni) {
    @Positive
    }

    @Positive
    public NodeSetDTM(NodeIterator iterator, XPathContext xctxt) {
    @Positive
    }

    @Positive
    public NodeSetDTM(NodeList nodeList, XPathContext xctxt) {
    @Positive
    }

    @Positive
    public NodeSetDTM(int node, DTMManager dtmManager) {
    @Positive
    }

    @Positive
    public void setEnvironment(Object environment);

    @Positive
    public int getRoot();

    @Positive
    public void setRoot(int context, Object environment);

    @Positive
    public Object clone() throws CloneNotSupportedException;

    @Positive
    public DTMIterator cloneWithReset() throws CloneNotSupportedException;

    @Positive
    public void reset();

    @Positive
    public int getWhatToShow();

    @Positive
    public DTMFilter getFilter();

    @Positive
    public boolean getExpandEntityReferences();

    @Positive
    public DTM getDTM(int nodeHandle);

    @Positive
    public DTMManager getDTMManager();

    @Positive
    public int nextNode();

    @Positive
    public int previousNode();

    @Positive
    public void detach();

    @Positive
    public void allowDetachToRelease(boolean allowRelease);

    @Positive
    public boolean isFresh();

    @Positive
    public void runTo(int index);

    @Positive
    public int item(int index);

    @Positive
    public int getLength();

    @Positive
    public void addNode(int n);

    @Positive
    public void insertNode(int n, int pos);

    @Positive
    public void removeNode(int n);

    @Positive
    public void addNodes(DTMIterator iterator);

    @Positive
    public void addNodesInDocOrder(DTMIterator iterator, XPathContext support);

    @Positive
    public int addNodeInDocOrder(int node, boolean test, XPathContext support);

    @Positive
    public int addNodeInDocOrder(int node, XPathContext support);

    @Positive
    public int size();

    @Positive
    public void addElement(int value);

    @Positive
    public void insertElementAt(int value, int at);

    @Positive
    public void appendNodes(NodeVector nodes);

    @Positive
    public void removeAllElements();

    @Positive
    public boolean removeElement(int s);

    @Positive
    public void removeElementAt(int i);

    @Positive
    public void setElementAt(int node, int index);

    @Positive
    public void setItem(int node, int index);

    @Positive
    public int elementAt(int i);

    @Positive
    @Pure
    @Positive
    public boolean contains(int s);

    @Positive
    public int indexOf(int elem, int index);

    @Positive
    public int indexOf(int elem);

    @Positive
    transient protected int m_next;

    @Positive
    public int getCurrentPos();

    @Positive
    public void setCurrentPos(int i);

    @Positive
    public int getCurrentNode();

    @Positive
    transient protected boolean m_mutable;

    @Positive
    transient protected boolean m_cacheNodes;

    @Positive
    protected int m_root;

    @Positive
    public boolean getShouldCacheNodes();

    @Positive
    public void setShouldCacheNodes(boolean b);

    @Positive
    public boolean isMutable();

    @Positive
    public int getLast();

    @Positive
    public void setLast(int last);

    @Positive
    public boolean isDocOrdered();

    @Positive
    public int getAxis();
    @Positive
}

// CFWR semantic augmentation - variant 1
