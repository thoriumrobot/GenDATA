/*
 * reserved comment block
 * DO NOT REMOVE OR ALTER!
 */
package org.jcp.xml.dsig.internal.dom;

import javax.xml.crypto.NodeSetData;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.NoSuchElementException;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;

public class DOMSubTreeData implements NodeSetData<Node> {

    public DOMSubTreeData(Node root, boolean excludeComments) {
    }

    @Override
    public Iterator<Node> iterator();

    public Node getRoot();

    public boolean excludeComments();

    static class DelayedNodeIterator implements Iterator<Node> {

        @Pure
        public boolean hasNext();

        @SideEffectsOnly("this")
        public Node next();

        public void remove();
    }
}
