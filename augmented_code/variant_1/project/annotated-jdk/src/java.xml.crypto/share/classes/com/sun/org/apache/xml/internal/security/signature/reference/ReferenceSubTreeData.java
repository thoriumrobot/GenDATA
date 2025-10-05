/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * reserved comment block
    @Positive
 * DO NOT REMOVE OR ALTER!
    @Positive
 */
    @Positive
package com.sun.org.apache.xml.internal.security.signature.reference;

    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import java.util.ListIterator;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import org.w3c.dom.NamedNodeMap;
    @Positive
import org.w3c.dom.Node;

    @Positive
public class ReferenceSubTreeData implements ReferenceNodeSetData {

    @Positive
    public ReferenceSubTreeData(Node root, boolean excludeComments) {
    @Positive
    }

    @Positive
    public Iterator<Node> iterator();

    @Positive
    public Node getRoot();

    @Positive
    public boolean excludeComments();

    @Positive
    static class DelayedNodeIterator implements Iterator<Node> {

    @Positive
        @Pure
    @Positive
        public boolean hasNext();

    @Positive
        @SideEffectsOnly("this")
    @Positive
        public Node next();

    @Positive
        public void remove();
    @Positive
    }
    @Positive
}
