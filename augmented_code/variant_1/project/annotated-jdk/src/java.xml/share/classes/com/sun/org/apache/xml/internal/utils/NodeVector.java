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
package com.sun.org.apache.xml.internal.utils;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.io.Serializable;
    @Positive
import com.sun.org.apache.xml.internal.dtm.DTM;

    @Positive
public class NodeVector implements Serializable, Cloneable {

    @Positive
    protected int m_firstFree;

    @Positive
    public NodeVector() {
    @Positive
    }

    @Positive
    public NodeVector(int blocksize) {
    @Positive
    }

    @Positive
    public Object clone() throws CloneNotSupportedException;

    @Positive
    public int size();

    @Positive
    public void addElement(int value);

    @Positive
    public final void push(int value);

    @Positive
    public final int pop();

    @Positive
    public final int popAndTop();

    @Positive
    public final void popQuick();

    @Positive
    public final int peepOrNull();

    @Positive
    public final void pushPair(int v1, int v2);

    @Positive
    public final void popPair();

    @Positive
    public final void setTail(int n);

    @Positive
    public final void setTailSub1(int n);

    @Positive
    public final int peepTail();

    @Positive
    public final int peepTailSub1();

    @Positive
    public void insertInOrder(int value);

    @Positive
    public void insertElementAt(int value, int at);

    @Positive
    public void appendNodes(NodeVector nodes);

    @Positive
    public void removeAllElements();

    @Positive
    public void RemoveAllNoClear();

    @Positive
    public boolean removeElement(int s);

    @Positive
    public void removeElementAt(int i);

    @Positive
    public void setElementAt(int node, int index);

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
    public void sort(int[] a, int lo0, int hi0) throws Exception;

    @Positive
    public void sort() throws Exception;
    @Positive
}
