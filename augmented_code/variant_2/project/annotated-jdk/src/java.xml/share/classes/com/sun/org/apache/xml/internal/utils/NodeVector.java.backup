/*
 * reserved comment block
 * DO NOT REMOVE OR ALTER!
 */
package com.sun.org.apache.xml.internal.utils;

import org.checkerframework.dataflow.qual.Pure;
import java.io.Serializable;
import com.sun.org.apache.xml.internal.dtm.DTM;

public class NodeVector implements Serializable, Cloneable {

    protected int m_firstFree;

    public NodeVector() {
    }

    public NodeVector(int blocksize) {
    }

    public Object clone() throws CloneNotSupportedException;

    public int size();

    public void addElement(int value);

    public final void push(int value);

    public final int pop();

    public final int popAndTop();

    public final void popQuick();

    public final int peepOrNull();

    public final void pushPair(int v1, int v2);

    public final void popPair();

    public final void setTail(int n);

    public final void setTailSub1(int n);

    public final int peepTail();

    public final int peepTailSub1();

    public void insertInOrder(int value);

    public void insertElementAt(int value, int at);

    public void appendNodes(NodeVector nodes);

    public void removeAllElements();

    public void RemoveAllNoClear();

    public boolean removeElement(int s);

    public void removeElementAt(int i);

    public void setElementAt(int node, int index);

    public int elementAt(int i);

    @Pure
    public boolean contains(int s);

    public int indexOf(int elem, int index);

    public int indexOf(int elem);

    public void sort(int[] a, int lo0, int hi0) throws Exception;

    public void sort() throws Exception;
}
