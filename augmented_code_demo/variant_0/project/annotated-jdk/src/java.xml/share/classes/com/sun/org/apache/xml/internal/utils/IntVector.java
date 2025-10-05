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
public class IntVector implements Cloneable {

    @Positive
    protected int m_blocksize;

    @Positive
    protected int[] m_map;

    @Positive
    protected int m_firstFree;

    @Positive
    protected int m_mapSize;

    @Positive
    public IntVector() {
    @Positive
    }

    @Positive
    public IntVector(int blocksize) {
    @Positive
    }

    @Positive
    public IntVector(int blocksize, int increaseSize) {
    @Positive
    }

    @Positive
    public IntVector(IntVector v) {
    @Positive
    }

    @Positive
    public final int size();

    @Positive
    public final void setSize(int sz);

    @Positive
    public final void addElement(int value);

    @Positive
    public final void addElements(int value, int numberOfElements);

    @Positive
    public final void addElements(int numberOfElements);

    @Positive
    public final void insertElementAt(int value, int at);

    @Positive
    public final void removeAllElements();

    @Positive
    public final boolean removeElement(int s);

    @Positive
    public final void removeElementAt(int i);

    @Positive
    public final void setElementAt(int value, int index);

    @Positive
    public final int elementAt(int i);

    @Positive
    @Pure
    @Positive
    public final boolean contains(int s);

    @Positive
    public final int indexOf(int elem, int index);

    @Positive
    public final int indexOf(int elem);

    @Positive
    public final int lastIndexOf(int elem);

    @Positive
    public Object clone() throws CloneNotSupportedException;
    @Positive
}

// CFWR semantic augmentation - variant 0
