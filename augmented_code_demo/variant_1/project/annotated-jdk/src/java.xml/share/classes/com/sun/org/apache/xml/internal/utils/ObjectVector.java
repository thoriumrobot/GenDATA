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
public class ObjectVector implements Cloneable {

    @Positive
    protected int m_blocksize;

    @Positive
    protected Object[] m_map;

    @Positive
    protected int m_firstFree;

    @Positive
    protected int m_mapSize;

    @Positive
    public ObjectVector() {
    @Positive
    }

    @Positive
    public ObjectVector(int blocksize) {
    @Positive
    }

    @Positive
    public ObjectVector(int blocksize, int increaseSize) {
    @Positive
    }

    @Positive
    public ObjectVector(ObjectVector v) {
    @Positive
    }

    @Positive
    public final int size();

    @Positive
    public final void setSize(int sz);

    @Positive
    public final void addElement(Object value);

    @Positive
    public final void addElements(Object value, int numberOfElements);

    @Positive
    public final void addElements(int numberOfElements);

    @Positive
    public final void insertElementAt(Object value, int at);

    @Positive
    public final void removeAllElements();

    @Positive
    public final boolean removeElement(Object s);

    @Positive
    public final void removeElementAt(int i);

    @Positive
    public final void setElementAt(Object value, int index);

    @Positive
    public final Object elementAt(int i);

    @Positive
    @Pure
    @Positive
    public final boolean contains(Object s);

    @Positive
    public final int indexOf(Object elem, int index);

    @Positive
    public final int indexOf(Object elem);

    @Positive
    public final int lastIndexOf(Object elem);

    @Positive
    public final void setToSize(int size);

    @Positive
    public Object clone() throws CloneNotSupportedException;
    @Positive
}

// CFWR semantic augmentation - variant 1
