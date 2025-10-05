/*
 * reserved comment block
 * DO NOT REMOVE OR ALTER!
 */
package com.sun.org.apache.xml.internal.utils;

import org.checkerframework.dataflow.qual.Pure;

public class ObjectVector implements Cloneable {

    protected int m_blocksize;

    protected Object[] m_map;

    protected int m_firstFree;

    protected int m_mapSize;

    public ObjectVector() {
    }

    public ObjectVector(int blocksize) {
    }

    public ObjectVector(int blocksize, int increaseSize) {
    }

    public ObjectVector(ObjectVector v) {
    }

    public final int size();

    public final void setSize(int sz);

    public final void addElement(Object value);

    public final void addElements(Object value, int numberOfElements);

    public final void addElements(int numberOfElements);

    public final void insertElementAt(Object value, int at);

    public final void removeAllElements();

    public final boolean removeElement(Object s);

    public final void removeElementAt(int i);

    public final void setElementAt(Object value, int index);

    public final Object elementAt(int i);

    @Pure
    public final boolean contains(Object s);

    public final int indexOf(Object elem, int index);

    public final int indexOf(Object elem);

    public final int lastIndexOf(Object elem);

    public final void setToSize(int size);

    public Object clone() throws CloneNotSupportedException;
}
