/*
 * reserved comment block
 * DO NOT REMOVE OR ALTER!
 */
package com.sun.org.apache.xml.internal.utils;

import org.checkerframework.dataflow.qual.Pure;

public class StringVector implements java.io.Serializable {

    protected int m_blocksize;

    protected String[] m_map;

    protected int m_firstFree;

    protected int m_mapSize;

    public StringVector() {
    }

    public StringVector(int blocksize) {
    }

    public int getLength();

    public final int size();

    public final void addElement(String value);

    public final String elementAt(int i);

    @Pure
    public final boolean contains(String s);

    @Pure
    public final boolean containsIgnoreCase(String s);

    public final void push(String s);

    public final String pop();

    public final String peek();
}
