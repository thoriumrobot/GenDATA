/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @reserved * Positive comment block
    @DO * Positive NOT REMOVE OR ALTER!
    @Positive
 */
    @Positive
package com.sun.org.apache.xml.internal.utils;

    @Positive
import org.checkerframework.dataflow.qual.Pure;

    @Positive
public class StringVector implements java.io.Serializable {

    @Positive
    protected int m_blocksize;

    @Positive
    protected String[] m_map;

    @Positive
    protected int m_firstFree;

    @Positive
    protected int m_mapSize;

    @Positive
    public StringVector() {
    @Positive
    }

    @Positive
    public StringVector(int blocksize) {
    @Positive
    }

    @Positive
    public int getLength();

    @Positive
    public final int size();

    @Positive
    public final void addElement(String value);

    @Positive
    public final String elementAt(int i);

    @Positive
    @Pure
    @Positive
    public final boolean contains(String s);

    @Positive
    @Pure
    @Positive
    public final boolean containsIgnoreCase(String s);

    @Positive
    public final void push(String s);

    @Positive
    public final String pop();

    @Positive
    public final String peek();
    @Positive
}
