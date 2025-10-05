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
public class StringToStringTableVector {

    @Positive
    public StringToStringTableVector() {
    @Positive
    }

    @Positive
    public StringToStringTableVector(int blocksize) {
    @Positive
    }

    @Positive
    public final int getLength();

    @Positive
    public final int size();

    @Positive
    public final void addElement(StringToStringTable value);

    @Positive
    public final String get(String key);

    @Positive
    @Pure
    @Positive
    public final boolean containsKey(String key);

    @Positive
    public final void removeLastElem();

    @Positive
    public final StringToStringTable elementAt(int i);

    @Positive
    @Pure
    @Positive
    public final boolean contains(StringToStringTable s);
    @Positive
}

// CFWR semantic augmentation - variant 1
