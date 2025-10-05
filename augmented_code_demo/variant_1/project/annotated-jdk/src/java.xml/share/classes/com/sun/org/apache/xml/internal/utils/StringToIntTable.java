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
public class StringToIntTable {

    @Positive
    public static final int INVALID_KEY;

    @Positive
    public StringToIntTable() {
    @Positive
    }

    @Positive
    public StringToIntTable(int blocksize) {
    @Positive
    }

    @Positive
    public final int getLength();

    @Positive
    public final void put(String key, int value);

    @Positive
    public final int get(String key);

    @Positive
    public final int getIgnoreCase(String key);

    @Positive
    @Pure
    @Positive
    public final boolean contains(String key);

    @Positive
    public final String[] keys();
    @Positive
}

// CFWR semantic augmentation - variant 1
