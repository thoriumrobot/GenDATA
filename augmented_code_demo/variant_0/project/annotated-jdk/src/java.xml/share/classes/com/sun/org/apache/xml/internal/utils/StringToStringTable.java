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
public class StringToStringTable {

    @Positive
    public StringToStringTable() {
    @Positive
    }

    @Positive
    public StringToStringTable(int blocksize) {
    @Positive
    }

    @Positive
    public final int getLength();

    @Positive
    public final void put(String key, String value);

    @Positive
    public final String get(String key);

    @Positive
    public final void remove(String key);

    @Positive
    public final String getIgnoreCase(String key);

    @Positive
    public final String getByValue(String val);

    @Positive
    public final String elementAt(int i);

    @Positive
    @Pure
    @Positive
    public final boolean contains(String key);

    @Positive
    @Pure
    @Positive
    public final boolean containsValue(String val);
    @Positive
}

// CFWR semantic augmentation - variant 0
