/*
    @Positive
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
    @Positive
 */
    @Positive
package com.sun.org.apache.xerces.internal.impl.xs.util;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import com.sun.org.apache.xerces.internal.xs.StringList;
    @Positive
import java.lang.reflect.Array;
    @Positive
import java.util.AbstractList;
    @Positive
import java.util.List;

    @Positive
@SuppressWarnings("unchecked")
    @Positive
public final class StringListImpl extends AbstractList<String> implements StringList {

    @Positive
    public static final StringListImpl EMPTY_LIST;

    @Positive
    public StringListImpl(List<String> v) {
    @Positive
    }

    @Positive
    public StringListImpl(String[] array, int length) {
    @Positive
    }

    @Positive
    public int getLength();

    @Positive
    @Pure
    @Positive
    public boolean contains(String item);

    @Positive
    public String item(int index);

    @Positive
    public String get(int index);

    @Positive
    public int size();

    @Positive
    public Object[] toArray();

    @Positive
    public Object[] toArray(Object[] a);
    @Positive
}

// CFWR semantic augmentation - variant 1
