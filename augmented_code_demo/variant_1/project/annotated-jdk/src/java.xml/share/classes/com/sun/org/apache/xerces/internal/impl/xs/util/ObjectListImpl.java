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
import com.sun.org.apache.xerces.internal.xs.datatypes.ObjectList;
    @Positive
import java.lang.reflect.Array;
    @Positive
import java.util.AbstractList;

    @Positive
@SuppressWarnings("unchecked")
    @Positive
public final class ObjectListImpl extends AbstractList<Object> implements ObjectList {

    @Positive
    public static final ObjectListImpl EMPTY_LIST;

    @Positive
    public ObjectListImpl(Object[] array, int length) {
    @Positive
    }

    @Positive
    public int getLength();

    @Positive
    @Pure
    @Positive
    public boolean contains(Object item);

    @Positive
    public Object item(int index);

    @Positive
    public Object get(int index);

    @Positive
    public int size();

    @Positive
    public Object[] toArray();

    @Positive
    public Object[] toArray(Object[] a);
    @Positive
}

// CFWR semantic augmentation - variant 1
