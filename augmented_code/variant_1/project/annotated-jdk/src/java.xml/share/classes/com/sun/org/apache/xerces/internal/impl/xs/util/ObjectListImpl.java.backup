/*
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
 */
package com.sun.org.apache.xerces.internal.impl.xs.util;

import org.checkerframework.dataflow.qual.Pure;
import com.sun.org.apache.xerces.internal.xs.datatypes.ObjectList;
import java.lang.reflect.Array;
import java.util.AbstractList;

@SuppressWarnings("unchecked")
public final class ObjectListImpl extends AbstractList<Object> implements ObjectList {

    public static final ObjectListImpl EMPTY_LIST;

    public ObjectListImpl(Object[] array, int length) {
    }

    public int getLength();

    @Pure
    public boolean contains(Object item);

    public Object item(int index);

    public Object get(int index);

    public int size();

    public Object[] toArray();

    public Object[] toArray(Object[] a);
}
