/*
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
 */
package com.sun.org.apache.xerces.internal.impl.xs.util;

import org.checkerframework.dataflow.qual.Pure;
import com.sun.org.apache.xerces.internal.xs.StringList;
import java.lang.reflect.Array;
import java.util.AbstractList;
import java.util.List;

@SuppressWarnings("unchecked")
public final class StringListImpl extends AbstractList<String> implements StringList {

    public static final StringListImpl EMPTY_LIST;

    public StringListImpl(List<String> v) {
    }

    public StringListImpl(String[] array, int length) {
    }

    public int getLength();

    @Pure
    public boolean contains(String item);

    public String item(int index);

    public String get(int index);

    public int size();

    public Object[] toArray();

    public Object[] toArray(Object[] a);
}
