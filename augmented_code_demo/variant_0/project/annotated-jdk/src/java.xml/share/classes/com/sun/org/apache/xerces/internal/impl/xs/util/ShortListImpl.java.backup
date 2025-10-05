/*
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
 */
package com.sun.org.apache.xerces.internal.impl.xs.util;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import com.sun.org.apache.xerces.internal.xs.ShortList;
import com.sun.org.apache.xerces.internal.xs.XSException;
import java.util.AbstractList;

public final class ShortListImpl extends AbstractList<Short> implements ShortList {

    public static final ShortListImpl EMPTY_LIST;

    public ShortListImpl(short[] array, int length) {
    }

    public int getLength();

    @Pure
    public boolean contains(short item);

    public short item(int index) throws XSException;

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public Short get(int index);

    public int size();
}
