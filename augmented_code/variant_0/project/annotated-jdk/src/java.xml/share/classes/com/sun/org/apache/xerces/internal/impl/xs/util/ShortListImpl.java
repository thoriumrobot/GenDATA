/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
    @Positive
 */
    @Positive
package com.sun.org.apache.xerces.internal.impl.xs.util;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import com.sun.org.apache.xerces.internal.xs.ShortList;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSException;
    @Positive
import java.util.AbstractList;

    @Positive
public final class ShortListImpl extends AbstractList<Short> implements ShortList {

    @Positive
    public static final ShortListImpl EMPTY_LIST;

    @Positive
    public ShortListImpl(short[] array, int length) {
    @Positive
    }

    @Positive
    public int getLength();

    @Positive
    @Pure
    @Positive
    public boolean contains(short item);

    @Positive
    public short item(int index) throws XSException;

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public Short get(int index);

    @Positive
    public int size();
    @Positive
}
