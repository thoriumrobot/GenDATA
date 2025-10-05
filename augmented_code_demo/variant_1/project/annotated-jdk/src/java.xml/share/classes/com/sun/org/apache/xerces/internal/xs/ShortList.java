/*
    @Positive
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
    @Positive
 */
    @Positive
package com.sun.org.apache.xerces.internal.xs;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.util.List;

    @Positive
public interface ShortList extends List<Short> {

    @Positive
    public int getLength();

    @Positive
    @Pure
    @Positive
    public boolean contains(short item);

    @Positive
    public short item(int index) throws XSException;
    @Positive
}

// CFWR semantic augmentation - variant 1
