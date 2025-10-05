/*
    @Positive
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
    @Positive
 */
    @Positive
package com.sun.org.apache.xerces.internal.impl.dv.util;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSException;
    @Positive
import com.sun.org.apache.xerces.internal.xs.datatypes.ByteList;
    @Positive
import java.util.AbstractList;

    @Positive
public class ByteListImpl extends AbstractList<Byte> implements ByteList {

    @Positive
    protected final byte[] data;

    @Positive
    protected String canonical;

    @Positive
    public ByteListImpl(byte[] data) {
    @Positive
    }

    @Positive
    public int getLength();

    @Positive
    @Pure
    @Positive
    public boolean contains(byte item);

    @Positive
    public byte item(int index) throws XSException;

    @Positive
    public Byte get(int index);

    @Positive
    public int size();

    @Positive
    public byte[] toByteArray();
    @Positive
}

// CFWR semantic augmentation - variant 1
