/*
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
 */
package com.sun.org.apache.xerces.internal.impl.dv.util;

import org.checkerframework.dataflow.qual.Pure;
import com.sun.org.apache.xerces.internal.xs.XSException;
import com.sun.org.apache.xerces.internal.xs.datatypes.ByteList;
import java.util.AbstractList;

public class ByteListImpl extends AbstractList<Byte> implements ByteList {

    protected final byte[] data;

    protected String canonical;

    public ByteListImpl(byte[] data) {
    }

    public int getLength();

    @Pure
    public boolean contains(byte item);

    public byte item(int index) throws XSException;

    public Byte get(int index);

    public int size();

    public byte[] toByteArray();
}
