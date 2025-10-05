/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
    @Positive
 */
    @Positive
package com.sun.org.apache.xerces.internal.xs.datatypes;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSException;
    @Positive
import java.util.List;

    @Positive
public interface ByteList extends List<Byte> {

    @Positive
    public int getLength();

    @Positive
    @Pure
    @Positive
    public boolean contains(byte item);

    @Positive
    public byte item(int index) throws XSException;

    @Positive
    public byte[] toByteArray();
    @Positive
}
