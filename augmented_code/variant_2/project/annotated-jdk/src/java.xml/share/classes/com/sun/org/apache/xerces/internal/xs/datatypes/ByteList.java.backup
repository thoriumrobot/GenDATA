/*
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
 */
package com.sun.org.apache.xerces.internal.xs.datatypes;

import org.checkerframework.dataflow.qual.Pure;
import com.sun.org.apache.xerces.internal.xs.XSException;
import java.util.List;

public interface ByteList extends List<Byte> {

    public int getLength();

    @Pure
    public boolean contains(byte item);

    public byte item(int index) throws XSException;

    public byte[] toByteArray();
}
