/*
    @Positive
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
    @Positive
 */
    @Positive
package com.sun.org.apache.xml.internal.serializer.dom3;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.List;
    @Positive
import org.w3c.dom.DOMStringList;

    @Positive
final class DOMStringListImpl implements DOMStringList {

    @Positive
    public String item(int index);

    @Positive
    public int getLength();

    @Positive
    @Pure
    @Positive
    public boolean contains(String param);

    @Positive
    public void add(String param);
    @Positive
}

// CFWR semantic augmentation - variant 1
