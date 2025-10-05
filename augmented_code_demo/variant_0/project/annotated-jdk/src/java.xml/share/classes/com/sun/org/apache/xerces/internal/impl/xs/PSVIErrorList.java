/*
    @Positive
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
    @Positive
 */
    @Positive
package com.sun.org.apache.xerces.internal.impl.xs;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import com.sun.org.apache.xerces.internal.xs.StringList;
    @Positive
import java.util.AbstractList;

    @Positive
final class PSVIErrorList extends AbstractList<String> implements StringList {

    @Positive
    public PSVIErrorList(String[] array, boolean even) {
    @Positive
    }

    @Positive
    @Pure
    @Positive
    public boolean contains(String item);

    @Positive
    public int getLength();

    @Positive
    public String item(int index);

    @Positive
    public String get(int index);

    @Positive
    public int size();
    @Positive
}

// CFWR semantic augmentation - variant 0
