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
import java.util.List;

    @Positive
public interface ObjectList extends List<Object> {

    @Positive
    public int getLength();

    @Positive
    @Pure
    @Positive
    public boolean contains(Object item);

    @Positive
    public Object item(int index);
    @Positive
}
