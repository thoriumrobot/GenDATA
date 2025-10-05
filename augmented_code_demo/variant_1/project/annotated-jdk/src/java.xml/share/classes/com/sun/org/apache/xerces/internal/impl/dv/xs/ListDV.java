/*
    @Positive
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
    @Positive
 */
    @Positive
package com.sun.org.apache.xerces.internal.impl.dv.xs;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import com.sun.org.apache.xerces.internal.impl.dv.InvalidDatatypeValueException;
    @Positive
import com.sun.org.apache.xerces.internal.impl.dv.ValidationContext;
    @Positive
import com.sun.org.apache.xerces.internal.xs.datatypes.ObjectList;
    @Positive
import java.util.AbstractList;

    @Positive
public class ListDV extends TypeValidator {

    @Positive
    public short getAllowedFacets();

    @Positive
    public Object getActualValue(String content, ValidationContext context) throws InvalidDatatypeValueException;

    @Positive
    public int getDataLength(Object value);

    @Positive
    final static class ListData extends AbstractList<Object> implements ObjectList {

    @Positive
        public ListData(Object[] data) {
    @Positive
        }

    @Positive
        public synchronized String toString();

    @Positive
        public int getLength();

    @Positive
        public boolean equals(Object obj);

    @Positive
        public int hashCode();

    @Positive
        @Pure
    @Positive
        public boolean contains(Object item);

    @Positive
        public Object item(int index);

    @Positive
        public Object get(int index);

    @Positive
        public int size();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
