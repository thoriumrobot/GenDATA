/*
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
 */
package com.sun.org.apache.xerces.internal.impl.dv.xs;

import org.checkerframework.dataflow.qual.Pure;
import com.sun.org.apache.xerces.internal.impl.dv.InvalidDatatypeValueException;
import com.sun.org.apache.xerces.internal.impl.dv.ValidationContext;
import com.sun.org.apache.xerces.internal.xs.datatypes.ObjectList;
import java.util.AbstractList;

public class ListDV extends TypeValidator {

    public short getAllowedFacets();

    public Object getActualValue(String content, ValidationContext context) throws InvalidDatatypeValueException;

    public int getDataLength(Object value);

    final static class ListData extends AbstractList<Object> implements ObjectList {

        public ListData(Object[] data) {
        }

        public synchronized String toString();

        public int getLength();

        public boolean equals(Object obj);

        public int hashCode();

        @Pure
        public boolean contains(Object item);

        public Object item(int index);

        public Object get(int index);

        public int size();
    }
}
