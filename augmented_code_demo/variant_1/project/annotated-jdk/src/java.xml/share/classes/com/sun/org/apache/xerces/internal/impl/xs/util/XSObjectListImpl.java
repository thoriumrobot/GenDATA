/*
    @Positive
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
    @Positive
 */
    @Positive
package com.sun.org.apache.xerces.internal.impl.xs.util;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSObject;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSObjectList;
    @Positive
import java.lang.reflect.Array;
    @Positive
import java.util.AbstractList;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.ListIterator;
    @Positive
import java.util.NoSuchElementException;

    @Positive
@SuppressWarnings("unchecked")
    @Positive
public class XSObjectListImpl extends AbstractList<XSObject> implements XSObjectList {

    @Positive
    public static final XSObjectListImpl EMPTY_LIST;

    @Positive
    static class EmptyIterator implements ListIterator<XSObject> {

    @Positive
        @Pure
    @Positive
        public boolean hasNext();

    @Positive
        @SideEffectsOnly("this")
    @Positive
        public XSObject next();

    @Positive
        public boolean hasPrevious();

    @Positive
        public XSObject previous();

    @Positive
        public int nextIndex();

    @Positive
        public int previousIndex();

    @Positive
        public void remove();

    @Positive
        public void set(XSObject object);

    @Positive
        public void add(XSObject object);
    @Positive
    }

    @Positive
    public XSObjectListImpl() {
    @Positive
    }

    @Positive
    public XSObjectListImpl(XSObject[] array, int length) {
    @Positive
    }

    @Positive
    public int getLength();

    @Positive
    public XSObject item(int index);

    @Positive
    public void clearXSObjectList();

    @Positive
    public void addXSObject(XSObject object);

    @Positive
    public void addXSObject(int index, XSObject object);

    @Positive
    @Pure
    @Positive
    public boolean contains(Object value);

    @Positive
    public XSObject get(int index);

    @Positive
    public int size();

    @Positive
    public Iterator<XSObject> iterator();

    @Positive
    public ListIterator<XSObject> listIterator();

    @Positive
    public ListIterator<XSObject> listIterator(int index);

    @Positive
    public Object[] toArray();

    @Positive
    public Object[] toArray(Object[] a);

    @Positive
    private final class XSObjectListIterator implements ListIterator<XSObject> {

    @Positive
        public XSObjectListIterator(int index) {
    @Positive
        }

    @Positive
        @Pure
    @Positive
        public boolean hasNext();

    @Positive
        @SideEffectsOnly("this")
    @Positive
        public XSObject next();

    @Positive
        public boolean hasPrevious();

    @Positive
        public XSObject previous();

    @Positive
        public int nextIndex();

    @Positive
        public int previousIndex();

    @Positive
        public void remove();

    @Positive
        public void set(XSObject o);

    @Positive
        public void add(XSObject o);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
