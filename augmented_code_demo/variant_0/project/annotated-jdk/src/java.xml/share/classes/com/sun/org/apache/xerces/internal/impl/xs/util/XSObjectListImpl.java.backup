/*
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
 */
package com.sun.org.apache.xerces.internal.impl.xs.util;

import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import com.sun.org.apache.xerces.internal.xs.XSObject;
import com.sun.org.apache.xerces.internal.xs.XSObjectList;
import java.lang.reflect.Array;
import java.util.AbstractList;
import java.util.Iterator;
import java.util.ListIterator;
import java.util.NoSuchElementException;

@SuppressWarnings("unchecked")
public class XSObjectListImpl extends AbstractList<XSObject> implements XSObjectList {

    public static final XSObjectListImpl EMPTY_LIST;

    static class EmptyIterator implements ListIterator<XSObject> {

        @Pure
        public boolean hasNext();

        @SideEffectsOnly("this")
        public XSObject next();

        public boolean hasPrevious();

        public XSObject previous();

        public int nextIndex();

        public int previousIndex();

        public void remove();

        public void set(XSObject object);

        public void add(XSObject object);
    }

    public XSObjectListImpl() {
    }

    public XSObjectListImpl(XSObject[] array, int length) {
    }

    public int getLength();

    public XSObject item(int index);

    public void clearXSObjectList();

    public void addXSObject(XSObject object);

    public void addXSObject(int index, XSObject object);

    @Pure
    public boolean contains(Object value);

    public XSObject get(int index);

    public int size();

    public Iterator<XSObject> iterator();

    public ListIterator<XSObject> listIterator();

    public ListIterator<XSObject> listIterator(int index);

    public Object[] toArray();

    public Object[] toArray(Object[] a);

    private final class XSObjectListIterator implements ListIterator<XSObject> {

        public XSObjectListIterator(int index) {
        }

        @Pure
        public boolean hasNext();

        @SideEffectsOnly("this")
        public XSObject next();

        public boolean hasPrevious();

        public XSObject previous();

        public int nextIndex();

        public int previousIndex();

        public void remove();

        public void set(XSObject o);

        public void add(XSObject o);
    }
}
