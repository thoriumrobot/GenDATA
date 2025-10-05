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
import com.sun.org.apache.xerces.internal.util.SymbolHash;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSNamedMap;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSObject;
    @Positive
import java.util.AbstractMap;
    @Positive
import java.util.AbstractSet;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.Map;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import java.util.Set;
    @Positive
import javax.xml.XMLConstants;
    @Positive
import javax.xml.namespace.QName;

    @Positive
public class XSNamedMapImpl extends AbstractMap<QName, XSObject> implements XSNamedMap {

    @Positive
    public static final XSNamedMapImpl EMPTY_MAP;

    @Positive
    public XSNamedMapImpl(String namespace, SymbolHash map) {
    @Positive
    }

    @Positive
    public XSNamedMapImpl(String[] namespaces, SymbolHash[] maps, int num) {
    @Positive
    }

    @Positive
    public XSNamedMapImpl(XSObject[] array, int length) {
    @Positive
    }

    @Positive
    public synchronized int getLength();

    @Positive
    public XSObject itemByName(String namespace, String localName);

    @Positive
    public synchronized XSObject item(int index);

    @Positive
    static boolean isEqual(String one, String two);

    @Positive
    @Pure
    @Positive
    public boolean containsKey(Object key);

    @Positive
    public XSObject get(Object key);

    @Positive
    public int size();

    @Positive
    public synchronized Set<Map.Entry<QName, XSObject>> entrySet();

    @Positive
    private static final class XSNamedMapEntry implements Map.Entry<QName, XSObject> {

    @Positive
        public XSNamedMapEntry(QName key, XSObject value) {
    @Positive
        }

    @Positive
        public QName getKey();

    @Positive
        public XSObject getValue();

    @Positive
        public XSObject setValue(XSObject value);

    @Positive
        public boolean equals(XSNamedMapEntry o);

    @Positive
        public int hashCode();

    @Positive
        public String toString();
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 1
