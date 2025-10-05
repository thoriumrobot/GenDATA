/*
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
 */
package com.sun.org.apache.xerces.internal.impl.xs.util;

import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import com.sun.org.apache.xerces.internal.util.SymbolHash;
import com.sun.org.apache.xerces.internal.xs.XSNamedMap;
import com.sun.org.apache.xerces.internal.xs.XSObject;
import java.util.AbstractMap;
import java.util.AbstractSet;
import java.util.Iterator;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;
import javax.xml.XMLConstants;
import javax.xml.namespace.QName;

public class XSNamedMapImpl extends AbstractMap<QName, XSObject> implements XSNamedMap {

    public static final XSNamedMapImpl EMPTY_MAP;

    public XSNamedMapImpl(String namespace, SymbolHash map) {
    }

    public XSNamedMapImpl(String[] namespaces, SymbolHash[] maps, int num) {
    }

    public XSNamedMapImpl(XSObject[] array, int length) {
    }

    public synchronized int getLength();

    public XSObject itemByName(String namespace, String localName);

    public synchronized XSObject item(int index);

    static boolean isEqual(String one, String two);

    @Pure
    public boolean containsKey(Object key);

    public XSObject get(Object key);

    public int size();

    public synchronized Set<Map.Entry<QName, XSObject>> entrySet();

    private static final class XSNamedMapEntry implements Map.Entry<QName, XSObject> {

        public XSNamedMapEntry(QName key, XSObject value) {
        }

        public QName getKey();

        public XSObject getValue();

        public XSObject setValue(XSObject value);

        public boolean equals(XSNamedMapEntry o);

        public int hashCode();

        public String toString();
    }
}
