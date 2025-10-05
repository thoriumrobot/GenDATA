/*
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
 */
package com.sun.org.apache.xerces.internal.impl.xs;

import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import com.sun.org.apache.xerces.internal.impl.Constants;
import com.sun.org.apache.xerces.internal.impl.xs.util.StringListImpl;
import com.sun.org.apache.xerces.internal.impl.xs.util.XSNamedMap4Types;
import com.sun.org.apache.xerces.internal.impl.xs.util.XSNamedMapImpl;
import com.sun.org.apache.xerces.internal.impl.xs.util.XSObjectListImpl;
import com.sun.org.apache.xerces.internal.util.SymbolHash;
import com.sun.org.apache.xerces.internal.util.XMLSymbols;
import com.sun.org.apache.xerces.internal.xs.StringList;
import com.sun.org.apache.xerces.internal.xs.XSAttributeDeclaration;
import com.sun.org.apache.xerces.internal.xs.XSAttributeGroupDefinition;
import com.sun.org.apache.xerces.internal.xs.XSConstants;
import com.sun.org.apache.xerces.internal.xs.XSElementDeclaration;
import com.sun.org.apache.xerces.internal.xs.XSIDCDefinition;
import com.sun.org.apache.xerces.internal.xs.XSModel;
import com.sun.org.apache.xerces.internal.xs.XSModelGroupDefinition;
import com.sun.org.apache.xerces.internal.xs.XSNamedMap;
import com.sun.org.apache.xerces.internal.xs.XSNamespaceItem;
import com.sun.org.apache.xerces.internal.xs.XSNamespaceItemList;
import com.sun.org.apache.xerces.internal.xs.XSNotationDeclaration;
import com.sun.org.apache.xerces.internal.xs.XSObject;
import com.sun.org.apache.xerces.internal.xs.XSObjectList;
import com.sun.org.apache.xerces.internal.xs.XSTypeDefinition;
import java.lang.reflect.Array;
import java.util.AbstractList;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.NoSuchElementException;

@SuppressWarnings("unchecked")
public final class XSModelImpl extends AbstractList<XSNamespaceItem> implements XSModel, XSNamespaceItemList {

    public XSModelImpl(SchemaGrammar[] grammars) {
    }

    public XSModelImpl(SchemaGrammar[] grammars, short s4sVersion) {
    }

    public StringList getNamespaces();

    public XSNamespaceItemList getNamespaceItems();

    public synchronized XSNamedMap getComponents(short objectType);

    public synchronized XSNamedMap getComponentsByNamespace(short objectType, String namespace);

    public XSTypeDefinition getTypeDefinition(String name, String namespace);

    public XSTypeDefinition getTypeDefinition(String name, String namespace, String loc);

    public XSAttributeDeclaration getAttributeDeclaration(String name, String namespace);

    public XSAttributeDeclaration getAttributeDeclaration(String name, String namespace, String loc);

    public XSElementDeclaration getElementDeclaration(String name, String namespace);

    public XSElementDeclaration getElementDeclaration(String name, String namespace, String loc);

    public XSAttributeGroupDefinition getAttributeGroup(String name, String namespace);

    public XSAttributeGroupDefinition getAttributeGroup(String name, String namespace, String loc);

    public XSModelGroupDefinition getModelGroupDefinition(String name, String namespace);

    public XSModelGroupDefinition getModelGroupDefinition(String name, String namespace, String loc);

    public XSIDCDefinition getIDCDefinition(String name, String namespace);

    public XSIDCDefinition getIDCDefinition(String name, String namespace, String loc);

    public XSNotationDeclaration getNotationDeclaration(String name, String namespace);

    public XSNotationDeclaration getNotationDeclaration(String name, String namespace, String loc);

    public synchronized XSObjectList getAnnotations();

    public boolean hasIDConstraints();

    public XSObjectList getSubstitutionGroup(XSElementDeclaration head);

    public int getLength();

    public XSNamespaceItem item(int index);

    public XSNamespaceItem get(int index);

    public int size();

    public Iterator<XSNamespaceItem> iterator();

    public ListIterator<XSNamespaceItem> listIterator();

    public ListIterator<XSNamespaceItem> listIterator(int index);

    public Object[] toArray();

    public Object[] toArray(Object[] a);

    private final class XSNamespaceItemListIterator implements ListIterator<XSNamespaceItem> {

        public XSNamespaceItemListIterator(int index) {
        }

        @Pure
        public boolean hasNext();

        @SideEffectsOnly("this")
        public XSNamespaceItem next();

        public boolean hasPrevious();

        public XSNamespaceItem previous();

        public int nextIndex();

        public int previousIndex();

        public void remove();

        public void set(XSNamespaceItem o);

        public void add(XSNamespaceItem o);
    }
}
