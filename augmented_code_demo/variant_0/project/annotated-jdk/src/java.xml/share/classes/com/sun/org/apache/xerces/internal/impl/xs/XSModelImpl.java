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
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import com.sun.org.apache.xerces.internal.impl.Constants;
    @Positive
import com.sun.org.apache.xerces.internal.impl.xs.util.StringListImpl;
    @Positive
import com.sun.org.apache.xerces.internal.impl.xs.util.XSNamedMap4Types;
    @Positive
import com.sun.org.apache.xerces.internal.impl.xs.util.XSNamedMapImpl;
    @Positive
import com.sun.org.apache.xerces.internal.impl.xs.util.XSObjectListImpl;
    @Positive
import com.sun.org.apache.xerces.internal.util.SymbolHash;
    @Positive
import com.sun.org.apache.xerces.internal.util.XMLSymbols;
    @Positive
import com.sun.org.apache.xerces.internal.xs.StringList;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSAttributeDeclaration;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSAttributeGroupDefinition;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSConstants;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSElementDeclaration;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSIDCDefinition;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSModel;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSModelGroupDefinition;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSNamedMap;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSNamespaceItem;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSNamespaceItemList;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSNotationDeclaration;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSObject;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSObjectList;
    @Positive
import com.sun.org.apache.xerces.internal.xs.XSTypeDefinition;
    @Positive
import java.lang.reflect.Array;
    @Positive
import java.util.AbstractList;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import java.util.ListIterator;
    @Positive
import java.util.NoSuchElementException;

    @Positive
@SuppressWarnings("unchecked")
    @Positive
public final class XSModelImpl extends AbstractList<XSNamespaceItem> implements XSModel, XSNamespaceItemList {

    @Positive
    public XSModelImpl(SchemaGrammar[] grammars) {
    @Positive
    }

    @Positive
    public XSModelImpl(SchemaGrammar[] grammars, short s4sVersion) {
    @Positive
    }

    @Positive
    public StringList getNamespaces();

    @Positive
    public XSNamespaceItemList getNamespaceItems();

    @Positive
    public synchronized XSNamedMap getComponents(short objectType);

    @Positive
    public synchronized XSNamedMap getComponentsByNamespace(short objectType, String namespace);

    @Positive
    public XSTypeDefinition getTypeDefinition(String name, String namespace);

    @Positive
    public XSTypeDefinition getTypeDefinition(String name, String namespace, String loc);

    @Positive
    public XSAttributeDeclaration getAttributeDeclaration(String name, String namespace);

    @Positive
    public XSAttributeDeclaration getAttributeDeclaration(String name, String namespace, String loc);

    @Positive
    public XSElementDeclaration getElementDeclaration(String name, String namespace);

    @Positive
    public XSElementDeclaration getElementDeclaration(String name, String namespace, String loc);

    @Positive
    public XSAttributeGroupDefinition getAttributeGroup(String name, String namespace);

    @Positive
    public XSAttributeGroupDefinition getAttributeGroup(String name, String namespace, String loc);

    @Positive
    public XSModelGroupDefinition getModelGroupDefinition(String name, String namespace);

    @Positive
    public XSModelGroupDefinition getModelGroupDefinition(String name, String namespace, String loc);

    @Positive
    public XSIDCDefinition getIDCDefinition(String name, String namespace);

    @Positive
    public XSIDCDefinition getIDCDefinition(String name, String namespace, String loc);

    @Positive
    public XSNotationDeclaration getNotationDeclaration(String name, String namespace);

    @Positive
    public XSNotationDeclaration getNotationDeclaration(String name, String namespace, String loc);

    @Positive
    public synchronized XSObjectList getAnnotations();

    @Positive
    public boolean hasIDConstraints();

    @Positive
    public XSObjectList getSubstitutionGroup(XSElementDeclaration head);

    @Positive
    public int getLength();

    @Positive
    public XSNamespaceItem item(int index);

    @Positive
    public XSNamespaceItem get(int index);

    @Positive
    public int size();

    @Positive
    public Iterator<XSNamespaceItem> iterator();

    @Positive
    public ListIterator<XSNamespaceItem> listIterator();

    @Positive
    public ListIterator<XSNamespaceItem> listIterator(int index);

    @Positive
    public Object[] toArray();

    @Positive
    public Object[] toArray(Object[] a);

    @Positive
    private final class XSNamespaceItemListIterator implements ListIterator<XSNamespaceItem> {

    @Positive
        public XSNamespaceItemListIterator(int index) {
    @Positive
        }

    @Positive
        @Pure
    @Positive
        public boolean hasNext();

    @Positive
        @SideEffectsOnly("this")
    @Positive
        public XSNamespaceItem next();

    @Positive
        public boolean hasPrevious();

    @Positive
        public XSNamespaceItem previous();

    @Positive
        public int nextIndex();

    @Positive
        public int previousIndex();

    @Positive
        public void remove();

    @Positive
        public void set(XSNamespaceItem o);

    @Positive
        public void add(XSNamespaceItem o);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
