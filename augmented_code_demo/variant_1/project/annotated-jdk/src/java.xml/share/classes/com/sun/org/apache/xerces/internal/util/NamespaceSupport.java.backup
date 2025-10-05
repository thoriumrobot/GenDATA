/*
 * Copyright (c) 2017, Oracle and/or its affiliates. All rights reserved.
 */
package com.sun.org.apache.xerces.internal.util;

import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import com.sun.org.apache.xerces.internal.xni.NamespaceContext;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

public class NamespaceSupport implements NamespaceContext {

    protected String[] fNamespace;

    protected int fNamespaceSize;

    protected int[] fContext;

    protected int fCurrentContext;

    protected String[] fPrefixes;

    public NamespaceSupport() {
    }

    public NamespaceSupport(NamespaceContext context) {
    }

    public void reset();

    public void pushContext();

    public void popContext();

    public boolean declarePrefix(String prefix, String uri);

    public String getURI(String prefix);

    public String getPrefix(String uri);

    public int getDeclaredPrefixCount();

    public String getDeclaredPrefixAt(int index);

    public Iterator<String> getPrefixes();

    public Enumeration<String> getAllPrefixes();

    public List<String> getPrefixes(String uri);

    @Pure
    public boolean containsPrefix(String prefix);

    @Pure
    public boolean containsPrefixInCurrentContext(String prefix);

    protected final class IteratorPrefixes implements Iterator<String> {

        public IteratorPrefixes(String[] prefixes, int size) {
        }

        @Pure
        public boolean hasNext();

        @SideEffectsOnly("this")
        public String next();

        public String toString();

        public void remove();
    }

    protected final class Prefixes implements Enumeration<String> {

        public Prefixes(String[] prefixes, int size) {
        }

        public boolean hasMoreElements();

        public String nextElement();

        public String toString();
    }
}
