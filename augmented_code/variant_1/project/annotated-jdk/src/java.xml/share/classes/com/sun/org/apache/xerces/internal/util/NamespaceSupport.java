/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 2017, Oracle and/or its affiliates. All rights reserved.
    @Positive
 */
    @Positive
package com.sun.org.apache.xerces.internal.util;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import com.sun.org.apache.xerces.internal.xni.NamespaceContext;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import java.util.NoSuchElementException;

    @Positive
public class NamespaceSupport implements NamespaceContext {

    @Positive
    protected String[] fNamespace;

    @Positive
    protected int fNamespaceSize;

    @Positive
    protected int[] fContext;

    @Positive
    protected int fCurrentContext;

    @Positive
    protected String[] fPrefixes;

    @Positive
    public NamespaceSupport() {
    @Positive
    }

    @Positive
    public NamespaceSupport(NamespaceContext context) {
    @Positive
    }

    @Positive
    public void reset();

    @Positive
    public void pushContext();

    @Positive
    public void popContext();

    @Positive
    public boolean declarePrefix(String prefix, String uri);

    @Positive
    public String getURI(String prefix);

    @Positive
    public String getPrefix(String uri);

    @Positive
    public int getDeclaredPrefixCount();

    @Positive
    public String getDeclaredPrefixAt(int index);

    @Positive
    public Iterator<String> getPrefixes();

    @Positive
    public Enumeration<String> getAllPrefixes();

    @Positive
    public List<String> getPrefixes(String uri);

    @Positive
    @Pure
    @Positive
    public boolean containsPrefix(String prefix);

    @Positive
    @Pure
    @Positive
    public boolean containsPrefixInCurrentContext(String prefix);

    @Positive
    protected final class IteratorPrefixes implements Iterator<String> {

    @Positive
        public IteratorPrefixes(String[] prefixes, int size) {
    @Positive
        }

    @Positive
        @Pure
    @Positive
        public boolean hasNext();

    @Positive
        @SideEffectsOnly("this")
    @Positive
        public String next();

    @Positive
        public String toString();

    @Positive
        public void remove();
    @Positive
    }

    @Positive
    protected final class Prefixes implements Enumeration<String> {

    @Positive
        public Prefixes(String[] prefixes, int size) {
    @Positive
        }

    @Positive
        public boolean hasMoreElements();

    @Positive
        public String nextElement();

    @Positive
        public String toString();
    @Positive
    }
    @Positive
}
