/*
 * Copyright (c) 2015, Oracle and/or its affiliates. All rights reserved.
 */
package com.sun.org.apache.xerces.internal.util;

import org.checkerframework.dataflow.qual.Pure;

public class SymbolTable {

    protected static final int TABLE_SIZE;

    protected static final int MAX_HASH_COLLISIONS;

    protected static final int MULTIPLIERS_SIZE;

    protected static final int MULTIPLIERS_MASK;

    protected Entry[] fBuckets;

    protected int fTableSize;

    protected transient int fCount;

    protected int fThreshold;

    protected float fLoadFactor;

    protected final int fCollisionThreshold;

    protected int[] fHashMultipliers;

    public SymbolTable(int initialCapacity, float loadFactor) {
    }

    public SymbolTable(int initialCapacity) {
    }

    public SymbolTable() {
    }

    public String addSymbol(String symbol);

    public String addSymbol(char[] buffer, int offset, int length);

    public int hash(String symbol);

    public int hash(char[] buffer, int offset, int length);

    protected void rehash();

    protected void rebalance();

    @Pure
    public boolean containsSymbol(String symbol);

    @Pure
    public boolean containsSymbol(char[] buffer, int offset, int length);

    protected static final class Entry {

        public final String symbol;

        public final char[] characters;

        public Entry next;

        public Entry(String symbol, Entry next) {
        }

        public Entry(char[] ch, int offset, int length, Entry next) {
        }
    }
}
