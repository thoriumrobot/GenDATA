/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2015, Oracle and/or its affiliates. All rights reserved.
    @Positive
 */
    @Positive
package com.sun.org.apache.xerces.internal.util;

    @Positive
import org.checkerframework.dataflow.qual.Pure;

    @Positive
public class SymbolTable {

    @Positive
    protected static final int TABLE_SIZE;

    @Positive
    protected static final int MAX_HASH_COLLISIONS;

    @Positive
    protected static final int MULTIPLIERS_SIZE;

    @Positive
    protected static final int MULTIPLIERS_MASK;

    @Positive
    protected Entry[] fBuckets;

    @Positive
    protected int fTableSize;

    @Positive
    protected transient int fCount;

    @Positive
    protected int fThreshold;

    @Positive
    protected float fLoadFactor;

    @Positive
    protected final int fCollisionThreshold;

    @Positive
    protected int[] fHashMultipliers;

    @Positive
    public SymbolTable(int initialCapacity, float loadFactor) {
    @Positive
    }

    @Positive
    public SymbolTable(int initialCapacity) {
    @Positive
    }

    @Positive
    public SymbolTable() {
    @Positive
    }

    @Positive
    public String addSymbol(String symbol);

    @Positive
    public String addSymbol(char[] buffer, int offset, int length);

    @Positive
    public int hash(String symbol);

    @Positive
    public int hash(char[] buffer, int offset, int length);

    @Positive
    protected void rehash();

    @Positive
    protected void rebalance();

    @Positive
    @Pure
    @Positive
    public boolean containsSymbol(String symbol);

    @Positive
    @Pure
    @Positive
    public boolean containsSymbol(char[] buffer, int offset, int length);

    @Positive
    protected static final class Entry {

    @Positive
        public final String symbol;

    @Positive
        public final char[] characters;

    @Positive
        public Entry next;

    @Positive
        public Entry(String symbol, Entry next) {
    @Positive
        }

    @Positive
        public Entry(char[] ch, int offset, int length, Entry next) {
    @Positive
        }
    @Positive
    }
    @Positive
}
